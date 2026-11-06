"""Pure-numpy trajectory timing for SimFirst execution (no Isaac imports).

Joint positions are radians; times are seconds. Gripper values use the CapApi
convention (1 = open, 0 = closed) unless a function says otherwise; sim-real
NRT files and the Isaac env use the opposite (0 = open, 1 = closed).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# Shared-limit guard in the env trips at the physical limit; leave headroom.
VELOCITY_MARGIN = 0.9


@dataclass(frozen=True)
class TimedWaypoints:
    """Fixed-period joint stream ready to be stepped one waypoint per control tick."""

    joint_positions: np.ndarray  # (N, dof)
    gripper_open: np.ndarray | None  # (N,), CapApi convention
    dt_s: float
    retime_ratio: float
    phase: tuple[str, ...] | None = None


def capapi_to_env_gripper(openness: float) -> float:
    return 1.0 - float(openness)


def env_to_capapi_gripper(closedness: float) -> float:
    return 1.0 - float(closedness)


def retime_for_velocity(
    positions: np.ndarray, dt_s: float, velocity_limits: Sequence[float], *, margin: float = VELOCITY_MARGIN
) -> tuple[float, float]:
    """Smallest dt >= ``dt_s`` whose finite-difference joint speeds stay under ``margin * limit``."""
    positions = np.asarray(positions, dtype=np.float64)
    limits = np.asarray(velocity_limits, dtype=np.float64) * float(margin)
    if positions.ndim != 2 or positions.shape[1] != limits.shape[0]:
        raise ValueError("velocity limits must match the joint dimension")
    if len(positions) < 2:
        return float(dt_s), 1.0
    steps = np.abs(np.diff(positions, axis=0)).max(axis=0)
    needed = float(np.max(steps / limits))
    new_dt = max(float(dt_s), needed)
    return new_dt, new_dt / float(dt_s)


def resample(
    positions: np.ndarray,
    source_dt_s: float,
    target_dt_s: float,
    gripper: np.ndarray | None = None,
    phase: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...] | None]:
    """Linear joint interpolation onto a ``target_dt_s`` grid; gripper and phase are held.

    The final waypoint is always kept so the stream ends exactly at the planned goal.
    """
    positions = np.asarray(positions, dtype=np.float64)
    count = len(positions)
    if count == 0:
        raise ValueError("trajectory is empty")
    duration = (count - 1) * float(source_dt_s)
    samples = int(np.floor(duration / float(target_dt_s) + 1e-9)) + 1
    times = np.arange(samples) * float(target_dt_s)
    if times[-1] < duration - 1e-9:
        times = np.append(times, duration)
    source_times = np.arange(count) * float(source_dt_s)
    out = np.stack([np.interp(times, source_times, positions[:, j]) for j in range(positions.shape[1])], axis=1)
    held = np.clip(np.searchsorted(source_times, times + 1e-9, side="right") - 1, 0, count - 1)
    grip = None if gripper is None else np.asarray(gripper, dtype=np.float64)[held]
    phases = None if phase is None else tuple(phase[i] for i in held)
    return out, grip, phases


def prepare(
    positions: np.ndarray,
    dt_s: float,
    control_period_s: float,
    velocity_limits: Sequence[float],
    *,
    gripper_open: np.ndarray | None = None,
    phase: Sequence[str] | None = None,
) -> TimedWaypoints:
    """Retime to respect joint speed limits, then resample to the env control period."""
    retimed_dt, ratio = retime_for_velocity(positions, dt_s, velocity_limits)
    if np.isclose(retimed_dt, control_period_s, rtol=0.0, atol=1e-9):
        joints = np.asarray(positions, dtype=np.float64)
        grip = None if gripper_open is None else np.asarray(gripper_open, dtype=np.float64)
        phases = None if phase is None else tuple(phase)
    else:
        joints, grip, phases = resample(positions, retimed_dt, control_period_s, gripper_open, phase)
    return TimedWaypoints(joints, grip, float(control_period_s), float(ratio), phases)


def from_nrt(data: Mapping[str, Any]) -> dict[str, Any]:
    """sim-real NRT trajectory JSON -> positions, dt and CapApi-convention gripper."""
    if data.get("action_type") != "absolute_joint_position":
        raise ValueError("NRT trajectory must be absolute_joint_position")
    if data.get("gripper_convention", "0_open_1_closed") != "0_open_1_closed":
        raise ValueError("unexpected NRT gripper convention")
    positions = np.asarray(data["q_target"], dtype=np.float64)
    closed = np.asarray(data["gripper_target"], dtype=np.float64)
    return {
        "joint_names": tuple(data["joint_names"]),
        "joint_positions": positions,
        "dt_s": 1.0 / float(data["control_hz"]),
        "gripper_open": 1.0 - closed,
        "phase": tuple(data.get("phase") or ()),
    }


def _se3_difference(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two 4x4 SE(3) matrices."""
    trans_mm = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0)
    cos = float(np.clip((np.trace(T_a[:3, :3].T @ T_b[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))
    return trans_mm, float(np.degrees(np.arccos(cos)))


def compare_with_real_frames(
    sim_q: np.ndarray,
    real_frames: Sequence[Mapping[str, Any]],
    *,
    sim_gripper_open: np.ndarray | None = None,
    fk_fn: Any = None,
) -> dict[str, Any]:
    """Compare per-waypoint simulated states against ``real/frames.jsonl`` rows aligned by ``source_index``.

    ``real_frames`` use the env/NRT gripper convention (0 = open, 1 = closed);
    ``sim_gripper_open`` uses the CapApi convention (1 = open, 0 = closed).
    """
    sim_q = np.asarray(sim_q, dtype=np.float64)
    pairs: list[tuple[int, Mapping[str, Any]]] = []
    for row in real_frames:
        idx = (row.get("requested") or {}).get("source_index")
        if idx is not None and 0 <= int(idx) < len(sim_q):
            pairs.append((int(idx), row))
    if not pairs:
        return {"paired_frames": 0}

    dq = np.asarray([sim_q[idx] - np.asarray(r["observation"]["q"], dtype=np.float64) for idx, r in pairs])
    q_rmse_per_joint = np.sqrt(np.mean(dq**2, axis=0))
    q_max_per_joint = np.max(np.abs(dq), axis=0)
    out: dict[str, Any] = {
        "paired_frames": len(pairs),
        "q_rmse_per_joint_mrad": np.round(q_rmse_per_joint * 1000.0, 3).tolist(),
        "q_rmse_mean_mrad": round(float(np.mean(q_rmse_per_joint) * 1000.0), 3),
        "q_rmse_max_mrad": round(float(np.max(q_rmse_per_joint) * 1000.0), 3),
        "q_peak_abs_mrad": round(float(np.max(q_max_per_joint) * 1000.0), 3),
    }
    if sim_gripper_open is not None:
        sim_closed = 1.0 - np.asarray(sim_gripper_open, dtype=np.float64)
        dg = np.asarray([sim_closed[idx] - float(r["observation"]["gripper"]) for idx, r in pairs])
        out["gripper_rmse"] = round(float(np.sqrt(np.mean(dg**2))), 4)

    if fk_fn is not None:
        trans_mm: list[float] = []
        rot_deg: list[float] = []
        for idx, r in pairs:
            T_sim = np.asarray(fk_fn(sim_q[idx]), dtype=np.float64)
            T_real = np.asarray(fk_fn(np.asarray(r["observation"]["q"], dtype=np.float64)), dtype=np.float64)
            t_err, r_err = _se3_difference(T_sim, T_real)
            trans_mm.append(t_err)
            rot_deg.append(r_err)
        out["tcp_trans_mean_mm"] = round(float(np.mean(trans_mm)), 3)
        out["tcp_trans_p95_mm"] = round(float(np.percentile(trans_mm, 95)), 3)
        out["tcp_trans_max_mm"] = round(float(np.max(trans_mm)), 3)
        out["tcp_rot_mean_deg"] = round(float(np.mean(rot_deg)), 3)
        out["tcp_rot_p95_deg"] = round(float(np.percentile(rot_deg, 95)), 3)
        out["tcp_rot_max_deg"] = round(float(np.max(rot_deg)), 3)
    return out

