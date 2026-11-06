"""Index of real-robot SimFirst recordings used to seed sim episodes.

Each record is one real attempt under ``real_ops/experiment/data_generation``.
Object poses are the real perception output (FoundationPose + hand-eye),
``T_world_object`` with world = robot base, column vectors, metres:
``p_world = T_world_object @ p_object``. They are estimates, not ground truth.

Build:
    python -m cap_harness.simfirst.real_episodes build <data_generation_dir> <manifest.jsonl>
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

TASKS = ("press_shampoo", "twist_lid", "insert_usb", "open_stapler")
SE3_ORTHO_TOL = 1e-3


@dataclass(frozen=True)
class RealEpisode:
    task: str
    episode_id: str
    group: str
    real_success: bool | None
    objects: dict[str, dict[str, Any]]
    robot_q: tuple[float, ...] | None
    gripper_width_m: float | None
    flags: tuple[str, ...]
    raw: dict[str, Any]

    def T_world_object(self, role: str = "object") -> np.ndarray:
        return np.asarray(self.objects[role]["T_world_object"], dtype=np.float64)

    @property
    def usable(self) -> bool:
        return not any(flag.startswith("bad_") for flag in self.flags)

    def load_trajectory(self, data_generation: Path) -> dict[str, Any] | None:
        """Real sim-real NRT ``trajectory.json`` (15 Hz, gripper 0 = open), if recorded."""
        rel = self.raw.get("trajectory")
        if not rel:
            return None
        return json.loads((Path(data_generation) / rel).read_text(encoding="utf-8"))

    def load_real_frames(self, data_generation: Path) -> list[dict[str, Any]] | None:
        """Per-step real execution trace ``real/frames.jsonl`` (15 Hz), if recorded."""
        path = Path(data_generation) / self.episode_id / "real" / "frames.jsonl"
        if not path.is_file():
            return None
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _se3_ok(matrix: Any) -> bool:
    try:
        T = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        return False
    R = T[:3, :3]
    return (
        np.allclose(R.T @ R, np.eye(3), atol=SE3_ORTHO_TOL)
        and abs(np.linalg.det(R) - 1.0) < SE3_ORTHO_TOL
        and np.allclose(T[3], [0, 0, 0, 1], atol=1e-9)
    )


def _task_of(attempt: dict[str, Any] | None, scene: dict[str, Any] | None, rel: str) -> str | None:
    for source in (attempt or {}, scene or {}):
        if source.get("task") in TASKS:
            return source["task"]
    return next((task for task in TASKS if rel.startswith(task)), None)


def _object_record(role: str, item: dict[str, Any], calibrations: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    flags: list[str] = []
    T = item.get("T_world_object")
    if not _se3_ok(T):
        flags.append(f"bad_se3:{role}")
    camera = item.get("source_camera")
    verdict = (calibrations.get(camera) or {}).get("verdict") if isinstance(camera, str) else None
    if verdict and verdict != "PASS":
        flags.append(f"calib_{verdict.lower()}:{role}:{camera}")
    dual = item.get("dual_pose_difference") or {}
    if dual.get("translation_mm", 0.0) > 20.0 or dual.get("rotation_deg", 0.0) > 10.0:
        flags.append(f"dual_disagree:{role}")
    resting = item.get("resting_projection") or {}
    return (
        {
            "T_world_object": T,
            "asset": item.get("asset"),
            "source_camera": camera,
            "source_calibration_verdict": verdict,
            "dual_pose_difference": dual or None,
            "resting_correction_deg": resting.get("correction_deg"),
            "resting_height_change_mm": resting.get("height_change_mm"),
            "support_z_m": resting.get("support_z_m"),
        },
        flags,
    )


def iter_attempt_dirs(root: Path) -> Iterator[Path]:
    seen: set[Path] = set()
    for marker in sorted(root.rglob("attempt.json")):
        directory = marker.parent
        if directory not in seen:
            seen.add(directory)
            yield directory


def build_record(directory: Path, root: Path) -> dict[str, Any] | None:
    rel = directory.relative_to(root).as_posix()
    attempt = _load(directory / "attempt.json")
    scene = _load(directory / "scene_state.json")
    outcome = _load(directory / "real" / "outcome.json")
    initial = _load(directory / "real" / "initial_state.json")
    task = _task_of(attempt, scene, rel)
    if task is None:
        return None
    flags: list[str] = []
    if scene is None:
        flags.append("bad_missing_scene_state")
    elif scene.get("frame") != "world":
        flags.append(f"bad_frame:{scene.get('frame')}")
    calibrations = (scene or {}).get("calibrations") or {}
    objects: dict[str, dict[str, Any]] = {}
    for role, item in ((scene or {}).get("objects") or {}).items():
        record, object_flags = _object_record(role, item, calibrations)
        objects[role] = record
        flags.extend(object_flags)
    if scene is not None and not objects:
        flags.append("bad_no_objects")
    if task == "insert_usb" and objects and "laptop" not in objects:
        flags.append("bad_missing_laptop")
    robot = (scene or {}).get("robot_initial") or {}
    q = robot.get("q") or ((initial or {}).get("observation") or {}).get("q")
    if q is not None and len(q) != 7:
        flags.append("bad_robot_q_dim")
    real_success = None
    if outcome is not None and "success" in outcome:
        real_success = bool(outcome["success"])
    elif attempt is not None and attempt.get("status") in {"success", "failed", "failure"}:
        real_success = attempt["status"] == "success"
        flags.append("success_from_attempt_status")
    else:
        flags.append("no_real_outcome")
    inhand = (attempt or {}).get("inhand_correction") or {}
    if inhand.get("applied"):
        flags.append("real_used_inhand_correction")
    trajectory = directory / "trajectory.json"
    if not trajectory.is_file():
        flags.append("no_trajectory")
    return {
        "task": task,
        "episode_id": rel,
        "group": rel.split("/")[0],
        "real_success": real_success,
        "real_success_source": (outcome or {}).get("source"),
        "attempt_status": (attempt or {}).get("status"),
        "execution_mode": (attempt or {}).get("execution_mode"),
        "pose_processing": (scene or {}).get("pose_processing"),
        "objects": objects,
        "robot_q": q,
        "gripper_width_m": robot.get("gripper_width_m"),
        "calibration_versions": {
            name: (cal or {}).get("calibrated_at") for name, cal in calibrations.items()
        },
        "calibration_verdicts": {name: (cal or {}).get("verdict") for name, cal in calibrations.items()},
        "trajectory": f"{rel}/trajectory.json" if trajectory.is_file() else None,
        "flags": sorted(set(flags)),
    }


def build_manifest(root: Path, out: Path) -> dict[str, Any]:
    records = [record for d in iter_attempt_dirs(root) if (record := build_record(d, root)) is not None]
    records.sort(key=lambda r: (r["task"], r["episode_id"]))
    # Early recordings used older assets (e.g. laptop2); the sim scene only has the current one.
    assets: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        for role, item in record["objects"].items():
            assets[(record["task"], role)][item["asset"]] += 1
    for record in records:
        for role, item in record["objects"].items():
            if item["asset"] != assets[(record["task"], role)].most_common(1)[0][0]:
                record["flags"] = sorted({*record["flags"], f"bad_legacy_asset:{role}"})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return summarize(records)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["task"]].append(record)
    for task, items in sorted(grouped.items()):
        usable = [r for r in items if not any(f.startswith("bad_") for f in r["flags"])]
        flag_counts = Counter(f.split(":")[0] for r in items for f in r["flags"])
        cameras = Counter(
            (o.get("source_camera") or "?") for r in usable for o in r["objects"].values()
        )
        per_task[task] = {
            "records": len(items),
            "usable": len(usable),
            "real_success": Counter(str(r["real_success"]) for r in usable),
            "groups": Counter(r["group"].split("_1")[0] for r in items),
            "source_camera": cameras,
            "flags": flag_counts,
        }
    return per_task


def load_manifest(path: Path, task: str | None = None, *, usable_only: bool = True) -> list[RealEpisode]:
    episodes: list[RealEpisode] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        if task is not None and raw["task"] != task:
            continue
        episode = RealEpisode(
            task=raw["task"],
            episode_id=raw["episode_id"],
            group=raw["group"],
            real_success=raw["real_success"],
            objects=raw["objects"],
            robot_q=tuple(raw["robot_q"]) if raw.get("robot_q") else None,
            gripper_width_m=raw.get("gripper_width_m"),
            flags=tuple(raw["flags"]),
            raw=raw,
        )
        if usable_only and not episode.usable:
            continue
        episodes.append(episode)
    return episodes


def real_scene_state(episode: RealEpisode) -> dict[str, Any]:
    """sim-real ``scene_state`` (schema 1, world = robot base) from a manifest record."""
    return {
        "schema_version": 1,
        "frame": "world",
        "task": episode.task,
        "objects": {
            role: {
                "asset": item["asset"],
                "source_camera": item["source_camera"],
                "T_world_object": item["T_world_object"],
            }
            for role, item in episode.objects.items()
        },
    }


def episode_for_seed(path: Path, task: str, seed: int) -> RealEpisode:
    episodes = load_manifest(path, task)
    if not episodes:
        raise LookupError(f"no usable real episodes for task {task!r} in {path}")
    return episodes[int(seed) % len(episodes)]


def main(argv: list[str]) -> int:
    if len(argv) != 3 or argv[0] != "build":
        print(__doc__)
        return 2
    summary = build_manifest(Path(argv[1]), Path(argv[2]))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=dict))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
