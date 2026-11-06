import numpy as np
import pytest

from cap_harness.simfirst.trajectory import (
    compare_with_real_frames,
    from_nrt,
    prepare,
    resample,
    retime_for_velocity,
)

LIMITS = [2.0] * 7


def test_retime_keeps_dt_when_under_limit():
    positions = np.zeros((5, 7))
    positions[:, 0] = np.linspace(0.0, 0.4, 5)
    dt, ratio = retime_for_velocity(positions, 0.1, LIMITS)
    assert dt == pytest.approx(0.1)
    assert ratio == pytest.approx(1.0)


def test_retime_stretches_fast_segments_to_margin():
    positions = np.zeros((2, 7))
    positions[1, 3] = 0.36  # 3.6 rad/s at dt 0.1, limit 2.0 * 0.9
    dt, ratio = retime_for_velocity(positions, 0.1, LIMITS)
    assert dt == pytest.approx(0.2)
    assert ratio == pytest.approx(2.0)


def test_resample_hits_goal_and_holds_gripper():
    positions = np.zeros((3, 7))
    positions[:, 0] = [0.0, 1.0, 2.0]
    out, grip, phase = resample(positions, 0.1, 1 / 15, np.array([1.0, 0.0, 0.0]), ["a", "b", "c"])
    assert out[0, 0] == pytest.approx(0.0)
    assert out[-1, 0] == pytest.approx(2.0)
    assert np.all(np.diff(out[:, 0]) > 0)
    assert grip[0] == 1.0 and grip[-1] == 0.0
    assert phase[0] == "a" and phase[-1] == "c"


def test_prepare_never_exceeds_limits_after_resampling():
    rng = np.random.default_rng(0)
    positions = np.cumsum(rng.normal(scale=0.2, size=(40, 7)), axis=0)
    timed = prepare(positions, 0.02, 1 / 15, LIMITS)
    speeds = np.abs(np.diff(timed.joint_positions, axis=0)).max(axis=0) / timed.dt_s
    assert np.all(speeds <= 0.9 * 2.0 + 1e-9)
    assert np.allclose(timed.joint_positions[-1], positions[-1])


def test_from_nrt_inverts_gripper_convention():
    data = {
        "action_type": "absolute_joint_position",
        "gripper_convention": "0_open_1_closed",
        "control_hz": 15,
        "joint_names": [f"joint{i}" for i in range(1, 8)],
        "q_target": [[0.0] * 7, [0.1] * 7],
        "gripper_target": [0.0, 1.0],
        "phase": ["approach", "close"],
    }
    parsed = from_nrt(data)
    assert parsed["dt_s"] == pytest.approx(1 / 15)
    assert parsed["gripper_open"].tolist() == [1.0, 0.0]


def test_compare_with_real_frames_computes_joint_and_tcp_errors():
    sim_q = np.zeros((3, 7))
    sim_q[1] = 0.01  # 10 mrad on all joints at step 1
    frames = [
        {"requested": {"source_index": 0}, "observation": {"q": [0.0] * 7, "gripper": 0.0}},
        {"requested": {"source_index": 1}, "observation": {"q": [0.0] * 7, "gripper": 1.0}},
    ]
    fk = lambda q: np.block([[np.eye(3), np.array([[q[0]], [0.0], [0.0]])], [np.zeros((1, 3)), np.ones((1, 1))]])
    rep = compare_with_real_frames(sim_q, frames, sim_gripper_open=np.array([1.0, 0.0, 0.0]), fk_fn=fk)
    assert rep["paired_frames"] == 2
    assert rep["q_peak_abs_mrad"] == pytest.approx(10.0)
    assert rep["gripper_rmse"] == pytest.approx(0.0)
    assert rep["tcp_trans_max_mm"] == pytest.approx(10.0)



def test_compare_with_real_frames_computes_joint_and_tcp_errors():
    sim_q = np.zeros((3, 7))
    sim_q[1] = 0.01  # 10 mrad on all joints at step 1
    frames = [
        {"requested": {"source_index": 0}, "observation": {"q": [0.0] * 7, "gripper": 0.0}},
        {"requested": {"source_index": 1}, "observation": {"q": [0.0] * 7, "gripper": 1.0}},
    ]
    fk = lambda q: np.block([[np.eye(3), np.array([[q[0]], [0.0], [0.0]])], [np.zeros((1, 3)), np.ones((1, 1))]])
    rep = compare_with_real_frames(sim_q, frames, sim_gripper_open=np.array([1.0, 0.0, 0.0]), fk_fn=fk)
    assert rep["paired_frames"] == 2
    assert rep["q_peak_abs_mrad"] == pytest.approx(10.0)
    assert rep["gripper_rmse"] == pytest.approx(0.0)
    assert rep["tcp_trans_max_mm"] == pytest.approx(10.0)

