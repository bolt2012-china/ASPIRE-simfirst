"""CapApi bridge over sim-real IsaacLabTaskEnv (A/C execution configs).

Must run under the Isaac 6 venv with::

    PYTHONPATH=/hy-tmp/work/ASPIRE/src:/hy-tmp/work/vendor/sim-real/src
    SIM_REAL_ROOT=/hy-tmp/work/vendor/sim-real
    SIMFIRST_MODE=aligned|nominal

``AppLauncher`` owns Kit; do not pre-boot a bare ``SimulationApp``.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.contracts import (
    ArmCommand,
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    StepResult,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode

from .mode import resolve_mode, resolve_sim_real_root
from .trajectory import TimedWaypoints, capapi_to_env_gripper, env_to_capapi_gripper
from .trajectory import prepare as prepare_waypoints

EMBODIMENT = "simfirst"
PRIMARY_ARM = "primary"
BASE_FRAME = "robot_base"
TRAJECTORY_TOLERANCE_RAD = 0.02


def _identity_pose(frame: str = BASE_FRAME) -> Pose:
    return Pose(
        position=np.zeros(3, dtype=np.float64),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        frame=frame,
    )


def _placeholder_camera(name: str = "agentview") -> CameraObservation:
    h, w = 48, 64
    return CameraObservation(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=np.zeros((h, w), dtype=np.float32),
        intrinsics=np.array(
            [[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        frame=name,
        camera_pose=_identity_pose(frame=name),
    )


def _ended_error(result: StepResult, message: str) -> ApiError:
    return ApiError(ErrorCode.TERMINATED if result.terminated else ErrorCode.TRUNCATED, message)


def _trajectory_result(
    result: StepResult,
    steps: int,
    goal: np.ndarray,
    timed: TimedWaypoints,
    tracking: float,
    succeeded: bool,
    trace: dict[str, Any] | None = None,
) -> ExecutionResult:
    error = float(np.max(np.abs(result.observation.robot_state.joint_positions[PRIMARY_ARM] - goal)))
    diagnostics: dict[str, Any] = {
        "joint_error_rad": error,
        "max_tracking_error_rad": tracking,
        "waypoints": int(len(timed.joint_positions)),
        "retime_ratio": timed.retime_ratio,
        "task_success": succeeded,
    }
    if trace is not None:
        diagnostics["trace"] = trace
    common = {
        "steps_executed": steps,
        "final_observation": result.observation,
        "terminated": result.terminated,
        "truncated": result.truncated,
        "final_errors": {PRIMARY_ARM: error},
        "diagnostics": diagnostics,
    }
    if result.terminated or result.truncated:
        return ExecutionResult(ok=False, error=_ended_error(result, "episode ended during trajectory execution"), **common)
    if error > TRAJECTORY_TOLERANCE_RAD:
        return ExecutionResult(
            ok=False,
            error=ApiError(ErrorCode.CONTROLLER_FAILED, "arm did not converge to the trajectory goal",
                           details={"joint_error_rad": error, "tolerance_rad": TRAJECTORY_TOLERANCE_RAD}),
            **common,
        )
    return ExecutionResult(ok=True, **common)


class CapApiCustomEnv:
    """Thin CapApi surface over a single-env IsaacLabTaskEnv."""

    def __init__(
        self,
        *,
        native_env: Any,
        app: Any,
        task_context: TaskContext | None,
        mode_name: str,
        execution_config: str,
    ) -> None:
        self._env = native_env
        self.app = app
        self._task_context = task_context
        self._mode_name = mode_name
        self._execution_config = execution_config
        self._closed = False
        self.real_episode: Any = None
        # CapApi convention (1 = open); the env resets with the gripper open.
        self._gripper_open = 1.0
        self._velocity_limits: tuple[float, ...] | None = None
        self._last_success = False

    @property
    def control_period_s(self) -> float:
        return 1.0 / float(self._env.config.policy_hz)

    def velocity_limits(self) -> tuple[float, ...]:
        """Shared physical joint-speed limits (rad/s) enforced by the env guard."""
        if self._velocity_limits is None:
            from sim_real_pipeline.control.physical_limits import physical_limits

            root = resolve_sim_real_root()
            self._velocity_limits = tuple(float(v) for v in physical_limits(str(root))["velocity"])
        return self._velocity_limits

    def object_poses(self) -> dict[str, np.ndarray]:
        """Simulated rigid roots as ``T_world_<role>`` (world = robot base, metres)."""
        from sim_real_pipeline.sim.lab_compat import as_torch, lab_quat_to_wxyz
        from scipy.spatial.transform import Rotation

        env = self._env
        bodies = {"object": env.task_body if env.task_body is not None else env.task}
        if getattr(env, "task_lid_body", None) is not None:
            bodies["lid"] = env.task_lid_body
        origin = as_torch(env.scene.env_origins)[0].detach().cpu().numpy()
        poses = {}
        for role, body in bodies.items():
            if body is None:
                continue
            state = as_torch(body.data.root_state_w)[0]
            w, x, y, z = lab_quat_to_wxyz(state[3:7]).detach().cpu().numpy().astype(np.float64)
            T = np.eye(4)
            T[:3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
            T[:3, 3] = state[:3].detach().cpu().numpy() - origin
            poses[role] = T
        return poses

    def reset(self) -> Observation:
        self._env.reset()
        self._gripper_open = 1.0
        self._last_success = False
        return self.get_observation()

    def get_observation(self) -> Observation:
        snap = self._env.observe(include_images=bool(self._env.config.visual_envs))
        return Observation(
            cameras=self._cameras_from_snapshot(snap),
            robot_state=self.get_robot_state(),
            task_context=self._task_context,
        )

    def get_robot_state(self) -> RobotState:
        env = self._env
        q = env._arm_joint_pos()[0].detach().cpu().numpy().astype(np.float64)
        dq = env.robot.data.joint_vel[0, env._arm_ids].detach().cpu().numpy().astype(np.float64)
        grip = env_to_capapi_gripper(float(env._measured_gripper()[0].detach().cpu().item()))
        # EE from wrist/flange if available; otherwise identity placeholder.
        ee = _identity_pose()
        try:
            # Prefer flange / ee body when present on the articulation.
            body_names = list(env.robot.body_names)
            hints = ("flange", "link7", "ee", "tool")
            idx = None
            for hint in hints:
                matches = [i for i, n in enumerate(body_names) if hint in n.lower()]
                if matches:
                    idx = matches[-1]
                    break
            if idx is not None:
                from sim_real_pipeline.sim.lab_compat import lab_quat_to_wxyz

                pos = env.robot.data.body_pos_w[0, idx].detach().cpu().numpy().astype(np.float64)
                quat = lab_quat_to_wxyz(env.robot.data.body_quat_w[0, idx]).detach().cpu().numpy().astype(np.float64)
                ee = Pose(position=pos, quaternion_wxyz=quat, frame=BASE_FRAME)
        except Exception:
            ee = _identity_pose()
        names = tuple(env.robot.joint_names[int(i)] for i in env._arm_ids.detach().cpu().tolist())
        return RobotState(
            joint_positions={PRIMARY_ARM: q},
            joint_velocities={PRIMARY_ARM: dq},
            end_effector_poses={PRIMARY_ARM: ee},
            gripper_positions={PRIMARY_ARM: grip},
            base_frame=BASE_FRAME,
            joint_names={PRIMARY_ARM: names},
            embodiment=EMBODIMENT,
        )

    def step(self, action: RobotAction) -> StepResult:
        import torch

        cmd = action.arms[PRIMARY_ARM] if PRIMARY_ARM in action.arms else next(iter(action.arms.values()))
        if cmd.gripper_position is not None:
            self._gripper_open = float(cmd.gripper_position)
        grip = capapi_to_env_gripper(self._gripper_open)
        vec = np.concatenate([np.asarray(cmd.target, dtype=np.float32), [np.float32(grip)]])
        tensor = torch.as_tensor(vec, device=self._env.device, dtype=torch.float32).unsqueeze(0)
        result = self._env.step(tensor)
        # DirectRLEnv returns (obs, reward, terminated, truncated, extras)
        if isinstance(result, tuple) and len(result) >= 4:
            reward = result[1]
            terminated = result[2]
            truncated = result[3]
            extras = result[4] if len(result) > 4 else {}
        else:
            reward = getattr(result, "reward", None)
            terminated = getattr(result, "terminated", False)
            truncated = getattr(result, "truncated", False)
            extras = getattr(result, "extras", {})
        term = bool(terminated[0].item()) if hasattr(terminated, "__getitem__") else bool(terminated)
        trunc = bool(truncated[0].item()) if hasattr(truncated, "__getitem__") else bool(truncated)
        rew = float(reward[0].item()) if hasattr(reward, "__getitem__") else (
            None if reward is None else float(reward)
        )
        success = False
        if isinstance(extras, Mapping):
            raw = extras.get("success")
            if raw is not None and hasattr(raw, "__getitem__"):
                success = bool(raw[0].item())
            elif raw is not None:
                success = bool(raw)
        self._last_success = bool(success) or self._last_success
        return StepResult(
            ok=True,
            observation=self.get_observation(),
            terminated=term,
            truncated=trunc,
            reward=rew,
            diagnostics={
                "simfirst_mode": self._mode_name,
                "execution_config": self._execution_config,
                "success": success,
            },
        )

    def check_success(self) -> bool:
        """Privileged task success from the last native extras / latch."""
        if self._last_success:
            return True
        env = self._env
        if env is None:
            return False
        extras = getattr(env, "extras", None)
        if isinstance(extras, Mapping) and "success" in extras:
            raw = extras["success"]
            if hasattr(raw, "__getitem__"):
                return bool(raw[0].item())
            return bool(raw)
        latch = getattr(env, "_success", None)
        if latch is not None and hasattr(latch, "__getitem__"):
            return bool(latch[0].item())
        return False

    def set_gripper(
        self, position: float, *, arm: str = "primary", max_steps: int = 30, settle_steps: int = 3
    ) -> ExecutionResult:
        """Hold the arm and command the gripper until the measured opening stops changing."""
        del arm
        q = self.get_robot_state().joint_positions[PRIMARY_ARM]
        action = RobotAction(arms={PRIMARY_ARM: ArmCommand(
            mode="joint_position", target=q, gripper_position=float(position), embodiment=EMBODIMENT)})
        previous = None
        still = 0
        result = None
        steps = 0
        for steps in range(1, max_steps + 1):
            result = self.step(action)
            if result.terminated or result.truncated:
                return ExecutionResult(
                    ok=False,
                    steps_executed=steps,
                    final_observation=result.observation,
                    terminated=result.terminated,
                    truncated=result.truncated,
                    error=_ended_error(result, "episode ended while moving the gripper"),
                )
            opening = result.observation.robot_state.gripper_positions[PRIMARY_ARM]
            still = still + 1 if previous is not None and abs(opening - previous) < 0.01 else 0
            previous = opening
            if still >= settle_steps:
                break
        return ExecutionResult(
            ok=True,
            steps_executed=steps,
            final_observation=None if result is None else result.observation,
            diagnostics={"gripper_open_measured": previous, "gripper_open_commanded": float(position)},
        )

    def execute_trajectory(self, trajectory: Trajectory) -> ExecutionResult:
        """Step a joint trajectory at the env control period, retimed to the shared speed limits."""
        if not isinstance(trajectory, Trajectory):
            raise ValueError("SimFirst executes single-arm Trajectory objects only")
        if trajectory.embodiment != EMBODIMENT:
            raise ValueError(f"trajectory embodiment {trajectory.embodiment!r} is not {EMBODIMENT!r}")
        if tuple(trajectory.joint_names) != tuple(self.get_robot_state().joint_names[PRIMARY_ARM]):
            raise ValueError("trajectory joint_names do not match the Rizon4 arm")
        timed = prepare_waypoints(
            trajectory.joint_positions,
            trajectory.dt_s,
            self.control_period_s,
            self.velocity_limits(),
            gripper_open=trajectory.gripper_positions,
        )
        return self.execute_waypoints(timed)

    def execute_waypoints(
        self, timed: TimedWaypoints, *, settle_steps: int = 15, record_trace: bool = False
    ) -> ExecutionResult:
        """One env step per waypoint, then hold the goal until it converges or ``settle_steps`` pass."""
        goal = timed.joint_positions[-1]
        tracking = 0.0
        succeeded = False
        result = None
        steps = 0
        q_trace: list[list[float]] = []
        dq_trace: list[list[float]] = []
        grip_trace: list[float] = []
        for index, target in enumerate(timed.joint_positions):
            grip = None if timed.gripper_open is None else float(timed.gripper_open[index])
            result = self.step(RobotAction(arms={PRIMARY_ARM: ArmCommand(
                mode="joint_position", target=target, gripper_position=grip, embodiment=EMBODIMENT)}))
            steps += 1
            succeeded = succeeded or bool((result.diagnostics or {}).get("success"))
            rs = result.observation.robot_state
            measured = rs.joint_positions[PRIMARY_ARM]
            tracking = max(tracking, float(np.max(np.abs(measured - target))))
            if record_trace:
                q_trace.append(np.asarray(measured, dtype=np.float64).tolist())
                dq_trace.append(np.asarray(rs.joint_velocities[PRIMARY_ARM], dtype=np.float64).tolist())
                grip_trace.append(float(rs.gripper_positions[PRIMARY_ARM]))
            if result.terminated or result.truncated:
                trace = {"q": q_trace, "dq": dq_trace, "gripper_open": grip_trace} if record_trace else None
                return _trajectory_result(result, steps, goal, timed, tracking, succeeded, trace)
        hold = RobotAction(arms={PRIMARY_ARM: ArmCommand(
            mode="joint_position", target=goal, gripper_position=None, embodiment=EMBODIMENT)})
        for _ in range(settle_steps):
            error = float(np.max(np.abs(result.observation.robot_state.joint_positions[PRIMARY_ARM] - goal)))
            if error <= TRAJECTORY_TOLERANCE_RAD:
                break
            result = self.step(hold)
            steps += 1
            succeeded = succeeded or bool((result.diagnostics or {}).get("success"))
            if result.terminated or result.truncated:
                break
        trace = {"q": q_trace, "dq": dq_trace, "gripper_open": grip_trace} if record_trace else None
        return _trajectory_result(result, steps, goal, timed, tracking, succeeded, trace)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._env.close()
        except Exception:
            pass
        self._env = None
        # App shutdown is owned by the adapter (SimulationApp.close can kill the interpreter).

    def _cameras_from_snapshot(self, snap: Any) -> dict[str, CameraObservation]:
        cameras: dict[str, CameraObservation] = {}
        images = getattr(snap, "images", {}) or {}
        if not images:
            cameras["agentview"] = _placeholder_camera("agentview")
            return cameras
        for name, tensor in images.items():
            # (N,H,W,3) float/uint8
            frame = tensor[0].detach().cpu().numpy()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            h, w = frame.shape[:2]
            cameras[name] = CameraObservation(
                rgb=frame,
                depth_m=np.zeros((h, w), dtype=np.float32),
                intrinsics=np.array(
                    [[float(w), 0.0, w / 2.0], [0.0, float(w), h / 2.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                ),
                frame=str(name),
                camera_pose=_identity_pose(frame=str(name)),
            )
        return cameras


def build_custom_env(
    *,
    task_name: str | None,
    seed: int,
    mode: str,
    execution_config: str,
    device: str,
    app: Any = None,
    enable_cameras: bool | None = None,
    num_envs: int = 1,
    task_context: TaskContext | None = None,
) -> CapApiCustomEnv:
    """Factory registered via ``SIMFIRST_CUSTOM_ENV_FACTORY``.

    Ignores a pre-built bare ``SimulationApp``; Lab ``AppLauncher`` must own Kit.
    """
    del app  # bare SimulationApp is incompatible with DirectRLEnv ownership
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("PRIVACY_CONSENT", "Y")

    root = resolve_sim_real_root()
    os.chdir(root)
    if str(root / "src") not in os.sys.path:
        os.sys.path.insert(0, str(root / "src"))

    resolved = resolve_mode(mode)
    task = task_name or "press_shampoo"
    cameras = (
        bool(int(os.environ.get("SIMFIRST_ENABLE_CAMERAS", "0")))
        if enable_cameras is None
        else bool(enable_cameras)
    )

    from sim_real_pipeline.core.config import ProjectPaths, load_yaml
    from sim_real_pipeline.sim.isaac_boot import boot_simulation_app
    from sim_real_pipeline.sim.task_plant.isaac_vec_env import IsaacVecEnvConfig
    from sim_real_pipeline.workflows.imitation_learning.custom_runtime import (
        load_execution_config,
    )

    paths = ProjectPaths.discover()
    task_cfg = load_yaml(paths.resolve(f"configs/tasks/{task}/task.yaml"))
    # Ensure type key expected by from_task_config.
    if "type" not in task_cfg:
        task_cfg = dict(task_cfg)
        task_cfg["type"] = task
    overrides = load_execution_config(paths, Path(execution_config))
    # Drop nulls so dataclass defaults remain usable.
    overrides = {k: v for k, v in overrides.items() if v is not None}

    # One real record per Kit process: the scene USD is authored with its poses.
    real_episode = None
    bundle = None
    manifest = os.environ.get("SIMFIRST_REAL_MANIFEST")
    if manifest:
        from sim_real_pipeline.core.config import load_bundle
        from sim_real_pipeline.evaluation.sim_screen import frozen_bundle

        from .real_episodes import episode_for_seed, real_scene_state

        real_episode = episode_for_seed(Path(manifest), task, seed)
        bundle = frozen_bundle(
            load_bundle(paths, f"tasks/{task}/data_generation/custom/config.yaml"),
            real_scene_state(real_episode),
        )
        overrides["arm_reset_noise_rad"] = 0
    if "action" in overrides and isinstance(overrides["action"], Mapping):
        from sim_real_pipeline.policies.action_codec import ActionCodecConfig

        overrides["action"] = ActionCodecConfig.from_dict(overrides["action"])

    cfg = IsaacVecEnvConfig.from_task_config(
        task_cfg,
        {},
        num_envs=int(num_envs),
        visual_envs=int(num_envs) if cameras else 0,
        device=str(device),
        seed=int(seed),
        headless=os.environ.get("SIMFIRST_HEADLESS", "1") not in {"0", "false", "False"},
        strip_cameras=not cameras,
        **overrides,
    )

    kit = boot_simulation_app(
        device=str(device),
        headless=cfg.headless,
        enable_cameras=cameras,
        use_lab_launcher=True,
    )

    if bundle is not None:
        import dataclasses

        from sim_real_pipeline.perception.scene_state import instantiate_scene

        scene_root = Path(os.environ.get("SIMFIRST_REAL_SCENE_DIR", "/hy-tmp/work/scenes/real_episodes"))
        # Per mode: A and C processes may author the same record concurrently.
        scene_dir = scene_root / task / real_episode.episode_id.replace("/", "__") / resolved.name
        cfg = dataclasses.replace(cfg, scene=str(instantiate_scene(paths, bundle, scene_dir)))

    # Import env only after AppLauncher has started Kit.
    from sim_real_pipeline.sim.task_plant.isaac_lab_env import IsaacLabTaskEnv

    native = IsaacLabTaskEnv(cfg, paths=paths, bundle=bundle)
    native._reuse_observations = True
    env = CapApiCustomEnv(
        native_env=native,
        app=kit,
        task_context=task_context,
        mode_name=resolved.name,
        execution_config=str(execution_config),
    )
    env.real_episode = real_episode
    return env


__all__ = ["CapApiCustomEnv", "build_custom_env"]
