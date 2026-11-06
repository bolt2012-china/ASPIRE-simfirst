"""EnvironmentAdapter bridging CapApi to SimFirst Custom (A/C modes).

* ``SIMFIRST_MODE=aligned`` (Condition C): ``custom_execution.yaml``
* ``SIMFIRST_MODE=nominal`` (Condition A): ``custom_execution_nominal.yaml``
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from cap_harness.contracts import (
    CameraObservation,
    ExecutionResult,
    Observation,
    Pose,
    RobotAction,
    RobotState,
    StepResult,
    SynchronizedTrajectory,
    TaskContext,
    Trajectory,
)
from cap_harness.errors import ApiError, ErrorCode

from .mode import SimFirstMode, execution_config_path, resolve_mode, resolve_sim_real_root
from .registry import SimFirstTaskMetadata, SimFirstTaskRegistry

EMBODIMENT = "simfirst"
PRIMARY_ARM = "primary"
BASE_FRAME = "robot_base"
CONTROL_HZ = 15.0


def _identity_pose(frame: str = BASE_FRAME) -> Pose:
    return Pose(
        position=np.zeros(3, dtype=np.float64),
        quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        frame=frame,
    )


def _zero_robot_state() -> RobotState:
    zeros7 = np.zeros(7, dtype=np.float64)
    return RobotState(
        joint_positions={PRIMARY_ARM: zeros7},
        joint_velocities={PRIMARY_ARM: zeros7},
        end_effector_poses={PRIMARY_ARM: _identity_pose()},
        gripper_positions={PRIMARY_ARM: 0.0},
        base_frame=BASE_FRAME,
        embodiment=EMBODIMENT,
    )


def _placeholder_camera(name: str = "agentview") -> CameraObservation:
    h, w = 48, 64
    return CameraObservation(
        rgb=np.zeros((h, w, 3), dtype=np.uint8),
        depth_m=np.zeros((h, w), dtype=np.float32),
        intrinsics=np.array([[100.0, 0.0, w / 2.0], [0.0, 100.0, h / 2.0], [0.0, 0.0, 1.0]]),
        frame=name,
        camera_pose=_identity_pose(frame=name),
    )


class SimFirstAdapter:
    """CapApi-facing wrapper over SimFirst Custom execution."""

    embodiment = EMBODIMENT
    registry_prefix = "simfirst"

    def __init__(
        self,
        *,
        mode: SimFirstMode | str | None = None,
        registry: SimFirstTaskRegistry | None = None,
        task_context: TaskContext | None = None,
        device: str | None = None,
        enable_cameras: bool | None = None,
        run_observer: object | None = None,
    ) -> None:
        self._mode = mode if isinstance(mode, SimFirstMode) else resolve_mode(mode)
        self._registry = registry or SimFirstTaskRegistry()
        self._task_context = task_context
        self._device = device or os.environ.get("SIMFIRST_DEVICE") or "cuda:0"
        if enable_cameras is None:
            # Cameras off by default for CapApi smoke; set SIMFIRST_ENABLE_CAMERAS=1 for RGB.
            enable_cameras = os.environ.get("SIMFIRST_ENABLE_CAMERAS", "0") not in {
                "0",
                "false",
                "False",
                "",
            }
        self._enable_cameras = bool(enable_cameras)
        self._run_observer = run_observer
        self._closed = False
        self._seed = 0
        self._metadata: SimFirstTaskMetadata | None = None
        self._sim_real_root = resolve_sim_real_root()
        self._execution_config = execution_config_path(self._mode)
        self._app: Any = None
        self._native_env: Any = None
        self._runtime_error: str | None = None
        self._step_count = 0
        self.control_frequency = CONTROL_HZ
        self.control_period_s = 1.0 / CONTROL_HZ

    @property
    def mode(self) -> SimFirstMode:
        return self._mode

    @property
    def native_env(self) -> Any:
        return self._native_env

    @property
    def arms(self) -> tuple[str, ...]:
        return (PRIMARY_ARM,)

    def resolve_arm(self, arm: str) -> str:
        name = str(arm)
        if name in {"primary", "arm", "robot0", PRIMARY_ARM}:
            return PRIMARY_ARM
        return name

    def reset(self, task_ref: object, seed: int) -> Observation:
        self._ensure_open()
        self._metadata = self._registry.resolve(task_ref)  # type: ignore[arg-type]
        self._seed = int(seed)
        self._step_count = 0
        self._task_context = TaskContext(
            suite=self._metadata.suite_name,
            task_id=self._metadata.task_id,
            task_name=self._metadata.task_name,
            language=self._metadata.language,
            family=self._metadata.family,
            metadata={
                "simfirst_mode": self._mode.name,
                "condition_id": self._mode.condition_id,
                "execution_config": str(self._execution_config),
                "sim_real_root": str(self._sim_real_root),
            },
        )
        self._ensure_runtime()
        if self._native_env is not None and hasattr(self._native_env, "reset"):
            self._native_env.reset()
        return self.get_observation()

    def get_task_context(self) -> TaskContext:
        if self._task_context is None:
            raise ApiError(ErrorCode.INVALID_REQUEST, "SimFirst adapter has no active task")
        return self._task_context

    def get_observation(self) -> Observation:
        self._ensure_open()
        if self._native_env is not None and hasattr(self._native_env, "get_observation"):
            return self._native_env.get_observation()
        return Observation(
            cameras={"agentview": _placeholder_camera()},
            robot_state=self.get_robot_state(),
            task_context=self._task_context,
        )

    def get_robot_state(self) -> RobotState:
        self._ensure_open()
        if self._native_env is not None and hasattr(self._native_env, "get_robot_state"):
            return self._native_env.get_robot_state()
        return _zero_robot_state()

    def step(self, action: RobotAction) -> StepResult:
        self._ensure_open()
        self._step_count += 1
        if self._native_env is not None and hasattr(self._native_env, "step"):
            return self._native_env.step(action)
        raise ApiError(
            ErrorCode.UNSUPPORTED,
            "SimFirst Custom env not attached; finish Isaac 6 install / set SIMFIRST_CUSTOM_ENV_FACTORY",
            details={"runtime_error": self._runtime_error or ""},
        )

    def execute_trajectory(self, trajectory: Trajectory | SynchronizedTrajectory) -> ExecutionResult:
        self._ensure_open()
        if self._native_env is not None and hasattr(self._native_env, "execute_trajectory"):
            return self._native_env.execute_trajectory(trajectory)
        raise ApiError(
            ErrorCode.UNSUPPORTED,
            "SimFirst trajectory execution requires Custom Isaac 6 env",
            details={"runtime_error": self._runtime_error or ""},
        )

    def set_gripper(self, position: float, *, arm: str = "primary") -> ExecutionResult:
        self._ensure_open()
        if self._native_env is not None and hasattr(self._native_env, "set_gripper"):
            return self._native_env.set_gripper(float(position), arm=self.resolve_arm(arm))
        return ExecutionResult(
            ok=True,
            steps_executed=0,
            diagnostics={"noop": True, "gripper": float(position), "mode": self._mode.name},
        )

    def set_grippers(self, positions: Mapping[str, float]) -> ExecutionResult:
        last = ExecutionResult(ok=True, steps_executed=0)
        for arm, value in positions.items():
            last = self.set_gripper(float(value), arm=str(arm))
        return last

    def check_success(self) -> bool:
        """Read the Custom env's privileged success latch (for CapApi finalize)."""
        if self._native_env is not None and hasattr(self._native_env, "check_success"):
            return bool(self._native_env.check_success())
        return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._native_env is not None and hasattr(self._native_env, "close"):
            try:
                self._native_env.close()
            except Exception:
                pass
        self._native_env = None
        if self._app is not None and hasattr(self._app, "close"):
            try:
                self._app.close()
            except Exception:
                pass
        self._app = None

    def _ensure_open(self) -> None:
        if self._closed:
            raise ApiError(ErrorCode.INVALID_REQUEST, "SimFirst adapter is closed")

    def _ensure_runtime(self) -> None:
        if self._native_env is not None or self._app is not None:
            return
        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
        os.environ.setdefault("ACCEPT_EULA", "Y")
        os.environ.setdefault("PRIVACY_CONSENT", "Y")
        try:
            self._native_env = self._try_build_custom_env()
            if self._native_env is not None and getattr(self._native_env, "app", None) is not None:
                self._app = self._native_env.app
        except Exception as exc:
            self._native_env = None
            self._app = None
            self._runtime_error = f"custom_env: {type(exc).__name__}: {exc}"

    def _try_build_custom_env(self) -> Any | None:
        src = self._sim_real_root / "src"
        src_s = str(src)
        if src_s not in os.sys.path:
            os.sys.path.insert(0, src_s)
        provenance = {
            "mode": self._mode.name,
            "condition_id": self._mode.condition_id,
            "execution_config": str(self._execution_config),
            "task": None if self._metadata is None else self._metadata.to_manifest_record(),
            "device": self._device,
        }
        dump = os.environ.get("SIMFIRST_PROVENANCE_PATH")
        if dump:
            path = Path(dump)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
        factory = os.environ.get(
            "SIMFIRST_CUSTOM_ENV_FACTORY",
            "cap_harness.simfirst.custom_env:build_custom_env",
        )
        if factory in {"", "0", "none", "None"}:
            return None
        module_name, _, attr = factory.partition(":")
        import importlib

        module = importlib.import_module(module_name)
        builder = getattr(module, attr)
        return builder(
            task_name=None if self._metadata is None else self._metadata.task_name,
            seed=self._seed,
            mode=self._mode.name,
            execution_config=str(self._execution_config),
            device=self._device,
            app=None,
            enable_cameras=self._enable_cameras,
            task_context=self._task_context,
        )


__all__ = ["EMBODIMENT", "SimFirstAdapter"]
