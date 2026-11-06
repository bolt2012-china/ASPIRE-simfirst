"""T1–T4 SimFirst contact tasks (press / twist / insert / stapler)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .mode import resolve_sim_real_root


class SimFirstRegistryError(ValueError):
    """Invalid SimFirst task reference."""


@dataclass(frozen=True, slots=True)
class SimFirstTaskMetadata:
    """Public metadata for one SimFirst contact task."""

    task_name: str
    language: str
    task_id: int
    family: str = "simfirst"
    scene_reset: str = "seeded"
    horizon: int = 1000
    task_yaml: str = ""

    @property
    def suite_name(self) -> str:
        return self.task_name

    @property
    def task_ref(self) -> str:
        return f"{self.task_name}:{self.task_id}"

    def to_manifest_record(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "horizon": self.horizon,
            "language": self.language,
            "scene_reset": self.scene_reset,
            "suite_name": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "task_ref": self.task_ref,
            "task_yaml": self.task_yaml,
        }


def _load_task_language(task_name: str) -> tuple[str, int, str]:
    """Read instruction + episode horizon from sim-real task.yaml when available."""
    root = resolve_sim_real_root()
    yaml_path = root / "configs" / "tasks" / task_name / "task.yaml"
    language = task_name.replace("_", " ")
    horizon = 1000
    if yaml_path.is_file():
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        language = str(payload.get("instruction") or payload.get("instruction_detail") or language)
        episode_s = float(payload.get("episode_length_s") or 60.0)
        # CapApi steps ~15 Hz policy tick for Custom; approximate horizon.
        horizon = max(100, int(episode_s * 15))
    return language, horizon, str(yaml_path)


def _task(task_name: str, task_id: int) -> SimFirstTaskMetadata:
    language, horizon, yaml_path = _load_task_language(task_name)
    return SimFirstTaskMetadata(
        task_name=task_name,
        language=language,
        task_id=task_id,
        horizon=horizon,
        task_yaml=yaml_path,
    )


# Fixed order matches experiment T1–T4.
SIMFIRST_TASKS: tuple[SimFirstTaskMetadata, ...] = (
    _task("press_shampoo", 0),
    _task("twist_lid", 1),
    _task("insert_usb", 2),
    _task("open_stapler", 3),
)


class SimFirstTaskRegistry:
    """Resolve only the four contact tasks used in the A/C comparison."""

    def __init__(self, tasks: tuple[SimFirstTaskMetadata, ...] = SIMFIRST_TASKS) -> None:
        self._by_name = {task.task_name: task for task in tasks}
        self._by_id = {task.task_id: task for task in tasks}
        if len(self._by_name) != len(tasks):
            raise SimFirstRegistryError("SimFirst task names must be unique")

    @property
    def available_tasks(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def enumerate_tasks(self) -> tuple[SimFirstTaskMetadata, ...]:
        return tuple(self._by_name.values())

    def resolve(
        self,
        task_ref: str | tuple[str, int] | SimFirstTaskMetadata,
        task_id: int | None = None,
    ) -> SimFirstTaskMetadata:
        if isinstance(task_ref, SimFirstTaskMetadata):
            return task_ref
        if isinstance(task_ref, tuple):
            if len(task_ref) != 2:
                raise SimFirstRegistryError("task tuple must be (task_name, task_id)")
            task_name, tuple_id = task_ref
            if task_id is not None and int(task_id) != int(tuple_id):
                raise SimFirstRegistryError("conflicting SimFirst task ids")
            task_id = int(tuple_id)
        else:
            task_name = str(task_ref)
            if ":" in task_name:
                task_name, encoded = task_name.rsplit(":", 1)
                encoded_id = int(encoded)
                if task_id is not None and int(task_id) != encoded_id:
                    raise SimFirstRegistryError("conflicting SimFirst task ids")
                task_id = encoded_id
        if task_name in self._by_name:
            task = self._by_name[task_name]
            if task_id is not None and int(task_id) != task.task_id:
                raise SimFirstRegistryError(
                    f"task {task_name!r} has id {task.task_id}, got {task_id}"
                )
            return task
        if task_id is not None and int(task_id) in self._by_id and not task_name:
            return self._by_id[int(task_id)]
        # Allow suite=task_name with task_id matching registry.
        if task_id is not None and int(task_id) in self._by_id:
            candidate = self._by_id[int(task_id)]
            if candidate.task_name == task_name or task_name in {"simfirst", "tasks"}:
                return candidate
        raise SimFirstRegistryError(
            f"unknown SimFirst task {task_name!r}; available: {', '.join(self.available_tasks)}"
        )


__all__ = [
    "SIMFIRST_TASKS",
    "SimFirstRegistryError",
    "SimFirstTaskMetadata",
    "SimFirstTaskRegistry",
]
