"""SimFirst (sim-real Custom) execution substrate for ASPIRE A/C comparison.

Condition C (aligned): ``SIMFIRST_MODE=aligned`` → Custom torque + identified plant.
Condition A (nominal): ``SIMFIRST_MODE=nominal`` → PhysX implicit PD, no plant residual.

Same T1–T4 tasks and CapApi surface; only the execution contract changes.
"""

from .mode import SimFirstMode, resolve_mode
from .registry import SIMFIRST_TASKS, SimFirstTaskMetadata, SimFirstTaskRegistry

__all__ = [
    "SIMFIRST_TASKS",
    "SimFirstMode",
    "SimFirstTaskMetadata",
    "SimFirstTaskRegistry",
    "resolve_mode",
]

# CapApiCustomEnv / build_custom_env are imported lazily so libero-only
# interpreters can resolve modes without Isaac.
