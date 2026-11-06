"""A/C mode switch for SimFirst execution substrate."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SimFirstMode:
    """Resolved Condition A vs C execution config."""

    name: str  # "aligned" | "nominal"
    condition_id: str  # "C" | "A"
    execution_config: str  # repo-relative yaml under sim-real
    description: str

    @property
    def is_aligned(self) -> bool:
        return self.name == "aligned"


ALIGNED = SimFirstMode(
    name="aligned",
    condition_id="C",
    execution_config="configs/runtime/custom_execution.yaml",
    description="ASPIRE + SimFirst (Custom torque + identified plant/PD)",
)

NOMINAL = SimFirstMode(
    name="nominal",
    condition_id="A",
    execution_config="configs/runtime/custom_execution_nominal.yaml",
    description="ASPIRE Nominal Sim (PhysX joint drive, no plant residual)",
)

_MODE_ALIASES = {
    "aligned": ALIGNED,
    "c": ALIGNED,
    "condition_c": ALIGNED,
    "simfirst": ALIGNED,
    "nominal": NOMINAL,
    "a": NOMINAL,
    "condition_a": NOMINAL,
    "baseline": NOMINAL,
}


def resolve_mode(value: str | None = None) -> SimFirstMode:
    """Resolve mode from explicit value or ``SIMFIRST_MODE`` / ``ASPIRE_SIMFIRST_MODE``."""
    raw = (
        value
        or os.environ.get("SIMFIRST_MODE")
        or os.environ.get("ASPIRE_SIMFIRST_MODE")
        or "aligned"
    )
    key = str(raw).strip().lower()
    try:
        return _MODE_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown SimFirst mode {raw!r}; expected one of {sorted(set(_MODE_ALIASES))}"
        ) from exc


def resolve_sim_real_root() -> Path:
    """Locate the sim-real checkout used as the execution substrate."""
    configured = os.environ.get("SIM_REAL_ROOT") or os.environ.get("PIPELINE_ROOT")
    if configured:
        path = Path(configured).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"SIM_REAL_ROOT does not exist: {path}")
        return path
    # Remote default used by this workspace.
    remote = Path("/hy-tmp/work/vendor/sim-real")
    if remote.is_dir():
        return remote
    # Local vendor checkout next to ASPIRE workspace tooling.
    local = Path(__file__).resolve().parents[4] / "vendor" / "sim-real"
    if local.is_dir():
        return local
    raise FileNotFoundError(
        "sim-real root not found; set SIM_REAL_ROOT=/path/to/sim-real"
    )


def execution_config_path(mode: SimFirstMode | None = None) -> Path:
    mode = mode or resolve_mode()
    root = resolve_sim_real_root()
    path = root / mode.execution_config
    if not path.is_file():
        raise FileNotFoundError(f"execution config missing for mode {mode.name}: {path}")
    return path
