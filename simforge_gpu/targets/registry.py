"""Registry for v2 agent targets and legacy deterministic targets."""

from __future__ import annotations

from simforge_gpu.targets.base import TargetInfo


_TARGETS: tuple[TargetInfo, ...] = (
    TargetInfo(
        name="py-torch",
        status="primary_agent_target",
        description="Python CPU simulation to PyTorch GPU implementation.",
        requires_external_agent=True,
    ),
    TargetInfo(
        name="r-torch",
        status="primary_agent_target",
        description="R CPU simulation to R torch GPU implementation.",
        requires_external_agent=True,
    ),
    TargetInfo(
        name="cupy",
        status="legacy_deterministic_target",
        description="Existing NumPy-to-CuPy deterministic MVP path.",
        requires_external_agent=False,
    ),
    TargetInfo(
        name="jax",
        status="planned_only",
        description="Roadmap target only.",
        requires_external_agent=True,
    ),
    TargetInfo(
        name="numba-cuda",
        status="planned_only",
        description="Roadmap target only.",
        requires_external_agent=True,
    ),
    TargetInfo(
        name="cudf",
        status="planned_only",
        description="Roadmap target only.",
        requires_external_agent=True,
    ),
)


def list_targets() -> tuple[TargetInfo, ...]:
    return _TARGETS


def get_target(name: str) -> TargetInfo:
    normalized = name.lower()
    aliases = {
        "legacy-cupy": "cupy",
        "torch": "py-torch",
        "numba": "numba-cuda",
    }
    normalized = aliases.get(normalized, normalized)
    for target in _TARGETS:
        if target.name == normalized:
            return target
    names = ", ".join(target.name for target in _TARGETS)
    raise ValueError(f"Unknown target '{name}'. Supported targets: {names}.")

