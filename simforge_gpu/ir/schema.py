"""Lightweight serializable schemas for SimForge GPU planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class UnsupportedFeature:
    code: str
    reason: str
    action: str
    category: str = "unsupported"
    source: str = "static_detector"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ConvertibleRegion:
    type: str
    line_start: int
    line_end: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GpuSuitability:
    gpu_suitable: bool = False
    confidence: str = "unknown"
    recommended_backend: str = "cupy"
    future_backend_candidates: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))


@dataclass(frozen=True)
class AnalysisIR:
    language: str = "python"
    imports: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    random_calls: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    convertible_regions: tuple[ConvertibleRegion, ...] = ()
    unsupported_features: tuple[UnsupportedFeature, ...] = ()
    gpu_suitability: GpuSuitability = field(default_factory=GpuSuitability)

    def to_dict(self) -> dict[str, Any]:
        return _json_ready(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value
