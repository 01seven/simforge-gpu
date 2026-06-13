"""Structured conversion plan schema and factory."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from simforge_gpu.backends.registry import get_backend
from simforge_gpu.ir.schema import AnalysisIR, UnsupportedFeature


@dataclass(frozen=True)
class ConversionPlan:
    summary: str
    source_file: str
    target_backend: str
    detected_patterns: tuple[dict[str, Any], ...]
    gpu_suitability: dict[str, Any]
    backend_status: dict[str, Any]
    changes: tuple[dict[str, str], ...]
    validation_strategy: str
    equivalence_level: str
    risks: tuple[str, ...]
    unsupported_features: tuple[UnsupportedFeature, ...]
    source_intent: dict[str, str] | None = None
    model_advisory: dict[str, Any] | None = None
    validation_notes: tuple[dict[str, str], ...] = ()
    model_risk_notes: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = _json_ready(asdict(self))
        for optional_field in (
            "source_intent",
            "model_advisory",
            "validation_notes",
            "model_risk_notes",
        ):
            if payload.get(optional_field) in (None, [], {}):
                payload.pop(optional_field, None)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def create_conversion_plan(
    ir: AnalysisIR,
    source_file: str,
    target_backend: str = "cupy",
    changes: tuple[dict[str, str], ...] = (),
    unsupported_features: tuple[UnsupportedFeature, ...] = (),
) -> ConversionPlan:
    backend = get_backend(target_backend)
    all_unsupported_features: list[UnsupportedFeature] = list(ir.unsupported_features)
    all_unsupported_features.extend(unsupported_features)
    if not backend.is_implemented:
        all_unsupported_features.append(
            UnsupportedFeature(
                code=f"--target {backend.name}",
                reason=backend.unsupported_reason(),
                action="Use --target cupy for the current supported backend.",
                category="backend_not_implemented",
                source="backend_policy",
            )
        )

    validation_strategy = "stochastic" if ir.random_calls else "deterministic"
    equivalence_level = "statistical" if ir.random_calls else "numerical"
    risks = (
        "CPU and GPU random numbers will not match elementwise.",
        "Floating point differences may occur.",
    ) if ir.random_calls else ("Floating point differences may occur.",)

    return ConversionPlan(
        summary="Convert NumPy simulation code to CuPy where safe.",
        source_file=source_file,
        target_backend=target_backend,
        detected_patterns=tuple(
            {
                "name": pattern,
                "parallelizable": pattern != "unknown",
                "reason": "Detected by static analysis.",
            }
            for pattern in ir.patterns
        ),
        gpu_suitability=ir.gpu_suitability.to_dict(),
        backend_status={
            "selected_backend": backend.name,
            "implemented": backend.is_implemented,
            "status": backend.status,
            "reason": backend.reason,
        },
        changes=changes,
        validation_strategy=validation_strategy,
        equivalence_level=equivalence_level,
        risks=risks,
        unsupported_features=tuple(_dedupe_unsupported(all_unsupported_features)),
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    return value


def _dedupe_unsupported(
    features: list[UnsupportedFeature],
) -> tuple[UnsupportedFeature, ...]:
    seen: set[tuple[str, str]] = set()
    deduped: list[UnsupportedFeature] = []
    for feature in features:
        key = (feature.code, feature.reason)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(feature)
    return tuple(deduped)
