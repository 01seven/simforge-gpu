"""Schema objects for external model suggestions.

Model suggestions are advisory input. This module only parses and validates the
shape of that input; it does not execute code, import GPU libraries, or call any
model provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from simforge_gpu.reporters.json_report import stable_json


SUPPORTED_SCHEMA_VERSION = "1.0"
MVP_FIT_VALUES = ("yes", "partial", "no", "unknown")
CONFIDENCE_VALUES = ("low", "medium", "high")
VALIDATION_TYPES = ("deterministic", "stochastic", "skipped")

REQUIRED_FIELDS = (
    "schema_version",
    "source_file",
    "source_intent",
    "suggested_backend",
    "mvp_fit",
    "confidence",
    "risks",
    "unsupported_hypotheses",
    "recommended_validation",
)


class SuggestionSchemaError(ValueError):
    """Raised when model_suggestion.json does not match the supported schema."""


@dataclass(frozen=True)
class ModelSuggestion:
    schema_version: str
    source_file: str
    source_intent: str
    suggested_backend: str
    mvp_fit: str
    confidence: str
    risks: list[object]
    unsupported_hypotheses: list[object]
    recommended_validation: dict[str, object]
    optional_fields: dict[str, object] = field(default_factory=dict)
    raw: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "source_file": self.source_file,
            "source_intent": self.source_intent,
            "suggested_backend": self.suggested_backend,
            "mvp_fit": self.mvp_fit,
            "confidence": self.confidence,
            "risks": self.risks,
            "unsupported_hypotheses": self.unsupported_hypotheses,
            "recommended_validation": self.recommended_validation,
        }
        payload.update(self.optional_fields)
        return payload

    def to_json(self) -> str:
        return stable_json(self.to_dict())


def parse_model_suggestion_json(payload: str) -> ModelSuggestion:
    try:
        raw = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SuggestionSchemaError(f"Invalid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise SuggestionSchemaError("Model suggestion must be a JSON object.")

    for field_name in REQUIRED_FIELDS:
        if field_name not in raw:
            raise SuggestionSchemaError(f"Missing required field: {field_name}")

    schema_version = _required_string(raw, "schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        raise SuggestionSchemaError(f"Unsupported schema_version: {schema_version}")

    source_file = _required_string(raw, "source_file")
    source_intent = _required_string(raw, "source_intent")
    suggested_backend = _required_string(raw, "suggested_backend").lower()
    mvp_fit = _enum_value(raw, "mvp_fit", MVP_FIT_VALUES)
    confidence = _enum_value(raw, "confidence", CONFIDENCE_VALUES)
    risks = _required_list(raw, "risks")
    unsupported_hypotheses = _required_list(raw, "unsupported_hypotheses")
    recommended_validation = _recommended_validation(raw["recommended_validation"])
    optional_fields = {
        key: value for key, value in raw.items() if key not in REQUIRED_FIELDS
    }

    return ModelSuggestion(
        schema_version=schema_version,
        source_file=source_file,
        source_intent=source_intent,
        suggested_backend=suggested_backend,
        mvp_fit=mvp_fit,
        confidence=confidence,
        risks=risks,
        unsupported_hypotheses=unsupported_hypotheses,
        recommended_validation=recommended_validation,
        optional_fields=optional_fields,
        raw=raw,
    )


def _required_string(payload: dict[str, object], field_name: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str) or not value.strip():
        raise SuggestionSchemaError(f"{field_name} must be a non-empty string.")
    return value.strip()


def _required_list(payload: dict[str, object], field_name: str) -> list[object]:
    value = payload[field_name]
    if not isinstance(value, list):
        raise SuggestionSchemaError(f"{field_name} must be a list.")
    return value


def _enum_value(
    payload: dict[str, object], field_name: str, allowed_values: tuple[str, ...]
) -> str:
    value = _required_string(payload, field_name)
    if value not in allowed_values:
        allowed = ", ".join(allowed_values)
        raise SuggestionSchemaError(f"{field_name} must be one of: {allowed}.")
    return value


def _recommended_validation(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SuggestionSchemaError("recommended_validation must be an object.")
    validation_type = value.get("type")
    if not isinstance(validation_type, str) or not validation_type.strip():
        raise SuggestionSchemaError("recommended_validation.type must be a non-empty string.")
    if validation_type not in VALIDATION_TYPES:
        allowed = ", ".join(VALIDATION_TYPES)
        raise SuggestionSchemaError(
            f"recommended_validation.type must be one of: {allowed}."
        )
    reason = value.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise SuggestionSchemaError("recommended_validation.reason must be a non-empty string.")
    return dict(value)
