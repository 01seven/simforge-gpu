"""Harness review for external model suggestions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from simforge_gpu.backends.base import UnsupportedBackendError
from simforge_gpu.backends.registry import get_backend
from simforge_gpu.ir.schema import UnsupportedFeature
from simforge_gpu.reporters.json_report import stable_json
from simforge_gpu.suggestions.schema import (
    ModelSuggestion,
    SuggestionSchemaError,
    parse_model_suggestion_json,
)


@dataclass(frozen=True)
class ReviewEntry:
    field: str
    reason: str
    source: str = "model"

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewWarning:
    field: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class SuggestionReview:
    schema_version: str
    status: str
    source_file: str
    suggestion_file: str
    accepted_fields: tuple[ReviewEntry, ...]
    warnings: tuple[ReviewWarning, ...]
    rejected_fields: tuple[ReviewEntry, ...]
    unsupported_features: tuple[dict[str, object], ...]
    suggestion: ModelSuggestion | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "source_file": self.source_file,
            "suggestion_file": self.suggestion_file,
            "accepted_fields": [entry.to_dict() for entry in self.accepted_fields],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "rejected_fields": [entry.to_dict() for entry in self.rejected_fields],
            "unsupported_features": list(self.unsupported_features),
        }

    def to_json(self) -> str:
        return stable_json(self.to_dict())

    @property
    def is_rejected(self) -> bool:
        return self.status == "REJECTED"


def review_model_suggestion_payload(
    payload: str,
    suggestion_file: str | Path,
    source_file: str | Path,
    target_backend: str,
    deterministic_unsupported: tuple[UnsupportedFeature, ...] = (),
) -> SuggestionReview:
    suggestion_path = Path(suggestion_file)
    source_path = Path(source_file)
    unsupported = _unsupported_from_static_detection(deterministic_unsupported)

    try:
        suggestion = parse_model_suggestion_json(payload)
    except SuggestionSchemaError as exc:
        return SuggestionReview(
            schema_version="1.0",
            status="REJECTED",
            source_file=_display_path(source_path),
            suggestion_file=_display_path(suggestion_path),
            accepted_fields=(),
            warnings=(),
            rejected_fields=(
                ReviewEntry(field="schema", reason=str(exc), source="harness"),
            ),
            unsupported_features=unsupported,
            suggestion=None,
        )

    warnings: list[ReviewWarning] = []
    rejected_fields: list[ReviewEntry] = []
    accepted_fields = [
        ReviewEntry(
            field="source_intent",
            reason="Non-empty source intent accepted as model-provided explanation.",
        ),
        ReviewEntry(
            field="risks",
            reason="Model risks accepted as advisory notes.",
        ),
        ReviewEntry(
            field="recommended_validation",
            reason="Recommended validation accepted as advisory validation note.",
        ),
    ]

    if not _paths_match(suggestion.source_file, source_path):
        rejected_fields.append(
            ReviewEntry(
                field="source_file",
                reason=(
                    f"Suggestion source_file {suggestion.source_file!r} does not match "
                    f"conversion input {_display_path(source_path)!r}."
                ),
                source="harness",
            )
        )

    try:
        suggested_backend = get_backend(suggestion.suggested_backend)
    except UnsupportedBackendError as exc:
        rejected_fields.append(
            ReviewEntry(field="suggested_backend", reason=str(exc), source="harness")
        )
        return _review(
            status="REJECTED",
            suggestion=suggestion,
            suggestion_path=suggestion_path,
            source_path=source_path,
            accepted_fields=(),
            warnings=tuple(warnings),
            rejected_fields=tuple(rejected_fields),
            unsupported=unsupported,
        )

    if not suggested_backend.is_implemented:
        rejected_fields.append(
            ReviewEntry(
                field="suggested_backend",
                reason=suggested_backend.unsupported_reason(),
                source="harness",
            )
        )
        unsupported.append(
            {
                "code": f"--target {suggested_backend.name}",
                "reason": suggested_backend.unsupported_reason(),
                "action": "Use --target cupy for the current supported backend.",
                "category": "backend_not_implemented",
                "source": "backend_policy",
            }
        )

    if suggestion.suggested_backend != target_backend.lower():
        warnings.append(
            ReviewWarning(
                field="suggested_backend",
                message=(
                    f"Model suggested {suggestion.suggested_backend}; harness target is "
                    f"{target_backend.lower()}."
                ),
            )
        )

    if suggestion.risks:
        warnings.append(
            ReviewWarning(
                field="risks",
                message=(
                    "Model risks are advisory and do not replace rule-detected "
                    "unsupported features."
                ),
            )
        )

    if suggestion.unsupported_hypotheses:
        warnings.append(
            ReviewWarning(
                field="unsupported_hypotheses",
                message="Model unsupported hypotheses are advisory model_hypothesis entries.",
            )
        )
        unsupported.extend(_unsupported_from_model_hypotheses(suggestion.unsupported_hypotheses))

    if any(entry.field == "source_file" for entry in rejected_fields):
        status = "REJECTED"
        accepted_for_status: tuple[ReviewEntry, ...] = ()
    elif warnings or rejected_fields or unsupported:
        status = "ACCEPTED_WITH_WARNINGS"
        accepted_for_status = tuple(accepted_fields)
    else:
        status = "ACCEPTED"
        accepted_for_status = tuple(accepted_fields)

    return _review(
        status=status,
        suggestion=suggestion,
        suggestion_path=suggestion_path,
        source_path=source_path,
        accepted_fields=accepted_for_status,
        warnings=tuple(warnings),
        rejected_fields=tuple(rejected_fields),
        unsupported=unsupported,
    )


def _review(
    status: str,
    suggestion: ModelSuggestion,
    suggestion_path: Path,
    source_path: Path,
    accepted_fields: tuple[ReviewEntry, ...],
    warnings: tuple[ReviewWarning, ...],
    rejected_fields: tuple[ReviewEntry, ...],
    unsupported: list[dict[str, object]],
) -> SuggestionReview:
    return SuggestionReview(
        schema_version="1.0",
        status=status,
        source_file=suggestion.source_file,
        suggestion_file=_display_path(suggestion_path),
        accepted_fields=accepted_fields,
        warnings=warnings,
        rejected_fields=rejected_fields,
        unsupported_features=tuple(unsupported),
        suggestion=suggestion,
    )


def _unsupported_from_static_detection(
    features: tuple[UnsupportedFeature, ...]
) -> list[dict[str, object]]:
    return [
        {
            **feature.to_dict(),
            "source": "static_detector",
        }
        for feature in features
    ]


def _unsupported_from_model_hypotheses(hypotheses: list[object]) -> list[dict[str, object]]:
    unsupported: list[dict[str, object]] = []
    for index, hypothesis in enumerate(hypotheses, start=1):
        if isinstance(hypothesis, dict):
            code = str(hypothesis.get("code", f"model_hypothesis_{index}"))
            reason = str(hypothesis.get("reason", hypothesis.get("message", ""))).strip()
            action = str(
                hypothesis.get("action", "Review this model hypothesis manually.")
            )
        else:
            code = f"model_hypothesis_{index}"
            reason = str(hypothesis)
            action = "Review this model hypothesis manually."
        unsupported.append(
            {
                "code": code,
                "reason": reason or "Model reported an unsupported hypothesis.",
                "action": action,
                "category": "model_hypothesis",
                "source": "model_hypothesis",
            }
        )
    return unsupported


def _paths_match(suggestion_source: str, source_path: Path) -> bool:
    left = _normalized_path(Path(suggestion_source))
    right = _normalized_path(source_path)
    return left == right or Path(suggestion_source).as_posix() == source_path.as_posix()


def _normalized_path(path: Path) -> str:
    return path.resolve(strict=False).as_posix().lower()


def _display_path(path: Path) -> str:
    return path.as_posix()
