"""Controlled merge from reviewed model suggestions into conversion plans."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from simforge_gpu.planners.conversion_plan import ConversionPlan
from simforge_gpu.suggestions.review import SuggestionReview


def merge_review_into_plan(
    plan: ConversionPlan, review: SuggestionReview
) -> ConversionPlan:
    """Return a plan with accepted advisory model fields added.

    The merge is intentionally additive. It does not change backend status,
    unsupported features, validation status, benchmark status, syntax status, or
    any transformation rule.
    """

    if review.is_rejected or review.suggestion is None:
        return plan

    suggestion = review.suggestion
    accepted_fields = [entry.field for entry in review.accepted_fields]
    rejected_fields = [entry.field for entry in review.rejected_fields]
    advisory: dict[str, Any] = {
        "suggestion_file": review.suggestion_file,
        "review_status": review.status,
        "accepted_fields": accepted_fields,
        "rejected_fields": rejected_fields,
    }
    for field_name in (
        "output_semantics",
        "benchmark_notes",
        "human_review_notes",
    ):
        if field_name in suggestion.optional_fields:
            advisory[field_name] = suggestion.optional_fields[field_name]

    return replace(
        plan,
        source_intent={
            "text": suggestion.source_intent,
            "source": "model",
        },
        model_advisory=advisory,
        validation_notes=_validation_notes(suggestion.recommended_validation),
        model_risk_notes=_risk_notes(suggestion.risks),
    )


def _validation_notes(recommended_validation: dict[str, object]) -> tuple[dict[str, str], ...]:
    return (
        {
            "type": str(recommended_validation["type"]),
            "message": str(recommended_validation["reason"]),
            "source": "model",
        },
    )


def _risk_notes(risks: list[object]) -> tuple[dict[str, Any], ...]:
    notes: list[dict[str, Any]] = []
    for risk in risks:
        if isinstance(risk, dict):
            note = dict(risk)
        else:
            note = {"message": str(risk)}
        note["source"] = "model"
        notes.append(note)
    return tuple(notes)
