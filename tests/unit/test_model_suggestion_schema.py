import json

import pytest

from simforge_gpu.suggestions.schema import (
    ModelSuggestion,
    SuggestionSchemaError,
    parse_model_suggestion_json,
)


def _minimal_suggestion() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "source_file": "examples/monte_carlo_pi/input_cpu.py",
        "source_intent": "Estimate pi with Monte Carlo sampling.",
        "suggested_backend": "cupy",
        "mvp_fit": "yes",
        "confidence": "high",
        "risks": [],
        "unsupported_hypotheses": [],
        "recommended_validation": {
            "type": "stochastic",
            "reason": "Compare scalar estimates with a stochastic tolerance.",
        },
    }


def test_parse_model_suggestion_accepts_required_core_and_preserves_optional_fields():
    payload = _minimal_suggestion()
    payload["output_semantics"] = "The script prints one floating-point estimate."
    payload["human_review_notes"] = ["Review tolerance before claiming correctness."]

    suggestion = parse_model_suggestion_json(json.dumps(payload))

    assert isinstance(suggestion, ModelSuggestion)
    assert suggestion.schema_version == "1.0"
    assert suggestion.source_file == "examples/monte_carlo_pi/input_cpu.py"
    assert suggestion.suggested_backend == "cupy"
    assert suggestion.optional_fields["output_semantics"] == (
        "The script prints one floating-point estimate."
    )
    assert suggestion.optional_fields["human_review_notes"] == [
        "Review tolerance before claiming correctness."
    ]
    assert json.loads(suggestion.to_json())["schema_version"] == "1.0"


def test_parse_model_suggestion_rejects_invalid_json():
    with pytest.raises(SuggestionSchemaError) as exc_info:
        parse_model_suggestion_json("{not json")

    assert "Invalid JSON" in str(exc_info.value)


def test_parse_model_suggestion_rejects_missing_required_field():
    payload = _minimal_suggestion()
    del payload["source_intent"]

    with pytest.raises(SuggestionSchemaError) as exc_info:
        parse_model_suggestion_json(json.dumps(payload))

    assert "Missing required field: source_intent" in str(exc_info.value)


def test_parse_model_suggestion_rejects_unsupported_schema_version():
    payload = _minimal_suggestion()
    payload["schema_version"] = "2.0"

    with pytest.raises(SuggestionSchemaError) as exc_info:
        parse_model_suggestion_json(json.dumps(payload))

    assert "Unsupported schema_version: 2.0" in str(exc_info.value)


def test_parse_model_suggestion_rejects_invalid_enum_values():
    payload = _minimal_suggestion()
    payload["confidence"] = "certain"

    with pytest.raises(SuggestionSchemaError) as exc_info:
        parse_model_suggestion_json(json.dumps(payload))

    assert "confidence must be one of" in str(exc_info.value)


def test_parse_model_suggestion_requires_recommended_validation_type_and_reason():
    payload = _minimal_suggestion()
    payload["recommended_validation"] = {"type": "stochastic"}

    with pytest.raises(SuggestionSchemaError) as exc_info:
        parse_model_suggestion_json(json.dumps(payload))

    assert "recommended_validation.reason" in str(exc_info.value)
