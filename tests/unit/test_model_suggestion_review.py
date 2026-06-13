import json
from pathlib import Path

from simforge_gpu.ir.schema import UnsupportedFeature
from simforge_gpu.suggestions.review import review_model_suggestion_payload


SOURCE = Path("examples/monte_carlo_pi/input_cpu.py")


def _suggestion(**overrides: object) -> str:
    payload: dict[str, object] = {
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
    payload.update(overrides)
    return json.dumps(payload)


def test_review_accepts_valid_minimal_suggestion_for_matching_cupy_target():
    review = review_model_suggestion_payload(
        _suggestion(),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "ACCEPTED"
    assert data["source_file"] == "examples/monte_carlo_pi/input_cpu.py"
    assert [entry["field"] for entry in data["accepted_fields"]] == [
        "source_intent",
        "risks",
        "recommended_validation",
    ]
    assert data["warnings"] == []
    assert data["rejected_fields"] == []


def test_review_accepts_valid_suggestion_with_advisory_risks_as_warnings():
    review = review_model_suggestion_payload(
        _suggestion(
            risks=[
                {
                    "code": "random_stream_difference",
                    "message": "CPU and GPU random streams will differ.",
                    "severity": "medium",
                }
            ],
        ),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "ACCEPTED_WITH_WARNINGS"
    assert data["warnings"][0]["field"] == "risks"
    assert "advisory" in data["warnings"][0]["message"]


def test_review_rejects_source_file_mismatch_after_path_normalization():
    review = review_model_suggestion_payload(
        _suggestion(source_file="examples/bootstrap_mean/input_cpu.py"),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "REJECTED"
    assert data["rejected_fields"][0]["field"] == "source_file"
    assert "does not match" in data["rejected_fields"][0]["reason"]


def test_review_rejects_invalid_json_as_schema_failure():
    review = review_model_suggestion_payload(
        "{not json",
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "REJECTED"
    assert data["rejected_fields"][0]["field"] == "schema"
    assert "Invalid JSON" in data["rejected_fields"][0]["reason"]


def test_review_rejects_unknown_backend():
    review = review_model_suggestion_payload(
        _suggestion(suggested_backend="magicgpu"),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "REJECTED"
    assert data["rejected_fields"][0]["field"] == "suggested_backend"
    assert "Unknown backend" in data["rejected_fields"][0]["reason"]


def test_review_does_not_accept_planned_backend_as_implemented():
    review = review_model_suggestion_payload(
        _suggestion(suggested_backend="torch"),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    data = review.to_dict()
    assert data["status"] == "ACCEPTED_WITH_WARNINGS"
    assert data["rejected_fields"][0]["field"] == "suggested_backend"
    assert data["unsupported_features"][0]["category"] == "backend_not_implemented"
    assert "planned but not implemented" in data["unsupported_features"][0]["reason"]


def test_review_preserves_rule_detected_unsupported_features():
    unsupported = (
        UnsupportedFeature(
            code="np.argsort",
            reason="np.argsort is outside the supported MVP mapping table.",
            action="Keep this operation on CPU or add explicit support later.",
            category="unsupported_numpy_api",
        ),
    )

    review = review_model_suggestion_payload(
        _suggestion(unsupported_hypotheses=[]),
        suggestion_file=Path("model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=unsupported,
    )

    data = review.to_dict()
    assert data["unsupported_features"][0]["code"] == "np.argsort"
    assert data["unsupported_features"][0]["source"] == "static_detector"
