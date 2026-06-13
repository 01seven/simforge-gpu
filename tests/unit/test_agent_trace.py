import json
from pathlib import Path

from simforge_gpu.suggestions.review import review_model_suggestion_payload
from simforge_gpu.tracing.agent_trace import build_agent_trace


SOURCE = Path("examples/monte_carlo_pi/input_cpu.py")


def _suggestion_payload() -> str:
    return json.dumps(
        {
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
    )


def test_build_agent_trace_records_accepted_and_rejected_model_fields():
    review = review_model_suggestion_payload(
        _suggestion_payload(),
        suggestion_file=Path("reports/model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    trace = build_agent_trace(
        source_file=SOURCE,
        target_backend="cupy",
        model_suggestion_path=Path("reports/model_suggestion.json"),
        suggestion_review_path=Path("reports/suggestion_review.json"),
        review=review,
        harness_gates=(
            {
                "name": "source_intake",
                "status": "PASSED",
                "artifact": "examples/monte_carlo_pi/input_cpu.py",
            },
            {
                "name": "suggestion_review",
                "status": review.status,
                "artifact": "reports/suggestion_review.json",
            },
        ),
        artifacts={
            "conversion_plan": "reports/conversion_plan.json",
            "suggestion_review": "reports/suggestion_review.json",
        },
    )

    data = trace.to_dict()
    assert data["schema_version"] == "1.0"
    assert data["source_file"] == "examples/monte_carlo_pi/input_cpu.py"
    assert data["target_backend"] == "cupy"
    assert data["accepted_model_fields"] == [
        "source_intent",
        "risks",
        "recommended_validation",
    ]
    assert data["rejected_model_fields"] == []
    assert data["harness_gates"][1]["status"] == "ACCEPTED"
    assert json.loads(trace.to_json())["artifacts"]["conversion_plan"] == (
        "reports/conversion_plan.json"
    )


def test_build_agent_trace_records_rejected_suggestion_gate():
    review = review_model_suggestion_payload(
        "{not json",
        suggestion_file=Path("reports/model_suggestion.json"),
        source_file=SOURCE,
        target_backend="cupy",
        deterministic_unsupported=(),
    )

    trace = build_agent_trace(
        source_file=SOURCE,
        target_backend="cupy",
        model_suggestion_path=Path("reports/model_suggestion.json"),
        suggestion_review_path=Path("reports/suggestion_review.json"),
        review=review,
        harness_gates=(
            {
                "name": "source_intake",
                "status": "PASSED",
                "artifact": "examples/monte_carlo_pi/input_cpu.py",
            },
            {
                "name": "suggestion_review",
                "status": review.status,
                "artifact": "reports/suggestion_review.json",
            },
        ),
        artifacts={"suggestion_review": "reports/suggestion_review.json"},
    )

    data = trace.to_dict()
    assert data["accepted_model_fields"] == []
    assert data["rejected_model_fields"] == ["schema"]
    assert data["harness_gates"][-1]["status"] == "REJECTED"
