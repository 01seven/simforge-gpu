import json

from simforge_gpu.ir.schema import AnalysisIR, GpuSuitability, UnsupportedFeature
from simforge_gpu.planners.conversion_plan import create_conversion_plan
from simforge_gpu.suggestions.merge import merge_review_into_plan
from simforge_gpu.suggestions.review import review_model_suggestion_payload


def _ir_with_unsupported() -> AnalysisIR:
    return AnalysisIR(
        imports=("numpy",),
        patterns=("monte_carlo_loop",),
        random_calls=("np.random.uniform",),
        outputs=("pi_estimate",),
        unsupported_features=(
            UnsupportedFeature(
                code="np.argsort",
                reason="np.argsort is outside the supported MVP mapping table.",
                action="Keep this operation on CPU or add explicit support later.",
                category="unsupported_numpy_api",
            ),
        ),
        gpu_suitability=GpuSuitability(
            gpu_suitable=True,
            confidence="medium",
            recommended_backend="cupy",
            future_backend_candidates=("torch",),
        ),
    )


def _suggestion_payload() -> str:
    return json.dumps(
        {
            "schema_version": "1.0",
            "source_file": "examples/monte_carlo_pi/input_cpu.py",
            "source_intent": "Estimate pi with Monte Carlo sampling.",
            "suggested_backend": "cupy",
            "mvp_fit": "yes",
            "confidence": "high",
            "risks": [
                {
                    "code": "random_stream_difference",
                    "message": "CPU and GPU random streams are not expected to match.",
                    "severity": "medium",
                }
            ],
            "unsupported_hypotheses": [],
            "recommended_validation": {
                "type": "stochastic",
                "reason": "Compare scalar estimates with a stochastic tolerance.",
            },
            "output_semantics": "The script prints a single scalar estimate.",
            "benchmark_notes": ["Benchmark only after validation is meaningful."],
            "human_review_notes": ["Review tolerance before using results."],
        }
    )


def _suggestion_with_overrides(**overrides: object) -> str:
    payload = json.loads(_suggestion_payload())
    payload.update(overrides)
    return json.dumps(payload)


def test_merge_review_adds_only_source_labeled_model_advisory_fields():
    plan = create_conversion_plan(
        _ir_with_unsupported(),
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
    )
    review = review_model_suggestion_payload(
        _suggestion_payload(),
        suggestion_file="projects/monte_carlo_pi/reports/model_suggestion.json",
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
        deterministic_unsupported=plan.unsupported_features,
    )

    merged = merge_review_into_plan(plan, review)
    data = merged.to_dict()

    assert data["target_backend"] == "cupy"
    assert data["backend_status"]["implemented"] is True
    assert data["unsupported_features"][0]["code"] == "np.argsort"
    assert data["source_intent"] == {
        "text": "Estimate pi with Monte Carlo sampling.",
        "source": "model",
    }
    assert data["model_advisory"]["review_status"] == "ACCEPTED_WITH_WARNINGS"
    assert data["model_advisory"]["accepted_fields"] == [
        "source_intent",
        "risks",
        "recommended_validation",
    ]
    assert data["validation_notes"] == [
        {
            "type": "stochastic",
            "message": "Compare scalar estimates with a stochastic tolerance.",
            "source": "model",
        }
    ]
    assert data["model_risk_notes"] == [
        {
            "code": "random_stream_difference",
            "message": "CPU and GPU random streams are not expected to match.",
            "severity": "medium",
            "source": "model",
        }
    ]


def test_merge_review_does_not_mutate_plan_for_rejected_suggestion():
    plan = create_conversion_plan(
        _ir_with_unsupported(),
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
    )
    review = review_model_suggestion_payload(
        "{not json",
        suggestion_file="projects/monte_carlo_pi/reports/model_suggestion.json",
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
        deterministic_unsupported=plan.unsupported_features,
    )

    merged = merge_review_into_plan(plan, review)

    assert merged.to_dict() == plan.to_dict()


def test_merge_review_does_not_merge_model_status_or_speedup_claims():
    plan = create_conversion_plan(
        _ir_with_unsupported(),
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
    )
    review = review_model_suggestion_payload(
        _suggestion_with_overrides(
            validation_status="PASSED",
            benchmark_status="PASSED",
            quality_gate="PASSED",
            speedup="999x",
        ),
        suggestion_file="projects/monte_carlo_pi/reports/model_suggestion.json",
        source_file="examples/monte_carlo_pi/input_cpu.py",
        target_backend="cupy",
        deterministic_unsupported=plan.unsupported_features,
    )

    merged = merge_review_into_plan(plan, review)
    data = merged.to_dict()
    rendered = json.dumps(data)

    assert "validation_status" not in rendered
    assert "benchmark_status" not in rendered
    assert "quality_gate" not in rendered
    assert "999x" not in rendered
    assert data["validation_notes"][0]["source"] == "model"
