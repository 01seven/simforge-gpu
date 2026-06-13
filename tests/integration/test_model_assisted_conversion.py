import json

from simforge_gpu.cli import main
from simforge_gpu.pipeline import inspect_project_status


EXAMPLE = "examples/monte_carlo_pi/input_cpu.py"


def _write_suggestion(path, **overrides):
    payload = {
        "schema_version": "1.0",
        "source_file": EXAMPLE,
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
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_review_suggestion_writes_review_artifacts_without_generated_code(tmp_path, capsys):
    suggestion = _write_suggestion(tmp_path / "model_suggestion.json")
    output_dir = tmp_path / "review_only"

    exit_code = main(
        [
            "review-suggestion",
            str(suggestion),
            "--source",
            EXAMPLE,
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Suggestion review: ACCEPTED_WITH_WARNINGS" in captured.out
    assert (output_dir / "reports" / "suggestion_review.json").exists()
    assert (output_dir / "reports" / "suggestion_review.md").exists()
    assert not (output_dir / "generated").exists()
    report = (output_dir / "reports" / "suggestion_review.md").read_text(encoding="utf-8")
    assert "Model advisory input is not treated as a trusted transformation." in report


def test_convert_with_accepted_suggestion_writes_advisory_artifacts_and_generated_code(
    tmp_path, capsys
):
    suggestion = _write_suggestion(tmp_path / "model_suggestion.json")
    output_dir = tmp_path / "accepted"

    exit_code = main(
        [
            "convert",
            EXAMPLE,
            "--target",
            "cupy",
            "--suggestion",
            str(suggestion),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Generated CuPy code:" in captured.out
    assert (output_dir / "generated" / "input_cpu_gpu.py").exists()
    assert (output_dir / "reports" / "model_suggestion.json").exists()
    assert (output_dir / "reports" / "suggestion_review.json").exists()
    assert (output_dir / "reports" / "suggestion_review.md").exists()
    assert (output_dir / "reports" / "agent_trace.json").exists()
    review = json.loads((output_dir / "reports" / "suggestion_review.json").read_text(encoding="utf-8"))
    assert review["suggestion_file"] == "reports/model_suggestion.json"

    plan = json.loads((output_dir / "reports" / "conversion_plan.json").read_text(encoding="utf-8"))
    assert plan["source_intent"]["source"] == "model"
    assert plan["model_advisory"]["review_status"] == "ACCEPTED_WITH_WARNINGS"
    assert plan["backend_status"]["implemented"] is True
    assert plan["validation_notes"][0]["source"] == "model"
    assert plan["model_risk_notes"][0]["source"] == "model"

    explanation = (output_dir / "reports" / "explanation_report.md").read_text(encoding="utf-8")
    assert "## Model Advisory Input" in explanation
    assert "Model advisory input is not treated as a trusted transformation." in explanation

    quality = (output_dir / "reports" / "quality_report.md").read_text(encoding="utf-8")
    assert "Suggestion review: ACCEPTED_WITH_WARNINGS" in quality

    trace = json.loads((output_dir / "reports" / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["accepted_model_fields"] == [
        "source_intent",
        "risks",
        "recommended_validation",
    ]
    assert trace["harness_gates"][3]["name"] == "suggestion_review"
    status = inspect_project_status(output_dir)
    assert status["artifacts"]["reports/model_suggestion.json"] == "present"
    assert status["artifacts"]["reports/suggestion_review.json"] == "present"
    assert status["artifacts"]["reports/suggestion_review.md"] == "present"
    assert status["artifacts"]["reports/agent_trace.json"] == "present"


def test_convert_with_rejected_suggestion_stops_before_code_generation(tmp_path, capsys):
    suggestion = _write_suggestion(
        tmp_path / "model_suggestion.json",
        source_file="examples/bootstrap_mean/input_cpu.py",
    )
    output_dir = tmp_path / "rejected"

    exit_code = main(
        [
            "convert",
            EXAMPLE,
            "--target",
            "cupy",
            "--suggestion",
            str(suggestion),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Suggestion review rejected" in captured.err
    assert (output_dir / "reports" / "suggestion_review.json").exists()
    assert (output_dir / "reports" / "suggestion_review.md").exists()
    assert (output_dir / "reports" / "agent_trace.json").exists()
    assert not (output_dir / "generated" / "input_cpu_gpu.py").exists()
    trace = json.loads((output_dir / "reports" / "agent_trace.json").read_text(encoding="utf-8"))
    assert trace["rejected_model_fields"] == ["source_file"]
