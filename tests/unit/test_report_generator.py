import json

from sim2gpu.ir.schema import UnsupportedFeature
from sim2gpu.reporters.markdown import (
    render_benchmark_report,
    render_explanation_report,
    render_unsupported_report,
)
from sim2gpu.reporters.json_report import stable_json


def test_stable_json_sorts_keys_and_indents():
    rendered = stable_json({"b": 2, "a": 1})

    assert rendered == '{\n  "a": 1,\n  "b": 2\n}\n'


def test_unsupported_report_lists_reason_and_action():
    report = render_unsupported_report(
        [
            UnsupportedFeature(
                code="--target torch",
                reason="TorchBackend is planned but not implemented in the MVP.",
                action="Use --target cupy.",
            )
        ]
    )

    assert "# Unsupported Report" in report
    assert "TorchBackend is planned" in report
    assert "Use --target cupy." in report


def test_benchmark_report_can_be_skipped_without_fake_speedup():
    report = render_benchmark_report(
        {"status": "SKIPPED", "reason": "No CUDA-compatible GPU detected."}
    )

    assert "Benchmark status: SKIPPED" in report
    assert "No CUDA-compatible GPU detected." in report
    assert "Speedup:" not in report


def test_benchmark_report_lists_trust_indicators():
    report = render_benchmark_report(
        {
            "status": "PASSED",
            "measurement_method": "subprocess_wall_time",
            "trust_level": "demo_only",
            "includes_transfer_overhead": False,
            "limitations": [
                "Includes Python process startup and CUDA initialization overhead.",
                "Does not isolate host/device transfer time.",
            ],
        }
    )

    assert "Measurement method: subprocess_wall_time" in report
    assert "Trust level: demo_only" in report
    assert "Includes transfer overhead: False" in report
    assert "Includes Python process startup" in report
    assert "Does not isolate host/device transfer time." in report


def test_explanation_report_summarizes_artifacts_without_claiming_speedup():
    report = render_explanation_report(
        {
            "summary": "Convert NumPy Monte Carlo code to CuPy where safe.",
            "validation_status": "SKIPPED",
            "benchmark_status": "SKIPPED",
            "artifacts": ["conversion_plan.json", "unsupported_report.md"],
        }
    )

    assert "Convert NumPy Monte Carlo code" in report
    assert "validation_status: SKIPPED" in report
    assert "benchmark_status: SKIPPED" in report
    assert "speedup" not in report.lower()
