import json
from pathlib import Path

from sim2gpu.cli import main
from sim2gpu.pipeline import collect_demo_status


def _write_project(root: Path, name: str, unsupported: str) -> Path:
    project = root / name
    reports = project / "reports"
    generated = project / "generated"
    reports.mkdir(parents=True)
    generated.mkdir()
    (generated / "input_cpu_gpu.py").write_text("import cupy as cp\n", encoding="utf-8")
    (reports / "conversion_plan.json").write_text(
        '{"target_backend": "cupy", "backend_status": {"status": "implemented"}}\n',
        encoding="utf-8",
    )
    (reports / "unsupported_report.md").write_text(unsupported, encoding="utf-8")
    (reports / "validation_report.md").write_text(
        "Validation status: SKIPPED\n", encoding="utf-8"
    )
    (reports / "benchmark_report.md").write_text(
        "Benchmark status: SKIPPED\n", encoding="utf-8"
    )
    (reports / "syntax_report.md").write_text(
        "Syntax status: PASSED\n", encoding="utf-8"
    )
    (reports / "quality_report.md").write_text(
        "Quality gate: PASSED_WITH_SKIPS\n", encoding="utf-8"
    )
    return project


def test_collect_demo_status_summarizes_project_reports(tmp_path):
    _write_project(tmp_path, "clean", "# Unsupported Report\n\nNo unsupported features detected.\n")
    _write_project(
        tmp_path,
        "partial",
        "# Unsupported Report\n\n## 1. `np.argsort`\n\nReason: unsupported\n",
    )

    rows = collect_demo_status(tmp_path)

    assert [row.name for row in rows] == ["clean", "partial"]
    assert rows[0].generated == "yes"
    assert rows[0].unsupported_count == 0
    assert rows[1].unsupported_count == 1
    assert rows[1].validation_status == "SKIPPED"
    assert rows[1].benchmark_status == "SKIPPED"


def test_demo_status_cli_prints_summary_table(tmp_path, capsys):
    _write_project(
        tmp_path,
        "partial",
        "# Unsupported Report\n\n## 1. `np.argsort`\n\nReason: unsupported\n",
    )

    exit_code = main(["demo-status", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Project" in captured.out
    assert "partial" in captured.out
    assert "unsupported" in captured.out
    assert "SKIPPED" in captured.out


def test_demo_status_cli_can_print_json(tmp_path, capsys):
    _write_project(
        tmp_path,
        "partial",
        "# Unsupported Report\n\n## 1. `np.argsort`\n\nReason: unsupported\n",
    )

    exit_code = main(["demo-status", str(tmp_path), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload == [
        {
            "name": "partial",
            "backend": "cupy",
            "backend_status": "implemented",
            "generated": "yes",
            "unsupported_count": 1,
            "validation_status": "SKIPPED",
            "benchmark_status": "SKIPPED",
        }
    ]


def test_inspect_project_cli_prints_artifacts_and_unsupported_items(tmp_path, capsys):
    project = _write_project(
        tmp_path,
        "partial",
        "# Unsupported Report\n\n## 1. `np.argsort`\n\nReason: unsupported\n\nAction: keep CPU-side.\n",
    )

    exit_code = main(["inspect-project", str(project)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# Project Inspection" in captured.out
    assert "Project: partial" in captured.out
    assert "generated/input_cpu_gpu.py" in captured.out
    assert "`np.argsort`" in captured.out
    assert "Validation status: SKIPPED" in captured.out
    assert "Benchmark status: SKIPPED" in captured.out
    assert "Syntax status: PASSED" in captured.out
    assert "Quality gate: PASSED_WITH_SKIPS" in captured.out
    assert "Next steps" in captured.out


def test_inspect_project_cli_can_print_json(tmp_path, capsys):
    project = _write_project(
        tmp_path,
        "partial",
        "# Unsupported Report\n\n## 1. `np.argsort`\n\nReason: unsupported\n\nAction: keep CPU-side.\n",
    )

    exit_code = main(["inspect-project", str(project), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["project"] == "partial"
    assert payload["backend"]["target_backend"] == "cupy"
    assert payload["backend"]["status"] == "implemented"
    assert payload["artifacts"]["generated/input_cpu_gpu.py"] == "present"
    assert payload["artifacts"]["reports/syntax_report.md"] == "present"
    assert payload["artifacts"]["reports/quality_report.md"] == "present"
    assert payload["artifacts"]["runs/validation.json"] == "missing"
    assert payload["artifacts"]["runs/benchmark.json"] == "missing"
    assert payload["unsupported"] == ["1. `np.argsort`"]
    assert payload["validation_status"] == "SKIPPED"
    assert payload["benchmark_status"] == "SKIPPED"
    assert payload["syntax_status"] == "PASSED"
    assert payload["quality_gate"] == "PASSED_WITH_SKIPS"
    assert "Review `unsupported_report.md`" in payload["next_steps"][0]


def test_check_artifacts_cli_passes_for_complete_project(tmp_path, capsys):
    project = _write_project(
        tmp_path,
        "complete",
        "# Unsupported Report\n\nNo unsupported features detected.\n",
    )
    for filename in ("analysis_ir.json", "explanation_report.md"):
        (project / "reports" / filename).write_text("ok\n", encoding="utf-8")

    exit_code = main(["check-artifacts", str(project)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Artifact check: PASSED" in captured.out
    assert "Missing artifacts: 0" in captured.out


def test_check_artifacts_cli_fails_for_missing_report(tmp_path, capsys):
    project = _write_project(
        tmp_path,
        "incomplete",
        "# Unsupported Report\n\nNo unsupported features detected.\n",
    )

    exit_code = main(["check-artifacts", str(project), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 1
    assert payload["status"] == "FAILED"
    assert "reports/analysis_ir.json" in payload["missing_artifacts"]
    assert "reports/explanation_report.md" in payload["missing_artifacts"]
    assert "reports/quality_report.md" not in payload["missing_artifacts"]
