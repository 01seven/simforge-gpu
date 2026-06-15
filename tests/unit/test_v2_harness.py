import json
from pathlib import Path

from simforge_gpu.cli import main


def _write_python_project(root: Path) -> None:
    root.mkdir()
    (root / "main.py").write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def estimate_pi(n=1000):",
                "    hits = 0",
                "    for _ in range(n):",
                "        x = np.random.uniform()",
                "        y = np.random.uniform()",
                "        if x * x + y * y <= 1.0:",
                "            hits += 1",
                "    return 4.0 * hits / n",
                "",
                "if __name__ == '__main__':",
                "    print(estimate_pi())",
            ]
        ),
        encoding="utf-8",
    )


def _write_r_project(root: Path) -> None:
    root.mkdir()
    (root / "main.R").write_text(
        "\n".join(
            [
                "set.seed(7)",
                "n <- 1000",
                "x <- runif(n)",
                "y <- runif(n)",
                "estimate <- 4 * mean(x * x + y * y <= 1)",
                "print(estimate)",
            ]
        ),
        encoding="utf-8",
    )


def test_inspect_python_project_writes_project_inspection(tmp_path, capsys):
    project = tmp_path / "python_project"
    _write_python_project(project)

    exit_code = main(["inspect", str(project), "--language", "python"])

    captured = capsys.readouterr()
    inspection_path = project / "reports" / "project_inspection.json"
    payload = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Project inspection:" in captured.out
    assert payload["language"] == "python"
    assert "main.py" in payload["detected_files"]
    assert "main.py" in payload["entrypoints"]
    assert "np.random" in payload["randomness_indicators"]
    assert "monte_carlo" in payload["simulation_indicators"]


def test_inspect_r_project_writes_project_inspection(tmp_path):
    project = tmp_path / "r_project"
    _write_r_project(project)

    exit_code = main(["inspect", str(project), "--language", "r", "--json"])

    inspection_path = project / "reports" / "project_inspection.json"
    payload = json.loads(inspection_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["language"] == "r"
    assert "main.R" in payload["detected_files"]
    assert "main.R" in payload["entrypoints"]
    assert "runif" in payload["randomness_indicators"]
    assert "monte_carlo" in payload["simulation_indicators"]


def test_generate_py_torch_plan_without_generating_torch_code(tmp_path):
    project = tmp_path / "python_project"
    _write_python_project(project)

    exit_code = main(
        ["plan", str(project), "--language", "python", "--target", "py-torch"]
    )

    plan_path = project / "reports" / "migration_plan.json"
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["target"] == "py-torch"
    assert payload["target_status"] == "primary_agent_target"
    assert payload["recommended_workflow"] == "external_agent"
    assert payload["agent_instructions_path"] == "agent/migration_task.md"
    assert not (project / "generated").exists()


def test_generate_r_torch_plan(tmp_path):
    project = tmp_path / "r_project"
    _write_r_project(project)

    exit_code = main(["plan", str(project), "--language", "r", "--target", "r-torch"])

    payload = json.loads((project / "reports" / "migration_plan.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["language"] == "r"
    assert payload["target"] == "r-torch"
    assert payload["target_status"] == "primary_agent_target"


def test_generate_codex_task_artifacts(tmp_path):
    project = tmp_path / "python_project"
    _write_python_project(project)

    exit_code = main(
        ["task", str(project), "--agent", "codex", "--target", "py-torch"]
    )

    agent_dir = project / "agent"
    request = json.loads((agent_dir / "migration_request.json").read_text(encoding="utf-8"))
    expected = json.loads((agent_dir / "expected_artifacts.json").read_text(encoding="utf-8"))
    task = (agent_dir / "migration_task.md").read_text(encoding="utf-8")
    constraints = (agent_dir / "constraints.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert request["agent"] == "codex"
    assert request["target"] == "py-torch"
    assert expected["required_outputs"] == ["modified_code", "agent_result.json"]
    assert "Do not fabricate validation results" in task
    assert "Do not fabricate benchmark results" in constraints


def test_collect_pending_agent_result_writes_trace(tmp_path):
    project = tmp_path / "python_project"
    _write_python_project(project)
    assert main(["task", str(project), "--agent", "codex", "--target", "py-torch"]) == 0

    exit_code = main(["collect", str(project)])

    trace = json.loads((project / "reports" / "agent_trace.json").read_text(encoding="utf-8"))
    result = json.loads((project / "agent" / "agent_result.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["status"] == "PENDING_AGENT_OUTPUT"
    assert trace["final_status"] == "PENDING_AGENT_OUTPUT"
    assert "not trusted" in " ".join(trace["trust_boundary_notes"]).lower()


def test_agent_validation_claims_do_not_become_harness_results(tmp_path):
    project = tmp_path / "python_project"
    _write_python_project(project)
    assert main(["task", str(project), "--agent", "codex", "--target", "py-torch"]) == 0
    (project / "agent" / "agent_result.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "agent": "codex",
                "status": "COMPLETED",
                "summary": "Claimed migration complete.",
                "files_modified": ["main.py"],
                "files_created": [],
                "claimed_changes": ["ported loops to torch"],
                "known_limitations": [],
                "validation_claims": ["PASSED"],
                "benchmark_claims": ["10x speedup"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    exit_code = main(["collect", str(project)])

    trace = json.loads((project / "reports" / "agent_trace.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert trace["final_status"] == "PENDING_HARNESS_VALIDATION"
    assert trace["validation_report_path"] == "reports/validation_report.md"
    assert trace["benchmark_report_path"] == "reports/benchmark_report.md"
    assert not (project / "runs" / "validation.json").exists()
    assert not (project / "runs" / "benchmark.json").exists()
