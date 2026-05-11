import json
from pathlib import Path

from sim2gpu.cli import main


EXAMPLE = Path("examples/monte_carlo_pi/input_cpu.py")


def test_analyze_writes_ir_artifact(tmp_path, capsys):
    exit_code = main(["analyze", str(EXAMPLE), "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    ir_path = tmp_path / "reports" / "analysis_ir.json"

    assert exit_code == 0
    assert str(ir_path) in captured.out
    assert ir_path.exists()

    ir = json.loads(ir_path.read_text(encoding="utf-8"))
    assert ir["language"] == "python"
    assert ir["imports"] == ["numpy"]
    assert "np.random.uniform" in ir["random_calls"]
    assert ir["gpu_suitability"]["recommended_backend"] == "cupy"


def test_convert_cupy_writes_vertical_slice_artifacts(tmp_path, capsys):
    exit_code = main(["convert", str(EXAMPLE), "--target", "cupy", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    generated = tmp_path / "generated" / "input_cpu_gpu.py"
    reports = tmp_path / "reports"

    assert exit_code == 0
    assert str(generated) in captured.out
    assert generated.exists()
    generated_source = generated.read_text(encoding="utf-8")
    assert "import cupy as cp" in generated_source
    assert "cp.random.uniform" in generated_source
    assert "cp.mean" in generated_source

    expected_reports = [
        "analysis_ir.json",
        "conversion_plan.json",
        "unsupported_report.md",
        "explanation_report.md",
        "validation_report.md",
        "benchmark_report.md",
        "syntax_report.md",
        "quality_report.md",
    ]
    for filename in expected_reports:
        assert (reports / filename).exists(), filename

    plan = json.loads((reports / "conversion_plan.json").read_text(encoding="utf-8"))
    assert plan["target_backend"] == "cupy"
    assert plan["backend_status"]["implemented"] is True
    assert plan["changes"]

    validation = (reports / "validation_report.md").read_text(encoding="utf-8")
    benchmark = (reports / "benchmark_report.md").read_text(encoding="utf-8")
    syntax = (reports / "syntax_report.md").read_text(encoding="utf-8")
    quality = (reports / "quality_report.md").read_text(encoding="utf-8")
    assert "Validation status: SKIPPED" in validation
    assert "Benchmark status: SKIPPED" in benchmark
    assert "Speedup:" not in benchmark
    assert "Syntax status: PASSED" in syntax
    assert "Generated source parsed successfully" in syntax
    assert "Quality gate: PASSED_WITH_SKIPS" in quality
    assert "Syntax status: PASSED" in quality
    assert "Validation status: SKIPPED" in quality
    assert "Benchmark status: SKIPPED" in quality


def test_convert_torch_writes_unsupported_report_without_generated_code(tmp_path, capsys):
    exit_code = main(["convert", str(EXAMPLE), "--target", "torch", "--output-dir", str(tmp_path)])

    captured = capsys.readouterr()
    unsupported_report = tmp_path / "reports" / "unsupported_report.md"

    assert exit_code == 2
    assert "TorchBackend is planned but not implemented in the MVP." in captured.err
    assert unsupported_report.exists()
    assert "TorchBackend is planned" in unsupported_report.read_text(encoding="utf-8")
    assert not (tmp_path / "generated" / "input_cpu_gpu.py").exists()


def test_pyproject_declares_console_script_entrypoint():
    pyproject = Path("pyproject.toml")

    assert pyproject.exists()
    text = pyproject.read_text(encoding="utf-8")
    assert 'sim2gpu = "sim2gpu.cli:main"' in text
