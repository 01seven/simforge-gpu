import json
from pathlib import Path

from sim2gpu.cli import main


def test_list_backends_prints_mvp_backend_status(capsys):
    exit_code = main(["list-backends"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "cupy   MVP backend / implemented" in captured.out
    assert "torch  planned" in captured.out
    assert "jax    planned" in captured.out
    assert "numba  planned" in captured.out
    assert "cudf   planned" in captured.out


def test_list_backends_can_print_json(capsys):
    exit_code = main(["list-backends", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload[0] == {
        "name": "cupy",
        "status": "MVP backend / implemented",
        "implemented": True,
    }
    assert payload[1]["name"] == "torch"
    assert payload[1]["status"] == "planned"
    assert payload[1]["implemented"] is False


def test_convert_rejects_torch_without_generating_fake_code(capsys):
    exit_code = main(
        ["convert", "examples/monte_carlo_pi/input_cpu.py", "--target", "torch"]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "TorchBackend is planned but not implemented in the MVP." in captured.err
    assert "Use --target cupy" in captured.err


def test_convert_dry_run_writes_plan_without_generated_code(tmp_path, capsys):
    output_dir = tmp_path / "dry_run"

    exit_code = main(
        [
            "convert",
            "examples/monte_carlo_pi/input_cpu.py",
            "--target",
            "cupy",
            "--dry-run",
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Dry run conversion plan:" in captured.out
    assert (output_dir / "reports" / "conversion_plan.json").exists()
    assert (output_dir / "reports" / "unsupported_report.md").exists()
    assert not (output_dir / "generated" / "input_cpu_gpu.py").exists()


def test_explain_reads_plan_and_can_write_report(tmp_path, capsys):
    convert_dir = tmp_path / "convert"
    main(
        [
            "convert",
            "examples/monte_carlo_pi/input_cpu.py",
            "--target",
            "cupy",
            "--output-dir",
            str(convert_dir),
        ]
    )
    output = tmp_path / "explanation_report.md"

    exit_code = main(
        [
            "explain",
            str(convert_dir / "reports" / "conversion_plan.json"),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Explanation report:" in captured.out
    assert output.exists()
    assert "Selected backend: cupy" in output.read_text(encoding="utf-8")


def test_validate_writes_no_gpu_skip_report(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated = tmp_path / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")
    output = tmp_path / "validation_report.md"

    exit_code = main(
        ["validate", str(original), str(generated), "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Validation report:" in captured.out
    report = output.read_text(encoding="utf-8")
    assert "Validation status: SKIPPED" in report
    assert "GPU execution disabled" in report


def test_validate_accepts_custom_tolerance_in_no_gpu_mode(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated = tmp_path / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")
    output = tmp_path / "validation_report.md"

    exit_code = main(
        [
            "validate",
            str(original),
            str(generated),
            "--tolerance",
            "0.25",
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Validation report:" in captured.out
    report = output.read_text(encoding="utf-8")
    assert "Validation status: SKIPPED" in report
    assert "Tolerance: 0.25" in report


def test_validate_writes_structured_run_artifact_in_no_gpu_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")

    exit_code = main(
        [
            "validate",
            str(original),
            str(generated),
            "--tolerance",
            "0.25",
        ]
    )

    payload = json.loads((tmp_path / "runs" / "validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "SKIPPED"
    assert payload["tolerance"] == 0.25
    assert payload["validation_type"] == "no-gpu-safe"


def test_validate_accepts_repeat_in_no_gpu_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")

    exit_code = main(
        [
            "validate",
            str(original),
            str(generated),
            "--repeat",
            "3",
        ]
    )

    report = (tmp_path / "reports" / "validation_report.md").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "runs" / "validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Validation status: SKIPPED" in report
    assert "Repeat count: 3" in report
    assert payload["status"] == "SKIPPED"
    assert payload["repeat"] == 3


def test_benchmark_writes_no_gpu_skip_report_without_speedup(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated = tmp_path / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")
    output = tmp_path / "benchmark_report.md"

    exit_code = main(
        ["benchmark", str(original), str(generated), "--output", str(output)]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "Benchmark report:" in captured.out
    report = output.read_text(encoding="utf-8")
    assert "Benchmark status: SKIPPED" in report
    assert "Speedup:" not in report


def test_benchmark_accepts_repeat_and_warmup_in_no_gpu_mode(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated = tmp_path / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")
    output = tmp_path / "benchmark_report.md"

    exit_code = main(
        [
            "benchmark",
            str(original),
            str(generated),
            "--repeat",
            "3",
            "--warmup",
            "1",
            "--output",
            str(output),
        ]
    )

    report = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Benchmark status: SKIPPED" in report
    assert "Repeat count: 3" in report
    assert "Warmup runs: 1" in report
    assert "Speedup:" not in report


def test_benchmark_writes_structured_run_artifact_in_no_gpu_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")

    exit_code = main(
        [
            "benchmark",
            str(original),
            str(generated),
            "--repeat",
            "3",
            "--warmup",
            "1",
        ]
    )

    payload = json.loads((tmp_path / "runs" / "benchmark.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "SKIPPED"
    assert payload["repeat"] == 3
    assert payload["warmup"] == 1
    assert payload["measurement_method"] == "not_run"
    assert payload["trust_level"] == "not_applicable"
    assert payload["includes_transfer_overhead"] is False
    assert "No benchmark measurement was executed." in payload["limitations"]


def test_benchmark_skip_report_lists_trust_indicators(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated = tmp_path / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")
    output = tmp_path / "benchmark_report.md"

    exit_code = main(
        ["benchmark", str(original), str(generated), "--output", str(output)]
    )

    report = output.read_text(encoding="utf-8")
    assert exit_code == 0
    assert "Measurement method: not_run" in report
    assert "Trust level: not_applicable" in report
    assert "No benchmark measurement was executed." in report


def test_validate_defaults_to_project_reports_dir_for_generated_file(tmp_path, capsys):
    original = Path("examples/monte_carlo_pi/input_cpu.py")
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "input_cpu_gpu.py"
    generated.write_text("import cupy as cp\n", encoding="utf-8")

    exit_code = main(["validate", str(original), str(generated)])

    captured = capsys.readouterr()
    expected = tmp_path / "reports" / "validation_report.md"
    assert exit_code == 0
    assert str(expected) in captured.out
    assert expected.exists()


def test_report_command_prints_release_summary(tmp_path, capsys):
    output_dir = tmp_path / "project"
    assert main(
        [
            "convert",
            "examples/monte_carlo_pi/input_cpu.py",
            "--target",
            "cupy",
            "--output-dir",
            str(output_dir),
        ]
    ) == 0

    exit_code = main(["report", str(output_dir)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# sim2gpu Project Report" in captured.out
    assert "Overall status:" in captured.out
    assert "Generated code: yes" in captured.out
    assert "Unsupported features: 0" in captured.out
    assert "Recommended next step:" in captured.out


def test_run_demo_generates_no_gpu_safe_vertical_slice(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("SIM2GPU_DISABLE_GPU", "1")

    exit_code = main(
        [
            "run-demo",
            "monte_carlo_pi",
            "--output-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()
    project = tmp_path / "monte_carlo_pi"
    assert exit_code == 0
    assert "Demo project:" in captured.out
    assert (project / "generated" / "input_cpu_gpu.py").exists()
    assert (project / "reports" / "conversion_plan.json").exists()
    assert (project / "runs" / "validation.json").exists()
    assert (project / "runs" / "benchmark.json").exists()
    assert "Overall status:" in captured.out


def test_init_example_copies_known_example(tmp_path, capsys):
    exit_code = main(
        ["init-example", "bootstrap_mean", "--output-dir", str(tmp_path)]
    )

    captured = capsys.readouterr()
    target = tmp_path / "bootstrap_mean"
    assert exit_code == 0
    assert "Example initialized:" in captured.out
    assert (target / "input_cpu.py").exists()
    assert (target / "README.md").exists()


def test_list_patterns_prints_supported_patterns(capsys):
    exit_code = main(["list-patterns"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "monte_carlo_independent_trials" in captured.out


def test_list_patterns_can_print_json(capsys):
    exit_code = main(["list-patterns", "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert "monte_carlo_independent_trials" in payload
    assert "permutation_test" in payload
