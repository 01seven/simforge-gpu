import os
import json
import pathlib
import site

import pytest

from sim2gpu.cli import main


pytestmark = pytest.mark.gpu


def _prepare_windows_cuda_dll_dirs() -> None:
    if os.name != "nt":
        return
    for root in site.getsitepackages():
        for rel in ("nvidia/cuda_nvrtc/bin", "nvidia/cuda_runtime/bin"):
            path = pathlib.Path(root) / rel
            if path.exists():
                os.add_dll_directory(str(path))


def _cupy_kernel_available() -> bool:
    try:
        _prepare_windows_cuda_dll_dirs()
        import cupy as cp

        old_cwd = pathlib.Path.cwd()
        nvrtc_bins = [
            pathlib.Path(root) / "nvidia" / "cuda_nvrtc" / "bin"
            for root in site.getsitepackages()
        ]
        for nvrtc_bin in nvrtc_bins:
            if nvrtc_bin.exists():
                os.chdir(nvrtc_bin)
                break
        try:
            value = float(cp.sum(cp.arange(5)).get())
        finally:
            os.chdir(old_cwd)
        return value == 10.0 and cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def test_validate_executes_generated_cupy_when_gpu_available(tmp_path, capsys):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "validate",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
        ]
    )

    captured = capsys.readouterr()
    report = (output_dir / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Validation report:" in captured.out
    assert "Validation status: PASSED" in report
    assert "Validation type: stochastic" in report
    assert "CPU output:" in report
    assert "GPU output:" in report


def test_benchmark_executes_generated_cupy_when_gpu_available(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "benchmark",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
        ]
    )

    report = (output_dir / "reports" / "benchmark_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Benchmark status: PASSED" in report
    assert "CPU runtime:" in report
    assert "GPU runtime:" in report
    assert "Speedup:" in report


def test_convert_validate_benchmark_flags_run_gpu_reports_when_available(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"

    exit_code = main(
        [
            "convert",
            "examples/monte_carlo_pi/input_cpu.py",
            "--target",
            "cupy",
            "--validate",
            "--benchmark",
            "--output-dir",
            str(output_dir),
        ]
    )

    validation = (output_dir / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    benchmark = (output_dir / "reports" / "benchmark_report.md").read_text(
        encoding="utf-8"
    )
    quality = (output_dir / "reports" / "quality_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Validation status: PASSED" in validation
    assert "Benchmark status: PASSED" in benchmark
    assert "Quality gate: PASSED" in quality


def test_gpu_validate_accepts_custom_tolerance(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "validate",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
            "--tolerance",
            "0.2",
        ]
    )

    report = (output_dir / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Validation status: PASSED" in report
    assert "Tolerance: 0.2" in report
    payload = json.loads((output_dir / "runs" / "validation.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASSED"
    assert payload["tolerance"] == 0.2
    assert payload["absolute_difference"] <= payload["tolerance"]


def test_gpu_validation_repeats_and_records_scalar_differences(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "validate",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
            "--repeat",
            "3",
            "--tolerance",
            "0.2",
        ]
    )

    report = (output_dir / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    payload = json.loads((output_dir / "runs" / "validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Validation status: PASSED" in report
    assert "Repeat count: 3" in report
    assert "Max absolute difference:" in report
    assert payload["status"] == "PASSED"
    assert payload["repeat"] == 3
    assert len(payload["absolute_differences"]) == 3
    assert payload["max_absolute_difference"] <= payload["tolerance"]


def test_gpu_validation_compares_json_array_outputs(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    original = tmp_path / "original.py"
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "generated_gpu.py"
    original.write_text(
        (
            "import json\n"
            "import numpy as np\n"
            "values = np.array([1.0, 2.0, 3.0])\n"
            "print(json.dumps(values.tolist()))\n"
        ),
        encoding="utf-8",
    )
    generated.write_text(
        (
            "import json\n"
            "import cupy as cp\n"
            "values = cp.array([1.0, 2.0, 3.0])\n"
            "print(json.dumps(cp.asnumpy(values).tolist()))\n"
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "validate",
            str(original),
            str(generated),
            "--tolerance",
            "0.000001",
        ]
    )

    report = (tmp_path / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    payload = json.loads((tmp_path / "runs" / "validation.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert "Validation status: PASSED" in report
    assert "Output kind: array" in report
    assert "Element count: 3" in report
    assert payload["status"] == "PASSED"
    assert payload["output_kind"] == "array"
    assert payload["element_count"] == 3
    assert payload["max_absolute_difference"] == 0.0


def test_gpu_benchmark_repeats_and_reports_median_runtime(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "benchmark",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
            "--repeat",
            "3",
            "--warmup",
            "1",
        ]
    )

    report = (output_dir / "reports" / "benchmark_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Benchmark status: PASSED" in report
    assert "Repeat count: 3" in report
    assert "Warmup runs: 1" in report
    assert "CPU median runtime:" in report
    assert "GPU median runtime:" in report
    assert "Speedup:" in report
    assert "Measurement method: subprocess_wall_time" in report
    assert "Trust level: demo_only" in report
    payload = json.loads((output_dir / "runs" / "benchmark.json").read_text(encoding="utf-8"))
    assert payload["status"] == "PASSED"
    assert payload["repeat"] == 3
    assert payload["warmup"] == 1
    assert "cpu_times_seconds" in payload
    assert len(payload["cpu_times_seconds"]) == 3
    assert payload["measurement_method"] == "subprocess_wall_time"
    assert payload["trust_level"] == "demo_only"
    assert payload["includes_transfer_overhead"] is False
    assert "Includes Python process startup and CUDA initialization overhead." in payload["limitations"]


@pytest.mark.parametrize(
    ("example_name", "tolerance"),
    [
        ("normal_mean_probability", "0.03"),
        ("random_walk", "250"),
    ],
)
def test_gpu_validation_covers_additional_safe_examples(tmp_path, example_name, tolerance):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / example_name
    assert (
        main(
            [
                "convert",
                f"examples/{example_name}/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "validate",
            f"examples/{example_name}/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
            "--tolerance",
            tolerance,
        ]
    )

    report = (output_dir / "reports" / "validation_report.md").read_text(
        encoding="utf-8"
    )
    assert exit_code == 0
    assert "Validation status: PASSED" in report
    assert f"Tolerance: {float(tolerance)}" in report
