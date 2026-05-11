import json
from pathlib import Path

from simforge_gpu.cli import main


EXAMPLES = (
    "monte_carlo_pi",
    "normal_mean_probability",
    "bootstrap_mean",
    "random_walk",
    "permutation_test",
)


def test_all_first_batch_examples_convert_to_artifacts(tmp_path):
    for name in EXAMPLES:
        output_dir = tmp_path / name
        exit_code = main(
            [
                "convert",
                str(Path("examples") / name / "input_cpu.py"),
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )

        reports = output_dir / "reports"
        generated = output_dir / "generated" / "input_cpu_gpu.py"
        assert exit_code == 0, name
        assert generated.exists(), name
        assert (reports / "analysis_ir.json").exists(), name
        assert (reports / "conversion_plan.json").exists(), name
        assert (reports / "unsupported_report.md").exists(), name
        assert (reports / "explanation_report.md").exists(), name
        assert (reports / "validation_report.md").exists(), name
        assert (reports / "benchmark_report.md").exists(), name
        assert (reports / "syntax_report.md").exists(), name
        assert (reports / "quality_report.md").exists(), name

        plan = json.loads((reports / "conversion_plan.json").read_text(encoding="utf-8"))
        assert plan["target_backend"] == "cupy"
        assert plan["backend_status"]["implemented"] is True


def test_partial_conversion_with_unsupported_numpy_api_keeps_numpy_import(tmp_path):
    output_dir = tmp_path / "permutation"

    exit_code = main(
        [
            "convert",
            "examples/permutation_test/input_cpu.py",
            "--target",
            "cupy",
            "--output-dir",
            str(output_dir),
        ]
    )

    generated_source = (output_dir / "generated" / "input_cpu_gpu.py").read_text(
        encoding="utf-8"
    )
    unsupported_report = (output_dir / "reports" / "unsupported_report.md").read_text(
        encoding="utf-8"
    )

    assert exit_code == 0
    assert "import cupy as cp" in generated_source
    assert "import numpy as np" in generated_source
    assert "WARNING: Partial conversion" in generated_source
    assert "np.argsort" in generated_source
    assert "np.argsort" in unsupported_report
