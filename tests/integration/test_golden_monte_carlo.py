from pathlib import Path

import pytest

from simforge_gpu.cli import main


@pytest.mark.parametrize(
    ("example_name", "golden_name"),
    [
        ("monte_carlo_pi", "monte_carlo_pi_expected_gpu.py"),
        ("normal_mean_probability", "normal_mean_probability_expected_gpu.py"),
        ("bootstrap_mean", "bootstrap_mean_expected_gpu.py"),
        ("random_walk", "random_walk_expected_gpu.py"),
        ("permutation_test", "permutation_test_expected_gpu.py"),
    ],
)
def test_example_generated_source_matches_golden(tmp_path, example_name, golden_name):
    exit_code = main(
        [
            "convert",
            f"examples/{example_name}/input_cpu.py",
            "--target",
            "cupy",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    generated = (tmp_path / "generated" / "input_cpu_gpu.py").read_text(
        encoding="utf-8"
    )
    golden = Path("tests/golden", golden_name).read_text(encoding="utf-8")
    assert generated == golden
