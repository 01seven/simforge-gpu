from pathlib import Path


EXAMPLES = (
    "monte_carlo_pi",
    "normal_mean_probability",
    "bootstrap_mean",
    "random_walk",
    "permutation_test",
)


def test_examples_include_numpy_cpu_fixture_and_readme():
    root = Path("examples")

    for name in EXAMPLES:
        example_dir = root / name
        source = example_dir / "input_cpu.py"
        readme = example_dir / "README.md"

        assert source.exists(), name
        assert readme.exists(), name
        assert "numpy" in source.read_text(encoding="utf-8")
        readme_text = readme.read_text(encoding="utf-8")
        assert "Example fixture" in readme_text
        assert "speedup" not in readme_text.lower()
