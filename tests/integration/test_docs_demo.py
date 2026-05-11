from pathlib import Path

from sim2gpu.cli import build_parser


def test_demo_doc_exists_and_lists_reproducible_no_gpu_commands():
    demo = Path("docs/DEMO.md")

    assert demo.exists()
    text = demo.read_text(encoding="utf-8")
    required = [
        "python -m pip install -e .",
        "sim2gpu doctor",
        "sim2gpu doctor --json",
        "sim2gpu list-backends",
        "sim2gpu list-backends --json",
        "sim2gpu list-patterns --json",
        "sim2gpu analyze examples/monte_carlo_pi/input_cpu.py",
        "sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy",
        "sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run",
        "sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --validate --benchmark",
        "sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target torch",
        "sim2gpu run-demo monte_carlo_pi",
        "sim2gpu report projects/monte_carlo_pi",
        "sim2gpu check-artifacts projects/monte_carlo_pi",
        "sim2gpu demo-status",
        "sim2gpu demo-status --json",
        "sim2gpu inspect-project projects/permutation_test",
        "sim2gpu inspect-project projects/permutation_test --json",
        "python -m pytest -q",
        "python -m pytest -m gpu -q",
    ]
    for command in required:
        assert command in text


def test_demo_doc_commands_are_declared_in_cli_help():
    parser = build_parser()
    help_text = parser.format_help()

    for command in [
        "analyze",
        "convert",
        "validate",
        "benchmark",
        "explain",
        "init-example",
        "run-demo",
        "report",
        "demo-status",
        "inspect-project",
        "check-artifacts",
        "doctor",
        "list-patterns",
        "list-backends",
    ]:
        assert command in help_text
