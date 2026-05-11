from pathlib import Path

from simforge_gpu.cli import build_parser


def test_demo_doc_exists_and_lists_reproducible_no_gpu_commands():
    demo = Path("docs/DEMO.md")

    assert demo.exists()
    text = demo.read_text(encoding="utf-8")
    required = [
        "python -m pip install -e .",
        "simforge doctor",
        "simforge doctor --json",
        "simforge list-backends",
        "simforge list-backends --json",
        "simforge list-patterns --json",
        "simforge analyze examples/monte_carlo_pi/input_cpu.py",
        "simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy",
        "simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run",
        "simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --validate --benchmark",
        "simforge convert examples/monte_carlo_pi/input_cpu.py --target torch",
        "simforge run-demo monte_carlo_pi",
        "simforge report projects/monte_carlo_pi",
        "simforge check-artifacts projects/monte_carlo_pi",
        "simforge demo-status",
        "simforge demo-status --json",
        "simforge inspect-project projects/permutation_test",
        "simforge inspect-project projects/permutation_test --json",
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
