"""Command-line interface for the sim2gpu MVP."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from sim2gpu.backends.cupy import SUPPORTED_PATTERNS
from sim2gpu.backends.registry import get_backend, list_backends
from sim2gpu.pipeline import (
    analyze_file,
    benchmark_files,
    check_project_artifacts,
    collect_demo_status,
    collect_doctor_report,
    collect_doctor_status,
    convert_file,
    explain_plan,
    init_example,
    inspect_project,
    inspect_project_status,
    project_report,
    run_demo,
    validate_files,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sim2gpu")
    subparsers = parser.add_subparsers(dest="command")

    analyze = subparsers.add_parser("analyze", help="Analyze a Python simulation file.")
    analyze.add_argument("input")
    analyze.add_argument("--output-dir")

    convert = subparsers.add_parser("convert", help="Convert a Python simulation file.")
    convert.add_argument("input")
    convert.add_argument("--target", default="cupy")
    convert.add_argument("--output-dir")
    convert.add_argument("--validate", action="store_true")
    convert.add_argument("--benchmark", action="store_true")
    convert.add_argument("--explain", action="store_true")
    convert.add_argument("--tolerance", type=float, default=0.15)
    convert.add_argument("--repeat", type=int, default=1)
    convert.add_argument("--warmup", type=int, default=0)
    convert.add_argument("--dry-run", action="store_true")

    validate = subparsers.add_parser("validate", help="Validate original and generated code.")
    validate.add_argument("original")
    validate.add_argument("generated")
    validate.add_argument("--output")
    validate.add_argument("--tolerance", type=float, default=0.15)
    validate.add_argument("--repeat", type=int, default=1)

    benchmark = subparsers.add_parser("benchmark", help="Benchmark original and generated code.")
    benchmark.add_argument("original")
    benchmark.add_argument("generated")
    benchmark.add_argument("--output")
    benchmark.add_argument("--repeat", type=int, default=1)
    benchmark.add_argument("--warmup", type=int, default=0)

    explain = subparsers.add_parser("explain", help="Explain a conversion plan.")
    explain.add_argument("conversion_plan")
    explain.add_argument("--output")

    init_example = subparsers.add_parser("init-example", help="Initialize an example project.")
    init_example.add_argument("name")
    init_example.add_argument("--output-dir")

    run_demo_parser = subparsers.add_parser("run-demo", help="Run an included end-to-end demo.")
    run_demo_parser.add_argument("name")
    run_demo_parser.add_argument("--output-dir")

    report = subparsers.add_parser("report", help="Print a user-facing project summary.")
    report.add_argument("project_dir")

    demo_status = subparsers.add_parser("demo-status", help="Summarize generated demo projects.")
    demo_status.add_argument("projects_dir", nargs="?", default="projects")
    demo_status.add_argument("--json", action="store_true", dest="json_output")

    inspect = subparsers.add_parser("inspect-project", help="Inspect one generated project.")
    inspect.add_argument("project_dir")
    inspect.add_argument("--json", action="store_true", dest="json_output")

    check_artifacts = subparsers.add_parser(
        "check-artifacts", help="Check generated project artifact completeness."
    )
    check_artifacts.add_argument("project_dir")
    check_artifacts.add_argument("--json", action="store_true", dest="json_output")

    doctor = subparsers.add_parser("doctor", help="Check the local no-GPU MVP environment.")
    doctor.add_argument("--json", action="store_true", dest="json_output")

    list_patterns = subparsers.add_parser(
        "list-patterns", help="List supported simulation patterns."
    )
    list_patterns.add_argument("--json", action="store_true", dest="json_output")

    list_backends_parser = subparsers.add_parser(
        "list-backends", help="List backend implementation status."
    )
    list_backends_parser.add_argument("--json", action="store_true", dest="json_output")
    return parser


def _print_skeleton(command: str) -> int:
    print(
        f"sim2gpu {command} skeleton is declared but not implemented yet.",
        file=sys.stderr,
    )
    return 2


def _handle_convert(args: argparse.Namespace) -> int:
    try:
        result = convert_file(
            args.input,
            target_backend=args.target,
            output_dir=args.output_dir,
            run_validation=args.validate,
            run_benchmark=args.benchmark,
            validation_tolerance=args.tolerance,
            benchmark_repeat=args.repeat,
            benchmark_warmup=args.warmup,
            dry_run=args.dry_run,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if result.exit_code != 0:
        print(result.message, file=sys.stderr)
        print(f"Unsupported report: {result.unsupported_report_path}", file=sys.stderr)
        return result.exit_code

    print(result.message)
    print(f"Conversion plan: {result.plan_path}")
    print(f"Unsupported report: {result.unsupported_report_path}")
    print(f"Validation report: {result.validation_report_path}")
    print(f"Benchmark report: {result.benchmark_report_path}")
    print(f"Explanation report: {result.explanation_report_path}")
    return 0


def _handle_analyze(args: argparse.Namespace) -> int:
    try:
        result = analyze_file(args.input, output_dir=args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Analysis IR: {result.ir_path}")
    return 0


def _handle_explain(args: argparse.Namespace) -> int:
    try:
        result = explain_plan(args.conversion_plan, output_path=args.output)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result.message)
    print(f"Explanation report: {result.output_path}")
    return 0


def _handle_validate(args: argparse.Namespace) -> int:
    try:
        result = validate_files(
            args.original,
            args.generated,
            output_path=args.output,
            tolerance=args.tolerance,
            repeat=args.repeat,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result.message)
    print(f"Validation report: {result.output_path}")
    return 0


def _handle_benchmark(args: argparse.Namespace) -> int:
    try:
        result = benchmark_files(
            args.original,
            args.generated,
            output_path=args.output,
            repeat=args.repeat,
            warmup=args.warmup,
        )
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result.message)
    print(f"Benchmark report: {result.output_path}")
    return 0


def _handle_init_example(args: argparse.Namespace) -> int:
    try:
        target_dir = init_example(args.name, output_dir=args.output_dir)
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"Example initialized: {target_dir}")
    return 0


def _handle_run_demo(args: argparse.Namespace) -> int:
    try:
        result = run_demo(args.name, output_dir=args.output_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result.message)
    return 0


def _handle_report(args: argparse.Namespace) -> int:
    try:
        print(project_report(args.project_dir))
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _handle_demo_status(args: argparse.Namespace) -> int:
    try:
        rows = collect_demo_status(args.projects_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True))
        return 0
    headers = (
        "Project",
        "Backend",
        "Status",
        "Generated",
        "unsupported",
        "Validation",
        "Benchmark",
    )
    print(
        f"{headers[0]:<28} {headers[1]:<8} {headers[2]:<12} {headers[3]:<9} "
        f"{headers[4]:<11} {headers[5]:<10} {headers[6]}"
    )
    for row in rows:
        unsupported = "missing" if row.unsupported_count < 0 else str(row.unsupported_count)
        print(
            f"{row.name:<28} {row.backend:<8} {row.backend_status:<12} "
            f"{row.generated:<9} {unsupported:<11} {row.validation_status:<10} "
            f"{row.benchmark_status}"
        )
    return 0


def _handle_inspect_project(args: argparse.Namespace) -> int:
    try:
        if args.json_output:
            print(json.dumps(inspect_project_status(args.project_dir), indent=2, sort_keys=True))
            return 0
        report = inspect_project(args.project_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(report)
    return 0


def _handle_check_artifacts(args: argparse.Namespace) -> int:
    try:
        status = check_project_artifacts(args.project_dir)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json_output:
        print(json.dumps(status, indent=2, sort_keys=True))
    else:
        print("# Artifact Check")
        print("")
        print(f"Project: {status['project']}")
        print(f"Artifact check: {status['status']}")
        missing = status["missing_artifacts"]
        missing_count = len(missing) if isinstance(missing, list) else 0
        print(f"Missing artifacts: {missing_count}")
        if missing:
            print("")
            print("Missing:")
            for artifact in missing:
                print(f"- {artifact}")
    return 0 if status["status"] == "PASSED" else 1


def _handle_doctor(args: argparse.Namespace) -> int:
    if args.json_output:
        print(json.dumps(collect_doctor_status(), indent=2, sort_keys=True))
        return 0
    print(collect_doctor_report())
    return 0


def _handle_list_backends(args: argparse.Namespace) -> int:
    if args.json_output:
        payload = []
        for backend in list_backends():
            status = "MVP backend / implemented" if backend.name == "cupy" else backend.status
            payload.append(
                {
                    "name": backend.name,
                    "status": status,
                    "implemented": backend.is_implemented,
                }
            )
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    for backend in list_backends():
        status = "MVP backend / implemented" if backend.name == "cupy" else backend.status
        print(f"{backend.name:<6} {status}")
    return 0


def _handle_list_patterns(args: argparse.Namespace) -> int:
    if args.json_output:
        print(json.dumps(list(SUPPORTED_PATTERNS), indent=2, sort_keys=True))
        return 0
    for pattern in SUPPORTED_PATTERNS:
        print(pattern)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help(sys.stderr)
        return 2

    if args.command == "list-backends":
        return _handle_list_backends(args)
    if args.command == "list-patterns":
        return _handle_list_patterns(args)
    if args.command == "analyze":
        return _handle_analyze(args)
    if args.command == "convert":
        return _handle_convert(args)
    if args.command == "explain":
        return _handle_explain(args)
    if args.command == "validate":
        return _handle_validate(args)
    if args.command == "benchmark":
        return _handle_benchmark(args)
    if args.command == "init-example":
        return _handle_init_example(args)
    if args.command == "run-demo":
        return _handle_run_demo(args)
    if args.command == "report":
        return _handle_report(args)
    if args.command == "demo-status":
        return _handle_demo_status(args)
    if args.command == "inspect-project":
        return _handle_inspect_project(args)
    if args.command == "check-artifacts":
        return _handle_check_artifacts(args)
    if args.command == "doctor":
        return _handle_doctor(args)

    return _print_skeleton(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
