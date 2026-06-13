"""SimForge GPU MVP pipeline orchestration."""

from __future__ import annotations

import ast
import json
import os
import shutil
import statistics
import sys
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from simforge_gpu.analyzers.python_ast import AnalysisFacts, analyze_source
from simforge_gpu.analyzers.unsupported_detector import detect_unsupported
from simforge_gpu.backends.registry import get_backend
from simforge_gpu.ir.schema import AnalysisIR, ConvertibleRegion, GpuSuitability, UnsupportedFeature
from simforge_gpu.planners.conversion_plan import ConversionPlan, create_conversion_plan
from simforge_gpu.reporters.markdown import (
    render_benchmark_report,
    render_explanation_report,
    render_quality_report,
    render_suggestion_review_report,
    render_syntax_report,
    render_unsupported_report,
    render_validation_report,
)
from simforge_gpu.reporters.json_report import stable_json
from simforge_gpu.runners.python_subprocess import cupy_kernel_probe, run_python_script
from simforge_gpu.suggestions.merge import merge_review_into_plan
from simforge_gpu.suggestions.review import (
    SuggestionReview,
    review_model_suggestion_payload,
)
from simforge_gpu.tracing.agent_trace import build_agent_trace, write_agent_trace
from simforge_gpu.transpilers.numpy_to_cupy import RewriteResult, rewrite_supported_numpy_calls


@dataclass(frozen=True)
class AnalyzeResult:
    output_dir: Path
    reports_dir: Path
    ir: AnalysisIR
    ir_path: Path


@dataclass(frozen=True)
class ConvertResult:
    exit_code: int
    output_dir: Path
    reports_dir: Path
    generated_path: Path | None
    plan_path: Path
    unsupported_report_path: Path
    explanation_report_path: Path
    validation_report_path: Path
    benchmark_report_path: Path
    syntax_report_path: Path
    quality_report_path: Path
    message: str


@dataclass(frozen=True)
class ReportResult:
    output_path: Path
    message: str


@dataclass(frozen=True)
class SuggestionReviewResult:
    output_dir: Path
    reports_dir: Path
    review: SuggestionReview
    model_suggestion_path: Path
    review_json_path: Path
    review_markdown_path: Path
    message: str


@dataclass(frozen=True)
class DemoStatusRow:
    name: str
    backend: str
    backend_status: str
    generated: str
    unsupported_count: int
    validation_status: str
    benchmark_status: str


def analyze_file(input_path: str | Path, output_dir: str | Path | None = None) -> AnalyzeResult:
    source_path = Path(input_path)
    source = _read_python_source(source_path)
    facts = analyze_source(source)
    unsupported = detect_unsupported(source, target_backend="cupy")
    ir = build_ir(facts, source, unsupported)

    root = _resolve_output_dir(source_path, output_dir)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ir_path = reports_dir / "analysis_ir.json"
    ir_path.write_text(ir.to_json(), encoding="utf-8")
    return AnalyzeResult(output_dir=root, reports_dir=reports_dir, ir=ir, ir_path=ir_path)


def review_suggestion_file(
    suggestion_path: str | Path,
    source_path: str | Path,
    output_dir: str | Path | None = None,
    target_backend: str = "cupy",
) -> SuggestionReviewResult:
    source_file = Path(source_path)
    source = _read_python_source(source_file)
    unsupported = tuple(detect_unsupported(source, target_backend=target_backend))
    root = _resolve_output_dir(source_file, output_dir, target_backend=target_backend)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    return _review_suggestion_for_reports(
        suggestion_path=Path(suggestion_path),
        source_path=source_file,
        reports_dir=reports_dir,
        target_backend=target_backend,
        deterministic_unsupported=unsupported,
    )


def convert_file(
    input_path: str | Path,
    target_backend: str = "cupy",
    output_dir: str | Path | None = None,
    run_validation: bool = False,
    run_benchmark: bool = False,
    validation_tolerance: float = 0.15,
    benchmark_repeat: int = 1,
    benchmark_warmup: int = 0,
    dry_run: bool = False,
    suggestion_path: str | Path | None = None,
) -> ConvertResult:
    source_path = Path(input_path)
    source = _read_python_source(source_path)
    facts = analyze_source(source)
    unsupported = list(detect_unsupported(source, target_backend=target_backend))
    ir = build_ir(facts, source, tuple(unsupported))

    root = _resolve_output_dir(source_path, output_dir, target_backend=target_backend)
    reports_dir = root / "reports"
    generated_dir = root / "generated"
    reports_dir.mkdir(parents=True, exist_ok=True)
    generated_dir.mkdir(parents=True, exist_ok=True)
    analysis_ir_path = reports_dir / "analysis_ir.json"
    analysis_ir_path.write_text(ir.to_json(), encoding="utf-8")

    suggestion_review: SuggestionReview | None = None
    model_suggestion_path: Path | None = None
    suggestion_review_json_path = reports_dir / "suggestion_review.json"
    suggestion_review_markdown_path = reports_dir / "suggestion_review.md"
    agent_trace_path = reports_dir / "agent_trace.json"
    if suggestion_path is not None:
        suggestion_result = _review_suggestion_for_reports(
            suggestion_path=Path(suggestion_path),
            source_path=source_path,
            reports_dir=reports_dir,
            target_backend=target_backend,
            deterministic_unsupported=tuple(unsupported),
        )
        suggestion_review = suggestion_result.review
        model_suggestion_path = suggestion_result.model_suggestion_path
        if suggestion_review.is_rejected:
            trace = build_agent_trace(
                source_file=source_path,
                target_backend=target_backend,
                model_suggestion_path=_relative_to(root, model_suggestion_path),
                suggestion_review_path=_relative_to(root, suggestion_review_json_path),
                review=suggestion_review,
                harness_gates=(
                    _gate("source_intake", "PASSED", _display_path(source_path)),
                    _gate("static_analysis", "PASSED", _relative_to(root, analysis_ir_path)),
                    _gate(
                        "suggestion_review",
                        suggestion_review.status,
                        _relative_to(root, suggestion_review_json_path),
                    ),
                ),
                artifacts={
                    "analysis_ir": _relative_to(root, analysis_ir_path),
                    "model_suggestion": _relative_to(root, model_suggestion_path),
                    "suggestion_review": _relative_to(root, suggestion_review_json_path),
                },
            )
            write_agent_trace(agent_trace_path, trace)
            return ConvertResult(
                exit_code=2,
                output_dir=root,
                reports_dir=reports_dir,
                generated_path=None,
                plan_path=reports_dir / "conversion_plan.json",
                unsupported_report_path=reports_dir / "unsupported_report.md",
                explanation_report_path=reports_dir / "explanation_report.md",
                validation_report_path=reports_dir / "validation_report.md",
                benchmark_report_path=reports_dir / "benchmark_report.md",
                syntax_report_path=reports_dir / "syntax_report.md",
                quality_report_path=reports_dir / "quality_report.md",
                message=f"Suggestion review rejected: {suggestion_review_json_path}",
            )

    backend = get_backend(target_backend)
    rewrite = RewriteResult(source=source, unsupported_features=(), changes=())
    generated_path: Path | None = None

    if backend.is_implemented:
        rewrite = rewrite_supported_numpy_calls(source)
        unsupported.extend(rewrite.unsupported_features)
        if suggestion_review is not None:
            unsupported.extend(_review_unsupported_for_plan(suggestion_review))
        if not dry_run:
            generated_source = _add_generation_header(
                rewrite.source, partial=bool(unsupported)
            )
            _syntax_check(generated_source, source_path)
            generated_path = generated_dir / f"{source_path.stem}_gpu.py"
            generated_path.write_text(generated_source, encoding="utf-8")

    plan = create_conversion_plan(
        ir,
        source_file=str(source_path),
        target_backend=target_backend,
        changes=rewrite.changes,
        unsupported_features=tuple(unsupported),
    )
    if suggestion_review is not None:
        plan = merge_review_into_plan(plan, suggestion_review)

    plan_path = reports_dir / "conversion_plan.json"
    unsupported_report_path = reports_dir / "unsupported_report.md"
    explanation_report_path = reports_dir / "explanation_report.md"
    validation_report_path = reports_dir / "validation_report.md"
    benchmark_report_path = reports_dir / "benchmark_report.md"
    syntax_report_path = reports_dir / "syntax_report.md"
    quality_report_path = reports_dir / "quality_report.md"

    plan_path.write_text(plan.to_json(), encoding="utf-8")
    unsupported_report_path.write_text(
        render_unsupported_report(plan.unsupported_features), encoding="utf-8"
    )
    validation_result = (
        _execute_validation(source_path, generated_path, tolerance=validation_tolerance)
        if run_validation and generated_path is not None
        else _validation_skip(plan)
    )
    benchmark_result = (
        _execute_benchmark(source_path, generated_path, repeat=benchmark_repeat, warmup=benchmark_warmup)
        if run_benchmark and generated_path is not None
        else _benchmark_skip(
            backend.display_name if backend.is_implemented else backend.name,
            repeat=benchmark_repeat,
            warmup=benchmark_warmup,
        )
    )
    _write_run_artifact(root / "runs" / "validation.json", validation_result)
    _write_run_artifact(root / "runs" / "benchmark.json", benchmark_result)
    validation_report_path.write_text(render_validation_report(validation_result), encoding="utf-8")
    benchmark_report_path.write_text(render_benchmark_report(benchmark_result), encoding="utf-8")
    syntax_result = _syntax_result(generated_path)
    if dry_run:
        syntax_result = {
            "status": "SKIPPED",
            "reason": "Dry run requested; generated source was not written.",
        }
    syntax_report_path.write_text(render_syntax_report(syntax_result), encoding="utf-8")
    quality_result = _quality_result(
        syntax_result=syntax_result,
        unsupported_count=len(plan.unsupported_features),
        validation_result=validation_result,
        benchmark_result=benchmark_result,
        suggestion_review_status=(
            suggestion_review.status if suggestion_review is not None else None
        ),
    )
    quality_report_path.write_text(render_quality_report(quality_result), encoding="utf-8")
    explanation_report_path.write_text(
        _render_explanation(
            plan,
            generated_path,
            validation_result,
            benchmark_result,
            suggestion_review=suggestion_review,
        ),
        encoding="utf-8",
    )

    if suggestion_review is not None and model_suggestion_path is not None:
        trace = build_agent_trace(
            source_file=source_path,
            target_backend=target_backend,
            model_suggestion_path=_relative_to(root, model_suggestion_path),
            suggestion_review_path=_relative_to(root, suggestion_review_json_path),
            review=suggestion_review,
            harness_gates=(
                _gate("source_intake", "PASSED", _display_path(source_path)),
                _gate("static_analysis", "PASSED", _relative_to(root, analysis_ir_path)),
                _gate(
                    "backend_policy",
                    "PASSED" if backend.is_implemented else "REJECTED",
                    _relative_to(root, plan_path),
                ),
                _gate(
                    "suggestion_review",
                    suggestion_review.status,
                    _relative_to(root, suggestion_review_json_path),
                ),
                _gate("syntax", str(syntax_result["status"]), _relative_to(root, syntax_report_path)),
                _gate(
                    "validation",
                    str(validation_result["status"]),
                    _relative_to(root, validation_report_path),
                ),
                _gate(
                    "benchmark",
                    str(benchmark_result["status"]),
                    _relative_to(root, benchmark_report_path),
                ),
            ),
            artifacts=_agent_trace_artifacts(
                root=root,
                analysis_ir_path=analysis_ir_path,
                plan_path=plan_path,
                unsupported_report_path=unsupported_report_path,
                explanation_report_path=explanation_report_path,
                validation_report_path=validation_report_path,
                benchmark_report_path=benchmark_report_path,
                syntax_report_path=syntax_report_path,
                quality_report_path=quality_report_path,
                model_suggestion_path=model_suggestion_path,
                suggestion_review_path=suggestion_review_json_path,
                generated_path=generated_path,
            ),
        )
        write_agent_trace(agent_trace_path, trace)

    if not backend.is_implemented:
        return ConvertResult(
            exit_code=2,
            output_dir=root,
            reports_dir=reports_dir,
            generated_path=None,
            plan_path=plan_path,
            unsupported_report_path=unsupported_report_path,
            explanation_report_path=explanation_report_path,
            validation_report_path=validation_report_path,
            benchmark_report_path=benchmark_report_path,
            syntax_report_path=syntax_report_path,
            quality_report_path=quality_report_path,
            message=backend.unsupported_reason(),
        )

    return ConvertResult(
        exit_code=0,
        output_dir=root,
        reports_dir=reports_dir,
        generated_path=generated_path,
        plan_path=plan_path,
        unsupported_report_path=unsupported_report_path,
        explanation_report_path=explanation_report_path,
        validation_report_path=validation_report_path,
        benchmark_report_path=benchmark_report_path,
        syntax_report_path=syntax_report_path,
        quality_report_path=quality_report_path,
        message=(
            f"Dry run conversion plan: {plan_path}"
            if dry_run
            else f"Generated CuPy code: {generated_path}"
        ),
    )


def run_demo(name: str, output_dir: str | Path | None = None) -> ReportResult:
    source_path = Path("examples") / name / "input_cpu.py"
    if not source_path.exists():
        known = ", ".join(sorted(path.name for path in Path("examples").iterdir() if path.is_dir()))
        raise ValueError(f"Unknown demo '{name}'. Available demos: {known}.")
    root = (Path(output_dir) if output_dir else Path("projects")) / name
    convert_file(
        source_path,
        target_backend="cupy",
        output_dir=root,
        run_validation=True,
        run_benchmark=True,
    )
    report = project_report(root)
    return ReportResult(output_path=root, message=f"Demo project: {root}\n\n{report}")


def _review_suggestion_for_reports(
    suggestion_path: Path,
    source_path: Path,
    reports_dir: Path,
    target_backend: str,
    deterministic_unsupported: tuple[UnsupportedFeature, ...],
) -> SuggestionReviewResult:
    _assert_existing_file(suggestion_path, "Model suggestion file")
    payload = suggestion_path.read_text(encoding="utf-8")
    model_suggestion_path = reports_dir / "model_suggestion.json"
    model_suggestion_path.write_text(payload, encoding="utf-8")
    review = review_model_suggestion_payload(
        payload,
        suggestion_file=Path(_relative_to(reports_dir.parent, model_suggestion_path)),
        source_file=source_path,
        target_backend=target_backend,
        deterministic_unsupported=deterministic_unsupported,
    )
    review_json_path = reports_dir / "suggestion_review.json"
    review_markdown_path = reports_dir / "suggestion_review.md"
    review_json_path.write_text(review.to_json(), encoding="utf-8")
    review_markdown_path.write_text(render_suggestion_review_report(review), encoding="utf-8")
    return SuggestionReviewResult(
        output_dir=reports_dir.parent,
        reports_dir=reports_dir,
        review=review,
        model_suggestion_path=model_suggestion_path,
        review_json_path=review_json_path,
        review_markdown_path=review_markdown_path,
        message=f"Suggestion review: {review.status}",
    )


def project_report(project_dir: str | Path) -> str:
    status = inspect_project_status(project_dir)
    artifacts = status["artifacts"]
    backend = status["backend"]
    unsupported_count = len(status["unsupported"]) if isinstance(status["unsupported"], list) else 0
    generated = "yes" if isinstance(artifacts, dict) and any(
        path.startswith("generated/") and artifact_status == "present"
        for path, artifact_status in artifacts.items()
    ) else "no"
    validation_status = str(status["validation_status"])
    benchmark_status = str(status["benchmark_status"])
    syntax_status = str(status["syntax_status"])
    quality_gate = str(status["quality_gate"])
    overall = _overall_project_status(
        generated=generated,
        unsupported_count=unsupported_count,
        validation_status=validation_status,
        syntax_status=syntax_status,
        quality_gate=quality_gate,
    )
    recommendation = _project_recommendation(
        unsupported_count=unsupported_count,
        validation_status=validation_status,
        benchmark_status=benchmark_status,
        generated=generated,
    )
    lines = [
        "# SimForge GPU Project Report",
        "",
        f"Project: {status['project']}",
        f"Path: {status['path']}",
        f"Overall status: {overall}",
        "",
        "## Summary",
        "",
        f"Target backend: {backend['target_backend'] if isinstance(backend, dict) else 'missing'}",
        f"Backend status: {backend['status'] if isinstance(backend, dict) else 'missing'}",
        f"Generated code: {generated}",
        f"Unsupported features: {unsupported_count}",
        f"Validation status: {validation_status}",
        f"Benchmark status: {benchmark_status}",
        f"Syntax status: {syntax_status}",
        f"Quality gate: {quality_gate}",
        "",
        f"Recommended next step: {recommendation}",
        "",
    ]
    return "\n".join(lines)


def explain_plan(
    conversion_plan_path: str | Path, output_path: str | Path | None = None
) -> ReportResult:
    plan_path = Path(conversion_plan_path)
    if not plan_path.exists():
        raise FileNotFoundError(f"Conversion plan not found: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = _render_plan_explanation(plan)
    destination = Path(output_path) if output_path else plan_path.with_name("explanation_report.md")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    return ReportResult(output_path=destination, message=report)


def validate_files(
    original_path: str | Path,
    generated_path: str | Path,
    output_path: str | Path | None = None,
    tolerance: float = 0.15,
    repeat: int = 1,
) -> ReportResult:
    original = Path(original_path)
    generated = Path(generated_path)
    _assert_existing_file(original, "Original file")
    _assert_existing_file(generated, "Generated file")
    result = _execute_validation(original, generated, tolerance=tolerance, repeat=repeat)
    report = render_validation_report(result)
    destination = (
        Path(output_path)
        if output_path
        else _default_report_path(generated, "validation_report.md")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    _write_run_artifact(_default_run_path(generated, "validation.json"), result)
    return ReportResult(output_path=destination, message=report)


def benchmark_files(
    original_path: str | Path,
    generated_path: str | Path,
    output_path: str | Path | None = None,
    repeat: int = 1,
    warmup: int = 0,
) -> ReportResult:
    original = Path(original_path)
    generated = Path(generated_path)
    _assert_existing_file(original, "Original file")
    _assert_existing_file(generated, "Generated file")
    result = _execute_benchmark(original, generated, repeat=repeat, warmup=warmup)
    report = render_benchmark_report(result)
    destination = (
        Path(output_path)
        if output_path
        else _default_report_path(generated, "benchmark_report.md")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report, encoding="utf-8")
    _write_run_artifact(_default_run_path(generated, "benchmark.json"), result)
    return ReportResult(output_path=destination, message=report)


def init_example(name: str, output_dir: str | Path | None = None) -> Path:
    source_dir = Path("examples") / name
    if not source_dir.exists():
        known = ", ".join(sorted(path.name for path in Path("examples").iterdir() if path.is_dir()))
        raise ValueError(f"Unknown example '{name}'. Available examples: {known}.")
    root = Path(output_dir) if output_dir else Path("projects") / "examples"
    target_dir = root / name
    if target_dir.exists():
        raise FileExistsError(f"Example target already exists: {target_dir}")
    target_dir.mkdir(parents=True)
    for filename in ("input_cpu.py", "README.md"):
        shutil.copy2(source_dir / filename, target_dir / filename)
    return target_dir


def collect_demo_status(projects_dir: str | Path = "projects") -> tuple[DemoStatusRow, ...]:
    root = Path(projects_dir)
    if not root.exists():
        raise FileNotFoundError(f"Projects directory not found: {root}")
    rows: list[DemoStatusRow] = []
    for project in sorted(path for path in root.iterdir() if path.is_dir()):
        reports = project / "reports"
        generated_dir = project / "generated"
        plan = _read_json_if_exists(reports / "conversion_plan.json")
        backend_status = plan.get("backend_status", {}) if isinstance(plan, dict) else {}
        if not isinstance(backend_status, dict):
            backend_status = {}
        rows.append(
            DemoStatusRow(
                name=project.name,
                backend=str(plan.get("target_backend", "missing")) if isinstance(plan, dict) else "missing",
                backend_status=str(backend_status.get("status", "missing")),
                generated="yes" if any(generated_dir.glob("*.py")) else "no",
                unsupported_count=_unsupported_count(reports / "unsupported_report.md"),
                validation_status=_report_status(
                    reports / "validation_report.md", "Validation status:"
                ),
                benchmark_status=_report_status(
                    reports / "benchmark_report.md", "Benchmark status:"
                ),
            )
        )
    return tuple(rows)


def inspect_project(project_dir: str | Path) -> str:
    status = inspect_project_status(project_dir)

    artifacts = status["artifacts"]
    backend = status["backend"]
    unsupported_items = status["unsupported"]
    next_steps = status["next_steps"]

    lines = [
        "# Project Inspection",
        "",
        f"Project: {status['project']}",
        f"Path: {status['path']}",
        "",
        "## Backend",
        "",
        f"Target backend: {backend['target_backend'] if isinstance(backend, dict) else 'missing'}",
        f"Backend status: {backend['status'] if isinstance(backend, dict) else 'missing'}",
        "",
        "## Artifacts",
        "",
    ]
    if isinstance(artifacts, dict):
        for artifact, artifact_status in artifacts.items():
            lines.append(f"- {artifact}: {artifact_status}")

    lines.extend(["", "## Unsupported", ""])
    if unsupported_items:
        lines.extend(f"- {item}" for item in unsupported_items)
    else:
        lines.append("- No unsupported features recorded.")

    lines.extend(
        [
            "",
            "## Validation And Benchmark",
            "",
            f"Validation status: {status['validation_status']}",
            f"Benchmark status: {status['benchmark_status']}",
            f"Syntax status: {status['syntax_status']}",
            f"Quality gate: {status['quality_gate']}",
            "",
            "## Next steps",
            "",
        ]
    )
    lines.extend(f"- {step}" for step in next_steps)
    lines.append("")
    return "\n".join(lines)


def inspect_project_status(project_dir: str | Path) -> dict[str, object]:
    project = Path(project_dir)
    if not project.exists():
        raise FileNotFoundError(f"Project directory not found: {project}")
    if not project.is_dir():
        raise ValueError(f"Project path is not a directory: {project}")

    reports = project / "reports"
    generated_dir = project / "generated"
    plan = _read_json_if_exists(reports / "conversion_plan.json")
    backend_status = plan.get("backend_status", {}) if isinstance(plan, dict) else {}
    if not isinstance(backend_status, dict):
        backend_status = {}

    generated_files = sorted(generated_dir.glob("*.py")) if generated_dir.exists() else []
    unsupported_items = _unsupported_items(reports / "unsupported_report.md")
    validation_status = _report_status(reports / "validation_report.md", "Validation status:")
    benchmark_status = _report_status(reports / "benchmark_report.md", "Benchmark status:")
    syntax_status = _report_status(reports / "syntax_report.md", "Syntax status:")
    quality_gate = _report_status(reports / "quality_report.md", "Quality gate:")

    artifacts: dict[str, str] = {}
    if generated_files:
        for generated in generated_files:
            artifacts[f"generated/{generated.name}"] = "present"
    else:
        artifacts["generated/*.py"] = "missing"
    artifact_paths = [
        reports / "analysis_ir.json",
        reports / "conversion_plan.json",
        reports / "unsupported_report.md",
        reports / "explanation_report.md",
        reports / "validation_report.md",
        reports / "benchmark_report.md",
        reports / "syntax_report.md",
        reports / "quality_report.md",
        project / "runs" / "validation.json",
        project / "runs" / "benchmark.json",
    ]
    advisory_artifact_paths = [
        reports / "model_suggestion.json",
        reports / "suggestion_review.json",
        reports / "suggestion_review.md",
        reports / "agent_trace.json",
    ]
    if any(artifact.exists() for artifact in advisory_artifact_paths):
        artifact_paths.extend(advisory_artifact_paths)
    for artifact in artifact_paths:
        status = "present" if artifact.exists() else "missing"
        try:
            display = artifact.relative_to(project)
        except ValueError:
            display = artifact
        artifacts[_display_path(display)] = status

    next_steps = []
    if unsupported_items:
        next_steps.append("Review `unsupported_report.md` before treating generated code as runnable.")
    if validation_status == "SKIPPED":
        next_steps.append("Run real validation only in an environment with required GPU dependencies.")
    if benchmark_status == "SKIPPED":
        next_steps.append("Run benchmark only after validation is meaningful; do not infer speedup from skipped reports.")
    if syntax_status not in {"PASSED", "SKIPPED"}:
        next_steps.append("Regenerate or inspect generated source before using this project.")
    if not unsupported_items and validation_status == "SKIPPED":
        next_steps.append("The generated code is a syntax-checked candidate, not a validated GPU result.")

    return {
        "project": project.name,
        "path": _display_path(project),
        "backend": {
            "target_backend": plan.get("target_backend", "missing") if isinstance(plan, dict) else "missing",
            "status": backend_status.get("status", "missing"),
        },
        "artifacts": artifacts,
        "unsupported": list(unsupported_items),
        "validation_status": validation_status,
        "benchmark_status": benchmark_status,
        "syntax_status": syntax_status,
        "quality_gate": quality_gate,
        "next_steps": next_steps,
    }


def check_project_artifacts(project_dir: str | Path) -> dict[str, object]:
    status = inspect_project_status(project_dir)
    artifacts = status["artifacts"]
    if not isinstance(artifacts, dict):
        artifacts = {}
    required = [
        "reports/analysis_ir.json",
        "reports/conversion_plan.json",
        "reports/unsupported_report.md",
        "reports/explanation_report.md",
        "reports/validation_report.md",
        "reports/benchmark_report.md",
        "reports/syntax_report.md",
        "reports/quality_report.md",
    ]
    backend = status["backend"]
    backend_status = backend.get("status", "missing") if isinstance(backend, dict) else "missing"
    if backend_status == "implemented":
        generated_present = any(
            path.startswith("generated/") and path.endswith(".py") and artifact_status == "present"
            for path, artifact_status in artifacts.items()
        )
        if not generated_present:
            required.append("generated/*.py")
    advisory_required = [
        "reports/model_suggestion.json",
        "reports/suggestion_review.json",
        "reports/suggestion_review.md",
        "reports/agent_trace.json",
    ]
    if any(artifacts.get(artifact) == "present" for artifact in advisory_required):
        required.extend(advisory_required)
    missing = [
        artifact
        for artifact in required
        if artifacts.get(artifact) != "present"
    ]
    return {
        "project": status["project"],
        "path": status["path"],
        "status": "PASSED" if not missing else "FAILED",
        "missing_artifacts": missing,
        "artifact_count": len(artifacts),
        "backend": status["backend"],
        "syntax_status": status["syntax_status"],
        "quality_gate": status["quality_gate"],
        "validation_status": status["validation_status"],
        "benchmark_status": status["benchmark_status"],
    }


def _overall_project_status(
    generated: str,
    unsupported_count: int,
    validation_status: str,
    syntax_status: str,
    quality_gate: str,
) -> str:
    if syntax_status == "FAILED" or quality_gate == "FAILED" or validation_status == "FAILED":
        return "FAILED"
    if generated != "yes":
        return "NO_GENERATED_CODE"
    if unsupported_count:
        return "NEEDS_REVIEW"
    if validation_status == "PASSED":
        return "READY"
    if validation_status == "SKIPPED":
        return "CANDIDATE_UNVALIDATED"
    return "UNKNOWN"


def _project_recommendation(
    unsupported_count: int,
    validation_status: str,
    benchmark_status: str,
    generated: str,
) -> str:
    if generated != "yes":
        return "Generate CuPy code with `simforge convert --target cupy`."
    if unsupported_count:
        return "Review unsupported_report.md before running generated code."
    if validation_status == "SKIPPED":
        return "Run validation in a CuPy/CUDA environment before trusting results."
    if validation_status == "FAILED":
        return "Inspect validation_report.md and do not use the generated code yet."
    if benchmark_status == "SKIPPED":
        return "Benchmark only after validation is meaningful; skipped benchmark is not a speedup claim."
    return "Review reports and use benchmark numbers only with their trust indicators."


def collect_doctor_report() -> str:
    """Return a no-GPU-safe environment report."""

    status = collect_doctor_status()
    lines = [
        "# SimForge GPU Doctor",
        "",
        f"Python: {status['python']}",
        f"Package import: {status['package_import']}",
        "",
        "## Backends",
        "",
    ]
    backends = status["backends"]
    if isinstance(backends, dict):
        for backend, backend_status in backends.items():
            lines.append(f"{backend:<6} {backend_status}")
    optional = status["optional_gpu_dependencies"]
    workspace = status["workspace"]
    lines.extend(
        [
            "",
            "## Optional GPU Dependencies",
            "",
            f"CuPy import: {optional['cupy_import'] if isinstance(optional, dict) else 'unknown'}",
            f"CUDA execution: {optional['cuda_execution'] if isinstance(optional, dict) else 'unknown'}",
            f"CUDA execution reason: {optional['cuda_execution_reason'] if isinstance(optional, dict) else 'unknown'}",
            "",
            "## Workspace",
            "",
            f"examples/: {workspace['examples'] if isinstance(workspace, dict) else 'unknown'}",
            f"projects/: {workspace['projects'] if isinstance(workspace, dict) else 'unknown'}",
            "",
            "## Notes",
            "",
            "- No-GPU MVP commands remain available without CuPy or CUDA.",
            "- Missing CuPy means generated code cannot be executed locally yet.",
            "- Benchmark reports must remain SKIPPED until real GPU execution is available.",
            "",
        ]
    )
    return "\n".join(lines)


def collect_doctor_status() -> dict[str, object]:
    """Return structured no-GPU-safe environment status."""

    cuda_available, cuda_reason = cupy_kernel_probe()
    return {
        "python": sys.version.split()[0],
        "package_import": "OK",
        "backends": {
            "cupy": "MVP backend / implemented",
            "torch": "planned",
            "jax": "planned",
            "numba": "planned",
            "cudf": "planned",
        },
        "optional_gpu_dependencies": {
            "cupy_import": "available" if find_spec("cupy") else "not installed",
            "cuda_execution": "available" if cuda_available else "unavailable",
            "cuda_execution_reason": cuda_reason,
        },
        "workspace": {
            "examples": "present" if Path("examples").exists() else "missing",
            "projects": "present" if Path("projects").exists() else "missing",
        },
        "notes": [
            "No-GPU MVP commands remain available without CuPy or CUDA.",
            "Missing CuPy means generated code cannot be executed locally yet.",
            "Benchmark reports must remain SKIPPED until real GPU execution is available.",
        ],
    }


def build_ir(
    facts: AnalysisFacts,
    source: str,
    unsupported_features: tuple[UnsupportedFeature, ...] = (),
) -> AnalysisIR:
    return AnalysisIR(
        imports=facts.imports,
        patterns=_detect_patterns(facts, source),
        random_calls=facts.random_calls,
        outputs=facts.outputs,
        convertible_regions=tuple(
            ConvertibleRegion(
                type=str(loop["type"]),
                line_start=int(loop["line_start"]),
                line_end=int(loop["line_end"]),
                strategy="batch_vectorization",
            )
            for loop in facts.for_loops
        ),
        unsupported_features=unsupported_features,
        gpu_suitability=_suitability(facts, source),
    )


def _detect_patterns(facts: AnalysisFacts, source: str) -> tuple[str, ...]:
    patterns: list[str] = []
    if "np.random.uniform" in facts.random_calls and "np.mean" in source:
        patterns.append("monte_carlo_independent_trials")
    if "np.random.normal" in facts.random_calls and "np.mean" in source:
        patterns.append("normal_mean_probability")
    if "np.random.binomial" in facts.random_calls and "np.sum" in source:
        patterns.append("random_walk")
    if facts.random_calls and not patterns:
        patterns.append("monte_carlo_loop")
    return tuple(patterns)


def _suitability(facts: AnalysisFacts, source: str) -> GpuSuitability:
    reasons = ["The source code is NumPy-style, so CuPy is the safest MVP backend."]
    warnings = [
        "Validation and benchmark execution are skipped in no-GPU MVP mode.",
        "TorchBackend may be useful in the future but is not implemented in the MVP.",
    ]
    if facts.random_calls:
        reasons.append("Detected NumPy random simulation calls.")
    if facts.for_loops:
        reasons.append("Detected Python-level loops that may be GPU candidates after analysis.")
    if "size=" in source:
        reasons.append("Detected vectorized random sampling with an explicit size argument.")
    return GpuSuitability(
        gpu_suitable=bool(facts.random_calls or "np." in source),
        confidence="medium" if facts.random_calls else "low",
        recommended_backend="cupy",
        future_backend_candidates=("torch",),
        reasons=tuple(reasons),
        warnings=tuple(warnings),
    )


def _read_python_source(source_path: Path) -> str:
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")
    if source_path.suffix != ".py":
        raise ValueError(f"Only Python .py files are supported in the MVP: {source_path}")
    return source_path.read_text(encoding="utf-8")


def _assert_existing_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")


def _default_report_path(generated_path: Path, filename: str) -> Path:
    if generated_path.parent.name == "generated":
        return generated_path.parent.parent / "reports" / filename
    return Path(filename)


def _default_run_path(generated_path: Path, filename: str) -> Path:
    if generated_path.parent.name == "generated":
        return generated_path.parent.parent / "runs" / filename
    return Path(filename)


def _write_run_artifact(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(stable_json(payload), encoding="utf-8")


def _review_unsupported_for_plan(review: SuggestionReview) -> list[UnsupportedFeature]:
    features: list[UnsupportedFeature] = []
    for feature in review.unsupported_features:
        source = str(feature.get("source", "model_hypothesis"))
        if source == "static_detector":
            continue
        features.append(
            UnsupportedFeature(
                code=str(feature.get("code", "model_hypothesis")),
                reason=str(feature.get("reason", "")),
                action=str(feature.get("action", "Review this model advisory item.")),
                category=str(feature.get("category", "model_hypothesis")),
                source=source,
            )
        )
    return features


def _gate(name: str, status: str, artifact: str) -> dict[str, object]:
    return {
        "name": name,
        "status": status,
        "artifact": artifact,
    }


def _agent_trace_artifacts(
    root: Path,
    analysis_ir_path: Path,
    plan_path: Path,
    unsupported_report_path: Path,
    explanation_report_path: Path,
    validation_report_path: Path,
    benchmark_report_path: Path,
    syntax_report_path: Path,
    quality_report_path: Path,
    model_suggestion_path: Path,
    suggestion_review_path: Path,
    generated_path: Path | None,
) -> dict[str, str]:
    artifacts = {
        "analysis_ir": _relative_to(root, analysis_ir_path),
        "conversion_plan": _relative_to(root, plan_path),
        "unsupported_report": _relative_to(root, unsupported_report_path),
        "explanation_report": _relative_to(root, explanation_report_path),
        "validation_report": _relative_to(root, validation_report_path),
        "benchmark_report": _relative_to(root, benchmark_report_path),
        "syntax_report": _relative_to(root, syntax_report_path),
        "quality_report": _relative_to(root, quality_report_path),
        "model_suggestion": _relative_to(root, model_suggestion_path),
        "suggestion_review": _relative_to(root, suggestion_review_path),
    }
    if generated_path is not None:
        artifacts["generated_source"] = _relative_to(root, generated_path)
    return artifacts


def _relative_to(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_output_dir(
    source_path: Path, output_dir: str | Path | None, target_backend: str = "cupy"
) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    base_name = source_path.parent.name or source_path.stem
    if target_backend != "cupy":
        base_name = f"{base_name}_{target_backend}"
    return Path("projects") / base_name


def _add_generation_header(source: str, partial: bool = False) -> str:
    warning = (
        "# WARNING: Partial conversion. Review unsupported_report.md before execution.\n"
        if partial
        else ""
    )
    return (
        "\"\"\"Generated by SimForge GPU MVP. Review reports before execution.\"\"\"\n"
        f"{warning}\n"
        + source
    )


def _syntax_check(source: str, source_path: Path) -> None:
    try:
        ast.parse(source)
    except SyntaxError as exc:
        raise SyntaxError(f"Generated code syntax check failed for {source_path}: {exc}") from exc


def _validation_skip(plan: ConversionPlan) -> dict[str, object]:
    return {
        "status": "SKIPPED",
        "validation_type": plan.validation_strategy,
        "equivalence_level": plan.equivalence_level,
        "reason": "No-GPU MVP mode does not execute generated GPU code.",
    }


def _benchmark_skip(backend_name: str, repeat: int = 1, warmup: int = 0) -> dict[str, object]:
    return {
        "status": "SKIPPED",
        "backend": backend_name,
        "repeat": repeat,
        "warmup": warmup,
        "reason": "No CUDA-compatible GPU benchmark was run in MVP no-GPU mode.",
        **_benchmark_trust_metadata(measured=False),
    }


def _execute_validation(
    original: Path,
    generated: Path,
    tolerance: float = 0.15,
    repeat: int = 1,
) -> dict[str, object]:
    repeat = max(1, repeat)
    if os.environ.get("SIMFORGE_DISABLE_GPU") == "1":
        return {
            "status": "SKIPPED",
            "validation_type": "no-gpu-safe",
            "equivalence_level": "not evaluated",
            "tolerance": tolerance,
            "repeat": repeat,
            "reason": "GPU execution disabled by SIMFORGE_DISABLE_GPU=1.",
        }
    available, reason = cupy_kernel_probe()
    if not available:
        return {
            "status": "SKIPPED",
            "validation_type": "no-gpu-safe",
            "equivalence_level": "not evaluated",
            "tolerance": tolerance,
            "repeat": repeat,
            "reason": f"CuPy GPU execution is unavailable: {reason}",
        }
    cpu_runs = [run_python_script(original) for _ in range(repeat)]
    gpu_runs = [run_python_script(generated, use_cuda_workdir=True) for _ in range(repeat)]
    failed_cpu = next((run for run in cpu_runs if run.status != "PASSED"), None)
    failed_gpu = next((run for run in gpu_runs if run.status != "PASSED"), None)
    if failed_cpu is not None or failed_gpu is not None:
        return {
            "status": "FAILED",
            "validation_type": "execution",
            "equivalence_level": "not evaluated",
            "repeat": repeat,
            "reason": _execution_failure_reason(failed_cpu or cpu_runs[0], failed_gpu or gpu_runs[0]),
        }
    cpu_outputs_raw = [run.parsed_output for run in cpu_runs]
    gpu_outputs_raw = [run.parsed_output for run in gpu_runs]
    comparisons = [
        _compare_numeric_outputs(cpu_output, gpu_output)
        for cpu_output, gpu_output in zip(cpu_outputs_raw, gpu_outputs_raw)
    ]
    failed_comparison = next((comparison for comparison in comparisons if not comparison["ok"]), None)
    if failed_comparison is not None:
        return {
            "status": "FAILED",
            "validation_type": "execution",
            "equivalence_level": "not evaluated",
            "repeat": repeat,
            "reason": str(failed_comparison["reason"]),
            "cpu_output": cpu_runs[-1].stdout.strip(),
            "gpu_output": gpu_runs[-1].stdout.strip(),
        }
    run_differences = [float(comparison["max_absolute_difference"]) for comparison in comparisons]
    all_differences = [
        difference
        for comparison in comparisons
        for difference in comparison["absolute_differences"]
    ]
    max_difference = max(all_differences)
    mean_difference = statistics.fmean(all_differences)
    last_comparison = comparisons[-1]
    output_kind = str(last_comparison["output_kind"])
    element_count = int(last_comparison["element_count"])
    return {
        "status": "PASSED" if max_difference <= tolerance else "FAILED",
        "validation_type": "stochastic",
        "equivalence_level": "statistical",
        "repeat": repeat,
        "output_kind": output_kind,
        "element_count": element_count,
        "cpu_output": cpu_outputs_raw[-1],
        "gpu_output": gpu_outputs_raw[-1],
        "cpu_outputs": cpu_outputs_raw,
        "gpu_outputs": gpu_outputs_raw,
        "absolute_difference": run_differences[-1],
        "absolute_differences": run_differences,
        "max_absolute_difference": max_difference,
        "mean_absolute_difference": mean_difference,
        "tolerance": tolerance,
        "reason": (
            "Repeated stochastic scalar estimates are compared with a loose tolerance."
            if repeat > 1
            else "Monte Carlo estimates are compared with a loose stochastic tolerance."
        ),
    }


def _compare_numeric_outputs(cpu_output: Any, gpu_output: Any) -> dict[str, object]:
    cpu_values = _flatten_numeric_output(cpu_output)
    gpu_values = _flatten_numeric_output(gpu_output)
    if cpu_values is None or gpu_values is None:
        return {
            "ok": False,
            "reason": "Could not parse numeric output from CPU and GPU runs.",
        }
    if len(cpu_values) != len(gpu_values):
        return {
            "ok": False,
            "reason": (
                "CPU and GPU outputs have different numeric element counts: "
                f"{len(cpu_values)} != {len(gpu_values)}."
            ),
        }
    differences = [
        abs(cpu_value - gpu_value)
        for cpu_value, gpu_value in zip(cpu_values, gpu_values)
    ]
    return {
        "ok": True,
        "output_kind": "scalar" if len(cpu_values) == 1 else "array",
        "element_count": len(cpu_values),
        "absolute_differences": differences,
        "max_absolute_difference": max(differences),
    }


def _flatten_numeric_output(output: Any) -> list[float] | None:
    if isinstance(output, bool) or output is None:
        return None
    if isinstance(output, (int, float)):
        return [float(output)]
    if isinstance(output, list):
        values: list[float] = []
        for item in output:
            flattened = _flatten_numeric_output(item)
            if flattened is None:
                return None
            values.extend(flattened)
        return values
    return None


def _execute_benchmark(
    original: Path,
    generated: Path,
    repeat: int = 1,
    warmup: int = 0,
) -> dict[str, object]:
    repeat = max(1, repeat)
    warmup = max(0, warmup)
    if os.environ.get("SIMFORGE_DISABLE_GPU") == "1":
        return {
            "status": "SKIPPED",
            "repeat": repeat,
            "warmup": warmup,
            "reason": "GPU execution disabled by SIMFORGE_DISABLE_GPU=1.",
            **_benchmark_trust_metadata(measured=False),
        }
    available, reason = cupy_kernel_probe()
    if not available:
        return {
            "status": "SKIPPED",
            "repeat": repeat,
            "warmup": warmup,
            "reason": f"CuPy GPU execution is unavailable: {reason}",
            **_benchmark_trust_metadata(measured=False),
        }
    for _ in range(warmup):
        run_python_script(original)
        run_python_script(generated, use_cuda_workdir=True)
    cpu_runs = [run_python_script(original) for _ in range(repeat)]
    gpu_runs = [run_python_script(generated, use_cuda_workdir=True) for _ in range(repeat)]
    failed_cpu = next((run for run in cpu_runs if run.status != "PASSED"), None)
    failed_gpu = next((run for run in gpu_runs if run.status != "PASSED"), None)
    if failed_cpu is not None or failed_gpu is not None:
        return {
            "status": "FAILED",
            "repeat": repeat,
            "warmup": warmup,
            "reason": _execution_failure_reason(failed_cpu or cpu_runs[0], failed_gpu or gpu_runs[0]),
            **_benchmark_trust_metadata(measured=True),
        }
    cpu_times = [run.runtime_seconds for run in cpu_runs]
    gpu_times = [run.runtime_seconds for run in gpu_runs]
    cpu_median = statistics.median(cpu_times)
    gpu_median = statistics.median(gpu_times)
    speedup = cpu_median / gpu_median if gpu_median > 0 else None
    result: dict[str, object] = {
        "status": "PASSED",
        "repeat": repeat,
        "warmup": warmup,
        "cpu_times_seconds": cpu_times,
        "gpu_times_seconds": gpu_times,
        "cpu_runtime": f"{cpu_median:.6f}s",
        "gpu_runtime": f"{gpu_median:.6f}s",
        "cpu_median_runtime": f"{cpu_median:.6f}s",
        "gpu_median_runtime": f"{gpu_median:.6f}s",
        "cpu_min_runtime": f"{min(cpu_times):.6f}s",
        "gpu_min_runtime": f"{min(gpu_times):.6f}s",
        "cpu_mean_runtime": f"{statistics.fmean(cpu_times):.6f}s",
        "gpu_mean_runtime": f"{statistics.fmean(gpu_times):.6f}s",
        "reason": "Measured by executing CPU and generated CuPy scripts locally.",
        **_benchmark_trust_metadata(measured=True),
    }
    if speedup is not None:
        result["speedup"] = f"{speedup:.3f}x"
    return result


def _benchmark_trust_metadata(measured: bool) -> dict[str, object]:
    if not measured:
        return {
            "measurement_method": "not_run",
            "trust_level": "not_applicable",
            "includes_transfer_overhead": False,
            "limitations": [
                "No benchmark measurement was executed.",
                "No speedup should be inferred from a skipped benchmark.",
            ],
        }
    return {
        "measurement_method": "subprocess_wall_time",
        "trust_level": "demo_only",
        "includes_transfer_overhead": False,
        "limitations": [
            "Includes Python process startup and CUDA initialization overhead.",
            "Does not isolate host/device transfer time.",
            "Does not use in-process synchronization around individual kernels.",
            "Use the reported speedup only as a local demo measurement, not a general performance claim.",
        ],
    }


def _execution_failure_reason(cpu: object, gpu: object) -> str:
    cpu_stderr = getattr(cpu, "stderr", "").strip()
    gpu_stderr = getattr(gpu, "stderr", "").strip()
    return (
        "CPU or GPU script execution failed. "
        f"CPU stderr: {cpu_stderr or 'none'}; GPU stderr: {gpu_stderr or 'none'}"
    )


def _syntax_result(generated_path: Path | None) -> dict[str, object]:
    if generated_path is None:
        return {
            "status": "SKIPPED",
            "reason": "No generated source was produced for this backend.",
        }
    return {
        "status": "PASSED",
        "source": _display_path(generated_path),
        "message": "Generated source parsed successfully.",
    }


def _quality_result(
    syntax_result: dict[str, object],
    unsupported_count: int,
    validation_result: dict[str, object],
    benchmark_result: dict[str, object],
    suggestion_review_status: str | None = None,
) -> dict[str, object]:
    syntax_status = str(syntax_result.get("status", "UNKNOWN"))
    validation_status = str(validation_result.get("status", "UNKNOWN"))
    benchmark_status = str(benchmark_result.get("status", "UNKNOWN"))
    quality_gate = "PASSED"
    notes = [
        "Validation and benchmark skips are explicit and are not correctness or speedup claims.",
    ]
    if syntax_status not in {"PASSED", "SKIPPED"}:
        quality_gate = "FAILED"
        notes.append("Generated source syntax did not pass the static quality gate.")
    if validation_status == "SKIPPED" or benchmark_status == "SKIPPED":
        quality_gate = "PASSED_WITH_SKIPS" if quality_gate != "FAILED" else quality_gate
        notes.append("At least one GPU execution gate was skipped.")
    if validation_status == "FAILED" or benchmark_status == "FAILED":
        quality_gate = "FAILED"
        notes.append("At least one execution gate failed.")
    if unsupported_count:
        notes.append("Unsupported features remain; review unsupported_report.md before execution.")
    result: dict[str, object] = {
        "quality_gate": quality_gate,
        "syntax_status": syntax_status,
        "unsupported_count": unsupported_count,
        "validation_status": validation_status,
        "benchmark_status": benchmark_status,
        "notes": notes,
    }
    if suggestion_review_status is not None:
        result["suggestion_review_status"] = suggestion_review_status
    return result


def _render_explanation(
    plan: ConversionPlan,
    generated_path: Path | None,
    validation_result: dict[str, object],
    benchmark_result: dict[str, object],
    suggestion_review: SuggestionReview | None = None,
) -> str:
    artifacts = [
        "conversion_plan.json",
        "unsupported_report.md",
        "validation_report.md",
        "benchmark_report.md",
        "explanation_report.md",
    ]
    if generated_path is not None:
        artifacts.insert(0, str(generated_path))
    report = render_explanation_report(
        {
            "summary": plan.summary,
            "validation_status": validation_result["status"],
            "benchmark_status": benchmark_result["status"],
            "artifacts": artifacts,
        }
    )
    details = [
        "",
        "## Backend",
        "",
        f"Selected backend: {plan.backend_status['selected_backend']}",
        f"Backend status: {plan.backend_status['status']}",
        "",
        "## Notes",
        "",
        "- LLM output is not used as a trusted source in this MVP path.",
        "- Rules transform only supported NumPy APIs.",
        "- Tests and reports decide whether a conversion is usable.",
        "",
    ]
    if suggestion_review is not None:
        details.extend(
            [
                "## Model Advisory Input",
                "",
                f"Suggestion review status: {suggestion_review.status}",
                "Accepted fields:",
            ]
        )
        if suggestion_review.accepted_fields:
            details.extend(f"- {entry.field}" for entry in suggestion_review.accepted_fields)
        else:
            details.append("- none")
        details.extend(["", "Rejected fields:"])
        if suggestion_review.rejected_fields:
            details.extend(f"- {entry.field}" for entry in suggestion_review.rejected_fields)
        else:
            details.append("- none")
        details.extend(
            [
                "",
                "Note: Model advisory input is not treated as a trusted transformation.",
                "",
            ]
        )
    return report + "\n".join(details)


def _render_plan_explanation(plan: dict[str, object]) -> str:
    backend_status = plan.get("backend_status", {})
    if not isinstance(backend_status, dict):
        backend_status = {}
    changes = plan.get("changes", [])
    risks = plan.get("risks", [])
    unsupported = plan.get("unsupported_features", [])
    lines = [
        "# Explanation Report",
        "",
        str(plan.get("summary", "No summary provided.")),
        "",
        "## Source",
        "",
        f"Source file: {plan.get('source_file', 'unknown')}",
        f"Target backend: {plan.get('target_backend', 'unknown')}",
        "",
        "## Backend",
        "",
        f"Selected backend: {backend_status.get('selected_backend', 'unknown')}",
        f"Backend status: {backend_status.get('status', 'unknown')}",
        "",
        "## Changes",
        "",
    ]
    if changes:
        for change in changes:
            if isinstance(change, dict):
                lines.append(
                    f"- {change.get('original', 'unknown')} -> {change.get('replacement', 'unknown')}: {change.get('reason', '')}"
                )
    else:
        lines.append("- No code changes are recorded in this plan.")
    lines.extend(["", "## Validation", ""])
    lines.append(f"Validation strategy: {plan.get('validation_strategy', 'unknown')}")
    lines.append(f"Equivalence level: {plan.get('equivalence_level', 'unknown')}")
    lines.extend(["", "## Risks", ""])
    if risks:
        lines.extend(f"- {risk}" for risk in risks)
    else:
        lines.append("- No risks recorded.")
    lines.extend(["", "## Unsupported Features", ""])
    if unsupported:
        for feature in unsupported:
            if isinstance(feature, dict):
                lines.append(
                    f"- `{feature.get('code', 'unknown')}`: {feature.get('reason', '')}"
                )
    else:
        lines.append("- No unsupported features recorded.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- LLM suggestions are not treated as trusted transformations.",
            "- Rules transform only supported regions and APIs.",
            "- Tests and reports decide whether generated code is usable.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _unsupported_count(path: Path) -> int:
    if not path.exists():
        return -1
    text = path.read_text(encoding="utf-8")
    if "No unsupported features detected." in text:
        return 0
    return sum(1 for line in text.splitlines() if line.startswith("## "))


def _unsupported_items(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8")
    if "No unsupported features detected." in text:
        return ()
    items: list[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            items.append(line.removeprefix("## ").strip())
    return tuple(items)


def _display_path(path: Path) -> str:
    return path.as_posix()


def _report_status(path: Path, prefix: str) -> str:
    if not path.exists():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return "unknown"
