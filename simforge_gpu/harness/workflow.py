"""Workflow orchestration for v2 inspect/plan/task/collect commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from simforge_gpu.agents.protocol import (
    build_agent_trace,
    build_migration_request,
    expected_artifacts_payload,
    read_agent_result,
    render_constraints_markdown,
    render_migration_task,
)
from simforge_gpu.harness.artifacts import (
    ensure_project_dirs,
    read_json_artifact,
    write_json_artifact,
)
from simforge_gpu.harness.schemas import (
    SCHEMA_VERSION,
    MigrationPlan,
    ProjectInspection,
)
from simforge_gpu.languages import detect_language, get_language_adapter
from simforge_gpu.targets import get_target


@dataclass(frozen=True)
class V2WorkflowResult:
    project_dir: Path
    artifact_path: Path
    message: str
    payload: dict[str, object]


def inspect_project(project_path: str | Path, language: str = "auto") -> V2WorkflowResult:
    root = _assert_project_path(project_path)
    selected_language = detect_language(root) if language == "auto" else language
    adapter = get_language_adapter(selected_language)
    inspection = adapter.inspect(root)
    paths = ensure_project_dirs(root)
    artifact_path = paths["reports"] / "project_inspection.json"
    payload = inspection.to_dict()
    write_json_artifact(artifact_path, payload)
    return V2WorkflowResult(
        project_dir=root,
        artifact_path=artifact_path,
        message=(
            f"Project inspection: {artifact_path}\n"
            f"Language: {inspection.language}\n"
            f"Detected files: {len(inspection.detected_files)}\n"
            f"Entrypoints: {', '.join(inspection.entrypoints) or 'none'}"
        ),
        payload=payload,
    )


def plan_project(
    project_path: str | Path,
    language: str = "auto",
    target: str = "py-torch",
) -> V2WorkflowResult:
    root = _assert_project_path(project_path)
    inspection = _load_or_inspect(root, language)
    target_info = get_target(target)
    plan = MigrationPlan(
        schema_version=SCHEMA_VERSION,
        language=inspection.language,
        target=target_info.name,
        target_status=target_info.status,
        recommended_workflow=(
            "legacy_deterministic_conversion"
            if target_info.status == "legacy_deterministic_target"
            else "external_agent"
        ),
        hotspots=_hotspots(inspection),
        candidate_gpu_regions=_candidate_regions(inspection),
        cpu_only_regions=tuple(inspection.io_indicators + inspection.plotting_indicators),
        risks=_risks(inspection, target_info.status),
        validation_strategy="stochastic_summary",
        benchmark_strategy="not_run_until_candidate_exists",
        agent_instructions_path="agent/migration_task.md",
    )
    paths = ensure_project_dirs(root)
    artifact_path = paths["reports"] / "migration_plan.json"
    payload = plan.to_dict()
    write_json_artifact(artifact_path, payload)
    return V2WorkflowResult(
        project_dir=root,
        artifact_path=artifact_path,
        message=(
            f"Migration plan: {artifact_path}\n"
            f"Target: {plan.target}\n"
            f"Target status: {plan.target_status}\n"
            "No torch code was generated."
        ),
        payload=payload,
    )


def create_agent_task(
    project_path: str | Path,
    agent: str,
    target: str,
    language: str = "auto",
) -> V2WorkflowResult:
    root = _assert_project_path(project_path)
    if language == "auto":
        language = _language_for_target(target)
    inspection = _load_or_inspect(root, language)
    plan_result = plan_project(root, language=inspection.language, target=target)
    plan = _plan_from_payload(plan_result.payload)
    request = build_migration_request(inspection, plan, agent)
    paths = ensure_project_dirs(root)
    request_path = paths["agent"] / "migration_request.json"
    task_path = paths["agent"] / "migration_task.md"
    constraints_path = paths["agent"] / "constraints.md"
    expected_path = paths["agent"] / "expected_artifacts.json"
    write_json_artifact(request_path, request.to_dict())
    task_path.write_text(render_migration_task(request, plan), encoding="utf-8")
    constraints_path.write_text(render_constraints_markdown(), encoding="utf-8")
    write_json_artifact(expected_path, expected_artifacts_payload())
    return V2WorkflowResult(
        project_dir=root,
        artifact_path=task_path,
        message=(
            f"Agent task: {task_path}\n"
            f"Migration request: {request_path}\n"
            "External agent was not launched."
        ),
        payload=request.to_dict(),
    )


def collect_agent_result(project_path: str | Path) -> V2WorkflowResult:
    root = _assert_project_path(project_path)
    paths = ensure_project_dirs(root)
    request_path = paths["agent"] / "migration_request.json"
    request = read_json_artifact(request_path) if request_path.exists() else {}
    agent = str(request.get("agent", "manual"))
    language = str(request.get("language", detect_language(root)))
    target = str(request.get("target", "py-torch"))
    result_path = paths["agent"] / "agent_result.json"
    result_existed = result_path.exists()
    agent_result = read_agent_result(result_path, agent)
    if not result_existed:
        write_json_artifact(result_path, agent_result.to_dict())
    trace = build_agent_trace(
        workflow_id=str(uuid4()),
        project_path=root.as_posix(),
        language=language,
        target=target,
        agent=agent,
        agent_result=agent_result,
    )
    trace_path = paths["reports"] / "agent_trace.json"
    write_json_artifact(trace_path, trace.to_dict())
    _write_pending_execution_reports(paths["reports"], trace.final_status)
    return V2WorkflowResult(
        project_dir=root,
        artifact_path=trace_path,
        message=(
            f"Agent trace: {trace_path}\n"
            f"Final status: {trace.final_status}\n"
            "Validation and benchmark were not run by collect."
        ),
        payload=trace.to_dict(),
    )


def _load_or_inspect(root: Path, language: str) -> ProjectInspection:
    reports = root / "reports"
    existing = reports / "project_inspection.json"
    if existing.exists():
        payload = read_json_artifact(existing)
        return _inspection_from_payload(payload)
    if language == "auto":
        language = detect_language(root)
    return get_language_adapter(language).inspect(root)


def _inspection_from_payload(payload: dict[str, object]) -> ProjectInspection:
    return ProjectInspection(
        project_path=str(payload.get("project_path", "")),
        language=str(payload.get("language", "unknown")),
        detected_files=tuple(str(item) for item in payload.get("detected_files", [])),
        entrypoints=tuple(str(item) for item in payload.get("entrypoints", [])),
        dependencies=tuple(str(item) for item in payload.get("dependencies", [])),
        simulation_indicators=tuple(str(item) for item in payload.get("simulation_indicators", [])),
        randomness_indicators=tuple(str(item) for item in payload.get("randomness_indicators", [])),
        io_indicators=tuple(str(item) for item in payload.get("io_indicators", [])),
        plotting_indicators=tuple(str(item) for item in payload.get("plotting_indicators", [])),
        unsupported_or_risky_features=tuple(
            str(item) for item in payload.get("unsupported_or_risky_features", [])
        ),
        notes=tuple(str(item) for item in payload.get("notes", [])),
    )


def _plan_from_payload(payload: dict[str, object]) -> MigrationPlan:
    return MigrationPlan(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        language=str(payload.get("language", "unknown")),
        target=str(payload.get("target", "py-torch")),
        target_status=str(payload.get("target_status", "unknown")),
        recommended_workflow=str(payload.get("recommended_workflow", "external_agent")),
        hotspots=tuple(str(item) for item in payload.get("hotspots", [])),
        candidate_gpu_regions=tuple(str(item) for item in payload.get("candidate_gpu_regions", [])),
        cpu_only_regions=tuple(str(item) for item in payload.get("cpu_only_regions", [])),
        risks=tuple(str(item) for item in payload.get("risks", [])),
        validation_strategy=str(payload.get("validation_strategy", "stochastic_summary")),
        benchmark_strategy=str(payload.get("benchmark_strategy", "not_run_until_candidate_exists")),
        agent_instructions_path=str(payload.get("agent_instructions_path", "agent/migration_task.md")),
    )


def _hotspots(inspection: ProjectInspection) -> tuple[str, ...]:
    hotspots = []
    if inspection.randomness_indicators:
        hotspots.append("random_sampling")
    if "repeated_random_trials" in inspection.simulation_indicators:
        hotspots.append("repeated_simulation_loop")
    return tuple(hotspots)


def _candidate_regions(inspection: ProjectInspection) -> tuple[str, ...]:
    regions = []
    if inspection.randomness_indicators:
        regions.append("random number generation")
    if inspection.simulation_indicators:
        regions.append("simulation aggregation")
    return tuple(regions)


def _risks(inspection: ProjectInspection, target_status: str) -> tuple[str, ...]:
    risks = list(inspection.unsupported_or_risky_features)
    if target_status == "primary_agent_target":
        risks.append("Requires semantic migration by an external agent; no automatic torch code is generated.")
    if target_status == "legacy_deterministic_target":
        risks.append("Legacy CuPy path only covers the existing supported NumPy subset.")
    if target_status == "planned_only":
        risks.append("Target is roadmap-only in this pass.")
    return tuple(risks)


def _language_for_target(target: str) -> str:
    if target == "r-torch":
        return "r"
    if target in {"py-torch", "cupy", "jax", "numba-cuda", "cudf"}:
        return "python"
    return "auto"


def _write_pending_execution_reports(reports_dir: Path, status: str) -> None:
    validation = [
        "# Validation Report",
        "",
        f"Validation status: {status}",
        "Validation type: pending_external_agent_candidate",
        "Equivalence level: not evaluated",
        "Reason: collect does not execute validation.",
        "",
    ]
    benchmark = [
        "# Benchmark Report",
        "",
        f"Benchmark status: {status}",
        "Measurement method: not_run",
        "Trust level: not_applicable",
        "Includes transfer overhead: False",
        "Reason: collect does not execute benchmarks.",
        "",
    ]
    (reports_dir / "validation_report.md").write_text("\n".join(validation), encoding="utf-8")
    (reports_dir / "benchmark_report.md").write_text("\n".join(benchmark), encoding="utf-8")


def _assert_project_path(project_path: str | Path) -> Path:
    root = Path(project_path)
    if not root.exists():
        raise FileNotFoundError(f"Project path not found: {root}")
    return root
