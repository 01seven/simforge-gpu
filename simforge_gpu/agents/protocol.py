"""External-agent task protocol for the v2 harness."""

from __future__ import annotations

import json
from pathlib import Path

from simforge_gpu.harness.schemas import (
    SCHEMA_VERSION,
    AgentResult,
    AgentTrace,
    MigrationPlan,
    MigrationRequest,
    ProjectInspection,
)


SUPPORTED_AGENTS = ("manual", "codex", "cursor")


def build_migration_request(
    inspection: ProjectInspection,
    plan: MigrationPlan,
    agent: str,
) -> MigrationRequest:
    _validate_agent(agent)
    return MigrationRequest(
        schema_version=SCHEMA_VERSION,
        project_path=inspection.project_path,
        language=inspection.language,
        target=plan.target,
        agent=agent,
        source_summary=_source_summary(inspection),
        entrypoints=inspection.entrypoints,
        performance_goals=(
            "Move performance-critical repeated simulation work to GPU tensors.",
            "Avoid unnecessary CPU/GPU transfers.",
            "Keep CPU-only I/O and plotting outside GPU regions.",
        ),
        correctness_requirements=(
            "Preserve simulation semantics and statistical meaning.",
            "Preserve user-facing outputs where possible.",
            "Document CPU-only or skipped sections.",
        ),
        validation_strategy=plan.validation_strategy,
        benchmark_strategy=plan.benchmark_strategy,
        constraints=protocol_constraints(),
        required_outputs=("modified_code", "agent_result.json"),
    )


def protocol_constraints() -> tuple[str, ...]:
    return (
        "Do not fabricate validation results.",
        "Do not fabricate benchmark results.",
        "Do not delete unsupported logic silently.",
        "Record modified and created files in agent_result.json.",
        "Treat validation_claims and benchmark_claims as agent claims only.",
    )


def render_migration_task(request: MigrationRequest, plan: MigrationPlan) -> str:
    lines = [
        "# SimForge GPU Migration Task",
        "",
        "## Project Summary",
        "",
        request.source_summary,
        "",
        "## Source Files And Entrypoints",
        "",
        "Entrypoints:",
    ]
    lines.extend(f"- {entrypoint}" for entrypoint in request.entrypoints or ("none detected",))
    lines.extend(
        [
            "",
            "## Target",
            "",
            f"- Target: {request.target}",
            f"- Target status: {plan.target_status}",
            f"- Agent: {request.agent}",
            "",
            "## Required Behavior Preservation",
            "",
        ]
    )
    lines.extend(f"- {requirement}" for requirement in request.correctness_requirements)
    lines.extend(["", "## Performance Objective", ""])
    lines.extend(f"- {goal}" for goal in request.performance_goals)
    lines.extend(["", "## Constraints", ""])
    lines.extend(f"- {constraint}" for constraint in request.constraints)
    lines.extend(
        [
            "",
            "## Required Agent Output",
            "",
            "- Modified code in the project workspace or source tree.",
            "- agent/agent_result.json describing changes and limitations.",
            "- Optional agent/agent_patch.diff if a diff is available.",
            "",
            "## Validation Guidance",
            "",
            "- CPU baseline and GPU candidate should be comparable.",
            "- Stochastic outputs should be compared statistically, not by exact random-number equality.",
            "- Do not mark validation or benchmark as passed unless the corresponding command actually ran.",
            "",
        ]
    )
    return "\n".join(lines)


def render_constraints_markdown() -> str:
    lines = ["# SimForge GPU Agent Constraints", ""]
    lines.extend(f"- {constraint}" for constraint in protocol_constraints())
    lines.extend(
        [
            "- Harness owns trust. Models propose and edit. Tests decide. Reports explain.",
            "",
        ]
    )
    return "\n".join(lines)


def expected_artifacts_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "required_outputs": ["modified_code", "agent_result.json"],
        "optional_outputs": ["agent_patch.diff", "notes.md"],
        "trusted_by_harness": False,
        "note": "Agent outputs are advisory until harness validation and benchmark commands run.",
    }


def pending_agent_result(agent: str) -> AgentResult:
    _validate_agent(agent)
    return AgentResult(
        schema_version=SCHEMA_VERSION,
        agent=agent,
        status="PENDING_AGENT_OUTPUT",
        summary="No external-agent result has been collected yet.",
        files_modified=(),
        files_created=(),
        claimed_changes=(),
        known_limitations=("External agent has not produced output.",),
        validation_claims=(),
        benchmark_claims=(),
    )


def read_agent_result(path: str | Path, agent: str) -> AgentResult:
    result_path = Path(path)
    if not result_path.exists():
        return pending_agent_result(agent)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    return AgentResult(
        schema_version=str(payload.get("schema_version", SCHEMA_VERSION)),
        agent=str(payload.get("agent", agent)),
        status=str(payload.get("status", "UNKNOWN")),
        summary=str(payload.get("summary", "")),
        files_modified=tuple(str(item) for item in payload.get("files_modified", [])),
        files_created=tuple(str(item) for item in payload.get("files_created", [])),
        claimed_changes=tuple(str(item) for item in payload.get("claimed_changes", [])),
        known_limitations=tuple(str(item) for item in payload.get("known_limitations", [])),
        validation_claims=tuple(str(item) for item in payload.get("validation_claims", [])),
        benchmark_claims=tuple(str(item) for item in payload.get("benchmark_claims", [])),
    )


def build_agent_trace(
    *,
    workflow_id: str,
    project_path: str,
    language: str,
    target: str,
    agent: str,
    agent_result: AgentResult,
) -> AgentTrace:
    final_status = (
        "PENDING_AGENT_OUTPUT"
        if agent_result.status == "PENDING_AGENT_OUTPUT"
        else "PENDING_HARNESS_VALIDATION"
    )
    return AgentTrace(
        schema_version=SCHEMA_VERSION,
        workflow_id=workflow_id,
        project_path=project_path,
        language=language,
        target=target,
        agent=agent,
        migration_request_path="agent/migration_request.json",
        migration_task_path="agent/migration_task.md",
        agent_result_path="agent/agent_result.json",
        validation_report_path="reports/validation_report.md",
        benchmark_report_path="reports/benchmark_report.md",
        final_status=final_status,
        trust_boundary_notes=(
            "External-agent validation and benchmark claims are not trusted harness results.",
            "Run simforge validate and simforge benchmark before claiming correctness or speedup.",
        ),
    )


def _source_summary(inspection: ProjectInspection) -> str:
    return (
        f"Language: {inspection.language}. "
        f"Files detected: {len(inspection.detected_files)}. "
        f"Simulation indicators: {', '.join(inspection.simulation_indicators) or 'none'}. "
        f"Risk indicators: {', '.join(inspection.unsupported_or_risky_features) or 'none'}."
    )


def _validate_agent(agent: str) -> None:
    if agent not in SUPPORTED_AGENTS:
        names = ", ".join(SUPPORTED_AGENTS)
        raise ValueError(f"Unknown agent '{agent}'. Supported agents: {names}.")

