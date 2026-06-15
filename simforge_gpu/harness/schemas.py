"""Plain Python schemas for the v2 external-agent harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass


SCHEMA_VERSION = "2.0"


@dataclass(frozen=True)
class ProjectInspection:
    project_path: str
    language: str
    detected_files: tuple[str, ...]
    entrypoints: tuple[str, ...]
    dependencies: tuple[str, ...]
    simulation_indicators: tuple[str, ...]
    randomness_indicators: tuple[str, ...]
    io_indicators: tuple[str, ...]
    plotting_indicators: tuple[str, ...]
    unsupported_or_risky_features: tuple[str, ...]
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class MigrationPlan:
    schema_version: str
    language: str
    target: str
    target_status: str
    recommended_workflow: str
    hotspots: tuple[str, ...]
    candidate_gpu_regions: tuple[str, ...]
    cpu_only_regions: tuple[str, ...]
    risks: tuple[str, ...]
    validation_strategy: str
    benchmark_strategy: str
    agent_instructions_path: str

    def to_dict(self) -> dict[str, object]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class MigrationRequest:
    schema_version: str
    project_path: str
    language: str
    target: str
    agent: str
    source_summary: str
    entrypoints: tuple[str, ...]
    performance_goals: tuple[str, ...]
    correctness_requirements: tuple[str, ...]
    validation_strategy: str
    benchmark_strategy: str
    constraints: tuple[str, ...]
    required_outputs: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class AgentResult:
    schema_version: str
    agent: str
    status: str
    summary: str
    files_modified: tuple[str, ...]
    files_created: tuple[str, ...]
    claimed_changes: tuple[str, ...]
    known_limitations: tuple[str, ...]
    validation_claims: tuple[str, ...]
    benchmark_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class AgentTrace:
    schema_version: str
    workflow_id: str
    project_path: str
    language: str
    target: str
    agent: str
    migration_request_path: str
    migration_task_path: str
    agent_result_path: str
    validation_report_path: str
    benchmark_report_path: str
    final_status: str
    trust_boundary_notes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return _dataclass_to_dict(self)


def _dataclass_to_dict(instance: object) -> dict[str, object]:
    data = asdict(instance)
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in data.items()
    }

