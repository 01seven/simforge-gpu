"""Stable agent trace artifact for model-assisted conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from simforge_gpu.reporters.json_report import stable_json
from simforge_gpu.suggestions.review import SuggestionReview


@dataclass(frozen=True)
class AgentTrace:
    schema_version: str
    source_file: str
    target_backend: str
    model_suggestion_path: str
    suggestion_review_path: str
    accepted_model_fields: tuple[str, ...]
    rejected_model_fields: tuple[str, ...]
    harness_gates: tuple[dict[str, object], ...]
    artifacts: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "source_file": self.source_file,
            "target_backend": self.target_backend,
            "model_suggestion_path": self.model_suggestion_path,
            "suggestion_review_path": self.suggestion_review_path,
            "accepted_model_fields": list(self.accepted_model_fields),
            "rejected_model_fields": list(self.rejected_model_fields),
            "harness_gates": list(self.harness_gates),
            "artifacts": self.artifacts,
        }

    def to_json(self) -> str:
        return stable_json(self.to_dict())


def build_agent_trace(
    source_file: str | Path,
    target_backend: str,
    model_suggestion_path: str | Path,
    suggestion_review_path: str | Path,
    review: SuggestionReview,
    harness_gates: tuple[dict[str, object], ...],
    artifacts: dict[str, str],
) -> AgentTrace:
    return AgentTrace(
        schema_version="1.0",
        source_file=_display_path(Path(source_file)),
        target_backend=target_backend,
        model_suggestion_path=_display_path(Path(model_suggestion_path)),
        suggestion_review_path=_display_path(Path(suggestion_review_path)),
        accepted_model_fields=tuple(entry.field for entry in review.accepted_fields),
        rejected_model_fields=tuple(entry.field for entry in review.rejected_fields),
        harness_gates=tuple(dict(gate) for gate in harness_gates),
        artifacts=dict(artifacts),
    )


def write_agent_trace(path: str | Path, trace: AgentTrace) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(trace.to_json(), encoding="utf-8")
    return destination


def _display_path(path: Path) -> str:
    return path.as_posix()
