"""Target metadata for v2 agent-harness workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TargetInfo:
    name: str
    status: str
    description: str
    requires_external_agent: bool

