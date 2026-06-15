"""Artifact read/write helpers for the v2 harness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from simforge_gpu.reporters.json_report import stable_json


def ensure_project_dirs(project_path: str | Path) -> dict[str, Path]:
    root = Path(project_path)
    reports = root / "reports"
    agent = root / "agent"
    workspace = root / "workspace"
    runs = root / "runs"
    for directory in (reports, agent, workspace, runs):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": root, "reports": reports, "agent": agent, "workspace": workspace, "runs": runs}


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(stable_json(payload), encoding="utf-8")
    return destination


def read_json_artifact(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def relative_artifact_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()

