"""Shared language adapter interface."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from simforge_gpu.harness.schemas import ProjectInspection


class LanguageAdapter(Protocol):
    language: str

    def inspect(self, project_path: str | Path) -> ProjectInspection:
        """Inspect a local project without executing user code."""


def detect_language(project_path: str | Path) -> str:
    root = Path(project_path)
    search_root = root.parent if root.is_file() else root
    if any(search_root.rglob("*.py")):
        return "python"
    if any(search_root.rglob("*.R")) or any(search_root.rglob("*.r")) or any(search_root.rglob("*.Rmd")):
        return "r"
    return "unknown"


def get_language_adapter(language: str) -> LanguageAdapter:
    normalized = language.lower()
    if normalized == "auto":
        raise ValueError("Language auto-detection requires a project path.")
    if normalized == "python":
        from simforge_gpu.languages.python_adapter import PythonAdapter

        return PythonAdapter()
    if normalized == "r":
        from simforge_gpu.languages.r_adapter import RAdapter

        return RAdapter()
    raise ValueError(f"Unsupported language '{language}'. Supported languages: python, r.")

