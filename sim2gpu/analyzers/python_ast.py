"""Small Python AST analyzer for MVP planning."""

from __future__ import annotations

import ast
from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisFacts:
    language: str
    imports: tuple[str, ...]
    numpy_aliases: tuple[str, ...]
    random_calls: tuple[str, ...]
    outputs: tuple[str, ...]
    for_loops: tuple[dict[str, int | str], ...]


def analyze_source(source: str) -> AnalysisFacts:
    tree = ast.parse(source)
    imports: list[str] = []
    numpy_aliases: list[str] = []
    random_calls: list[str] = []
    outputs_with_lines: list[tuple[int, str]] = []
    for_loops: list[dict[str, int | str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "numpy":
                    imports.append("numpy")
                    numpy_aliases.append(alias.asname or "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            imports.append("numpy")
        elif isinstance(node, ast.For):
            for_loops.append(
                {
                    "type": "for_loop",
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                }
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    outputs_with_lines.append((node.lineno, target.id))
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name and ".random." in call_name:
                random_calls.append(call_name)

    return AnalysisFacts(
        language="python",
        imports=tuple(dict.fromkeys(imports)),
        numpy_aliases=tuple(dict.fromkeys(numpy_aliases)),
        random_calls=tuple(dict.fromkeys(random_calls)),
        outputs=tuple(name for _, name in sorted(outputs_with_lines)),
        for_loops=tuple(for_loops),
    )


def _call_name(node: ast.AST) -> str | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None
