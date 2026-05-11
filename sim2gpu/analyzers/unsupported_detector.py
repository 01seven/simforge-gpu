"""Rule-based unsupported feature detector for the MVP boundary."""

from __future__ import annotations

import ast

from sim2gpu.backends.cupy import SUPPORTED_NUMPY_APIS
from sim2gpu.backends.registry import get_backend
from sim2gpu.ir.schema import UnsupportedFeature


def detect_unsupported(source: str, target_backend: str = "cupy") -> tuple[UnsupportedFeature, ...]:
    features: list[UnsupportedFeature] = []

    try:
        backend = get_backend(target_backend)
        if not backend.is_implemented:
            features.append(
                UnsupportedFeature(
                    code=f"--target {backend.name}",
                    reason=backend.unsupported_reason(),
                    action="Use --target cupy for the current supported backend.",
                    category="backend_not_implemented",
                )
            )
    except ValueError as exc:
        features.append(
            UnsupportedFeature(
                code=f"--target {target_backend}",
                reason=str(exc),
                action="Use one of the listed backend names.",
                category="unsupported_backend",
            )
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return tuple(
            features
            + [
                UnsupportedFeature(
                    code="python source",
                    reason=f"Source could not be parsed: {exc.msg}.",
                    action="Fix syntax before running sim2gpu.",
                    category="syntax_error",
                )
            ]
        )

    seen: set[str] = {feature.code for feature in features}

    def add(code: str, reason: str, action: str, category: str) -> None:
        if code in seen:
            return
        seen.add(code)
        features.append(
            UnsupportedFeature(
                code=code,
                reason=reason,
                action=action,
                category=category,
            )
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root == "pandas":
                    add("import pandas", "pandas-heavy pipelines are outside MVP scope.", "Keep dataframe work on CPU or wait for roadmap support.", "pandas_pipeline")
                elif root == "matplotlib":
                    add("import matplotlib", "Plotting is outside MVP conversion scope.", "Keep plotting on CPU after transferring final results.", "plotting")
                elif root == "multiprocessing":
                    add("import multiprocessing", "multiprocessing conversion is outside MVP scope.", "Run multiprocessing code on CPU or refactor first.", "multiprocessing")
                elif root == "threading":
                    add("import threading", "threading conversion is outside MVP scope.", "Run threading code on CPU or refactor first.", "threading")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root == "matplotlib":
                add("from matplotlib import ...", "Plotting is outside MVP conversion scope.", "Keep plotting on CPU after transferring final results.", "plotting")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name == "open":
                add("open(...)", "File I/O is outside MVP conversion scope.", "Keep file I/O as CPU-side preprocessing or postprocessing.", "file_io")
            elif call_name == "eval":
                add("eval(...)", "Dynamic eval is outside MVP conversion scope.", "Remove dynamic execution before conversion.", "dynamic_execution")
            elif call_name == "exec":
                add("exec(...)", "Dynamic exec is outside MVP conversion scope.", "Remove dynamic execution before conversion.", "dynamic_execution")
            elif call_name and call_name.startswith("plt."):
                add("plt.*", "Plotting is outside MVP conversion scope.", "Keep plotting on CPU after transferring final results.", "plotting")
            elif call_name and call_name.startswith("np.") and call_name not in SUPPORTED_NUMPY_APIS:
                add(call_name, "NumPy API is not in the MVP supported mapping table.", "Leave this call unchanged or add a supported mapping with tests.", "unsupported_numpy_api")
        elif isinstance(node, ast.ClassDef):
            add(
                f"class {node.name}",
                "class-heavy code is outside MVP conversion scope.",
                "Refactor the simulation kernel into a simple function before conversion.",
                "class_heavy",
            )
        elif isinstance(node, ast.For):
            _detect_loop_risks(node, add)

    return tuple(features)


def _detect_loop_risks(node: ast.For, add) -> None:
    assigned: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Assign):
            for target in child.targets:
                if isinstance(target, ast.Name):
                    assigned.add(target.id)
                    if _name_used(child.value, target.id):
                        add(
                            f"{target.id} = {target.id} + ...",
                            "Detected a sequential dependency loop that may not be safely parallelizable.",
                            "Keep this loop on CPU or rewrite it with an explicit parallel strategy.",
                            "sequential_dependency",
                        )
        elif isinstance(child, ast.AugAssign) and isinstance(child.target, ast.Name):
            add(
                f"{child.target.id} augmented assignment",
                "Detected a sequential dependency loop that may not be safely parallelizable.",
                "Keep this loop on CPU or rewrite it with an explicit parallel strategy.",
                "sequential_dependency",
            )
        elif isinstance(child, ast.Call):
            call_name = _call_name(child.func)
            if call_name and call_name.endswith(".append"):
                add(
                    f"{call_name}(...)",
                    "Detected a side-effect-heavy loop that mutates Python containers.",
                    "Collect results with array operations or keep this loop on CPU.",
                    "side_effect_loop",
                )


def _name_used(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


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
