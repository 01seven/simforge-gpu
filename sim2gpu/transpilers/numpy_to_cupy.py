"""Conservative NumPy to CuPy API mapping helpers."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from sim2gpu.backends.cupy import SUPPORTED_NUMPY_APIS
from sim2gpu.ir.schema import UnsupportedFeature

SUPPORTED_NUMPY_TO_CUPY = {
    api: api.replace("np.", "cp.", 1) for api in SUPPORTED_NUMPY_APIS
}


@dataclass(frozen=True)
class MappingResult:
    original: str
    replacement: str | None
    unsupported_feature: UnsupportedFeature | None = None


@dataclass(frozen=True)
class RewriteResult:
    source: str
    unsupported_features: tuple[UnsupportedFeature, ...]
    changes: tuple[dict[str, str], ...] = ()


def map_numpy_api(api_name: str) -> MappingResult:
    replacement = SUPPORTED_NUMPY_TO_CUPY.get(api_name)
    if replacement is not None:
        return MappingResult(original=api_name, replacement=replacement)
    return MappingResult(
        original=api_name,
        replacement=None,
        unsupported_feature=UnsupportedFeature(
            code=api_name,
            reason="NumPy API is not in the MVP supported mapping table.",
            action="Leave this call unchanged or add an explicit supported mapping with tests.",
            category="unsupported_numpy_api",
        ),
    )


def rewrite_supported_numpy_calls(source: str) -> RewriteResult:
    unsupported: list[UnsupportedFeature] = []
    changes: list[dict[str, str]] = []
    rewritten = source

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return RewriteResult(
            source=rewritten,
            unsupported_features=tuple(unsupported),
            changes=tuple(changes),
        )

    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = _call_name(node.func)
        if not call_name or not call_name.startswith("np."):
            continue
        if call_name in SUPPORTED_NUMPY_TO_CUPY or call_name in seen:
            continue
        seen.add(call_name)
        feature = map_numpy_api(call_name).unsupported_feature
        if feature is not None:
            unsupported.append(feature)

    rewritten = _rewrite_import(source, keep_numpy=bool(unsupported))
    for api in sorted(SUPPORTED_NUMPY_TO_CUPY, key=len, reverse=True):
        replacement = SUPPORTED_NUMPY_TO_CUPY[api]
        if api in rewritten:
            rewritten = rewritten.replace(api, replacement)
            changes.append(
                {
                    "type": "api_mapping",
                    "original": api,
                    "replacement": replacement,
                    "reason": "Supported MVP NumPy API mapped to CuPy.",
                }
            )
    if rewritten != source:
        changes = [
            {
                "type": "import_mapping",
                "original": "import numpy as np",
                "replacement": "import cupy as cp",
                "reason": "CuPy provides a NumPy-compatible GPU array namespace.",
            }
        ] + changes

    return RewriteResult(
        source=rewritten,
        unsupported_features=tuple(unsupported),
        changes=tuple(changes),
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


def _rewrite_import(source: str, keep_numpy: bool) -> str:
    if "import numpy as np" not in source:
        return source
    replacement = (
        "import numpy as np\nimport cupy as cp"
        if keep_numpy
        else "import cupy as cp"
    )
    return source.replace("import numpy as np", replacement, 1)
