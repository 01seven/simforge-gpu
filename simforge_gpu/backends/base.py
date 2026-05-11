"""Shared backend metadata types.

These types intentionally avoid importing GPU libraries. Backend selection and
planning must be testable in no-GPU environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class BackendInfo:
    """Static status for a target backend."""

    name: str
    display_name: str
    status: str
    reason: str
    supported_patterns: tuple[str, ...] = ()
    supported_numpy_apis: tuple[str, ...] = ()

    @property
    def is_implemented(self) -> bool:
        return self.status == "implemented"

    def unsupported_reason(self) -> str:
        if self.is_implemented:
            return ""
        return (
            f"{self.display_name} is planned but not implemented in the MVP.\n"
            "Use --target cupy for the current supported backend."
        )

    def validate_environment(self) -> dict[str, object]:
        """Return import-level environment metadata without doing GPU work."""

        if not self.is_implemented:
            return {
                "available": False,
                "status": self.status,
                "reason": self.unsupported_reason(),
            }
        if self.name == "cupy" and find_spec("cupy") is None:
            return {
                "available": False,
                "status": "missing_dependency",
                "reason": "CuPy is not installed; no-GPU planning can still run.",
            }
        return {
            "available": True,
            "status": "import_spec_found",
            "reason": "Backend import spec found; CUDA execution is checked later.",
        }

    def map_function_call(self, numpy_api: str) -> str | None:
        """Map a supported NumPy API name to the backend namespace."""

        if not self.is_implemented:
            raise NotImplementedError(self.unsupported_reason())
        if numpy_api not in self.supported_numpy_apis:
            return None
        return numpy_api.replace("np.", "cp.", 1)


class UnsupportedBackendError(ValueError):
    """Raised when a backend name is not known to SimForge GPU."""

    def __init__(self, backend_name: str, supported_names: list[str]) -> None:
        self.backend_name = backend_name
        self.supported_names = supported_names
        names = ", ".join(supported_names)
        super().__init__(f"Unknown backend '{backend_name}'. Supported backends: {names}.")

    def to_report_entry(self) -> dict[str, str]:
        return {
            "code": f"--target {self.backend_name}",
            "reason": str(self),
            "action": "Use one of the listed supported backend names.",
        }
