"""Backend metadata for SimForge GPU."""

from simforge_gpu.backends.base import BackendInfo, UnsupportedBackendError
from simforge_gpu.backends.registry import get_backend, list_backends

__all__ = [
    "BackendInfo",
    "UnsupportedBackendError",
    "get_backend",
    "list_backends",
]
