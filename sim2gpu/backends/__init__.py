"""Backend metadata for sim2gpu."""

from sim2gpu.backends.base import BackendInfo, UnsupportedBackendError
from sim2gpu.backends.registry import get_backend, list_backends

__all__ = [
    "BackendInfo",
    "UnsupportedBackendError",
    "get_backend",
    "list_backends",
]
