"""Backend registry for implemented and planned targets."""

from simforge_gpu.backends.base import BackendInfo, UnsupportedBackendError
from simforge_gpu.backends.cudf import CUDF_BACKEND
from simforge_gpu.backends.cupy import CUPY_BACKEND
from simforge_gpu.backends.jax import JAX_BACKEND
from simforge_gpu.backends.numba_cuda import NUMBA_CUDA_BACKEND
from simforge_gpu.backends.torch import TORCH_BACKEND


_BACKENDS: tuple[BackendInfo, ...] = (
    CUPY_BACKEND,
    TORCH_BACKEND,
    JAX_BACKEND,
    NUMBA_CUDA_BACKEND,
    CUDF_BACKEND,
)


def list_backends() -> tuple[BackendInfo, ...]:
    """Return all known backends in stable display order."""

    return _BACKENDS


def get_backend(name: str) -> BackendInfo:
    """Return backend metadata by name or raise a structured unsupported error."""

    normalized = name.lower()
    for backend in _BACKENDS:
        if backend.name == normalized:
            return backend
    raise UnsupportedBackendError(normalized, [backend.name for backend in _BACKENDS])
