"""Planned-only Numba-CUDA backend metadata."""

from sim2gpu.backends.base import BackendInfo

NUMBA_CUDA_BACKEND = BackendInfo(
    name="numba",
    display_name="NumbaCudaBackend",
    status="planned",
    reason="Future backend for custom CUDA kernel workflows.",
)
