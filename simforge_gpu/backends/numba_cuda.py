"""Planned-only Numba-CUDA backend metadata."""

from simforge_gpu.backends.base import BackendInfo

NUMBA_CUDA_BACKEND = BackendInfo(
    name="numba",
    display_name="NumbaCudaBackend",
    status="planned",
    reason="Future backend for custom CUDA kernel workflows.",
)
