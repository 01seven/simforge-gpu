"""Planned-only JAX backend metadata."""

from simforge_gpu.backends.base import BackendInfo

JAX_BACKEND = BackendInfo(
    name="jax",
    display_name="JaxBackend",
    status="planned",
    reason="Future backend for functional compiled numerical workloads.",
)
