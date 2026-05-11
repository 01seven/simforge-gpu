"""Planned-only Torch backend metadata.

This module intentionally does not implement torch conversion.
"""

from simforge_gpu.backends.base import BackendInfo

TORCH_BACKEND = BackendInfo(
    name="torch",
    display_name="TorchBackend",
    status="planned",
    reason="Future backend for tensor-heavy simulation and PyTorch integration.",
)
