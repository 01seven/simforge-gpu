"""Planned-only cuDF backend metadata."""

from sim2gpu.backends.base import BackendInfo

CUDF_BACKEND = BackendInfo(
    name="cudf",
    display_name="CudfBackend",
    status="planned",
    reason="Future backend for dataframe-heavy GPU workflows.",
)
