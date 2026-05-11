import pytest

from sim2gpu.backends.registry import get_backend, list_backends


def test_list_backends_separates_implemented_and_planned_backends():
    backends = {backend.name: backend for backend in list_backends()}

    assert backends["cupy"].status == "implemented"
    assert backends["cupy"].is_implemented is True

    for name in ["torch", "jax", "numba", "cudf"]:
        assert backends[name].status == "planned"
        assert backends[name].is_implemented is False
        assert "not implemented in the MVP" in backends[name].unsupported_reason()


def test_get_backend_returns_structured_unknown_backend_error():
    with pytest.raises(ValueError) as exc_info:
        get_backend("unknown")

    assert "Unknown backend 'unknown'" in str(exc_info.value)
    assert "cupy, torch, jax, numba, cudf" in str(exc_info.value)
