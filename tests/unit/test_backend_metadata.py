import pytest

from simforge_gpu.backends.registry import get_backend


def test_cupy_backend_declares_supported_metadata_without_importing_cupy():
    backend = get_backend("cupy")

    assert backend.name == "cupy"
    assert backend.status == "implemented"
    assert "np.mean" in backend.supported_numpy_apis
    assert "np.random.normal" in backend.supported_numpy_apis
    assert "monte_carlo_independent_trials" in backend.supported_patterns

    environment = backend.validate_environment()
    assert set(environment) == {"available", "status", "reason"}


def test_torch_backend_is_planned_only_and_cannot_map_calls():
    backend = get_backend("torch")

    assert backend.status == "planned"
    assert "TorchBackend is planned but not implemented in the MVP." in (
        backend.unsupported_reason()
    )
    with pytest.raises(NotImplementedError):
        backend.map_function_call("np.mean")
