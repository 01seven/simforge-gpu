from sim2gpu.transpilers.numpy_to_cupy import map_numpy_api, rewrite_supported_numpy_calls


def test_map_numpy_api_maps_only_supported_calls():
    mapped = map_numpy_api("np.mean")
    unsupported = map_numpy_api("np.linalg.inv")

    assert mapped.replacement == "cp.mean"
    assert mapped.unsupported_feature is None
    assert unsupported.replacement is None
    assert unsupported.unsupported_feature.code == "np.linalg.inv"


def test_rewrite_supported_numpy_calls_reports_unknown_calls_without_hard_convert():
    source = """
import numpy as np

x = np.random.normal(size=10)
y = np.mean(x)
z = np.linalg.inv(np.eye(2))
"""

    result = rewrite_supported_numpy_calls(source)

    assert "import cupy as cp" in result.source
    assert "cp.random.normal" in result.source
    assert "cp.mean" in result.source
    assert "np.linalg.inv" in result.source
    assert result.unsupported_features[0].code == "np.linalg.inv"
