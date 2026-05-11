import json

from sim2gpu.ir.schema import AnalysisIR, ConvertibleRegion, GpuSuitability


def test_analysis_ir_serializes_to_stable_json():
    ir = AnalysisIR(
        imports=("numpy",),
        patterns=("monte_carlo_loop",),
        random_calls=("np.random.normal",),
        outputs=("estimate",),
        convertible_regions=(
            ConvertibleRegion(
                type="for_loop",
                line_start=6,
                line_end=8,
                strategy="batch_vectorization",
            ),
        ),
        gpu_suitability=GpuSuitability(
            gpu_suitable=True,
            confidence="medium",
            recommended_backend="cupy",
            future_backend_candidates=("torch",),
            reasons=("Detected independent simulation trials.",),
            warnings=("Vectorization may increase memory use.",),
        ),
    )

    data = json.loads(ir.to_json())

    assert data["language"] == "python"
    assert data["imports"] == ["numpy"]
    assert data["convertible_regions"][0]["strategy"] == "batch_vectorization"
    assert data["gpu_suitability"]["future_backend_candidates"] == ["torch"]
