import json

from sim2gpu.ir.schema import AnalysisIR, GpuSuitability
from sim2gpu.planners.conversion_plan import create_conversion_plan


def test_create_conversion_plan_records_backend_status_and_unsupported_target():
    ir = AnalysisIR(
        imports=("numpy",),
        patterns=("monte_carlo_loop",),
        random_calls=("np.random.uniform",),
        outputs=("pi_estimate",),
        gpu_suitability=GpuSuitability(
            gpu_suitable=True,
            confidence="medium",
            recommended_backend="cupy",
            future_backend_candidates=("torch",),
        ),
    )

    plan = create_conversion_plan(ir, source_file="input.py", target_backend="torch")
    data = json.loads(plan.to_json())

    assert data["source_file"] == "input.py"
    assert data["target_backend"] == "torch"
    assert data["backend_status"]["implemented"] is False
    assert data["unsupported_features"][0]["code"] == "--target torch"
    assert data["unsupported_features"][0]["category"] == "backend_not_implemented"
    assert data["equivalence_level"] == "statistical"
