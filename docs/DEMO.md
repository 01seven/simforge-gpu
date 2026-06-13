# GPU-Capable MVP Demo

This walkthrough shows the current `SimForge GPU` MVP. The core workflow still runs
without CUDA, CuPy, or a real LLM API. When CuPy/CUDA are available, the same
commands can execute generated CuPy code for validation and benchmark reports.

## 1. Install Locally

```bash
python -m pip install -e .
```

Optional GPU execution:

```bash
python -m pip install -e ".[gpu]"
```

## 2. Inspect Supported Backends And Patterns

```bash
simforge doctor
simforge doctor --json
simforge list-backends
simforge list-backends --json
simforge list-patterns
simforge list-patterns --json
```

Expected backend status:

```text
cupy   MVP backend / implemented
torch  planned
jax    planned
numba  planned
cudf   planned
```

## 3. Analyze The Monte Carlo Pi Example

```bash
simforge analyze examples/monte_carlo_pi/input_cpu.py
```

This writes:

```text
projects/monte_carlo_pi/reports/analysis_ir.json
```

## 4. Convert To The MVP CuPy Target

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --validate --benchmark
```

This writes:

```text
projects/monte_carlo_pi/generated/input_cpu_gpu.py
projects/monte_carlo_pi/reports/conversion_plan.json
projects/monte_carlo_pi/reports/unsupported_report.md
projects/monte_carlo_pi/reports/explanation_report.md
projects/monte_carlo_pi/reports/validation_report.md
projects/monte_carlo_pi/reports/benchmark_report.md
projects/monte_carlo_pi/reports/syntax_report.md
projects/monte_carlo_pi/reports/quality_report.md
projects/monte_carlo_pi/runs/validation.json
projects/monte_carlo_pi/runs/benchmark.json
```

Without GPU execution, validation and benchmark reports are explicit `SKIPPED`
reports. With CuPy/CUDA available, `--validate --benchmark` runs the CPU and
generated CuPy scripts and writes measured reports.

Use `--dry-run` when you want only the conversion plan and unsupported report
before writing generated GPU code.

## 5. Optional Model-Assisted Harness Flow

An external coding agent may write `model_suggestion.json`. The harness reviews
that local file; it does not call a model API.

Create a local sample suggestion:

```bash
python - <<'PY'
from pathlib import Path

target = Path("projects/monte_carlo_pi/reports/model_suggestion.json")
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    """{
  "schema_version": "1.0",
  "source_file": "examples/monte_carlo_pi/input_cpu.py",
  "source_intent": "Estimate pi with Monte Carlo sampling.",
  "suggested_backend": "cupy",
  "mvp_fit": "yes",
  "confidence": "high",
  "risks": [
    {
      "code": "random_stream_difference",
      "message": "CPU and GPU random streams are not expected to match.",
      "severity": "medium"
    }
  ],
  "unsupported_hypotheses": [],
  "recommended_validation": {
    "type": "stochastic",
    "reason": "Compare scalar estimates with a stochastic tolerance."
  }
}
""",
    encoding="utf-8",
)
PY
```

Review the suggestion without generating code:

```bash
simforge review-suggestion projects/monte_carlo_pi/reports/model_suggestion.json --source examples/monte_carlo_pi/input_cpu.py --output-dir projects/monte_carlo_pi
```

Run model-assisted conversion:

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --suggestion projects/monte_carlo_pi/reports/model_suggestion.json
```

This writes:

```text
projects/monte_carlo_pi/reports/model_suggestion.json
projects/monte_carlo_pi/reports/suggestion_review.json
projects/monte_carlo_pi/reports/suggestion_review.md
projects/monte_carlo_pi/reports/agent_trace.json
```

`model_suggestion.json` is model opinion. `suggestion_review.json` is the
harness review. `conversion_plan.json` remains the accepted plan.

## 6. Confirm Torch Is Planned Only

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py --target torch
```

Expected behavior:

- The command returns a planned / not implemented message.
- It does not generate torch code.
- It writes an unsupported report under `projects/monte_carlo_pi_torch/`.

## 7. Standalone Reports

```bash
simforge explain projects/monte_carlo_pi/reports/conversion_plan.json
simforge validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py
simforge validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --tolerance 0.2
simforge validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --repeat 3 --tolerance 0.2
simforge benchmark examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --repeat 5 --warmup 1
simforge report projects/monte_carlo_pi
simforge check-artifacts projects/monte_carlo_pi
```

In no-GPU mode:

- `validate` writes `Validation status: SKIPPED`.
- `benchmark` writes `Benchmark status: SKIPPED`.
- No speedup is reported.

In GPU-capable mode:

- `validate` can write `Validation status: PASSED` or `FAILED`.
- `validate --repeat` records repeated scalar outputs and max/mean absolute
  differences in `runs/validation.json`.
- `benchmark` can write measured CPU runtime, GPU runtime, median/min/mean
  runtime summaries, raw timing samples in `runs/benchmark.json`, and speedup.
- Benchmark reports also include trust indicators. Current successful runs are
  marked `subprocess_wall_time` / `demo_only`, so speedup is a local demo
  measurement rather than a general performance claim.
- `check-artifacts` verifies report and generated-source completeness.
- `report` prints a short release-style summary with an overall status and next
  recommended action.

## 8. Run The One-Command Demo

```bash
simforge run-demo monte_carlo_pi
```

This runs conversion, validation, benchmark, and final project reporting for the
included Monte Carlo pi example. In no-GPU mode the execution gates are skipped
with explicit reasons.

## 9. Summarize Demo Projects

```bash
simforge demo-status
simforge demo-status --json
```

This prints a table with backend, generated-file presence, unsupported count,
validation status, and benchmark status. The `--json` option emits the same
project summary as machine-readable JSON for CI or agent workflows.

## 10. Inspect A Partial Conversion

```bash
simforge inspect-project projects/permutation_test
simforge inspect-project projects/permutation_test --json
```

The permutation-test example is intentionally useful for demoing partial
conversion. It should report unsupported items such as `np.argsort` and a
sequential counter update. The generated source includes a partial-conversion
warning and keeps `import numpy as np` because an unsupported NumPy call remains.
The `--json` option emits the same inspection as structured JSON.

## 11. Run Tests

```bash
python -m pytest -q
python -m pytest -m "not gpu" -q
python -m pytest -m gpu -q
```

`not gpu` tests run without CUDA and without a real LLM API. GPU-marked tests
require CuPy/CUDA and are optional.
