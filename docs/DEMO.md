# GPU-Capable MVP Demo

This walkthrough shows the current `sim2gpu` MVP. The core workflow still runs
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
sim2gpu doctor
sim2gpu doctor --json
sim2gpu list-backends
sim2gpu list-backends --json
sim2gpu list-patterns
sim2gpu list-patterns --json
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
sim2gpu analyze examples/monte_carlo_pi/input_cpu.py
```

This writes:

```text
projects/monte_carlo_pi/reports/analysis_ir.json
```

## 4. Convert To The MVP CuPy Target

```bash
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --validate --benchmark
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

## 5. Confirm Torch Is Planned Only

```bash
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target torch
```

Expected behavior:

- The command returns a planned / not implemented message.
- It does not generate torch code.
- It writes an unsupported report under `projects/monte_carlo_pi_torch/`.

## 6. Standalone Reports

```bash
sim2gpu explain projects/monte_carlo_pi/reports/conversion_plan.json
sim2gpu validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py
sim2gpu validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --tolerance 0.2
sim2gpu validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --repeat 3 --tolerance 0.2
sim2gpu benchmark examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --repeat 5 --warmup 1
sim2gpu report projects/monte_carlo_pi
sim2gpu check-artifacts projects/monte_carlo_pi
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

## 7. Run The One-Command Demo

```bash
sim2gpu run-demo monte_carlo_pi
```

This runs conversion, validation, benchmark, and final project reporting for the
included Monte Carlo pi example. In no-GPU mode the execution gates are skipped
with explicit reasons.

## 8. Summarize Demo Projects

```bash
sim2gpu demo-status
sim2gpu demo-status --json
```

This prints a table with backend, generated-file presence, unsupported count,
validation status, and benchmark status. The `--json` option emits the same
project summary as machine-readable JSON for CI or agent workflows.

## 9. Inspect A Partial Conversion

```bash
sim2gpu inspect-project projects/permutation_test
sim2gpu inspect-project projects/permutation_test --json
```

The permutation-test example is intentionally useful for demoing partial
conversion. It should report unsupported items such as `np.argsort` and a
sequential counter update. The generated source includes a partial-conversion
warning and keeps `import numpy as np` because an unsupported NumPy call remains.
The `--json` option emits the same inspection as structured JSON.

## 10. Run Tests

```bash
python -m pytest -q
python -m pytest -m "not gpu" -q
python -m pytest -m gpu -q
```

`not gpu` tests run without CUDA and without a real LLM API. GPU-marked tests
require CuPy/CUDA and are optional.
