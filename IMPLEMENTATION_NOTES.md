# Implementation Notes

This file records MVP implementation status, intentional limitations, and
planned work.

## Implemented In The MVP

- Local package metadata via `pyproject.toml`.
- CLI entry point: `simforge = "simforge_gpu.cli:main"`.
- `simforge --help`.
- `simforge list-backends`.
- `simforge list-backends --json`.
- `simforge list-patterns`.
- `simforge list-patterns --json`.
- `simforge analyze <input.py>`.
- `simforge convert <input.py> --target cupy`.
- `simforge convert <input.py> --target cupy --dry-run`.
- `simforge convert <input.py> --target cupy --validate --benchmark` for
  optional GPU execution when CuPy/CUDA are available.
- `simforge convert <input.py> --target torch` as an explicit unsupported path.
- `simforge explain <conversion_plan.json>`.
- `simforge validate <original.py> <generated_gpu.py>` as optional real GPU
  validation with no-GPU skip fallback.
- `simforge validate <original.py> <generated_gpu.py> --tolerance <float>`.
- `simforge validate <original.py> <generated_gpu.py> --repeat <n>` for
  repeated scalar stochastic validation.
- `simforge benchmark <original.py> <generated_gpu.py>` as optional real GPU
  benchmark with no-GPU skip fallback.
- `simforge benchmark <original.py> <generated_gpu.py> --repeat <n> --warmup <n>`.
- `simforge init-example <name>`.
- `simforge run-demo <name>`.
- `simforge report <project_dir>`.
- `simforge demo-status`.
- `simforge demo-status --json`.
- `simforge inspect-project <project_dir>`.
- `simforge inspect-project <project_dir> --json`.
- `simforge check-artifacts <project_dir>`.
- `simforge check-artifacts <project_dir> --json`.
- `simforge doctor`.
- `simforge doctor --json`.
- Standalone validation and benchmark reports default to the matching project
  `reports/` directory when the generated file is under `projects/*/generated/`.
- Standalone validation and benchmark commands also write structured run
  artifacts to `runs/validation.json` and `runs/benchmark.json` when the
  generated file is under a project `generated/` directory.
- Backend registry with CuPy implemented and torch/jax/numba/cudf planned.
- Backend and pattern discovery can be emitted as JSON.
- Metadata-only CuPy backend with supported NumPy API and pattern lists.
- Planned-only backend stubs that do not generate code.
- Python AST analyzer for imports, NumPy aliases, loops, random calls, and
  assignment outputs.
- Lightweight IR JSON output.
- Conversion plan JSON output.
- Conservative NumPy-to-CuPy rewrite for explicitly supported APIs.
- Unsupported report, explanation report, validation skip report, and benchmark
  skip report.
- Syntax report and quality report artifacts.
- Optional CuPy/CUDA runner for generated scripts.
- GPU-marked tests for real Monte Carlo pi validation and benchmark execution.
- GPU-marked tests for `normal_mean_probability` and `random_walk` validation.
- Repeated benchmark reports with warmup, median, min, and mean runtime fields.
- Structured benchmark run artifacts with raw CPU/GPU timing samples.
- Benchmark trust indicators in Markdown and JSON, including measurement
  method, trust level, transfer-overhead status, and limitations.
- Repeated scalar validation reports with raw output samples, absolute
  differences, max difference, and mean difference.
- JSON/list numeric output validation for simple array-like outputs printed by
  CPU and generated CuPy scripts.
- Conservative unsupported detection for unsupported backends, unsupported NumPy
  APIs, pandas, plotting, file I/O, dynamic execution, class-heavy code,
  side-effect-heavy loops, and sequential dependency loops.
- Standard unsupported categories for stable reports and JSON artifacts.
- All first-batch examples run through the no-GPU conversion pipeline and
  produce generated source plus reports.
- All first-batch generated source outputs are covered by golden tests.
- Generated demo projects can be summarized with `simforge demo-status`.
- Generated demo project summaries can be emitted as JSON for CI or agent
  workflows.
- A single generated project can be inspected with `simforge inspect-project`.
- Single-project inspection can be emitted as JSON for CI or agent workflows.
- Local no-GPU environment status can be checked with `simforge doctor`.
- Local no-GPU environment status can be emitted as JSON with
  `simforge doctor --json`.
- `docs/DEMO.md` provides a reproducible no-GPU MVP walkthrough.
- Partial conversion keeps `import numpy as np` when unsupported `np.*` calls
  remain in generated code.
- Generated partial conversions include a source-level warning that points users
  to `unsupported_report.md`.
- no-GPU unit and integration tests.

## Intentional MVP Limitations

- Generated CuPy code is syntax checked but not executed in no-GPU mode.
- `simforge doctor` checks package/workspace status and optional CuPy/CUDA
  execution readiness.
- Validation is reported as `SKIPPED` when GPU execution is unavailable or
  disabled.
- Benchmark is reported as `SKIPPED` when GPU execution is unavailable or
  disabled.
- Speedup is reported only after a real local benchmark run.
- Small demo benchmarks may show GPU slower than CPU because process startup,
  CUDA initialization, and kernel compilation overhead dominate.
- Benchmark trust level is currently `demo_only` for successful GPU runs because
  measurements use subprocess wall time and do not isolate transfer overhead or
  individual kernels.
- TorchBackend is planned only and does not generate torch code.
- JAX, Numba-CUDA, cuDF, and R support are roadmap only.
- Standalone `validate` and `benchmark` commands are no-GPU-safe and become
  real execution commands when CuPy/CUDA execution is available.
- The rewrite engine is intentionally conservative and only maps explicitly
  supported NumPy APIs.
- `convert --dry-run` does not write generated GPU code; it is for planning and
  unsupported review.

## Artifact Layout

Default output for `examples/monte_carlo_pi/input_cpu.py` goes to:

```text
projects/monte_carlo_pi/
  generated/input_cpu_gpu.py
  reports/analysis_ir.json
  reports/conversion_plan.json
  reports/unsupported_report.md
  reports/explanation_report.md
  reports/validation_report.md
  reports/benchmark_report.md
  reports/syntax_report.md
  reports/quality_report.md
  runs/validation.json
  runs/benchmark.json
```

## Next Work

- Improve AST analysis for more loop patterns and output detection.
- Add transfer-overhead accounting and repeated in-process benchmark execution.
- Broaden real GPU validation beyond simple JSON/list numeric outputs.
