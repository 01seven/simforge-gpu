# SimForge GPU

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Backend: CuPy](https://img.shields.io/badge/backend-CuPy-2f6f9f)](docs/BACKEND_STRATEGY.md)
[![Status: MVP](https://img.shields.io/badge/status-MVP-orange)](docs/MVP_SCOPE.md)

`SimForge GPU` is a local CPU-to-GPU migration harness for AI coding agents and
a correctness-first Python toolkit for NumPy-based statistical simulations.

```text
Model proposes. Harness constrains. Tools execute. Tests decide. Reports explain.
```

It is not a universal compiler and not a model API wrapper. External agents may
write advisory `model_suggestion.json` files, but SimForge GPU validates that
advice, enforces backend policy, performs deterministic CuPy rewrites, and keeps
`conversion_plan.json` as the harness-accepted plan.

## Why It Exists

Many simulations, such as Monte Carlo estimates, bootstrap resampling,
permutation tests, and random walks, are naturally parallel. Moving them safely
from NumPy to GPU code still requires careful handling of supported APIs, random
streams, validation, benchmark claims, and unsupported behavior.

SimForge GPU makes that migration auditable:

- analyze a single-file NumPy simulation script;
- create a conversion plan before code generation;
- rewrite only the supported MVP NumPy API subset to CuPy;
- syntax-check generated code before presenting it as runnable;
- report unsupported features instead of hiding them;
- validate and benchmark only when meaningful execution is available;
- explain the risks, skipped checks, and generated artifacts.

## Quick Start

```bash
python -m pip install -e .

simforge list-backends
simforge run-demo monte_carlo_pi
simforge report projects/monte_carlo_pi
```

In a no-GPU environment, validation and benchmark reports are written as
`SKIPPED` with clear reasons. Optional GPU execution requires:

```bash
python -m pip install -e ".[gpu]"
```

## Current MVP

Implemented:

- single-file Python simulation analysis;
- conservative NumPy-to-CuPy conversion;
- local review of external `model_suggestion.json` artifacts;
- conversion, unsupported, validation, benchmark, syntax, quality, and
  explanation reports;
- optional real CuPy validation and benchmark when CuPy/CUDA are available;
- no-GPU-safe tests and CLI behavior.

Planned only:

- TorchBackend;
- JAX;
- Numba-CUDA;
- cuDF;
- broader validation and richer benchmark accounting.

Planned backends are never used for generated code in the MVP.

## Workflow

Deterministic conversion:

```text
input.py
  -> static analysis
  -> conversion_plan.json
  -> deterministic CuPy rewrite
  -> syntax / validation / benchmark / quality / explanation reports
```

Model-assisted conversion:

```text
input.py
  -> static analysis
  -> model_suggestion.json
  -> suggestion_review.json / suggestion_review.md
  -> accepted advisory fields merged into conversion_plan.json
  -> deterministic CuPy rewrite
  -> agent_trace.json and reports
```

The trust boundary is strict:

- `model_suggestion.json` is external model opinion;
- `suggestion_review.json` is the harness review of that opinion;
- `conversion_plan.json` is the accepted plan;
- model advice cannot change backend status, remove unsupported features, mark
  validation or benchmark as passed, or create speedup claims.

## Common Commands

```bash
# Analyze and convert
simforge analyze examples/monte_carlo_pi/input_cpu.py
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run

# Optional model-assisted path
simforge review-suggestion model_suggestion.json --source examples/monte_carlo_pi/input_cpu.py
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --suggestion model_suggestion.json

# Reports and checks
simforge explain projects/monte_carlo_pi/reports/conversion_plan.json
simforge validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py
simforge benchmark examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py
simforge inspect-project projects/monte_carlo_pi
simforge check-artifacts projects/monte_carlo_pi

# Discovery
simforge doctor
simforge list-backends
simforge list-patterns
```

Machine-readable JSON is available for discovery and inspection commands, such
as `list-backends --json`, `doctor --json`, `demo-status --json`,
`inspect-project --json`, and `check-artifacts --json`.

## Generated Artifacts

For `projects/monte_carlo_pi/`, a conversion can write:

```text
generated/input_cpu_gpu.py
reports/analysis_ir.json
reports/conversion_plan.json
reports/unsupported_report.md
reports/validation_report.md
reports/benchmark_report.md
reports/syntax_report.md
reports/quality_report.md
reports/explanation_report.md
runs/validation.json
runs/benchmark.json
```

When a model suggestion is supplied, it also writes:

```text
reports/model_suggestion.json
reports/suggestion_review.json
reports/suggestion_review.md
reports/agent_trace.json
```

## Supported Scope

Supported:

- single Python files;
- NumPy-style statistical simulation code;
- simple independent trials and repeated sampling;
- scalar output validation;
- simple numeric JSON/list output validation;
- a small, explicit NumPy API subset mapped conservatively to CuPy.

First-batch examples:

- `monte_carlo_pi`
- `normal_mean_probability`
- `bootstrap_mean`
- `random_walk`
- `permutation_test`

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for example-specific status.

## Explicitly Unsupported

The MVP does not support:

- general Python or R to GPU compilation;
- real Torch, JAX, Numba-CUDA, or cuDF conversion;
- pandas-heavy pipelines;
- plotting conversion;
- file I/O as a GPU conversion target;
- network, database, multiprocessing, or multithreading conversion;
- class-heavy code or dynamic `eval` / `exec`;
- direct OpenAI or other model API calls from `simforge`;
- fake speedup numbers or unvalidated performance claims.

Unsupported behavior is written to reports with stable categories such as
`backend_not_implemented`, `unsupported_numpy_api`, `pandas_pipeline`,
`plotting`, `file_io`, `dynamic_execution`, `side_effect_loop`, and
`sequential_dependency`.

## Validation And Benchmark Honesty

Validation can be `PASS`, `FAIL`, or `SKIPPED`. Stochastic simulations are
checked statistically rather than by elementwise random-number equality.

Benchmark reports include speedup only after real CPU/GPU execution. Skipped
benchmarks never imply performance improvement. Current successful benchmark
runs are marked `subprocess_wall_time` / `demo_only`, so they are local demo
measurements rather than general performance claims.

## Tests

Recommended no-GPU check:

```bash
python -m pytest -m "not gpu" -q
```

Full local suite:

```bash
python -m pytest -q
```

Optional GPU tests:

```bash
python -m pytest -m gpu -q
```

## Documentation

- [Demo walkthrough](docs/DEMO.md)
- [MVP scope](docs/MVP_SCOPE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Backend strategy](docs/BACKEND_STRATEGY.md)
- [Validation strategy](docs/VALIDATION_STRATEGY.md)
- [Testing strategy](docs/TESTING_STRATEGY.md)
- [Roadmap](docs/ROADMAP.md)

## License

MIT License. See [LICENSE](LICENSE).
