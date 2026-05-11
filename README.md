# sim2gpu

`sim2gpu` is a correctness-first workflow and Python toolkit for migrating
NumPy-based statistical simulation code from CPU to GPU with CuPy, while
producing conversion plans, unsupported reports, validation reports, benchmark
reports, and human-readable explanations.

```text
LLM suggests. Rules transform. Tests decide. Reports explain.
```

## MVP Status

| Area | Status |
| --- | --- |
| Python NumPy simulation analysis | Implemented |
| NumPy to CuPy conversion | Implemented for an explicit MVP API subset |
| Real CuPy validation and benchmark | Implemented when CuPy/CUDA are available |
| no-GPU analysis, planning, reporting, and tests | Implemented |
| TorchBackend | Planned only, no fake torch code |
| JAX, Numba-CUDA, cuDF, R support | Roadmap only |

`sim2gpu` is not a universal compiler. It is a conservative migration assistant
for small, auditable simulation scripts.

## Why This Exists

Statistical simulation code often repeats many independent trials: Monte Carlo
estimates, bootstrap resampling, permutation tests, random walks, and repeated
sampling experiments. These workloads can be good GPU candidates, but moving
them safely requires more than replacing `np` with `cp`.

`sim2gpu` focuses on the pieces that make a migration reviewable:

- static analysis of NumPy simulation scripts;
- a structured conversion plan before code generation;
- explicit unsupported reports for unsafe code;
- no-GPU-safe validation and benchmark behavior;
- real CuPy execution when GPU dependencies are available;
- reports that explain what changed and what remains uncertain.

## Install

Local no-GPU development:

```bash
python -m pip install -e .
```

Optional GPU execution:

```bash
python -m pip install -e ".[gpu]"
```

The default install does not require CuPy or CUDA. Analyzer, planner,
transpiler, reporter, CLI, and no-GPU tests work without GPU hardware.

## Try The MVP In 60 Seconds

```bash
sim2gpu list-backends
sim2gpu run-demo monte_carlo_pi
sim2gpu report projects/monte_carlo_pi
```

In a no-GPU environment, validation and benchmark reports are marked `SKIPPED`
with reasons. In a CuPy/CUDA environment, the same workflow can execute the CPU
and generated CuPy scripts.

## Core Workflow

```bash
sim2gpu analyze examples/monte_carlo_pi/input_cpu.py
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy
sim2gpu validate examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py
sim2gpu benchmark examples/monte_carlo_pi/input_cpu.py projects/monte_carlo_pi/generated/input_cpu_gpu.py --repeat 5 --warmup 1
sim2gpu explain projects/monte_carlo_pi/reports/conversion_plan.json
sim2gpu inspect-project projects/monte_carlo_pi
```

Torch is intentionally rejected in the MVP:

```bash
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target torch
```

That command writes an unsupported report and does not generate torch code.

## What Gets Generated

For a project such as `projects/monte_carlo_pi/`, `sim2gpu` writes:

```text
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

Every conversion has a plan. Unsupported features are reported rather than
silently skipped.

## Example: Monte Carlo Pi

Input:

```python
import numpy as np

n = 100_000
x = np.random.uniform(size=n)
y = np.random.uniform(size=n)
inside = x * x + y * y <= 1.0
pi_estimate = 4.0 * np.mean(inside)
print(pi_estimate)
```

The MVP can generate a CuPy candidate, validate the resulting scalar estimate
with stochastic tolerance, and benchmark CPU/GPU scripts when GPU execution is
available.

## Supported MVP Scope

Supported code shape:

- single-file Python scripts;
- NumPy-based statistical simulations;
- simple independent trials and repeated sampling;
- scalar output validation;
- simple numeric JSON/list output validation;
- conservative NumPy API rewrites to CuPy.

First-batch examples:

- `monte_carlo_pi`
- `normal_mean_probability`
- `bootstrap_mean`
- `random_walk`
- `permutation_test`

See [docs/EXAMPLES.md](docs/EXAMPLES.md) for example-specific conversion
status.

## Unsupported In The MVP

The MVP does not support:

- general R or Python to GPU compilation;
- real TorchBackend conversion;
- JAX, Numba-CUDA, or cuDF conversion;
- pandas-heavy pipelines;
- plotting conversion;
- file I/O as a GPU conversion target;
- network, database, multiprocessing, or multithreading conversion;
- class-heavy or dynamic `eval` / `exec` conversion;
- fake benchmark speedups or unvalidated performance claims.

Unsupported reports include stable categories such as
`backend_not_implemented`, `unsupported_numpy_api`, `pandas_pipeline`,
`plotting`, `file_io`, `dynamic_execution`, `side_effect_loop`, and
`sequential_dependency`.

## CLI Reference

| Command | Purpose |
| --- | --- |
| `sim2gpu analyze input.py` | Write static analysis IR |
| `sim2gpu convert input.py --target cupy` | Generate CuPy code and reports |
| `sim2gpu convert input.py --target cupy --dry-run` | Plan only, no generated GPU file |
| `sim2gpu convert input.py --target torch` | Planned-only rejection path |
| `sim2gpu validate original.py generated_gpu.py` | Validate CPU/GPU outputs or skip safely |
| `sim2gpu benchmark original.py generated_gpu.py` | Benchmark CPU/GPU scripts or skip safely |
| `sim2gpu explain conversion_plan.json` | Render a human-readable explanation |
| `sim2gpu run-demo monte_carlo_pi` | Run an included end-to-end demo |
| `sim2gpu report projects/monte_carlo_pi` | Print a concise project summary |
| `sim2gpu demo-status` | Summarize generated demo projects |
| `sim2gpu inspect-project projects/name` | Inspect artifacts and next steps |
| `sim2gpu check-artifacts projects/name` | CI-friendly artifact completeness check |
| `sim2gpu doctor` | Inspect local package and optional GPU status |
| `sim2gpu list-backends` | Show backend status |
| `sim2gpu list-patterns` | Show supported simulation patterns |

Machine-readable JSON is available for:

```bash
sim2gpu list-backends --json
sim2gpu list-patterns --json
sim2gpu demo-status --json
sim2gpu inspect-project projects/monte_carlo_pi --json
sim2gpu check-artifacts projects/monte_carlo_pi --json
sim2gpu doctor --json
```

## Validation

Validation is correctness-first:

- deterministic code should use shape, dtype, and numerical tolerance checks;
- stochastic simulation should use statistical tolerance, not elementwise random
  number equality;
- skipped validation is explicit and includes a reason;
- validation can write both Markdown and machine-readable JSON.

Useful options:

```bash
sim2gpu validate original.py generated_gpu.py --tolerance 0.2
sim2gpu validate original.py generated_gpu.py --repeat 3 --tolerance 0.2
```

The current GPU-capable MVP validates scalar outputs and simple numeric
JSON/list outputs.

## Benchmark

Benchmarking is intentionally cautious. A benchmark report may include CPU
runtime, GPU runtime, median/min/mean summaries, raw timing samples, and speedup
only after real execution.

Skipped benchmarks do not report speedup.

Successful MVP benchmark reports are marked:

```text
measurement_method: subprocess_wall_time
trust_level: demo_only
```

That means timings include Python process startup and CUDA initialization
overhead, do not isolate host/device transfer time, and should not be treated as
general performance claims.

## Tests

Recommended no-GPU CI command:

```bash
python -m pytest -m "not gpu" -q
```

Full local test suite:

```bash
python -m pytest -q
```

Optional GPU tests:

```bash
python -m pytest -m gpu -q
```

Use `SIM2GPU_DISABLE_GPU=1` to force validation and benchmark commands into
safe skipped-report mode.

## Documentation

- [Project brief](docs/PROJECT_BRIEF.md)
- [MVP scope](docs/MVP_SCOPE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Backend strategy](docs/BACKEND_STRATEGY.md)
- [Validation strategy](docs/VALIDATION_STRATEGY.md)
- [Testing strategy](docs/TESTING_STRATEGY.md)
- [Demo walkthrough](docs/DEMO.md)
- [Release checklist](docs/RELEASE_CHECKLIST.md)
- [Roadmap](docs/ROADMAP.md)

## Roadmap

Near-term:

- richer static analysis and memory suitability checks;
- broader validation beyond scalar and simple JSON/list outputs;
- in-process benchmark runner with transfer-overhead accounting.

Later:

- real TorchBackend design;
- R support;
- JAX;
- Numba-CUDA;
- cuDF;
- editor and agent workflow integrations.

## Contributing

Contributions should preserve the project boundary:

- no fake backend implementations;
- no silent unsupported behavior;
- no performance claims without validation and benchmark context;
- tests must not depend on a real LLM service;
- no-GPU tests should remain first-class.

## License

MIT License. See [LICENSE](LICENSE).
