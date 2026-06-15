# SimForge GPU

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Primary: Torch Agent Harness](https://img.shields.io/badge/primary-Torch%20agent%20harness-blue)](docs/V2_PIVOT.md)
[![Legacy: CuPy MVP](https://img.shields.io/badge/legacy-CuPy%20MVP-2f6f9f)](docs/BACKEND_STRATEGY.md)

`SimForge GPU` is a torch-first agent harness for migrating local R/Python CPU
simulation code to GPU implementations using external coding agents.

```text
Harness owns trust. Models propose and edit. Tests decide. Reports explain.
```

It is not a universal compiler and not a built-in model API wrapper. In the v2
direction, SimForge GPU inspects projects, generates migration plans and
external-agent tasks, collects agent output, and keeps validation, benchmarking,
reports, audit traces, and trust boundaries under harness control.

The existing deterministic NumPy-to-CuPy MVP is preserved as a legacy path.

## v2 Quick Start

```bash
python -m pip install -e .

simforge inspect examples/python_monte_carlo_torch --language python
simforge plan examples/python_monte_carlo_torch --language python --target py-torch
simforge task examples/python_monte_carlo_torch --agent codex --target py-torch

# User or an external coding agent edits the project and writes agent_result.json.

simforge collect examples/python_monte_carlo_torch
```

For R torch task generation:

```bash
simforge inspect examples/r_monte_carlo_torch --language r
simforge plan examples/r_monte_carlo_torch --language r --target r-torch
simforge task examples/r_monte_carlo_torch --agent codex --target r-torch
```

This first v2 pass does not call Codex, OpenAI, Anthropic, Cursor, Claude Code,
or any other model API. It writes task artifacts for external agents to use.

## Legacy CuPy Quick Start

```bash
simforge list-backends
simforge run-demo monte_carlo_pi
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run
simforge report projects/monte_carlo_pi
```

In a no-GPU environment, validation and benchmark reports are written as
`SKIPPED` with clear reasons. Optional CuPy execution requires:

```bash
python -m pip install -e ".[gpu]"
```

## Current Scope

Primary v2 direction:

- project-level inspection for lightweight Python and R simulation projects;
- `py-torch` and `r-torch` as primary external-agent targets;
- task artifacts for manual, Codex, and Cursor-style external agents;
- pending/honest validation and benchmark states until real harness commands run;
- agent traces that separate agent claims from harness results.

Legacy deterministic MVP:

- single-file Python simulation analysis;
- conservative NumPy-to-CuPy conversion for a small supported API subset;
- conversion, unsupported, validation, benchmark, syntax, quality, and
  explanation reports;
- optional real CuPy validation and benchmark when CuPy/CUDA are available;
- no-GPU-safe tests and CLI behavior.

Planned only:

- automatic external-agent invocation;
- full Python-to-PyTorch transpilation;
- full R-to-R-torch migration;
- JAX, Numba-CUDA, and cuDF conversion;
- richer validation and transfer-overhead-aware benchmarking.

## v2 Workflow

```text
Local R/Python project
  -> inspect
  -> plan
  -> task
  -> external agent modifies code
  -> collect
  -> validate
  -> benchmark
  -> report / audit trace
```

The trust boundary is strict:

- `migration_request.json` and `migration_task.md` are harness-generated
  instructions;
- `agent_result.json` is external-agent output and remains advisory;
- agent validation and benchmark claims are not harness results;
- speedup appears only after real benchmark execution.

## Common Commands

```bash
# v2 agent-harness path
simforge inspect examples/python_monte_carlo_torch --language python
simforge plan examples/python_monte_carlo_torch --language python --target py-torch
simforge task examples/python_monte_carlo_torch --agent codex --target py-torch
simforge collect examples/python_monte_carlo_torch

# Legacy deterministic CuPy path
simforge analyze examples/monte_carlo_pi/input_cpu.py
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run

# Optional model-advisory path for the legacy CuPy workflow
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

Machine-readable JSON is available for several commands, including
`inspect --json`, `plan --json`, `task --json`, `collect --json`,
`list-backends --json`, `doctor --json`, `demo-status --json`,
`inspect-project --json`, and `check-artifacts --json`.

## v2 Artifacts

For a v2 project, commands may write:

```text
agent/migration_request.json
agent/migration_task.md
agent/constraints.md
agent/expected_artifacts.json
agent/agent_result.json
reports/project_inspection.json
reports/migration_plan.json
reports/agent_trace.json
reports/validation_report.md
reports/benchmark_report.md
workspace/
runs/
```

`collect` may write pending validation and benchmark reports, but it does not
execute validation or benchmarking.

## Explicitly Unsupported

This pass does not support:

- general Python or R to GPU compilation;
- automatic torch code generation;
- direct OpenAI, Anthropic, Cursor, or other model API calls from `simforge`;
- fake validation, fake benchmark results, or inferred speedup;
- pandas-heavy pipelines, plotting conversion, file I/O conversion, network or
  database conversion, multiprocessing conversion, or dynamic execution.

Unsupported behavior must be explicit in reports and task constraints.

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

- [v2 pivot](docs/V2_PIVOT.md)
- [Agent protocol](docs/AGENT_PROTOCOL.md)
- [Demo walkthrough](docs/DEMO.md)
- [MVP scope](docs/MVP_SCOPE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Backend strategy](docs/BACKEND_STRATEGY.md)
- [Validation strategy](docs/VALIDATION_STRATEGY.md)
- [Testing strategy](docs/TESTING_STRATEGY.md)
- [Roadmap](docs/ROADMAP.md)

## License

MIT License. See [LICENSE](LICENSE).

