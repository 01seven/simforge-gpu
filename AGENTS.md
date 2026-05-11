# AGENTS.md

This file is the repository-level operating guide for AI coding agents working on
`sim2gpu`.

## Project Positioning

`sim2gpu` is a correctness-first, AI-agent-assisted workflow and Python toolkit
for migrating NumPy-based statistical simulation code from CPU to GPU, with
conversion plans, validation reports, benchmarks, and explanations.

Core rule:

```text
LLM suggests. Rules transform. Tests decide. Reports explain.
```

## MVP Boundary

The MVP supports only:

- Single-file Python simulation scripts.
- NumPy-style statistical simulation code.
- A small, explicit subset of NumPy APIs.
- Rule-based NumPy to CuPy conversion planning and later transformation.
- Structured conversion plans, validation reports, benchmark reports,
  explanation reports, and unsupported reports.
- No-GPU development and CI for analyzer, planner, transpiler, reporter, and
  backend-selection behavior.

## Out Of Scope For MVP

Do not implement or imply support for:

- General R or Python to GPU compilation.
- Real TorchBackend conversion.
- JAX, Numba-CUDA, or cuDF conversion.
- pandas-heavy pipelines.
- File I/O, plotting, network, database, multiprocessing, or multithreading
  conversion.
- Complex class-heavy code, dynamic `eval` / `exec`, complex closures, or
  arbitrary third-party package migration.
- Automatic mathematical proof of equivalence.
- Fake speedup numbers or GPU benchmark claims without measurement.

## Backend Strategy

MVP backend:

- `CuPyBackend`: implemented target for NumPy-style simulation code.

Planned backends:

- `TorchBackend`: planned only; important for future tensor-heavy simulation,
  PyTorch model integration, autograd, `torch.compile`, AMP, and distributed
  workflows.
- `JaxBackend`: planned only.
- `NumbaCudaBackend`: planned only.
- `CudfBackend`: planned only.

Never generate fake Torch, JAX, Numba-CUDA, or cuDF conversion code. If a user
selects an unimplemented backend, return a clear unsupported / not implemented
message and include it in the unsupported report.

## Correctness First

- Every conversion must have a conversion plan.
- Generated code must be syntax checked before it is presented as runnable.
- Unsupported features must be explicit; never silently skip unsafe code.
- LLM output is advisory and must not be the sole trusted source.
- Reports must explain risks, unsupported features, and validation status.

## Validation First

- Deterministic code should be checked with shape, dtype, and numerical
  tolerance comparisons.
- Stochastic simulation should be checked statistically, not by elementwise
  random-number equality.
- Validation can pass, fail, or be skipped with a clear reason.
- Performance claims must not be made when validation fails or is skipped.

## No-GPU Development

Agents must keep the project useful without CUDA hardware:

- Analyzer, planner, transpiler, reporter, CLI, and backend-selection tests must
  run without a GPU.
- GPU benchmark and real CuPy execution tests must be optional.
- Missing GPU should produce skipped benchmark status, not a crash.
- Tests must not depend on real LLM API output.

## Recommended Commands

Use these once tests exist:

```bash
python -m pytest tests/unit tests/integration
python -m pytest -m "not gpu"
python -m pytest -m gpu
```

Use the GPU-marked command only in an environment with CUDA and the required GPU
dependencies.

## Before Every Implementation

Before changing code, read:

- `AGENTS.md`
- `TASKS.md`
- The relevant file in `docs/`
- Any relevant workflow in `skills/sim2gpu/`

Confirm the task stays inside the MVP boundary.

## After Every Modification

Summarize:

- Files created or changed.
- What is planning/documentation versus executable implementation.
- Commands run and results.
- Tests run and results, or why tests were not run.
- Any unsupported behavior intentionally left unsupported.
