# Testing Strategy

The MVP must be testable without a GPU and without a real LLM API.

## No-GPU CI

No-GPU CI should cover:

- AST analyzer behavior.
- IR serialization.
- Conversion plan schema.
- Unsupported detection.
- Backend registry and unsupported backend behavior.
- Rule-based mapping metadata.
- Report generation.
- CLI command parsing.
- Model suggestion schema validation, review, merge policy, and trace artifacts.

No-GPU tests must not import CuPy as a required dependency.

## Optional GPU Tests

Optional GPU tests may cover:

- CuPy availability.
- Real generated CuPy execution.
- GPU validation.
- GPU benchmark measurement.

These tests should be marked, for example:

```bash
python -m pytest -m gpu
```

The current GPU-capable milestone includes GPU-marked tests for the Monte Carlo
pi vertical slice. They execute CPU and generated CuPy scripts when CuPy/CUDA
kernel execution is available, and cover scalar validation repeats plus
benchmark repeats.

Default CI should use:

```bash
python -m pytest -m "not gpu"
```

## Unit Tests

Unit tests should focus on isolated modules: analyzer functions, backend status,
IR models, planner behavior, unsupported detector rules, and reporters.

## Golden File Tests

Golden tests should compare stable JSON artifacts such as conversion plans and
unsupported reports. JSON output should be deterministic.

## Snapshot Tests

Snapshot tests can verify Markdown reports when the report wording is expected to
remain stable.

## Integration Tests

Integration tests should run a no-GPU pipeline from source fixture to analysis,
planning, unsupported report, and explanation report. They should not require
actual GPU execution.

No-GPU tests also verify that skipped validation and benchmark commands write
machine-readable `runs/validation.json` and `runs/benchmark.json` artifacts
without executing generated CuPy code.

GPU integration tests must be marked `gpu` and skipped when CuPy/CUDA execution
is unavailable. Default no-GPU behavior remains protected by
`python -m pytest -m "not gpu"`.

## Backend Selection Tests

Backend tests must verify:

- `cupy` is the only implemented MVP backend.
- `torch`, `jax`, `numba`, and `cudf` are planned.
- Planned backends do not generate target code.

## Unsupported Backend Tests

Tests must assert that selecting `--target torch` produces a structured
unsupported result and an actionable message.

## Model Suggestion Fixture Tests

Model-assisted harness tests use local `model_suggestion.json` fixtures. CI must
never depend on real LLM output, network access, model SDKs, API keys, or model
availability.

Tests must verify that model suggestions cannot override backend policy,
unsupported detection, validation status, benchmark status, syntax status,
quality status, or speedup claims.
