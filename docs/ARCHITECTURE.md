# Architecture

SimForge GPU is pivoting from a CuPy-first deterministic MVP into a
torch-first external-agent harness while preserving the legacy CuPy path.

```text
Harness owns trust. Models propose and edit. Tests decide. Reports explain.
```

## v2 Layer Model

### Harness Layer

The harness owns project state and artifacts. It creates inspections, migration
plans, task files, collection traces, validation reports, benchmark reports, and
audit records. It does not call model APIs in this pass.

Current modules:

- `simforge_gpu/harness/schemas.py`
- `simforge_gpu/harness/artifacts.py`
- `simforge_gpu/harness/workflow.py`

### Language Adapter Layer

Language adapters perform lightweight, no-execution inspection. They find source
files, entrypoints, dependencies, simulation indicators, randomness indicators,
I/O, plotting, and risky features.

Current modules:

- `simforge_gpu/languages/python_adapter.py`
- `simforge_gpu/languages/r_adapter.py`

The R adapter is intentionally textual in this pass. It is not a full R parser.

### Target Layer

The v2 target registry separates primary agent targets from legacy and roadmap
targets.

| Target | Status | Behavior |
| --- | --- | --- |
| `py-torch` | `primary_agent_target` | Generate external-agent task artifacts for Python to PyTorch migration. |
| `r-torch` | `primary_agent_target` | Generate external-agent task artifacts for R to R torch migration. |
| `cupy` | `legacy_deterministic_target` | Preserve the existing deterministic NumPy-to-CuPy MVP. |
| `jax` | `planned_only` | Roadmap only. |
| `numba-cuda` | `planned_only` | Roadmap only. |
| `cudf` | `planned_only` | Roadmap only. |

### Agent Protocol Layer

The agent protocol writes instructions for external agents and collects their
output without trusting it as validation evidence.

Current artifacts:

- `agent/migration_request.json`
- `agent/migration_task.md`
- `agent/constraints.md`
- `agent/expected_artifacts.json`
- `agent/agent_result.json`
- `reports/agent_trace.json`

### Validation And Benchmark Layer

Validation and benchmarking remain harness responsibilities. Agent claims are
recorded only as claims. Validation and benchmark states must be `PASS`, `FAIL`,
`SKIPPED`, `NOT_RUN`, or `PENDING_AGENT_OUTPUT` unless real execution occurs.

The existing CuPy validation and benchmark runners remain available for legacy
generated code. v2 task generation writes pending reports when no candidate is
available.

### Legacy CuPy Layer

The existing deterministic pipeline remains intact:

```text
input.py
  -> AST analysis
  -> conversion_plan.json
  -> conservative NumPy-to-CuPy rewrite
  -> syntax / validation / benchmark / quality / explanation reports
```

Existing modules such as `analyzers`, `ir`, `planners`, `transpilers`,
`backends`, `reporters`, `runners`, `suggestions`, and `tracing` continue to
support this legacy MVP.

## v2 Workflow

```text
Local R/Python project
  -> inspect
  -> plan
  -> generate agent task
  -> external agent modifies code
  -> collect agent result
  -> validate
  -> benchmark
  -> audit report
```

`inspect`, `plan`, `task`, and `collect` are no-GPU-safe. They do not execute
source code, generated code, GPU code, or model APIs.

## Inputs, Outputs, And Trust Gates

| Step | Input | Output | Gate |
| --- | --- | --- | --- |
| Inspect | Local project path | `reports/project_inspection.json` | Files and indicators recorded without execution. |
| Plan | Inspection and target | `reports/migration_plan.json` | Target status and risks explicit. |
| Task | Plan and agent name | `agent/*` task artifacts | External agent not launched. |
| Collect | Optional `agent_result.json` | `reports/agent_trace.json` | Agent claims separated from harness results. |
| Validate | CPU baseline and candidate | Validation report / run artifact | Real execution or honest skip/pending state. |
| Benchmark | CPU baseline and candidate | Benchmark report / run artifact | Speedup only after real measurement. |

## v2 Schema Families

The v2 harness uses plain dataclasses and stable JSON:

- `ProjectInspection`
- `MigrationPlan`
- `MigrationRequest`
- `AgentResult`
- `AgentTrace`

The schema version for this first v2 pass is `2.0`.

