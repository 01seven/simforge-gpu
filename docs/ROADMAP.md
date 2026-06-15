# Roadmap

## Completed Legacy MVP

- Repository scaffold, AGENTS.md, workflow docs, examples, and tests.
- CLI for `analyze`, `convert`, `validate`, `benchmark`, `explain`,
  `review-suggestion`, `init-example`, `run-demo`, `report`, `demo-status`,
  `inspect-project`, `check-artifacts`, `doctor`, `list-patterns`, and
  `list-backends`.
- Backend registry for legacy CuPy and planned Torch/JAX/Numba-CUDA/cuDF.
- Python AST analyzer, lightweight IR, unsupported detector, conversion plan,
  report generation, and conservative NumPy-to-CuPy rewrite.
- Optional CuPy/CUDA validation and benchmark execution when available.
- No-GPU skipped validation and benchmark reports with trust indicators.
- Structured `runs/validation.json` and `runs/benchmark.json` for the legacy
  deterministic path.
- Release demo commands, dry-run conversion, project summary reports, and
  no-GPU CI command coverage.

## v2 Pivot Milestone

The v2 direction repositions SimForge GPU as a torch-first external-agent
harness while preserving the legacy CuPy MVP.

Implemented in the first v2 pass:

- `simforge inspect` for lightweight Python and R project inspection.
- `simforge plan` for v2 migration plans.
- `simforge task` for manual/Codex/Cursor-ready external-agent task artifacts.
- `simforge collect` for pending or collected agent traces.
- `py-torch` and `r-torch` as `primary_agent_target`.
- `cupy` as `legacy_deterministic_target`.
- `jax`, `numba-cuda`, and `cudf` as `planned_only` v2 targets.
- Python and R torch-target examples for task generation.
- No-GPU tests for the v2 workflow.

## Near-Term v2 Work

- Add richer project-level validation plans.
- Add user-provided candidate discovery in `workspace/`.
- Connect v2 `validate` and `benchmark` to project directories, not only
  original/generated file pairs.
- Add audit report rendering for v2 traces.
- Expand Python and R adapter heuristics while keeping no-execution behavior.

## Later Agent Automation

- Optional local launch integration for Codex, Claude Code, Cursor, or similar
  tools.
- Strict sandboxing and explicit user consent before automated agent execution.
- Better patch ingestion and conflict reporting.
- Agent run provenance and replay metadata.

## Later Torch Work

- Real PyTorch examples with manually authored candidates.
- Validation patterns for stochastic tensor simulations.
- GPU memory analysis and transfer-overhead reporting.
- Guidance for `torch.compile`, mixed precision, and device placement.

## Later R Work

- Real R torch examples with manually authored candidates.
- Stronger R project inspection.
- Optional parser-backed R analysis if textual inspection becomes insufficient.

## Roadmap Targets

JAX, Numba-CUDA, and cuDF remain roadmap-only. They should not produce generated
code until the harness has target-specific planning, validation, unsupported
handling, and benchmark evidence.

