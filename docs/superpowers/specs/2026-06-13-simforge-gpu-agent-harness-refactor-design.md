# SimForge GPU Agent Harness Refactor Design

## Purpose

This document is the design baseline for refactoring `SimForge GPU` into a
local CPU-to-GPU migration harness for AI coding agents.

It is written for a future Codex goal-driven development session. The goal
runner should treat this document as the product and architecture contract for
the refactor. It is not a sprint plan and it intentionally does not divide the
work into release phases. Instead, it defines the target shape, trust model,
artifact contracts, command behavior, implementation protocol, and acceptance
criteria that each development step must preserve.

The product positioning should become:

```text
SimForge GPU is a local CPU-to-GPU migration harness for AI coding agents.
```

The core rule should evolve from the current:

```text
LLM suggests. Rules transform. Tests decide. Reports explain.
```

to the harness-oriented form:

```text
Model proposes. Harness constrains. Tools execute. Tests decide. Reports explain.
```

The refactor should keep the current correctness-first spirit. It should add a
clear place for external model advice without allowing that advice to become a
trusted transformation.

## Product Inspiration

The design is inspired by the `harness + model = agent` framing used by
projects such as `hugohe3/ppt-master`.

The relevant idea is not PowerPoint generation itself. The relevant idea is
that the open-source project owns the workflow, artifacts, and local tools, while
an external AI coding agent supplies model intelligence. The model quality sets
the ceiling, but the harness supplies discipline.

For `SimForge GPU`, that means:

- Codex, Claude Code, Cursor, Copilot, or another external coding agent may read
  the repository, inspect source files, and produce structured advice.
- `SimForge GPU` must remain the local harness that validates advice, enforces
  backend policy, performs deterministic transformations, runs checks, and
  writes auditable reports.
- The first repo-native version must not become an OpenAI API wrapper.

## Non-Negotiable Principles

The refactor must preserve these rules:

- Every conversion still has a formal `conversion_plan.json`.
- Generated code is still syntax checked before being presented as runnable.
- Unsupported features are still explicit and never silently skipped.
- Model output is advisory and is never the sole trusted source.
- Reports explain risks, unsupported features, validation status, benchmark
  status, and skipped checks.
- Analyzer, planner, transpiler, reporter, CLI, backend-selection behavior, and
  no-GPU tests must run without CUDA.
- Tests must not depend on real LLM API output.
- CuPy remains the only implemented MVP backend.
- Torch, JAX, Numba-CUDA, and cuDF remain planned only.
- No fake Torch, JAX, Numba-CUDA, or cuDF conversion code may be generated.
- No speedup may be claimed without a real measured benchmark and validation
  context.
- Skipped validation or benchmark is not a correctness or performance claim.

## Explicit Non-Goals

Do not implement these as part of this refactor unless the user explicitly
changes the product direction:

- Direct OpenAI API calls from `simforge`.
- API key management.
- A hosted SaaS workflow.
- A real TorchBackend.
- JAX, Numba-CUDA, or cuDF conversion.
- General Python-to-GPU or R-to-GPU compilation.
- LLM-generated code paths that bypass the deterministic gates.
- Automatic acceptance of model-proposed patches.
- Performance claims from skipped, failed, or demo-only benchmark paths.

## Current Baseline To Preserve

The current repository already contains the beginning of a harness:

- `AGENTS.md` defines the repository-level correctness rules.
- `skills/simforge_gpu/SKILL.md` defines a strict migration workflow.
- `simforge_gpu/pipeline.py` orchestrates analysis, planning, conversion,
  reports, validation, benchmark, and project inspection.
- `simforge_gpu/analyzers/` performs AST and unsupported-feature analysis.
- `simforge_gpu/backends/` separates implemented and planned backends.
- `simforge_gpu/transpilers/numpy_to_cupy.py` performs conservative CuPy rewrites.
- `simforge_gpu/reporters/` writes Markdown and JSON reports.
- `simforge_gpu/runners/` executes optional CPU/GPU scripts when available.
- The existing no-GPU test path is `python -m pytest -m "not gpu" -q`.

This refactor should extend that harness rather than replacing it with a
black-box model workflow.

## Target Architecture

The target architecture has three layers.

### Agent Workflow Layer

This layer tells external AI coding agents how to work inside the repository.
It lives in:

```text
AGENTS.md
skills/simforge_gpu/SKILL.md
docs/*
docs/superpowers/specs/*
```

It defines:

- What external agents may do.
- What external agents must not do.
- Which artifacts they may write.
- Which CLI commands they should run.
- Which gates decide whether work is accepted.

### Harness Toolkit Layer

This is the executable Python package:

```text
simforge_gpu/
```

It owns:

- static analysis;
- backend policy;
- suggestion validation;
- formal conversion planning;
- deterministic CuPy rewrite;
- syntax checking;
- optional validation and benchmark execution;
- report generation;
- project inspection;
- artifact completeness checks.

### External Model Layer

This layer is outside `simforge`.

It may be Codex, Claude Code, Cursor, Copilot, Gemini CLI, or another
agent-capable environment. It supplies reasoning and writes advisory artifacts,
but it does not own the final trust decision.

For the repo-native harness, the external model interacts by reading files,
writing `model_suggestion.json`, and running local CLI commands. `simforge`
does not need to call the model directly.

## New Conceptual Flow

The model-assisted flow is:

```text
Input Python file
  -> static analysis by harness
  -> optional external model review
  -> model_suggestion.json written by external agent
  -> suggestion schema validation by harness
  -> backend policy enforcement by harness
  -> suggestion_review.json / suggestion_review.md
  -> accepted advisory fields merged into conversion_plan.json
  -> deterministic CuPy rewrite
  -> syntax report
  -> validation report
  -> benchmark report
  -> explanation report
  -> agent_trace.json
```

The deterministic flow remains:

```text
Input Python file
  -> static analysis
  -> conversion_plan.json
  -> deterministic CuPy rewrite
  -> syntax / validation / benchmark / reports
```

The deterministic flow must keep working without a model suggestion.

## Trust Model

The trust boundary is the most important part of the refactor.

These artifacts have different trust levels:

```text
model_suggestion.json = external model opinion
suggestion_review.json = harness review of that opinion
conversion_plan.json = formal plan accepted by the harness
unsupported_report.md = formal unsupported behavior record
validation_report.md = formal validation status
benchmark_report.md = formal benchmark status
agent_trace.json = audit trail connecting the above
```

The model may propose. The harness decides what is accepted.

### Advisory Artifact

`model_suggestion.json` is never trusted by itself. It may contain useful
semantic interpretation, risk notes, and validation suggestions, but it cannot
override deterministic checks.

### Formal Plan

`conversion_plan.json` is the accepted planning artifact. It may include
model-derived fields only after the harness validates them and marks their
source.

### Rules Beat Model Claims

If the unsupported detector flags `np.argsort`, the model cannot make that
unsupported item disappear.

If the backend registry says `torch` is planned, the model cannot make Torch
implemented.

If validation is skipped, the model cannot claim correctness.

If benchmark is skipped, failed, or demo-only, the model cannot claim speedup.

## Required New Package Boundaries

Add focused modules rather than growing `pipeline.py` without limit.

Recommended module layout:

```text
simforge_gpu/
  suggestions/
    __init__.py
    schema.py
    review.py
    merge.py
  tracing/
    __init__.py
    agent_trace.py
```

### `suggestions/schema.py`

Owns structured types for model suggestions and review results.

Responsibilities:

- represent the model suggestion schema;
- parse JSON payloads;
- provide stable `to_dict` / `to_json` output;
- avoid importing GPU libraries or model SDKs.

### `suggestions/review.py`

Owns validation of `model_suggestion.json`.

Responsibilities:

- validate required fields;
- validate enum values;
- validate source path consistency;
- validate backend policy through the existing backend registry;
- create warning and rejection entries;
- produce `suggestion_review.json` data.

### `suggestions/merge.py`

Owns the controlled merge from suggestion review into conversion plan inputs.

Responsibilities:

- accept only allowed advisory fields;
- preserve `source: "model"` metadata;
- never overwrite backend implementation status;
- never remove rule-detected unsupported features;
- never overwrite syntax, validation, benchmark, or quality statuses.

### `tracing/agent_trace.py`

Owns audit trail data.

Responsibilities:

- record the source file;
- record target backend;
- record suggestion artifact path;
- record suggestion review artifact path;
- record accepted and rejected model fields;
- record harness gate statuses;
- write stable JSON.

## Model Suggestion Schema

`model_suggestion.json` should use a layered schema: a small required core plus
optional detailed fields.

### Required Fields

Example:

```json
{
  "schema_version": "1.0",
  "source_file": "examples/monte_carlo_pi/input_cpu.py",
  "source_intent": "Estimate pi by sampling points in the unit square and measuring the fraction inside the unit circle.",
  "suggested_backend": "cupy",
  "mvp_fit": "yes",
  "confidence": "high",
  "risks": [
    {
      "code": "random_stream_difference",
      "message": "CPU NumPy and GPU CuPy random streams are not expected to match elementwise.",
      "severity": "medium"
    }
  ],
  "unsupported_hypotheses": [],
  "recommended_validation": {
    "type": "stochastic",
    "reason": "The output is a Monte Carlo scalar estimate; compare tolerance-bounded summary outputs rather than elementwise random samples."
  }
}
```

Required field rules:

- `schema_version` must be a supported string. Start with `"1.0"`.
- `source_file` must match the conversion input after path normalization.
- `source_intent` must be a non-empty string.
- `suggested_backend` must be a known backend name.
- `mvp_fit` must be one of `yes`, `partial`, `no`, or `unknown`.
- `confidence` must be one of `low`, `medium`, or `high`.
- `risks` must be a list.
- `unsupported_hypotheses` must be a list.
- `recommended_validation` must include `type` and `reason`.
- `recommended_validation.type` must be one of `deterministic`,
  `stochastic`, or `skipped`.

### Optional Fields

Example:

```json
{
  "api_mapping_notes": [
    {
      "original": "np.random.uniform",
      "suggested": "cp.random.uniform",
      "reason": "Supported by the current CuPy MVP mapping table."
    }
  ],
  "loop_strategy_notes": [],
  "memory_risk_notes": [
    {
      "message": "The default sample size is small; memory risk appears low for the fixture.",
      "severity": "low"
    }
  ],
  "output_semantics": "The script prints a single floating-point estimate.",
  "benchmark_notes": [
    "Benchmark only after validation is meaningful."
  ],
  "human_review_notes": [
    "Review tolerance before using this as a real statistical validation."
  ]
}
```

Optional field rules:

- Unknown optional fields may be preserved in raw trace but should not be merged
  into `conversion_plan.json` unless explicitly allowed.
- Optional fields must not be allowed to change backend status or support
  policy.
- Optional fields must not be allowed to claim validation or benchmark success.

## Suggestion Review Schema

`suggestion_review.json` records the harness review of model advice.

Example:

```json
{
  "schema_version": "1.0",
  "status": "ACCEPTED_WITH_WARNINGS",
  "source_file": "examples/monte_carlo_pi/input_cpu.py",
  "suggestion_file": "projects/monte_carlo_pi/reports/model_suggestion.json",
  "accepted_fields": [
    {
      "field": "source_intent",
      "reason": "Non-empty source intent accepted as model-provided explanation.",
      "source": "model"
    },
    {
      "field": "recommended_validation",
      "reason": "Validation type is compatible with detected random calls.",
      "source": "model"
    }
  ],
  "warnings": [
    {
      "field": "risks",
      "message": "Model risks are advisory and do not replace rule-detected unsupported features."
    }
  ],
  "rejected_fields": [],
  "unsupported_features": []
}
```

Allowed statuses:

```text
ACCEPTED
ACCEPTED_WITH_WARNINGS
REJECTED
```

Status rules:

- `ACCEPTED`: schema is valid and no policy warnings were needed.
- `ACCEPTED_WITH_WARNINGS`: schema is valid enough to use, but warnings or
  unsupported advisory claims were recorded.
- `REJECTED`: required fields are missing, source file does not match, schema
  version is unsupported, JSON cannot be parsed, backend is unknown, or another
  hard policy rule fails.

Backend policy rules:

- `suggested_backend: "cupy"` may be accepted if the CLI target is also `cupy`
  or if the harness can explain a safe target resolution.
- `suggested_backend: "torch"` must not be accepted as implemented. It should
  produce a planned-backend warning or unsupported feature.
- Unknown backends should reject the suggestion or mark the backend field
  rejected, depending on whether the rest of the suggestion is usable.

## Suggestion Markdown Report

`suggestion_review.md` should be human-readable.

Recommended structure:

```text
# Suggestion Review

Review status: ACCEPTED_WITH_WARNINGS
Source file: examples/monte_carlo_pi/input_cpu.py
Suggestion file: projects/monte_carlo_pi/reports/model_suggestion.json

## Accepted Fields

- source_intent: accepted as advisory explanation.
- recommended_validation: accepted as advisory validation note.

## Warnings

- risks: Model risks do not replace rule-detected unsupported features.

## Rejected Fields

- No rejected fields.

## Unsupported

- No unsupported model suggestions.
```

The report must never imply that model acceptance equals conversion correctness.

## Agent Trace Schema

`agent_trace.json` should connect model advice, harness gates, and generated
artifacts.

Example:

```json
{
  "schema_version": "1.0",
  "source_file": "examples/monte_carlo_pi/input_cpu.py",
  "target_backend": "cupy",
  "model_suggestion_path": "projects/monte_carlo_pi/reports/model_suggestion.json",
  "suggestion_review_path": "projects/monte_carlo_pi/reports/suggestion_review.json",
  "accepted_model_fields": [
    "source_intent",
    "risks",
    "recommended_validation",
    "output_semantics"
  ],
  "rejected_model_fields": [],
  "harness_gates": [
    {
      "name": "source_intake",
      "status": "PASSED",
      "artifact": "examples/monte_carlo_pi/input_cpu.py"
    },
    {
      "name": "static_analysis",
      "status": "PASSED",
      "artifact": "reports/analysis_ir.json"
    },
    {
      "name": "backend_policy",
      "status": "PASSED",
      "artifact": "reports/conversion_plan.json"
    },
    {
      "name": "suggestion_review",
      "status": "ACCEPTED_WITH_WARNINGS",
      "artifact": "reports/suggestion_review.json"
    },
    {
      "name": "syntax",
      "status": "PASSED",
      "artifact": "reports/syntax_report.md"
    },
    {
      "name": "validation",
      "status": "SKIPPED",
      "artifact": "reports/validation_report.md"
    },
    {
      "name": "benchmark",
      "status": "SKIPPED",
      "artifact": "reports/benchmark_report.md"
    }
  ],
  "artifacts": {
    "analysis_ir": "reports/analysis_ir.json",
    "conversion_plan": "reports/conversion_plan.json",
    "unsupported_report": "reports/unsupported_report.md",
    "explanation_report": "reports/explanation_report.md",
    "quality_report": "reports/quality_report.md"
  }
}
```

Trace rules:

- Always write `agent_trace.json` for `convert --suggestion`.
- Do not require `agent_trace.json` for the deterministic no-suggestion path
  unless the implementation chooses to add it consistently.
- If suggestion review is rejected, trace should still record the rejected
  suggestion and stopped gate.

## Conversion Plan Integration

The formal conversion plan should remain the source of truth for accepted
conversion behavior.

Add model-aware fields without breaking existing consumers. Prefer additive
fields such as:

```json
{
  "source_intent": {
    "text": "Estimate pi by Monte Carlo sampling.",
    "source": "model"
  },
  "model_advisory": {
    "suggestion_file": "reports/model_suggestion.json",
    "review_status": "ACCEPTED_WITH_WARNINGS",
    "accepted_fields": ["source_intent", "risks", "recommended_validation"]
  },
  "validation_notes": [
    {
      "message": "Use stochastic scalar tolerance rather than elementwise random equality.",
      "source": "model"
    }
  ]
}
```

If modifying the existing `ConversionPlan` dataclass, do so in a backward
compatible way:

- Existing fields must remain present.
- Existing tests expecting current fields should still pass after fixture
  updates where necessary.
- Existing deterministic conversions should still produce stable plans.
- New fields should be optional or have safe defaults.

## Merge Rules

Allowed model-derived plan additions:

- `source_intent`
- model-sourced risk notes;
- validation notes;
- output semantics;
- benchmark caution notes;
- human review notes;
- unsupported hypotheses as advisory entries, if clearly marked as model
  hypotheses.

Disallowed model-derived plan changes:

- replacing `target_backend`;
- changing backend implementation status;
- changing supported API mapping tables;
- removing rule-detected unsupported features;
- marking validation as passed;
- marking benchmark as passed;
- adding speedup;
- changing syntax status;
- changing quality gate status.

When in doubt, preserve the model content in `agent_trace.json` but do not merge
it into `conversion_plan.json`.

## CLI Behavior

### Deterministic Convert

This command must keep its current behavior:

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy
```

It should not require model suggestions, API keys, or network access.

### Model-Assisted Convert

Add:

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py \
  --target cupy \
  --suggestion projects/monte_carlo_pi/reports/model_suggestion.json
```

Expected behavior:

1. Read the source file.
2. Run static analysis.
3. Run unsupported detection.
4. Read and parse the model suggestion.
5. Validate the suggestion against schema and source file.
6. Enforce backend policy.
7. Write `suggestion_review.json`.
8. Write `suggestion_review.md`.
9. Merge accepted advisory fields into the formal conversion plan.
10. Run deterministic CuPy rewrite if the backend is implemented.
11. Syntax check generated code.
12. Run validation or write skipped validation report using current behavior.
13. Run benchmark or write skipped benchmark report using current behavior.
14. Write `agent_trace.json`.
15. Write explanation and quality reports.

If suggestion review is rejected:

- do not generate GPU code from that suggestion;
- write `suggestion_review.json`;
- write `suggestion_review.md`;
- write `agent_trace.json` if enough context exists;
- return a non-zero exit code unless a deliberate `--ignore-suggestion-errors`
  option is later designed and approved.

Do not add `--ignore-suggestion-errors` unless the user explicitly asks for it.

### Suggestion Review Command

Add:

```bash
simforge review-suggestion path/to/model_suggestion.json --source input.py
```

Recommended options:

```bash
simforge review-suggestion path/to/model_suggestion.json \
  --source examples/monte_carlo_pi/input_cpu.py \
  --output-dir projects/monte_carlo_pi
```

Behavior:

1. Read the source file.
2. Read and parse the model suggestion.
3. Validate schema and source file match.
4. Enforce backend policy.
5. Write `reports/suggestion_review.json`.
6. Write `reports/suggestion_review.md`.
7. Print the review status and report paths.

This command must not:

- generate GPU code;
- write `generated/*.py`;
- write validation or benchmark reports;
- claim conversion success.

### CLI Error Handling

Use clear exit codes:

- `0`: accepted or accepted with warnings where the command can proceed.
- `1`: artifact completeness or quality failure if matching existing behavior.
- `2`: user input, unsupported backend, rejected suggestion, missing file, or
  schema validation error.

Existing CLI behavior should not be broken.

## Report Behavior

### Explanation Report

When a model suggestion is used, the explanation report should include a concise
section:

```text
## Model Advisory Input

Suggestion review status: ACCEPTED_WITH_WARNINGS
Accepted fields:
- source_intent
- recommended_validation

Rejected fields:
- none

Note: Model advisory input is not treated as a trusted transformation.
```

### Unsupported Report

Unsupported report should include both deterministic unsupported entries and
accepted model unsupported hypotheses only if they are clearly marked.

Use source labels:

```text
Source: static_detector
Source: backend_policy
Source: model_hypothesis
```

Model hypotheses must not erase deterministic unsupported entries.

### Quality Report

Quality report should include suggestion review status when present:

```text
Suggestion review: ACCEPTED_WITH_WARNINGS
```

Suggestion warnings should not by themselves fail the quality gate unless they
represent a hard policy problem.

## Documentation Updates Required

Update documentation to reflect the harness positioning:

```text
README.md
docs/ARCHITECTURE.md
docs/MVP_SCOPE.md
docs/DEMO.md
skills/simforge_gpu/SKILL.md
AGENTS.md if needed
```

Documentation must say:

- `SimForge GPU` is a harness, not a model.
- The first repo-native harness does not directly call OpenAI or any model API.
- External agents write `model_suggestion.json`.
- The harness validates and selectively accepts model advice.
- `conversion_plan.json` remains the formal accepted plan.
- Tests and reports decide whether generated code is usable.
- Planned backends remain unsupported for code generation.

Avoid implying:

- built-in ChatGPT integration;
- automatic model calls;
- real Torch/JAX/Numba/cuDF support;
- validated speedups without measurement;
- correctness when validation is skipped.

## Goal-Driven Development Protocol For Codex

When using Codex goal mode to implement this refactor, follow this protocol.

### 1. Re-read Current Project Contracts

Before editing code, read:

```text
AGENTS.md
TASKS.md
docs/ARCHITECTURE.md
docs/MVP_SCOPE.md
docs/BACKEND_STRATEGY.md
docs/VALIDATION_STRATEGY.md
skills/simforge_gpu/SKILL.md
this design document
```

Confirm that the immediate change keeps the project inside the MVP boundary.

### 2. Preserve The Deterministic Baseline

Before adding model-assisted behavior, understand the current deterministic
path:

```text
simforge analyze
simforge convert --target cupy
simforge convert --target torch
simforge validate
simforge benchmark
simforge report
simforge inspect-project
simforge check-artifacts
```

Do not break existing commands while adding suggestion support.

### 3. Add Tests Before Or Alongside Behavior

Use tests to pin the trust boundary.

At minimum, add tests for:

- valid suggestion is accepted;
- valid suggestion with warnings is accepted with warnings;
- missing required field rejects suggestion;
- invalid JSON rejects suggestion;
- unsupported schema version rejects suggestion;
- source file mismatch rejects suggestion;
- unknown backend rejects suggestion;
- planned backend such as `torch` is not accepted as implemented;
- model risks are merged into plan with `source: "model"`;
- rule-detected unsupported features remain present even if model omits them;
- `review-suggestion` writes review artifacts and no generated code;
- `convert --suggestion` writes review artifacts, plan, generated code, and
  trace when accepted;
- rejected suggestion does not generate code;
- no-GPU validation and benchmark skip behavior remains unchanged.

### 4. Keep New Code Focused

Prefer new focused modules:

```text
simforge_gpu/suggestions/schema.py
simforge_gpu/suggestions/review.py
simforge_gpu/suggestions/merge.py
simforge_gpu/tracing/agent_trace.py
```

Avoid placing all new logic into `pipeline.py`.

`pipeline.py` may orchestrate the flow, but it should not own the suggestion
schema, review rules, merge policy, or trace schema.

### 5. Make Artifacts Stable

All new JSON artifacts should be deterministic:

- stable key ordering;
- stable enum values;
- stable list ordering where possible;
- no timestamps unless explicitly required by the user;
- no machine-specific absolute paths unless existing project conventions require
  them.

Prefer existing `stable_json` helpers where appropriate.

### 6. Preserve No-GPU Safety

Suggestion review must never require:

- CuPy;
- CUDA;
- real GPU hardware;
- model SDKs;
- network access;
- API keys.

`python -m pytest -m "not gpu" -q` must remain the primary verification command.

### 7. Update Reports And Docs Last Enough To Match Behavior

Update user-facing docs after command behavior and artifact shapes are clear
enough to describe accurately.

Do not document commands that do not exist.

Do not claim model integration that is only conceptual.

### 8. Verify And Review

For each meaningful implementation slice, run focused tests first, then the
no-GPU suite:

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -m "not gpu" -q
python -m pytest -m "not gpu" -q
```

If the environment cannot run one of these commands, record why.

Before declaring completion, inspect:

```bash
git status --short
git diff --stat
```

Confirm only intended files changed.

## Acceptance Criteria

The refactor is acceptable when all of the following are true.

### Product Positioning

- README and architecture docs describe SimForge as a local harness for AI coding
  agents.
- Docs clearly say the first repo-native harness does not directly call OpenAI
  or any other model API.
- Docs clearly distinguish external model advice from harness-accepted plans.

### Artifact Contracts

- `model_suggestion.json` schema is documented and implemented.
- `suggestion_review.json` is written for suggestion review.
- `suggestion_review.md` is written for human review.
- `agent_trace.json` is written for `convert --suggestion`.
- JSON artifacts are stable and test-covered.

### CLI Behavior

- Existing `simforge convert input.py --target cupy` still works.
- Existing planned-backend rejection still works.
- New `simforge convert input.py --target cupy --suggestion suggestion.json`
  works for accepted suggestions.
- New `simforge review-suggestion suggestion.json --source input.py` works and
  does not generate GPU code.
- Rejected suggestions stop before code generation.

### Trust Boundary

- Model suggestions cannot override backend registry status.
- Model suggestions cannot remove deterministic unsupported features.
- Model suggestions cannot mark validation or benchmark as passed.
- Model suggestions cannot create speedup claims.
- Merged model fields are marked with `source: "model"` or equivalent metadata.

### No-GPU Safety

- Suggestion review works without CuPy, CUDA, network access, model SDKs, or API
  keys.
- Existing no-GPU skipped validation and benchmark behavior remains unchanged.
- `python -m pytest -m "not gpu" -q` passes.

### Documentation

- `skills/simforge_gpu/SKILL.md` tells external agents how to write and submit
  `model_suggestion.json`.
- Demo docs include a copy-pasteable model-assisted flow that uses a local sample
  suggestion file or explains where an external agent writes one.
- Backend docs continue to reject fake planned-backend conversion.

## Test Matrix

Recommended new tests:

```text
tests/unit/test_model_suggestion_schema.py
tests/unit/test_model_suggestion_review.py
tests/unit/test_model_suggestion_merge.py
tests/unit/test_agent_trace.py
tests/unit/test_cli.py
tests/integration/test_model_assisted_conversion.py
```

Specific test cases:

```text
valid minimal suggestion -> ACCEPTED
valid detailed suggestion -> ACCEPTED
valid suggestion with advisory risk -> ACCEPTED_WITH_WARNINGS
missing schema_version -> REJECTED
unsupported schema_version -> REJECTED
source_file mismatch -> REJECTED
invalid suggested_backend -> REJECTED
suggested_backend torch -> not accepted as implemented
suggested_backend cupy with target cupy -> accepted
model risk merged with source=model
recommended_validation merged as advisory note only
model unsupported hypothesis preserved but cannot erase static unsupported
review-suggestion writes suggestion_review.json
review-suggestion writes suggestion_review.md
review-suggestion does not write generated/*.py
convert --suggestion writes agent_trace.json
convert --suggestion preserves existing reports
convert --suggestion rejected suggestion stops code generation
convert without suggestion remains backward compatible
```

## Example Goal Prompt

A future user may start a Codex goal with a prompt like:

```text
Read docs/superpowers/specs/2026-06-13-simforge-gpu-agent-harness-refactor-design.md
and refactor SimForge GPU into the repo-native agent harness described there.
Proceed step by step. Preserve the deterministic no-GPU path, add model
suggestion review and merge support, update docs, and verify with no-GPU tests.
Ask before making product-scope changes that conflict with the design.
```

Codex should then use this document as the contract and proceed through small
reviewable changes.

## Open Questions For Future Review

These questions should be reviewed during goal-driven development if they affect
implementation details:

- Should rejected suggestions return exit code `2` consistently with other CLI
  input errors?
- Should `agent_trace.json` be written for deterministic conversions too, or
  only for model-assisted conversions?
- Should `model_suggestion.json` be copied into `reports/` when the user passes
  a suggestion from another location?
- Should model unsupported hypotheses appear in `unsupported_report.md`, or only
  in `suggestion_review.md` and `agent_trace.json`?
- Should `conversion_plan.json` grow model-aware fields directly, or should the
  model advisory stay in a nested `model_advisory` object?

Default answers unless the user says otherwise:

- rejected suggestions return exit code `2`;
- trace is required for model-assisted conversions only;
- suggestion input is copied to `reports/model_suggestion.json` for auditability;
- model unsupported hypotheses appear in `suggestion_review.md` and
  `agent_trace.json`, and in `unsupported_report.md` only when clearly labeled;
- accepted model fields live under additive `source_intent`, `validation_notes`,
  and `model_advisory` fields.

## Final Design Sentence

External models provide intelligence. SimForge provides discipline.

