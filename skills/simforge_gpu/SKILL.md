# SimForge GPU Agent Workflow

This file defines the strict workflow for agents migrating statistical
simulation code with `SimForge GPU`. It is a project workflow outline, not a complete
published ChatGPT Skill package.

Core principle:

```text
Model proposes. Harness constrains. Tools execute. Tests decide. Reports explain.
```

## Non-Negotiable Rules

- Do not convert code without a conversion plan.
- Do not silently skip unsupported code.
- Do not implement or fake TorchBackend in the MVP.
- Do not claim speedup without a real benchmark.
- Do not require a GPU for analysis, planning, reporting, or no-GPU tests.
- Do not rely on real LLM output for CI.
- Do not call a model API from `simforge`; external agents write advisory
  artifacts locally.
- Treat `model_suggestion.json` as opinion and `conversion_plan.json` as the
  harness-accepted plan.

## Step 1: Source Code Intake

Agent action:

- Confirm the source file exists.
- Confirm the input is Python.
- Read the file and preserve its path for reports.

Do not:

- Modify the source file in place.
- Guess missing files.

Input:

- User-provided source path.

Output:

- Source text and source metadata.

Gate:

- File exists and language is recognized as Python.

Failure behavior:

- Stop and report the missing or unsupported input.

## Step 2: Static Analysis

Agent action:

- Parse the file with Python AST tooling.
- Detect imports, NumPy aliases, loops, random calls, candidate outputs, and
  obvious side effects.

Do not:

- Execute the source during static analysis.

Input:

- Source text.

Output:

- Static analysis facts.

Gate:

- Imports, loops, random calls, and outputs are recorded or explicitly marked
  unknown.

Failure behavior:

- Produce an analysis failure entry and stop conversion planning if parsing
  fails.

## Step 3: GPU Suitability Analysis

Agent action:

- Estimate whether the simulation is a plausible GPU candidate.
- Consider trial count, loop independence, side effects, memory risk, and
  transfer overhead.

Do not:

- Promise speedup based only on code shape.

Input:

- Static analysis facts.

Output:

- Suitability report with confidence, reasons, warnings, recommended backend,
  and future backend candidates.

Gate:

- Suitability result includes reasons and warnings.

Failure behavior:

- Continue only if unsupported risks are clearly reported.

## Step 4: Backend Selection

Agent action:

- Select the requested backend or default to CuPy for MVP.
- Check backend implementation status.

Do not:

- Generate code for planned-only backends.

Input:

- Target backend request and suitability result.

Output:

- Backend status.

Gate:

- Target backend is implemented or explicitly rejected.

Failure behavior:

- For `torch`, `jax`, `numba`, or `cudf`, report planned / not implemented and
  stop code generation.

## Optional Step 4A: Model Suggestion Review

Agent action:

- If using external model advice, write `model_suggestion.json` with schema
  version `1.0`.
- Run `simforge review-suggestion model_suggestion.json --source input.py` to
  produce `suggestion_review.json` and `suggestion_review.md` without
  generating GPU code.
- For model-assisted conversion, run
  `simforge convert input.py --target cupy --suggestion model_suggestion.json`.

Do not:

- Put API keys, network calls, or model SDK calls in the harness.
- Let model advice change backend implementation status.
- Let model advice remove rule-detected unsupported features.
- Let model advice mark validation, benchmark, syntax, or quality gates as
  passed.

Input:

- Source file and external-agent `model_suggestion.json`.

Output:

- `reports/model_suggestion.json`
- `reports/suggestion_review.json`
- `reports/suggestion_review.md`
- `reports/agent_trace.json` for `convert --suggestion`

Gate:

- Suggestion review is `ACCEPTED`, `ACCEPTED_WITH_WARNINGS`, or `REJECTED`.

Failure behavior:

- Rejected suggestions stop before code generation and return an input-error
  exit code.

## Step 5: Conversion Plan

Agent action:

- Generate a structured conversion plan before transformation.
- Include patterns, changes, validation strategy, equivalence level, risks, and
  unsupported features.
- Merge accepted model advice only as source-labeled advisory fields.

Do not:

- Treat model suggestions as trusted transformations.
- Let model suggestions overwrite backend policy, unsupported detection,
  validation, benchmark, syntax, or quality gate status.

Input:

- IR, suitability result, backend status, unsupported findings.

Output:

- `conversion_plan.json`.

Gate:

- Plan includes risks and unsupported features, even if lists are empty.

Failure behavior:

- If no safe conversion exists, produce an unsupported report instead of
  generated code.

## Step 6: GPU Code Generation

Agent action:

- Apply rule-based transformations only for supported APIs and regions.
- Generate CuPy code for supported MVP regions.
- Preserve or report CPU-side code when appropriate.

Do not:

- Hard-convert unknown APIs.
- Generate torch, jax, numba, or cudf code in the MVP.

Input:

- Source code and conversion plan.

Output:

- Generated Python file or partial conversion status.

Gate:

- Generated file exists and passes syntax check.

Failure behavior:

- Stop before validation and report syntax or unsupported failure.

## Step 7: Validation

Agent action:

- Run deterministic or stochastic validation depending on the plan.
- Use statistical checks for random simulations.

Do not:

- Require elementwise equality for stochastic random simulations.
- Claim correctness if validation was skipped.

Input:

- Original code, generated code, expected outputs, validation configuration.

Output:

- Validation report.

Gate:

- Validation is `PASS`, `FAIL`, or `SKIPPED` with reason.

Failure behavior:

- If validation fails, report failure and avoid performance claims.

## Step 8: Benchmark

Agent action:

- Benchmark only after validation is acceptable or when explicitly requested for
  diagnostic purposes.
- Include warmup, transfer overhead, problem size, and environment status.

Do not:

- Invent speedup numbers.
- Fail no-GPU workflows because CUDA is unavailable.

Input:

- Runnable CPU/GPU scripts and benchmark configuration.

Output:

- Benchmark report.

Gate:

- Benchmark is completed or skipped with a clear reason.

Failure behavior:

- Missing GPU produces `SKIPPED`, not a misleading failure.

## Step 9: Explanation Report

Agent action:

- Explain what changed, why it changed, what was unsupported, and how validation
  should be interpreted.

Do not:

- Hide risks or skipped checks.

Input:

- Conversion plan, unsupported report, validation report, benchmark report.

Output:

- Explanation report.

Gate:

- Report lists artifacts, status, risks, unsupported features, and next steps.

Failure behavior:

- If upstream artifacts are missing, explain which gate failed.

## Step 10: Final Summary

Agent action:

- Summarize artifacts, changed files, validation status, benchmark status, and
  unsupported features.

Do not:

- Overstate project capabilities.

Input:

- All generated artifacts and command results.

Output:

- Final user-facing summary.

Gate:

- All artifacts are listed and current status is clear.

Failure behavior:

- Clearly name any skipped step and why it was skipped.
