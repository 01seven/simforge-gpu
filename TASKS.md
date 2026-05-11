# SimForge GPU MVP Task Breakdown

Each task must be small enough to implement and test independently. Phase 1 must
not require a real GPU, real LLM API, or real benchmark data.

## Status Legend

- `[x]` complete in the current MVP.
- `[ ]` planned or not implemented.

## [x] Task 1: Repo Scaffold

Create the package, docs, skills, examples, projects, and tests directories.

Acceptance criteria:

- Required directories exist.
- Empty directories are retained with `.gitkeep`.
- No core converter logic is implemented.

## [x] Task 2: AGENTS.md And Docs

Create repository-level agent rules and planning documents.

Acceptance criteria:

- `AGENTS.md` states MVP boundaries and backend policy.
- `docs/` includes project brief, scope, architecture, backend, validation,
  testing, and roadmap docs.
- Docs explicitly prohibit fake TorchBackend and silent unsupported handling.

## [x] Task 3: Package Skeleton

Create minimal Python package files.

Acceptance criteria:

- `simforge_gpu/__init__.py` exists.
- `simforge_gpu/cli.py` exists as a placeholder only.
- Importing the package does not require CuPy, CUDA, or an LLM client.

## [x] Task 4: CLI Skeleton

Add CLI command structure without full behavior.

Acceptance criteria:

- Commands are declared for `analyze`, `convert`, `validate`, `benchmark`,
  `explain`, `init-example`, `list-patterns`, and `list-backends`.
- `list-backends` can show `cupy implemented` and planned backend statuses.
- Unsupported targets return clear not-implemented messages.

## [x] Task 5: Backend Registry

Implement a backend registry that separates implemented and planned backends.

Acceptance criteria:

- Registry returns `CuPyBackend` as implemented.
- Registry returns planned status for torch, jax, numba, and cudf.
- Requesting an unknown backend returns a structured unsupported error.
- Tests run without importing CuPy.

## [x] Task 6: CuPyBackend Stub

Create a real MVP backend stub for CuPy API mapping metadata.

Acceptance criteria:

- Stub declares supported NumPy APIs and patterns.
- Environment validation can report CuPy/CUDA unavailable without crashing.
- No GPU work is performed in no-GPU tests.

## [x] Task 7: TorchBackend Planned Stub

Create a planned-only TorchBackend object.

Acceptance criteria:

- It never generates torch conversion code.
- It reports planned / not implemented.
- CLI and reports include the unsupported reason.

## [x] Task 8: AST Analyzer

Analyze a Python source file with the standard `ast` module.

Acceptance criteria:

- Detects NumPy imports and aliases.
- Detects simple `for` loops and `np.random` calls.
- Detects candidate output assignments.
- Produces deterministic structured analysis.

## [x] Task 9: IR Schema

Define the lightweight intermediate representation.

Acceptance criteria:

- IR captures language, imports, patterns, random calls, outputs,
  convertible regions, unsupported features, and GPU suitability.
- IR can serialize to JSON.
- Unit tests validate default and populated IR examples.

## [x] Task 10: Unsupported Detector

Detect code outside the MVP boundary.

Acceptance criteria:

- Flags pandas, plotting, file I/O, networking, multiprocessing, threading,
  `eval`, `exec`, and unimplemented backends.
- Reports code, reason, and recommended action.
- Does not silently drop unsupported code.

## [x] Task 11: Conversion Plan Schema

Create a structured conversion plan model.

Acceptance criteria:

- Plan includes source file, target backend, patterns, suitability, backend
  status, changes, validation strategy, equivalence level, risks, and
  unsupported features.
- Plans serialize to stable JSON for golden tests.

## [x] Task 12: Simple NumPy To CuPy Mapping

Implement conservative API mapping for the first supported NumPy APIs.

Acceptance criteria:

- Maps only explicitly supported APIs.
- Unknown APIs become unsupported report entries.
- Mapping can be tested without CuPy installed.

## [x] Task 13: Report Generator

Generate Markdown and JSON reports from structured data.

Acceptance criteria:

- Produces conversion plan, validation, benchmark, explanation, and unsupported
  report artifacts.
- Skipped validation or benchmark includes a reason.
- Reports do not invent speedups or correctness claims.

## [x] Task 14: No-GPU Tests

Add deterministic tests for planning and reporting behavior.

Acceptance criteria:

- `python -m pytest -m "not gpu"` passes without CUDA.
- Tests use fixtures and mocks, not real LLM calls.
- Planned backends are tested as unsupported for conversion.

## [x] Task 15: Example Fixtures

Add small example source files and expected planning artifacts.

Acceptance criteria:

- Examples cover Monte Carlo pi, normal mean probability, bootstrap mean,
  random walk, and permutation test.
- Expected artifacts are clearly marked as examples or golden fixtures.
- No example claims real benchmark results without execution.

## [x] Task 16: MVP Vertical Slice CLI

Wire analyzer, IR, planner, CuPy rewrite, reports, and generated file output
through the CLI.

Acceptance criteria:

- `simforge analyze examples/monte_carlo_pi/input_cpu.py` writes
  `analysis_ir.json`.
- `simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy` writes a
  generated CuPy file and all MVP reports.
- `simforge convert examples/monte_carlo_pi/input_cpu.py --target torch` writes an
  unsupported report and does not generate torch code.
- `python -m pytest` passes in a no-GPU environment.

## [x] Task 29: Static Syntax And Quality Reports

Add static quality artifacts for generated projects.

Acceptance criteria:

- `syntax_report.md` is written for conversions.
- `quality_report.md` summarizes syntax, unsupported, validation, and benchmark
  status.
- `inspect-project` and `check-artifacts` include the new reports.
- Tests cover report creation and inspection.

## [x] Task 30: Artifact Check CLI

Add a CI-friendly artifact completeness command.

Acceptance criteria:

- `simforge check-artifacts <project_dir>` returns success for complete projects.
- Missing required reports return exit code 1.
- `--json` emits structured status.
- The command does not execute GPU code.

## [x] Task 31: Optional GPU Validation

Run generated CuPy code and compare stochastic outputs when CuPy and CUDA are
available.

Acceptance criteria:

- `simforge validate original.py generated_gpu.py` executes CPU/GPU scripts when
  CuPy kernel execution is available.
- Missing or disabled GPU produces `SKIPPED`, not a crash.
- Monte Carlo pi validation uses stochastic tolerance.
- GPU tests are marked with `pytest.mark.gpu`.

## [x] Task 32: Optional GPU Benchmark Runner

Measure CPU and GPU runtime only when CuPy/CUDA execution is available.

Acceptance criteria:

- `simforge benchmark original.py generated_gpu.py` writes measured runtime and
  speedup only after real execution.
- Missing or disabled GPU produces `SKIPPED`.
- `simforge convert --validate --benchmark` can update reports in one command.
- GPU tests cover the benchmark path.

## [x] Task 33: Configurable Scalar GPU Validation

Add user-tunable scalar validation tolerance.

Acceptance criteria:

- `simforge validate original.py generated_gpu.py --tolerance <float>` is
  accepted.
- Reports include the configured tolerance for passed, failed, and skipped
  validation.
- GPU tests cover the custom tolerance path.

## [x] Task 34: Repeated Benchmark Reporting

Add repeat and warmup controls to benchmark execution.

Acceptance criteria:

- `simforge benchmark original.py generated_gpu.py --repeat <n> --warmup <n>` is
  accepted.
- Reports include repeat count, warmup count, median runtime, min runtime, mean
  runtime, and speedup after real execution.
- No-GPU skipped reports include repeat and warmup settings without speedup.

## [x] Task 35: Broader Scalar GPU Validation

Extend real GPU validation beyond Monte Carlo pi for safe scalar examples.

Acceptance criteria:

- `normal_mean_probability` validates on GPU with a stochastic tolerance.
- `random_walk` validates on GPU with a loose stochastic tolerance.
- Partial examples remain unsupported/partial rather than forced into real
  validation.

## [x] Task 36: Structured Run Artifacts And Repeated Validation

Persist machine-readable validation and benchmark execution results.

Acceptance criteria:

- Standalone `validate` writes `runs/validation.json` next to project reports
  when the generated file is under `generated/`.
- Standalone `benchmark` writes `runs/benchmark.json` next to project reports
  when the generated file is under `generated/`.
- `convert --validate --benchmark` writes the same structured run artifacts.
- Real benchmark JSON includes raw repeated CPU/GPU timing samples.
- `simforge validate ... --repeat <n>` records repeated scalar validation
  differences and summary statistics.
- no-GPU skipped validation and benchmark runs still write structured artifacts
  with clear skip reasons.

## [x] Task 37: JSON Array Output Validation

Compare simple numeric JSON/list outputs from CPU and generated CuPy scripts.

Acceptance criteria:

- Script runner parses JSON output from the final stdout line before falling
  back to scalar float parsing.
- Validation compares scalar and list-shaped numeric outputs with the same
  tolerance policy.
- Validation reports include output kind and element count.
- `runs/validation.json` records array/list output comparison summaries.
- Tests cover parser behavior and GPU array-output validation.

## [ ] Post-MVP: Broader Real GPU Validation

Extend real validation beyond scalar printed outputs.

## [x] Task 38: Benchmark Trust Indicators

Make benchmark trust and measurement limitations explicit.

Acceptance criteria:

- Benchmark Markdown reports include measurement method, trust level, transfer
  overhead status, and limitations.
- `runs/benchmark.json` includes the same trust metadata for passed, failed,
  and skipped benchmark paths.
- Skipped benchmarks use `not_run` / `not_applicable` metadata and do not imply
  speedup.
- GPU benchmark tests verify demo-only subprocess measurement metadata.
- no-GPU tests verify skipped benchmark trust metadata.

## [x] Task 39: Release Demo Commands

Add small release-readiness commands without expanding backend scope.

Acceptance criteria:

- `simforge convert <input.py> --target cupy --dry-run` writes planning reports
  without generated GPU code.
- Unsupported features include stable categories such as
  `backend_not_implemented`, `unsupported_numpy_api`, `pandas_pipeline`,
  `plotting`, `file_io`, `dynamic_execution`, `side_effect_loop`, and
  `sequential_dependency`.
- `simforge report <project_dir>` prints a user-facing summary with overall
  status and recommended next step.
- `simforge run-demo monte_carlo_pi` runs the included demo workflow and remains
  no-GPU-safe.
- README and demo docs include a 60-second release path.

## [ ] Post-MVP: Rich Benchmark Runner

Add transfer accounting and in-process execution.

## [x] Task 17: Standalone Explain CLI

Read a `conversion_plan.json` file and render a human-readable explanation.

Acceptance criteria:

- `simforge explain projects/monte_carlo_pi/reports/conversion_plan.json` prints
  an explanation.
- `--output explanation_report.md` writes the report.
- No GPU or LLM API is required.
- Tests cover the CLI path.

## [x] Task 18: Standalone No-GPU Validate CLI

Provide a safe validation command for no-GPU environments.

Acceptance criteria:

- `simforge validate original.py generated_gpu.py` writes a validation report.
- The report is explicitly `SKIPPED` in no-GPU MVP mode.
- The command does not execute generated GPU code.
- Tests cover skipped validation output.

## [x] Task 19: Standalone No-GPU Benchmark CLI

Provide a safe benchmark command for no-GPU environments.

Acceptance criteria:

- `simforge benchmark original.py generated_gpu.py` writes a benchmark report.
- The report is explicitly `SKIPPED` in no-GPU MVP mode.
- The report does not contain fake speedup values.
- Tests cover skipped benchmark output.

## [x] Task 20: Init Example CLI

Copy an included example into a project workspace.

Acceptance criteria:

- `simforge init-example <name>` supports all first-batch examples.
- Output goes to `projects/examples/<name>` by default.
- Tests cover copying a known example.

## [x] Task 21: Unsupported Detector Polish

Expand conservative unsupported detection rules.

Acceptance criteria:

- Flags unsupported backend, unsupported NumPy API, pandas, plotting, file I/O,
  dynamic `eval` / `exec`, class-heavy code, side-effect-heavy loops, and
  sequential dependency loops.
- Tests cover the conservative rules.

## [x] Task 22: First-Batch Example Conversion Coverage

Run every first-batch example through the no-GPU conversion pipeline.

Acceptance criteria:

- `monte_carlo_pi`, `normal_mean_probability`, `bootstrap_mean`, `random_walk`,
  and `permutation_test` all produce generated source plus MVP reports.
- Partial conversions with unsupported NumPy APIs keep required CPU-side imports.
- Golden coverage protects the Monte Carlo generated CuPy output.
- Golden coverage protects generated source for all first-batch examples.

## [x] Task 23: Demo Status CLI

Summarize generated project artifacts in a terminal-friendly table.

Acceptance criteria:

- `simforge demo-status` scans `projects/` by default.
- The summary includes backend, backend status, generated-file presence,
  unsupported count, validation status, and benchmark status.
- The command is no-GPU and report-driven; it does not execute generated code.
- Tests cover summary parsing and CLI output.

## [x] Task 24: Inspect Project CLI

Inspect one generated project in detail.

Acceptance criteria:

- `simforge inspect-project projects/<name>` prints artifact paths, backend
  status, unsupported entries, validation status, benchmark status, and next
  steps.
- The command is no-GPU and report-driven; it does not execute generated code.
- Tests cover CLI output.

## [x] Task 25: No-GPU Demo Walkthrough

Document a copy-pasteable GitHub demo path.

Acceptance criteria:

- `docs/DEMO.md` lists install, backend listing, analysis, conversion,
  planned-backend rejection, explain, validate, benchmark, demo status,
  inspect-project, and tests.
- The walkthrough explicitly says validation and benchmark are skipped in
  no-GPU mode.
- Tests verify the documented commands remain present.

## [x] Task 26: Doctor CLI

Add a no-GPU-safe environment inspection command.

Acceptance criteria:

- `simforge doctor` prints Python/package status, backend status, optional CuPy
  availability, and workspace checks.
- The command does not import CuPy directly or require CUDA.
- Tests cover report content and CLI output.

## [x] Task 27: Machine-Readable No-GPU Status

Add JSON output for no-GPU demo inspection commands.

Acceptance criteria:

- `simforge doctor --json` prints structured environment status.
- `simforge demo-status --json` prints structured project summary rows.
- `simforge inspect-project <project_dir> --json` prints structured
  single-project status.
- JSON output does not execute GPU code or import CuPy directly.
- Tests cover the JSON CLI paths.

## [x] Task 28: Machine-Readable Discovery Commands

Add JSON output for backend and pattern discovery.

Acceptance criteria:

- `simforge list-backends --json` prints structured backend status.
- `simforge list-patterns --json` prints supported pattern names.
- JSON output does not import CuPy, call GPU code, or require an LLM.
- Tests cover both discovery JSON paths.
