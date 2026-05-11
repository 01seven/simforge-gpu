# Roadmap

## MVP

- Complete: repository scaffold.
- Complete: agent workflow guide.
- Complete: Python package skeleton and local install metadata.
- Complete: CLI for `analyze`, `convert`, `list-patterns`, and `list-backends`.
- Complete: backend registry.
- Complete: implemented CuPyBackend metadata and conservative mapping.
- Complete: planned-only stubs for Torch, JAX, Numba-CUDA, and cuDF.
- Complete: AST analyzer.
- Complete: lightweight IR.
- Complete: basic GPU suitability metadata.
- Complete: conversion plan schema.
- Complete: unsupported detector and report.
- Complete: Markdown/JSON report generation.
- Complete: no-GPU test suite.
- Complete: Monte Carlo pi vertical slice with generated CuPy source and skipped
  validation/benchmark reports.
- Complete: standalone `explain`, `validate`, `benchmark`, and `init-example`
  CLI polish for no-GPU demo use.
- Complete: conservative unsupported detection for class-heavy code,
  side-effect-heavy loops, sequential dependency loops, and unsupported NumPy
  APIs.
- Complete: no-GPU conversion artifact coverage for all first-batch examples,
  with conservative partial-conversion import handling.
- Complete: golden generated-source coverage for all first-batch examples.
- Complete: report-driven `demo-status` CLI for generated project summaries.
- Complete: report-driven `inspect-project` CLI for single-project drill-downs.
- Complete: no-GPU demo walkthrough in `docs/DEMO.md`.
- Complete: no-GPU `doctor` CLI for local environment inspection.
- Complete: JSON status output for `doctor`, `demo-status`, and
  `inspect-project`.
- Complete: JSON discovery output for `list-backends` and `list-patterns`.
- Complete: static `syntax_report.md` and `quality_report.md` artifacts.
- Complete: `check-artifacts` CLI for project artifact completeness.
- Complete: optional CuPy/CUDA validation and benchmark execution for generated
  Monte Carlo pi code.
- Complete: GPU-marked tests for real validation and benchmark paths.
- Complete: configurable scalar validation tolerance.
- Complete: repeated benchmark measurements with warmup and median/min/mean
  runtime summaries.
- Complete: real GPU validation coverage for `normal_mean_probability` and
  `random_walk`.
- Complete: structured `runs/validation.json` and `runs/benchmark.json`
  artifacts for standalone and convert-triggered execution paths.
- Complete: repeated scalar stochastic validation with max/mean absolute
  difference summaries.
- Complete: simple numeric JSON/list output validation for generated CuPy
  scripts.
- Complete: benchmark trust indicators for skipped and successful benchmark
  paths.
- Complete: release demo commands, including dry-run conversion, project
  summary reports, and one-command demo workflow.

## v0.2

- Broader real GPU validation beyond scalar and simple JSON/list outputs.
- Better loop and output detection.
- More example fixtures.
- Richer validation report rendering.
- Stable CLI artifact layout.

## v0.3

- Safer vectorization strategies.
- Memory suitability analysis.
- More statistical validation checks.
- Transfer-overhead accounting.
- In-process benchmark execution with transfer-overhead accounting.

## TorchBackend

TorchBackend remains planned until the project can provide trustworthy tensor
mapping, validation, and unsupported behavior. It should not be implemented by
renaming NumPy calls to torch calls without semantic review.

## R Support

Future R support may parse R simulation code into an intermediate representation
or a Python NumPy-like representation before targeting GPU backends. It is not in
the MVP.

## JAX

JAX may be useful for functional, compiled numerical workloads. It requires a
separate backend strategy and is roadmap only.

## Numba-CUDA

Numba-CUDA may help when simulations need custom kernels. It is a deeper
compiler-style path and is roadmap only.

## cuDF

cuDF may be useful for dataframe-heavy GPU workflows. pandas-heavy conversion is
outside the MVP and belongs to future work.

## VS Code And Agent Workflow Integration

Future work may include editor tasks, agent prompts, project templates, and
workflow commands for tools such as Codex, Claude Code, Cursor, Kiro, and VS Code
Copilot.
