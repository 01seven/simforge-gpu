# Explanation Report

Convert NumPy simulation code to CuPy where safe.

validation_status: SKIPPED
benchmark_status: SKIPPED

Artifacts:
- conversion_plan.json
- unsupported_report.md
- validation_report.md
- benchmark_report.md
- explanation_report.md

## Backend

Selected backend: torch
Backend status: planned

## Notes

- LLM output is not used as a trusted source in this MVP path.
- Rules transform only supported NumPy APIs.
- Tests and reports decide whether a conversion is usable.
