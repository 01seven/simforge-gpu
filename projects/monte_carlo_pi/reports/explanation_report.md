# Explanation Report

Convert NumPy simulation code to CuPy where safe.

validation_status: PASSED
benchmark_status: PASSED

Artifacts:
- projects\monte_carlo_pi\generated\input_cpu_gpu.py
- conversion_plan.json
- unsupported_report.md
- validation_report.md
- benchmark_report.md
- explanation_report.md

## Backend

Selected backend: cupy
Backend status: implemented

## Notes

- LLM output is not used as a trusted source in this MVP path.
- Rules transform only supported NumPy APIs.
- Tests and reports decide whether a conversion is usable.
