# Example Status

The MVP includes five first-batch examples. Each example has a CPU NumPy input,
can be analyzed, and can produce CuPy-oriented generated source plus reports
without requiring CUDA. The `monte_carlo_pi` example also has a real optional
GPU validation/benchmark vertical slice when CuPy/CUDA execution is available.

For a full command walkthrough, see [DEMO.md](DEMO.md).

## Status Table

| Example | Conversion status | Notes |
| --- | --- | --- |
| `monte_carlo_pi` | Full conservative rewrite | Supported APIs: `np.random.uniform`, `np.mean` |
| `normal_mean_probability` | Full conservative rewrite | Supported APIs: `np.random.normal`, `np.mean` |
| `random_walk` | Full conservative rewrite | Supported APIs: `np.random.binomial`, `np.maximum`, `np.sum` |
| `bootstrap_mean` | Partial conversion | Python list mutation is reported as a side-effect-heavy loop |
| `permutation_test` | Partial conversion | Sequential counter update and unsupported `np.argsort` are reported |

## Artifact Pattern

For an example named `<name>`, demo artifacts are written under:

```text
projects/<name>/
  generated/input_cpu_gpu.py
  reports/analysis_ir.json
  reports/conversion_plan.json
  reports/unsupported_report.md
  reports/explanation_report.md
  reports/validation_report.md
  reports/benchmark_report.md
  reports/syntax_report.md
  reports/quality_report.md
```

## Golden Coverage

Generated source for all first-batch examples is covered by golden tests in
`tests/golden/`. These golden files protect the conservative rewrite behavior and
the partial-conversion warnings.

## Demo Status

Use the CLI to summarize generated project artifacts:

```bash
simforge demo-status
```

The command reports backend status, generated file presence, unsupported count,
validation status, and benchmark status. It reads artifacts only and does not
execute generated GPU code.

For one project, use:

```bash
simforge inspect-project projects/permutation_test
```

This prints artifact presence, unsupported entries, validation and benchmark
status, and suggested next steps.

To verify artifact completeness:

```bash
simforge check-artifacts projects/monte_carlo_pi
```
