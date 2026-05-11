# Validation Strategy

Validation is a core feature, not an optional polish step. A conversion is not
trustworthy until its outputs are checked or explicitly marked unvalidated.

## Deterministic Validation

Use deterministic validation for code without random simulation behavior.

Checks may include:

- Shape comparison.
- Dtype comparison.
- `np.allclose`.
- Absolute tolerance.
- Relative tolerance.
- Summary statistic comparison.

Status values:

- `PASS`
- `FAIL`
- `SKIPPED`

Skipped validation must include a reason.

The GPU-capable MVP executes standalone validation when CuPy/CUDA kernel
execution is available. If CuPy is missing, CUDA is unavailable, or
`SIM2GPU_DISABLE_GPU=1` is set, validation remains `SKIPPED` with a clear
reason.

## Stochastic Validation

Use stochastic validation for random simulations. CPU NumPy and GPU CuPy can use
different random number generators, floating-point order, and parallel execution
behavior. Elementwise equality is not expected.

The first real GPU validation paths compare scalar estimates for Monte Carlo pi,
normal mean probability, and random walk examples with user-configurable
stochastic tolerances. These are smoke-level statistical checks, not proof of
full distributional equivalence.

Standalone validation accepts `--repeat <n>` for scalar stochastic examples. The
report records repeat count, max absolute difference, and mean absolute
difference. The matching `runs/validation.json` artifact also preserves raw CPU
outputs, raw GPU outputs, and per-run absolute differences.

The runner parses JSON from the final stdout line before falling back to the
last scalar float. This allows the MVP to compare simple numeric list outputs
such as `[mean, std, variance]` without pretending to validate arbitrary Python
objects.

Checks may include:

- Final estimate difference.
- Mean difference.
- Variance difference.
- Quantile comparison.
- Confidence interval overlap.
- Kolmogorov-Smirnov test where appropriate.
- Repeated validation runs.
- User-defined tolerance.
- Machine-readable run artifacts.
- Numeric JSON/list output comparison for small summary arrays.

## Equivalence Levels

```text
Level 1: API / syntactic equivalence
Level 2: numerical equivalence
Level 3: statistical equivalence
```

Reports must state the equivalence level and why it was chosen.

## Validation Report Template

```text
# Validation Report

Validation type: stochastic
Equivalence level: statistical
Status: PASS

CPU estimate: 0.3142
GPU estimate: 0.3150
Absolute difference: 0.0008
Tolerance: 0.005

Checks:
- mean difference: PASS
- variance difference: PASS
- quantile comparison: PASS

Notes:
- CPU and GPU random samples are not expected to match elementwise.
- Floating-point differences may occur.
```
