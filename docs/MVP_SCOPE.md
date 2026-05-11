# MVP Scope

## Supported In The MVP

- Single Python input files.
- NumPy-based statistical simulation code.
- Simple independent trials.
- Simple vectorized NumPy expressions.
- Basic GPU suitability analysis.
- Lightweight IR generation.
- Conversion plan generation.
- Explicit unsupported reports.
- Conservative NumPy to CuPy mapping.
- No-GPU tests for planning and reporting behavior.

## Not Supported In The MVP

- General R support.
- General Python application migration.
- pandas-heavy pipelines.
- Plotting conversion.
- File I/O as a conversion target.
- External APIs, network calls, or database calls.
- multiprocessing or multithreading conversion.
- Complex OOP-heavy code.
- Complex closures.
- Dynamic `eval` or `exec`.
- Arbitrary third-party package conversion.
- Sequential simulations with iteration-to-iteration dependencies that cannot
  be safely parallelized.
- Real TorchBackend, JAX, Numba-CUDA, or cuDF conversion.

## First Supported NumPy APIs

```text
np.array
np.zeros
np.ones
np.arange
np.linspace
np.mean
np.var
np.std
np.sum
np.sqrt
np.exp
np.log
np.maximum
np.minimum
np.random.normal
np.random.uniform
np.random.binomial
```

## First Supported Simulation Patterns

```text
monte_carlo_independent_trials
simple_bootstrap_mean
random_walk
normal_mean_probability
permutation_test
```

## Must Produce Unsupported

The tool must produce an unsupported report entry when it detects:

- A backend selected outside the implemented MVP backend.
- NumPy APIs outside the supported mapping table.
- pandas, plotting, file I/O, networking, databases, multiprocessing, threading,
  `eval`, or `exec`.
- Complex side effects in a candidate loop.
- Output variables that cannot be identified.
- A conversion that would require pretending Torch, JAX, Numba-CUDA, or cuDF
  support exists.
- A likely GPU memory explosion from unsafe full vectorization.
