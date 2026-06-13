# Rich Benchmark Engine Design

## Goal

Add a v0.2 benchmark path that produces more trustworthy local performance
measurements while preserving the existing no-GPU-safe behavior and conservative
reporting rules.

The feature covers three related benchmark improvements:

- in-process callable timing to reduce Python process startup noise;
- optional host/device transfer overhead measurement when it can be isolated;
- workload-size sweeps to show CPU/GPU crossover behavior.

## Non-Goals

This does not add new GPU backends, fake Torch/JAX/Numba/cuDF conversion, or
general Python profiling. It does not claim universal speedups. It does not
change validation semantics or require CUDA for no-GPU CI.

## User-Facing CLI

The existing subprocess benchmark remains the default:

```bash
simforge benchmark original.py generated_gpu.py
```

New richer benchmark options:

```bash
simforge benchmark original.py generated_gpu.py \
  --method in-process \
  --sizes 10000,100000,1000000 \
  --repeat 5 \
  --warmup 1 \
  --include-transfer
```

`simforge convert --benchmark` accepts matching options:

```bash
simforge convert input.py --target cupy --benchmark \
  --benchmark-method in-process \
  --benchmark-sizes 10000,100000 \
  --include-transfer
```

`--method subprocess` keeps the current behavior. `--method in-process` requires
both files to expose a callable `run` function. The runner will call
`run(size)` when a size sweep is configured and `run()` when no size is given.

## Architecture

Add an in-process benchmark runner in the runner layer, separate from the
existing subprocess script runner. The new runner loads the original and
generated files as isolated Python modules, finds `run`, performs warmups, and
records repeated CPU and GPU timings with `time.perf_counter`.

GPU timing synchronizes CuPy before and after each generated call when CuPy is
available. This prevents asynchronous kernel launch timing from looking faster
than completed GPU work.

Transfer measurement is explicit and conservative:

- If `--include-transfer` is not set, transfer status is `not_requested`.
- If the GPU callable returns a CuPy array, transfer time is measured by
  converting the returned value to host memory with `cp.asnumpy`.
- If the GPU callable already returns a Python scalar, transfer status is
  `embedded_in_result` because the generated code likely transferred during
  `float(...)`.
- If CuPy is unavailable or the return value cannot be classified, transfer
  status is `unsupported`.

The subprocess benchmark path remains available and keeps its existing
`subprocess_wall_time` / `demo_only` trust indicators.

## Data Flow

`cli.py` parses benchmark options and passes them to `pipeline.py`.

`pipeline.py` chooses one of two measurement paths:

- subprocess benchmark: current implementation;
- in-process benchmark: new callable runner.

Both paths write the same artifact locations:

- `reports/benchmark_report.md`;
- `runs/benchmark.json` when the generated file belongs to a generated project.

For in-process runs, the JSON result includes a top-level summary and a
`size_sweep` list. Each size entry includes raw CPU/GPU samples, median/min/mean
runtime summaries, optional transfer samples, and speedup only when both CPU and
GPU medians are valid.

## Error Handling

No-GPU behavior remains `SKIPPED`, not a crash.

In-process benchmark returns `FAILED` with a clear reason when:

- either module cannot be loaded;
- either file lacks a callable `run`;
- the callable raises an exception;
- CuPy execution is requested but cannot be initialized.

No speedup is reported for skipped or failed runs. Transfer numbers are omitted
unless measured.

## Reports

Markdown benchmark reports render:

- method, trust level, repeat, warmup;
- top-level median CPU/GPU runtime and speedup;
- size-sweep rows when present;
- transfer overhead status and measured transfer summaries when present;
- limitations that explain whether timings are subprocess demo timings or
  in-process local comparison timings.

Machine-readable JSON includes the same information. Report text must continue
to avoid unsupported performance claims.

## Tests

No-GPU tests cover:

- CLI parsing for `--method`, `--sizes`, `--include-transfer`, and convert
  equivalents;
- skipped in-process benchmark artifacts when GPU is disabled;
- report rendering for size sweeps and transfer statuses;
- callable-runner failure for missing `run`.

GPU-marked tests cover:

- in-process benchmark execution for a generated CuPy example;
- repeat and warmup handling;
- size sweep artifact shape;
- transfer status for scalar-returning examples.

Default CI remains:

```bash
python -m pytest -m "not gpu"
```

Optional GPU verification remains:

```bash
python -m pytest -m gpu
```
