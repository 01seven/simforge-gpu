# Benchmark Report

Benchmark status: PASSED

Repeat count: 3
Warmup runs: 1

Measurement method: subprocess_wall_time
Trust level: demo_only
Includes transfer overhead: False

CPU runtime: 0.305995s
GPU runtime: 1.885493s
CPU median runtime: 0.305995s
GPU median runtime: 1.885493s
CPU min runtime: 0.303946s
GPU min runtime: 1.869094s
CPU mean runtime: 0.305494s
GPU mean runtime: 1.890362s
Speedup: 0.162x
Reason: Measured by executing CPU and generated CuPy scripts locally.

Limitations:
- Includes Python process startup and CUDA initialization overhead.
- Does not isolate host/device transfer time.
- Does not use in-process synchronization around individual kernels.
- Use the reported speedup only as a local demo measurement, not a general performance claim.
