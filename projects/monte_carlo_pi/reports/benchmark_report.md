# Benchmark Report

Benchmark status: PASSED

Repeat count: 3
Warmup runs: 1

Measurement method: subprocess_wall_time
Trust level: demo_only
Includes transfer overhead: False

CPU runtime: 0.304871s
GPU runtime: 1.854971s
CPU median runtime: 0.304871s
GPU median runtime: 1.854971s
CPU min runtime: 0.302176s
GPU min runtime: 1.853872s
CPU mean runtime: 0.304966s
GPU mean runtime: 1.893094s
Speedup: 0.164x
Reason: Measured by executing CPU and generated CuPy scripts locally.

Limitations:
- Includes Python process startup and CUDA initialization overhead.
- Does not isolate host/device transfer time.
- Does not use in-process synchronization around individual kernels.
- Use the reported speedup only as a local demo measurement, not a general performance claim.
