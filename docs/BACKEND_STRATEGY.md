# Backend Strategy

## Why CuPy Is The MVP Default

CuPy is the safest first backend because it mirrors much of the NumPy API and is
well aligned with NumPy-style statistical simulation. For the MVP, this reduces
semantic risk and lets the project focus on planning, validation, and reporting.

## Why TorchBackend Matters Later

TorchBackend is important for future users whose simulations are tensor-heavy or
need to connect with PyTorch models. It may support future workflows involving:

- Autograd.
- PyTorch model integration.
- `torch.compile`.
- Mixed precision.
- Distributed training or simulation pipelines.

TorchBackend is not implemented in the MVP.

## Backend Abstraction

Planned backend interface:

```text
Backend
  - name
  - status
  - supported_patterns
  - supported_numpy_apis
  - map_imports()
  - map_function_call()
  - map_random_call()
  - handle_array_to_cpu()
  - generate_prelude()
  - validate_environment()
  - benchmark_capabilities()
  - unsupported_reason()
```

Backends must be metadata-driven enough that analyzer, planner, and tests can run
without importing actual GPU libraries.

## Backend Status

| Backend | MVP status | Behavior |
| --- | --- | --- |
| `CuPyBackend` | Implemented | Supported target for MVP conversion |
| `TorchBackend` | Planned | Return not implemented; do not generate torch code |
| `JaxBackend` | Planned | Return not implemented |
| `NumbaCudaBackend` | Planned | Return not implemented |
| `CudfBackend` | Planned | Return not implemented |

## CLI Behavior For Unimplemented Backends

If a user runs:

```bash
sim2gpu convert input.py --target torch
```

The CLI should return:

```text
TorchBackend is planned but not implemented in the MVP.
Use --target cupy for the current supported backend.
```

The same reason must appear in the unsupported report. The command must not
produce fake torch code.
