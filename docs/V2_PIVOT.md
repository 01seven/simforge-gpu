# v2 Pivot

SimForge GPU began as a correctness-first deterministic NumPy-to-CuPy MVP. That
legacy path remains useful, but it is too narrow for the broader goal of helping
users migrate real local simulation projects.

The v2 pivot makes SimForge GPU a torch-first external-agent harness.

## What Changes

Primary direction:

- Python CPU simulation to PyTorch GPU implementation through external agents.
- R CPU simulation to R torch GPU implementation through external agents.
- Harness-generated project inspections, migration plans, task files, collection
  traces, validation reports, benchmark reports, and audit artifacts.

The harness does not become a universal compiler. It constrains and verifies the
work done by external coding agents.

## What Stays

The existing CuPy deterministic MVP stays in place as the legacy path:

```bash
simforge convert examples/monte_carlo_pi/input_cpu.py --target cupy
```

This path remains useful for supported NumPy-style single-file simulations and
continues to run without a GPU for analysis, planning, reporting, and skipped
execution artifacts.

## What Is Intentionally Unsupported

This first v2 pass does not:

- automatically call Codex, OpenAI, Anthropic, Cursor, or Claude Code;
- generate PyTorch or R torch code automatically;
- claim validation success from agent output;
- claim benchmark speedup without real measurement;
- support JAX, Numba-CUDA, or cuDF beyond roadmap metadata.

## First v2 Workflow

```bash
simforge inspect ./my_project --language python
simforge plan ./my_project --language python --target py-torch
simforge task ./my_project --agent codex --target py-torch
simforge collect ./my_project
```

For R:

```bash
simforge inspect ./my_r_project --language r
simforge plan ./my_r_project --language r --target r-torch
simforge task ./my_r_project --agent codex --target r-torch
```

The generated task artifacts are designed to be pasted into an external coding
agent or used as local instructions.

