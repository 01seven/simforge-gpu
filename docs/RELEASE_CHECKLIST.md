# Release Checklist

Use this checklist before publishing the MVP.

## Scope

- [ ] Confirm the release only claims Python NumPy simulation to CuPy support.
- [ ] Confirm Torch, JAX, Numba-CUDA, cuDF, and R remain documented as planned.
- [ ] Confirm benchmark reports keep `demo_only` trust indicators unless a richer runner is added.
- [ ] Confirm unsupported behavior is explicit and categorized.

## Local Verification

```bash
python -m pip install -e .
python -m pytest -q
python -m pytest -m "not gpu" -q
sim2gpu --help
sim2gpu list-backends
sim2gpu convert examples/monte_carlo_pi/input_cpu.py --target cupy --dry-run
sim2gpu run-demo monte_carlo_pi
sim2gpu report projects/monte_carlo_pi
```

Optional GPU verification:

```bash
python -m pip install -e ".[gpu]"
python -m pytest -m gpu -q
```

## Repository Hygiene

- [ ] `LICENSE` is present.
- [ ] `.gitignore` excludes caches, virtualenvs, build outputs, and local scratch artifacts.
- [ ] `.github/workflows/ci.yml` runs no-GPU tests.
- [ ] `__pycache__/`, `.pytest_cache/`, and local scratch JSON files are absent from version control.
- [ ] README quick start matches current CLI behavior.

## Demo Artifacts

- [ ] Decide whether to commit `projects/` demo artifacts.
- [ ] If committing demo artifacts, confirm benchmark numbers are clearly documented as local demo outputs.
- [ ] If not committing demo artifacts, confirm `sim2gpu run-demo monte_carlo_pi` regenerates them.

## Final Notes

- [ ] Do not publish real TorchBackend claims.
- [ ] Do not publish benchmark speedup claims without validation and trust context.
- [ ] Keep `LLM suggests. Rules transform. Tests decide. Reports explain.` visible in README.

