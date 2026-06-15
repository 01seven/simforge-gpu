# Python Monte Carlo Torch Target Example

This example is a CPU NumPy Monte Carlo simulation intended for the v2
external-agent harness.

Generate task artifacts:

```bash
simforge inspect examples/python_monte_carlo_torch --language python
simforge plan examples/python_monte_carlo_torch --language python --target py-torch
simforge task examples/python_monte_carlo_torch --agent codex --target py-torch
```

No PyTorch migration is generated automatically in this pass.

