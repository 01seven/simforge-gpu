# R Monte Carlo Torch Target Example

This example is a CPU R Monte Carlo simulation intended for the v2 external-agent
harness.

Generate task artifacts:

```bash
simforge inspect examples/r_monte_carlo_torch --language r
simforge plan examples/r_monte_carlo_torch --language r --target r-torch
simforge task examples/r_monte_carlo_torch --agent codex --target r-torch
```

No R torch migration is generated automatically in this pass.

