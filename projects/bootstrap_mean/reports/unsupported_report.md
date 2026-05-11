# Unsupported Report

## 1. `estimates.append(...)`

Category: side_effect_loop

Reason: Detected a side-effect-heavy loop that mutates Python containers.

Action: Collect results with array operations or keep this loop on CPU.
