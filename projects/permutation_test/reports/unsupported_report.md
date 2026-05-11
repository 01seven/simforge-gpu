# Unsupported Report

## 1. `count augmented assignment`

Category: sequential_dependency

Reason: Detected a sequential dependency loop that may not be safely parallelizable.

Action: Keep this loop on CPU or rewrite it with an explicit parallel strategy.

## 2. `np.argsort`

Category: unsupported_numpy_api

Reason: NumPy API is not in the MVP supported mapping table.

Action: Leave this call unchanged or add a supported mapping with tests.
