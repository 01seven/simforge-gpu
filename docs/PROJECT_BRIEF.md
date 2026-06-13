# Project Brief

## One-Sentence Introduction

`SimForge GPU` is a local CPU-to-GPU migration harness for AI coding agents and
a correctness-first Python toolkit for migrating NumPy-based statistical
simulation code from CPU to GPU.

## Target Users

- Statistics students.
- Data science students.
- Researchers who write simulation scripts.
- Python users who understand NumPy but do not want to write CUDA kernels.
- AI coding agents that need a strict workflow for safe GPU migration.

## Core Pain

Many statistical simulations are naturally parallel, but moving them from CPU
NumPy to GPU code requires knowledge of CUDA, GPU memory behavior, random number
differences, and validation design. Users often need help deciding whether GPU
migration is safe before they need help making it fast.

## Project Value

`SimForge GPU` gives users an auditable migration workflow:

- Analyze the source simulation.
- Decide whether it is suitable for GPU migration.
- Generate a structured conversion plan.
- Transform only supported regions.
- Validate deterministic or stochastic equivalence.
- Benchmark only when the environment supports it.
- Explain changes, unsupported features, and risks.
- Review external model advice as a local artifact without letting that advice
  bypass deterministic checks.

## Why This Is Not A General Code Converter

General code conversion is too broad for a correctness-first MVP. Arbitrary
Python or R programs may include side effects, I/O, plotting, network calls,
class-heavy behavior, dynamic execution, and dependencies that do not map safely
to GPU arrays. `SimForge GPU` intentionally begins with NumPy-style statistical
simulation and an explicit supported API list.

## Why This Is Not A Model Wrapper

The first repo-native harness does not call OpenAI or any other model API.
External agents may write `model_suggestion.json`, but the harness validates it,
records `suggestion_review.json`, and keeps `conversion_plan.json` as the
accepted plan.
