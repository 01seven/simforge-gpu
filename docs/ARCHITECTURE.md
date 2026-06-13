# Architecture

## Three-Layer Design

### Agent Workflow Layer

The Agent Workflow Layer constrains AI coding agents. It lives in files such as
`AGENTS.md`, `skills/simforge_gpu/SKILL.md`, and design documents under
`docs/`.

It defines:

- The strict migration pipeline.
- Quality gates for every step.
- What agents may and may not convert.
- Required artifacts.
- Failure behavior for unsupported code.
- How external agents may write advisory `model_suggestion.json` artifacts.

### Python Toolkit Layer

The Python Toolkit Layer is the executable package.

Planned modules:

- `analyzers`: parse Python and detect simulation features.
- `ir`: define lightweight intermediate representation.
- `planners`: create conversion plans.
- `transpilers`: perform rule-based transformations.
- `backends`: describe implemented and planned target backends.
- `validators`: compare CPU and GPU outputs.
- `reporters`: write JSON and Markdown reports.
- `runners`: execute CPU/GPU code and benchmarks where available.
- `suggestions`: parse, review, and merge external model suggestions.
- `tracing`: write model-assisted conversion audit trails.

### External Model Layer

The External Model Layer is outside `simforge`. It may be Codex, Claude Code,
Cursor, Copilot, or another AI coding agent. It can inspect source code and
write `model_suggestion.json`, but it does not own the trust decision.

The first repo-native harness does not call OpenAI or any other model API. The
local harness reviews model advice, enforces backend policy, performs
deterministic transformations, and writes reports.

## Pipeline

```text
Input Python file
  -> Source intake
  -> Python parser / AST analyzer
  -> Simulation pattern detector
  -> GPU suitability analyzer
  -> Intermediate analysis representation
  -> Optional model suggestion review
  -> Conversion planner
  -> Unsupported feature detector
  -> Backend selection
  -> CuPy code generator
  -> Syntax check
  -> CPU/GPU runner
  -> Validation
  -> Benchmark
  -> Reports
  -> Optional agent trace
```

The deterministic path omits model suggestion review and keeps working without
API keys, network access, CuPy, or CUDA.

The model-assisted path adds:

```text
model_suggestion.json
  -> suggestion schema validation
  -> backend policy enforcement
  -> suggestion_review.json / suggestion_review.md
  -> accepted advisory fields merged into conversion_plan.json
  -> agent_trace.json
```

## Inputs, Outputs, And Quality Gates

| Step | Input | Output | Quality gate |
| --- | --- | --- | --- |
| Source intake | File path | Source text | File exists and language is Python |
| Static analysis | Source text | AST facts | Imports, loops, random calls, outputs recorded |
| Suitability | AST facts | Suitability result | Recommendation includes reasons and warnings |
| Planning | IR | Conversion plan | Plan includes backend status, risks, unsupported |
| Suggestion review | model_suggestion.json | suggestion_review.json / .md | Model advice is accepted, warned, or rejected |
| Backend selection | Plan target | Backend status | Target is implemented or explicitly rejected |
| Generation | Plan and source | Generated source | Syntax check passes or partial conversion is reported |
| Validation | CPU/GPU outputs | Validation report | PASS, FAIL, or SKIPPED with reason |
| Benchmark | Runners | Benchmark report | Measured or SKIPPED with reason |
| Explanation | All artifacts | Explanation report | Reports changed code, risks, and unsupported features |
| Trace | Model-assisted artifacts | agent_trace.json | Trace links suggestion review, gates, and artifacts |

Validation and benchmark also write machine-readable run artifacts when a
generated file belongs to a project workspace:

```text
projects/<name>/runs/validation.json
projects/<name>/runs/benchmark.json
```

Skipped run artifacts keep the same status and reason as the Markdown reports.
Successful validation artifacts include scalar outputs, absolute differences,
tolerance, and repeat count. Successful benchmark artifacts include repeat and
warmup settings plus raw CPU/GPU timing samples.

## Lightweight IR

```json
{
  "language": "python",
  "imports": ["numpy"],
  "patterns": ["monte_carlo_loop"],
  "random_calls": ["np.random.normal"],
  "outputs": ["estimate"],
  "convertible_regions": [
    {
      "type": "for_loop",
      "line_start": 6,
      "line_end": 9,
      "strategy": "batch_vectorization"
    }
  ],
  "unsupported_features": [],
  "gpu_suitability": {
    "gpu_suitable": true,
    "confidence": "medium",
    "recommended_backend": "cupy",
    "future_backend_candidates": ["torch"]
  }
}
```

## Conversion Plan Schema

```json
{
  "summary": "Convert NumPy simulation code to CuPy where safe.",
  "source_file": "input.py",
  "target_backend": "cupy",
  "detected_patterns": [],
  "gpu_suitability": {},
  "backend_status": {},
  "changes": [],
  "validation_strategy": "stochastic",
  "equivalence_level": "statistical",
  "risks": [],
  "unsupported_features": [],
  "source_intent": {
    "text": "Model-provided source intent when accepted.",
    "source": "model"
  },
  "model_advisory": {
    "suggestion_file": "reports/model_suggestion.json",
    "review_status": "ACCEPTED_WITH_WARNINGS",
    "accepted_fields": ["source_intent", "risks", "recommended_validation"]
  },
  "validation_notes": [
    {
      "type": "stochastic",
      "message": "Model-provided validation note.",
      "source": "model"
    }
  ]
}
```

Model-aware fields are additive. They cannot replace backend status, remove
detected unsupported features, mark validation or benchmark as passed, or add
speedup claims.

## Unsupported Report Schema

```json
{
  "unsupported_features": [
    {
      "code": "--target torch",
      "reason": "TorchBackend is planned but not implemented in the MVP.",
      "action": "Use --target cupy for the current supported backend."
    }
  ]
}
```
