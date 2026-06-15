# Agent Protocol

The v2 agent protocol lets SimForge GPU generate task artifacts for external
coding agents without treating those agents as trusted validators.

## Trust Boundary

```text
Harness owns trust. Models propose and edit. Tests decide. Reports explain.
```

External agents may modify code and write `agent_result.json`. Their validation
or benchmark claims are recorded as claims only. The harness must run validation
and benchmarking before reporting correctness or speedup.

## Task Artifacts

`simforge task <project> --agent codex --target py-torch` writes:

```text
agent/migration_request.json
agent/migration_task.md
agent/constraints.md
agent/expected_artifacts.json
```

The CLI does not launch Codex, Cursor, Claude Code, OpenAI, Anthropic, or any
other model API in this pass.

## Migration Request Schema

`migration_request.json` includes:

- `schema_version`
- `project_path`
- `language`
- `target`
- `agent`
- `source_summary`
- `entrypoints`
- `performance_goals`
- `correctness_requirements`
- `validation_strategy`
- `benchmark_strategy`
- `constraints`
- `required_outputs`

## Agent Result Schema

External agents should write:

```json
{
  "schema_version": "2.0",
  "agent": "codex",
  "status": "COMPLETED",
  "summary": "Short summary of the migration attempt.",
  "files_modified": ["path/to/file.py"],
  "files_created": [],
  "claimed_changes": ["Moved repeated sampling to torch tensors."],
  "known_limitations": ["Validation has not been run by the harness."],
  "validation_claims": [],
  "benchmark_claims": []
}
```

`validation_claims` and `benchmark_claims` are never harness results.

## Collect Behavior

`simforge collect <project>` looks for:

```text
agent/agent_result.json
agent/agent_patch.diff
```

If `agent_result.json` is absent, collect writes a pending result with
`PENDING_AGENT_OUTPUT`. If an agent result is present, collect writes
`reports/agent_trace.json` with `PENDING_HARNESS_VALIDATION` until real harness
validation and benchmark commands run.

## Required Agent Constraints

External-agent task files include these constraints:

- Do not fabricate validation results.
- Do not fabricate benchmark results.
- Do not delete unsupported logic silently.
- Record modified and created files in `agent_result.json`.
- Preserve simulation semantics and statistical meaning.
- Keep CPU-only I/O and plotting outside GPU regions.

