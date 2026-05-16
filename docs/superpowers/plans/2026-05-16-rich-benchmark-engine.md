# Rich Benchmark Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional in-process benchmark method with transfer-overhead reporting and workload-size sweeps while preserving no-GPU-safe defaults.

**Architecture:** Keep the existing subprocess benchmark as the default path in `pipeline.py`. Add a focused in-process runner under `simforge_gpu/runners/` that loads `run` callables from CPU and generated GPU scripts, synchronizes CuPy timings, and returns the same report dictionary shape plus richer sweep and transfer metadata.

**Tech Stack:** Python standard library (`argparse`, `importlib`, `statistics`, `time`), optional CuPy imported only after a GPU availability check, pytest unit tests, GPU-marked integration tests.

---

## File Structure

- Modify `simforge_gpu/cli.py`: add benchmark CLI options and pass them into pipeline functions.
- Modify `simforge_gpu/pipeline.py`: parse benchmark sizes, validate method combinations, route to subprocess or in-process benchmark, preserve artifact paths.
- Create `simforge_gpu/runners/in_process_benchmark.py`: load `run` callables, execute repeated timings, synchronize CuPy, classify transfer overhead.
- Modify `simforge_gpu/reporters/markdown.py`: render size sweep rows and transfer metadata.
- Modify `tests/unit/test_cli.py`: cover CLI parsing, skipped no-GPU artifacts, and invalid benchmark option combinations.
- Modify `tests/unit/test_report_generator.py`: cover size sweep and transfer report rendering.
- Create `tests/unit/test_in_process_benchmark.py`: cover size parsing and missing-callable failure helpers without requiring CuPy.
- Modify `tests/integration/test_gpu_execution.py`: cover real in-process GPU benchmark when CuPy/CUDA execution is available.
- Modify `README.md`, `docs/DEMO.md`, `docs/ROADMAP.md`, and `docs/TESTING_STRATEGY.md`: document the new local benchmark method and its trust limits.

---

### Task 1: CLI Option Parsing And Pipeline Parameters

**Files:**
- Modify: `simforge_gpu/cli.py`
- Modify: `simforge_gpu/pipeline.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write failing CLI tests**

Add these tests to `tests/unit/test_cli.py`:

```python
def test_benchmark_accepts_in_process_options_in_no_gpu_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMFORGE_DISABLE_GPU", "1")
    original = tmp_path / "original.py"
    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated = generated_dir / "generated_gpu.py"
    original.write_text("def run(n=10):\n    return n\nprint(run())\n", encoding="utf-8")
    generated.write_text("def run(n=10):\n    return n\nprint(run())\n", encoding="utf-8")

    exit_code = main(
        [
            "benchmark",
            str(original),
            str(generated),
            "--method",
            "in-process",
            "--sizes",
            "10,100",
            "--repeat",
            "3",
            "--warmup",
            "1",
            "--include-transfer",
        ]
    )

    payload = json.loads((tmp_path / "runs" / "benchmark.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["status"] == "SKIPPED"
    assert payload["measurement_method"] == "not_run"
    assert payload["requested_method"] == "in-process"
    assert payload["sizes"] == [10, 100]
    assert payload["include_transfer"] is True


def test_benchmark_rejects_sizes_for_subprocess_method(tmp_path, capsys):
    original = tmp_path / "original.py"
    generated = tmp_path / "generated_gpu.py"
    original.write_text("print(1)\n", encoding="utf-8")
    generated.write_text("print(1)\n", encoding="utf-8")

    exit_code = main(
        [
            "benchmark",
            str(original),
            str(generated),
            "--method",
            "subprocess",
            "--sizes",
            "10,100",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--sizes requires --method in-process" in captured.err
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_cli.py::test_benchmark_accepts_in_process_options_in_no_gpu_mode tests/unit/test_cli.py::test_benchmark_rejects_sizes_for_subprocess_method -q
```

Expected: FAIL because the CLI does not recognize `--method`, `--sizes`, or `--include-transfer`.

- [ ] **Step 3: Implement CLI arguments**

In `simforge_gpu/cli.py`, add benchmark options:

```python
benchmark.add_argument("--method", choices=("subprocess", "in-process"), default="subprocess")
benchmark.add_argument("--sizes")
benchmark.add_argument("--include-transfer", action="store_true")
```

Add convert benchmark options:

```python
convert.add_argument("--benchmark-method", choices=("subprocess", "in-process"), default="subprocess")
convert.add_argument("--benchmark-sizes")
convert.add_argument("--include-transfer", action="store_true")
```

Pass the values into `convert_file` and `benchmark_files` using keyword arguments:

```python
benchmark_method=args.benchmark_method,
benchmark_sizes=args.benchmark_sizes,
benchmark_include_transfer=args.include_transfer,
```

and:

```python
method=args.method,
sizes=args.sizes,
include_transfer=args.include_transfer,
```

- [ ] **Step 4: Implement pipeline parameter acceptance and validation**

In `simforge_gpu/pipeline.py`, extend `convert_file`, `benchmark_files`, `_execute_benchmark`, and `_benchmark_skip` with:

```python
benchmark_method: str = "subprocess",
benchmark_sizes: str | None = None,
benchmark_include_transfer: bool = False,
```

and:

```python
method: str = "subprocess",
sizes: str | None = None,
include_transfer: bool = False,
```

Add `_parse_benchmark_sizes`:

```python
def _parse_benchmark_sizes(raw: str | None) -> list[int] | None:
    if raw is None or raw.strip() == "":
        return None
    sizes: list[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            raise ValueError("Benchmark sizes must be comma-separated positive integers.")
        value = int(text)
        if value <= 0:
            raise ValueError("Benchmark sizes must be positive integers.")
        sizes.append(value)
    return sizes
```

Validate before execution:

```python
parsed_sizes = _parse_benchmark_sizes(sizes)
if method == "subprocess" and parsed_sizes is not None:
    raise ValueError("--sizes requires --method in-process.")
```

- [ ] **Step 5: Run focused CLI tests**

Run:

```bash
python -m pytest tests/unit/test_cli.py::test_benchmark_accepts_in_process_options_in_no_gpu_mode tests/unit/test_cli.py::test_benchmark_rejects_sizes_for_subprocess_method -q
```

Expected: PASS.

---

### Task 2: In-Process Benchmark Runner

**Files:**
- Create: `simforge_gpu/runners/in_process_benchmark.py`
- Modify: `simforge_gpu/pipeline.py`
- Test: `tests/unit/test_in_process_benchmark.py`

- [ ] **Step 1: Write runner unit tests**

Create `tests/unit/test_in_process_benchmark.py` with:

```python
import pathlib

from simforge_gpu.runners.in_process_benchmark import (
    load_run_callable,
    run_in_process_benchmark,
)


def test_load_run_callable_reports_missing_run(tmp_path):
    script = tmp_path / "script.py"
    script.write_text("value = 1\n", encoding="utf-8")

    result = load_run_callable(script, module_name="missing_run_test")

    assert result.status == "FAILED"
    assert "does not define a callable run" in result.reason


def test_in_process_benchmark_skips_when_gpu_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMFORGE_DISABLE_GPU", "1")
    original = tmp_path / "original.py"
    generated = tmp_path / "generated_gpu.py"
    original.write_text("def run(n=10):\n    return n\n", encoding="utf-8")
    generated.write_text("def run(n=10):\n    return n\n", encoding="utf-8")

    result = run_in_process_benchmark(
        original,
        generated,
        repeat=2,
        warmup=1,
        sizes=[10, 100],
        include_transfer=True,
    )

    assert result["status"] == "SKIPPED"
    assert result["requested_method"] == "in-process"
    assert result["sizes"] == [10, 100]
    assert result["include_transfer"] is True
    assert result["measurement_method"] == "not_run"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/unit/test_in_process_benchmark.py -q
```

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement the runner module**

Create `simforge_gpu/runners/in_process_benchmark.py` with these public functions and result shape:

```python
from __future__ import annotations

import importlib.util
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from simforge_gpu.runners.python_subprocess import (
    activate_cuda_dll_directories,
    cupy_kernel_probe,
    prepare_cuda_subprocess_environment,
)


@dataclass(frozen=True)
class CallableLoadResult:
    status: str
    run: Callable[..., Any] | None
    reason: str


def load_run_callable(path: Path, module_name: str) -> CallableLoadResult:
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            return CallableLoadResult("FAILED", None, f"Could not load module spec for {path}.")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        return CallableLoadResult("FAILED", None, f"Failed to load {path}: {exc}")
    run = getattr(module, "run", None)
    if not callable(run):
        return CallableLoadResult("FAILED", None, f"{path} does not define a callable run function.")
    return CallableLoadResult("PASSED", run, "")
```

Implement `run_in_process_benchmark` so it returns `SKIPPED` before importing CuPy when `SIMFORGE_DISABLE_GPU=1`; otherwise it probes CuPy, loads both callables, times repeats, and returns `PASSED` with `size_sweep`.

- [ ] **Step 4: Route pipeline to in-process runner**

Import and call the new runner from `_execute_benchmark` when `method == "in-process"`:

```python
from simforge_gpu.runners.in_process_benchmark import run_in_process_benchmark
```

```python
if method == "in-process":
    return run_in_process_benchmark(
        original,
        generated,
        repeat=repeat,
        warmup=warmup,
        sizes=parsed_sizes,
        include_transfer=include_transfer,
    )
```

- [ ] **Step 5: Run runner tests**

Run:

```bash
python -m pytest tests/unit/test_in_process_benchmark.py -q
```

Expected: PASS.

---

### Task 3: Size Sweep And Transfer Result Shape

**Files:**
- Modify: `simforge_gpu/runners/in_process_benchmark.py`
- Modify: `tests/unit/test_in_process_benchmark.py`

- [ ] **Step 1: Add a pure CPU helper test for summary shape**

Add a test that monkeypatches the runner's GPU probe and CuPy import helpers so no real CuPy is needed:

```python
def test_in_process_result_contains_size_sweep_without_transfer(tmp_path, monkeypatch):
    original = tmp_path / "original.py"
    generated = tmp_path / "generated_gpu.py"
    original.write_text("def run(n=10):\n    return n + 1\n", encoding="utf-8")
    generated.write_text("def run(n=10):\n    return n + 1\n", encoding="utf-8")
    monkeypatch.setattr("simforge_gpu.runners.in_process_benchmark.cupy_kernel_probe", lambda: (True, "mock"))
    monkeypatch.setattr("simforge_gpu.runners.in_process_benchmark._import_cupy", lambda: None)

    result = run_in_process_benchmark(
        original,
        generated,
        repeat=2,
        warmup=1,
        sizes=[10, 20],
        include_transfer=False,
    )

    assert result["status"] == "PASSED"
    assert result["measurement_method"] == "in_process_callable"
    assert result["trust_level"] == "comparative_local"
    assert result["transfer_overhead_status"] == "not_requested"
    assert [row["size"] for row in result["size_sweep"]] == [10, 20]
    assert len(result["size_sweep"][0]["cpu_times_seconds"]) == 2
    assert "speedup" in result["size_sweep"][0]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run:

```bash
python -m pytest tests/unit/test_in_process_benchmark.py::test_in_process_result_contains_size_sweep_without_transfer -q
```

Expected: FAIL until the runner returns the requested shape.

- [ ] **Step 3: Complete runner timing implementation**

Implement helpers in `simforge_gpu/runners/in_process_benchmark.py`:

```python
def _call_run(run: Callable[..., Any], size: int | None) -> Any:
    if size is None:
        return run()
    return run(size)


def _time_call(run: Callable[..., Any], size: int | None, cp: Any | None) -> tuple[float, Any]:
    if cp is not None:
        cp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    value = _call_run(run, size)
    if cp is not None:
        cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - start, value


def _summary(times: list[float], prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_times_seconds": times,
        f"{prefix}_median_runtime": f"{statistics.median(times):.6f}s",
        f"{prefix}_min_runtime": f"{min(times):.6f}s",
        f"{prefix}_mean_runtime": f"{statistics.fmean(times):.6f}s",
    }
```

For each size row, include `cpu_times_seconds`, `gpu_times_seconds`, median/min/mean runtime fields, and `speedup` when the GPU median is positive. Mirror the first sweep row's runtime fields at the top level for compatibility with existing reports.

- [ ] **Step 4: Run all runner unit tests**

Run:

```bash
python -m pytest tests/unit/test_in_process_benchmark.py -q
```

Expected: PASS.

---

### Task 4: Markdown Report Rendering

**Files:**
- Modify: `simforge_gpu/reporters/markdown.py`
- Test: `tests/unit/test_report_generator.py`

- [ ] **Step 1: Write report rendering tests**

Add to `tests/unit/test_report_generator.py`:

```python
def test_benchmark_report_renders_size_sweep_and_transfer_status():
    report = render_benchmark_report(
        {
            "status": "PASSED",
            "repeat": 2,
            "warmup": 1,
            "measurement_method": "in_process_callable",
            "trust_level": "comparative_local",
            "includes_transfer_overhead": True,
            "transfer_overhead_status": "embedded_in_result",
            "cpu_median_runtime": "0.010000s",
            "gpu_median_runtime": "0.005000s",
            "speedup": "2.000x",
            "size_sweep": [
                {
                    "size": 10,
                    "cpu_median_runtime": "0.010000s",
                    "gpu_median_runtime": "0.005000s",
                    "speedup": "2.000x",
                    "transfer_overhead_status": "embedded_in_result",
                }
            ],
            "limitations": ["Local in-process timings are machine-specific."],
        }
    )

    assert "Measurement method: in_process_callable" in report
    assert "Transfer overhead status: embedded_in_result" in report
    assert "Size sweep:" in report
    assert "| 10 | 0.010000s | 0.005000s | 2.000x | embedded_in_result |" in report
```

- [ ] **Step 2: Run the report test and verify it fails**

Run:

```bash
python -m pytest tests/unit/test_report_generator.py::test_benchmark_report_renders_size_sweep_and_transfer_status -q
```

Expected: FAIL until Markdown rendering supports the new fields.

- [ ] **Step 3: Implement Markdown rendering**

In `render_benchmark_report`, after trust indicators, render:

```python
if "transfer_overhead_status" in result:
    lines.append(f"Transfer overhead status: {result['transfer_overhead_status']}")
```

After scalar runtime fields, render `size_sweep`:

```python
size_sweep = result.get("size_sweep", [])
if isinstance(size_sweep, list) and size_sweep:
    lines.extend([
        "",
        "Size sweep:",
        "",
        "| Size | CPU median | GPU median | Speedup | Transfer overhead |",
        "| --- | --- | --- | --- | --- |",
    ])
    for row in size_sweep:
        if isinstance(row, dict):
            lines.append(
                f"| {row.get('size', 'default')} | "
                f"{row.get('cpu_median_runtime', 'unknown')} | "
                f"{row.get('gpu_median_runtime', 'unknown')} | "
                f"{row.get('speedup', 'n/a')} | "
                f"{row.get('transfer_overhead_status', result.get('transfer_overhead_status', 'unknown'))} |"
            )
```

- [ ] **Step 4: Run report tests**

Run:

```bash
python -m pytest tests/unit/test_report_generator.py -q
```

Expected: PASS.

---

### Task 5: GPU Integration Coverage

**Files:**
- Modify: `tests/integration/test_gpu_execution.py`

- [ ] **Step 1: Add GPU-marked integration test**

Add this test to `tests/integration/test_gpu_execution.py`:

```python
def test_gpu_in_process_benchmark_size_sweep_when_available(tmp_path):
    if not _cupy_kernel_available():
        pytest.skip("CuPy kernel execution is not available in this environment.")
    output_dir = tmp_path / "monte_carlo_pi"
    assert (
        main(
            [
                "convert",
                "examples/monte_carlo_pi/input_cpu.py",
                "--target",
                "cupy",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )

    exit_code = main(
        [
            "benchmark",
            "examples/monte_carlo_pi/input_cpu.py",
            str(output_dir / "generated" / "input_cpu_gpu.py"),
            "--method",
            "in-process",
            "--sizes",
            "1000,2000",
            "--repeat",
            "2",
            "--warmup",
            "1",
            "--include-transfer",
        ]
    )

    payload = json.loads((output_dir / "runs" / "benchmark.json").read_text(encoding="utf-8"))
    report = (output_dir / "reports" / "benchmark_report.md").read_text(encoding="utf-8")
    assert exit_code == 0
    assert payload["status"] == "PASSED"
    assert payload["measurement_method"] == "in_process_callable"
    assert payload["trust_level"] == "comparative_local"
    assert [row["size"] for row in payload["size_sweep"]] == [1000, 2000]
    assert payload["transfer_overhead_status"] in {"embedded_in_result", "measured", "not_requested"}
    assert "Size sweep:" in report
```

- [ ] **Step 2: Run GPU test when possible**

Run:

```bash
python -m pytest tests/integration/test_gpu_execution.py::test_gpu_in_process_benchmark_size_sweep_when_available -q
```

Expected: PASS when CuPy/CUDA kernel execution is available, SKIPPED otherwise.

---

### Task 6: Documentation And Full Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/DEMO.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/TESTING_STRATEGY.md`

- [ ] **Step 1: Update docs**

Add concise documentation that:

- existing benchmark defaults to subprocess demo timing;
- `--method in-process` requires `run` callables;
- `--sizes` performs local workload-size sweep;
- `--include-transfer` only reports transfer overhead when it can be isolated;
- no-GPU paths still skip safely.

- [ ] **Step 2: Run no-GPU verification**

Run:

```bash
python -m pytest -m "not gpu" -q
```

Expected: PASS.

- [ ] **Step 3: Run GPU verification if available**

Run:

```bash
python -m pytest -m gpu -q
```

Expected: PASS or SKIPPED when CuPy/CUDA kernel execution is unavailable.

- [ ] **Step 4: Inspect git changes**

Run:

```bash
git status --short
git diff --stat
```

Expected: only planned benchmark, test, and documentation files are changed.
