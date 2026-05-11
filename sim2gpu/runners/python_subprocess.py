"""Subprocess runners for optional CPU/GPU execution."""

from __future__ import annotations

import os
import re
import json
import site
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScriptRun:
    status: str
    stdout: str
    stderr: str
    returncode: int
    runtime_seconds: float
    numeric_output: float | None
    parsed_output: Any


def cupy_kernel_probe() -> tuple[bool, str]:
    code = """
from sim2gpu.runners.python_subprocess import activate_cuda_dll_directories, prepare_cuda_subprocess_environment
import os
activate_cuda_dll_directories()
cwd, env = prepare_cuda_subprocess_environment()
if cwd:
    os.chdir(cwd)
import cupy as cp
value = float(cp.sum(cp.arange(5)).get())
count = cp.cuda.runtime.getDeviceCount()
print(f"devices={count}; value={value}")
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=30,
        env=prepare_cuda_subprocess_environment()[1],
    )
    if completed.returncode == 0:
        return True, completed.stdout.strip()
    return False, (completed.stderr or completed.stdout).strip()


def run_python_script(path: Path, use_cuda_workdir: bool = False) -> ScriptRun:
    script = path.resolve()
    cwd = str(script.parent)
    env = os.environ.copy()
    if use_cuda_workdir:
        cuda_cwd, env = prepare_cuda_subprocess_environment()
        if cuda_cwd is not None:
            cwd = str(cuda_cwd)
    start = time.perf_counter()
    command = [sys.executable, str(script)]
    if use_cuda_workdir:
        command = [
            sys.executable,
            "-c",
            (
                "from sim2gpu.runners.python_subprocess import "
                "activate_cuda_dll_directories; "
                "activate_cuda_dll_directories(); "
                f"exec(compile(open({str(script)!r}, encoding='utf-8').read(), {str(script)!r}, 'exec'))"
            ),
        ]
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    runtime = time.perf_counter() - start
    parsed_output = parse_script_output(completed.stdout)
    return ScriptRun(
        status="PASSED" if completed.returncode == 0 else "FAILED",
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
        runtime_seconds=runtime,
        numeric_output=_last_float(completed.stdout),
        parsed_output=parsed_output,
    )


def prepare_cuda_subprocess_environment() -> tuple[Path | None, dict[str, str]]:
    env = os.environ.copy()
    cuda_workdir: Path | None = None
    for root in site.getsitepackages():
        root_path = Path(root)
        for rel in (
            "nvidia/cuda_nvrtc/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/curand/bin",
        ):
            dll_dir = root_path / rel
            if dll_dir.exists():
                env["PATH"] = f"{dll_dir}{os.pathsep}{env.get('PATH', '')}"
                if rel == "nvidia/cuda_nvrtc/bin" and cuda_workdir is None:
                    cuda_workdir = dll_dir
    return cuda_workdir, env


def activate_cuda_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    for root in site.getsitepackages():
        root_path = Path(root)
        for rel in (
            "nvidia/cuda_nvrtc/bin",
            "nvidia/cuda_runtime/bin",
            "nvidia/curand/bin",
        ):
            dll_dir = root_path / rel
            if dll_dir.exists():
                os.add_dll_directory(str(dll_dir))


def _last_float(text: str) -> float | None:
    matches = re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    if not matches:
        return None
    return float(matches[-1])


def parse_script_output(text: str) -> Any:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        try:
            return json.loads(lines[-1])
        except json.JSONDecodeError:
            pass
    return _last_float(text)
