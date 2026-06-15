"""Lightweight Python project inspection for the v2 harness."""

from __future__ import annotations

from pathlib import Path

from simforge_gpu.harness.schemas import ProjectInspection


class PythonAdapter:
    language = "python"

    def inspect(self, project_path: str | Path) -> ProjectInspection:
        root = Path(project_path)
        files = _find_files(root)
        texts = {file: file.read_text(encoding="utf-8") for file in files}
        relative_files = tuple(_relative(root, file) for file in files)
        entrypoints = tuple(
            _relative(root, file)
            for file, text in texts.items()
            if file.name == "main.py"
            or file.name.startswith("input_")
            or 'if __name__ == "__main__"' in text
            or "if __name__ == '__main__'" in text
        )
        dependencies = _unique(
            dependency
            for text in texts.values()
            for dependency, markers in _PYTHON_DEPENDENCIES.items()
            if any(marker in text for marker in markers)
        )
        randomness = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _PYTHON_RANDOMNESS.items()
            if any(marker in text for marker in markers)
        )
        simulation = _simulation_indicators(texts, randomness)
        io_indicators = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _PYTHON_IO.items()
            if any(marker in text for marker in markers)
        )
        plotting = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _PYTHON_PLOTTING.items()
            if any(marker in text for marker in markers)
        )
        risky = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _PYTHON_RISKY.items()
            if any(marker in text for marker in markers)
        )
        notes = ("Lightweight textual inspection only; Python code was not executed.",)
        return ProjectInspection(
            project_path=root.as_posix(),
            language=self.language,
            detected_files=relative_files,
            entrypoints=entrypoints,
            dependencies=dependencies,
            simulation_indicators=simulation,
            randomness_indicators=randomness,
            io_indicators=io_indicators,
            plotting_indicators=plotting,
            unsupported_or_risky_features=risky,
            notes=notes,
        )


_PYTHON_DEPENDENCIES = {
    "numpy": ("import numpy", "from numpy"),
    "scipy": ("import scipy", "from scipy"),
    "pandas": ("import pandas", "from pandas"),
    "torch": ("import torch", "from torch"),
    "matplotlib": ("matplotlib", "pyplot"),
}

_PYTHON_RANDOMNESS = {
    "np.random": ("np.random", "numpy.random"),
    "random": ("import random", " random."),
}

_PYTHON_IO = {
    "file_io": ("open(", "read_csv", "to_csv", "np.load", "np.save"),
}

_PYTHON_PLOTTING = {
    "matplotlib": ("matplotlib", "plt."),
}

_PYTHON_RISKY = {
    "pandas_pipeline": ("import pandas", "from pandas"),
    "plotting": ("matplotlib", "plt."),
    "file_io": ("open(", "read_csv", "to_csv"),
    "multiprocessing": ("multiprocessing", "ProcessPoolExecutor"),
    "network_or_database": ("requests.", "socket.", "sqlite3", "sqlalchemy"),
    "dynamic_execution": ("eval(", "exec("),
}


def _find_files(root: Path) -> tuple[Path, ...]:
    if root.is_file() and root.suffix == ".py":
        return (root,)
    if not root.exists():
        raise FileNotFoundError(f"Project path not found: {root}")
    return tuple(sorted(path for path in root.rglob("*.py") if path.is_file()))


def _simulation_indicators(texts: dict[Path, str], randomness: tuple[str, ...]) -> tuple[str, ...]:
    indicators: list[str] = []
    combined = "\n".join(texts.values()).lower()
    if any(keyword in combined for keyword in ("monte", "estimate_pi", "pi", "simulation")):
        indicators.append("monte_carlo")
    if "bootstrap" in combined:
        indicators.append("bootstrap")
    if "permutation" in combined:
        indicators.append("permutation")
    if "random_walk" in combined or "random walk" in combined:
        indicators.append("random_walk")
    if randomness and ("for " in combined or "range(" in combined):
        indicators.append("repeated_random_trials")
    return _unique(indicators)


def _relative(root: Path, path: Path) -> str:
    base = root.parent if root.is_file() else root
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def _unique(values) -> tuple[str, ...]:
    seen: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)

