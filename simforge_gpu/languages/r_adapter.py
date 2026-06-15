"""Lightweight R project inspection for the v2 harness."""

from __future__ import annotations

from pathlib import Path

from simforge_gpu.harness.schemas import ProjectInspection


class RAdapter:
    language = "r"

    def inspect(self, project_path: str | Path) -> ProjectInspection:
        root = Path(project_path)
        files = _find_files(root)
        texts = {file: file.read_text(encoding="utf-8") for file in files}
        relative_files = tuple(_relative(root, file) for file in files)
        entrypoints = tuple(
            _relative(root, file)
            for file in files
            if file.name.lower() == "main.r"
            or "scripts" in file.parts
            or "R" in file.parts
            or file.name.startswith("input_")
        )
        dependencies = _unique(
            dependency
            for text in texts.values()
            for dependency, markers in _R_DEPENDENCIES.items()
            if any(marker in text for marker in markers)
        )
        randomness = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _R_RANDOMNESS.items()
            if any(marker in text for marker in markers)
        )
        simulation = _simulation_indicators(texts, randomness)
        io_indicators = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _R_IO.items()
            if any(marker in text for marker in markers)
        )
        plotting = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _R_PLOTTING.items()
            if any(marker in text for marker in markers)
        )
        risky = _unique(
            indicator
            for text in texts.values()
            for indicator, markers in _R_RISKY.items()
            if any(marker in text for marker in markers)
        )
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
            notes=("Lightweight textual inspection only; R code was not executed.",),
        )


_R_DEPENDENCIES = {
    "torch": ("library(torch)", "require(torch)"),
    "data.table": ("library(data.table)", "require(data.table)"),
    "dplyr": ("library(dplyr)", "require(dplyr)"),
    "ggplot2": ("library(ggplot2)", "require(ggplot2)", "ggplot("),
    "parallel": ("library(parallel)", "mclapply", "parLapply"),
}

_R_RANDOMNESS = {
    "set.seed": ("set.seed",),
    "rnorm": ("rnorm(",),
    "runif": ("runif(",),
    "sample": ("sample(",),
    "replicate": ("replicate(",),
}

_R_IO = {
    "file_io": ("read.csv", "write.csv", "readRDS", "saveRDS", "file("),
}

_R_PLOTTING = {
    "ggplot2": ("ggplot(", "library(ggplot2)"),
    "plotting": ("plot(", "hist("),
}

_R_RISKY = {
    "plotting": ("ggplot(", "plot(", "hist("),
    "file_io": ("read.csv", "write.csv", "readRDS", "saveRDS"),
    "parallel": ("parallel", "mclapply", "parLapply"),
    "external_process": ("system(", "system2("),
    "dynamic_execution": ("eval(", "source("),
}


def _find_files(root: Path) -> tuple[Path, ...]:
    if root.is_file() and root.suffix.lower() in {".r", ".rmd"}:
        return (root,)
    if not root.exists():
        raise FileNotFoundError(f"Project path not found: {root}")
    paths = {
        path
        for pattern in ("*.R", "*.r", "*.Rmd")
        for path in root.rglob(pattern)
        if path.is_file()
    }
    return tuple(sorted(paths))


def _simulation_indicators(texts: dict[Path, str], randomness: tuple[str, ...]) -> tuple[str, ...]:
    combined = "\n".join(texts.values()).lower()
    indicators: list[str] = []
    if randomness:
        indicators.append("monte_carlo")
    if "bootstrap" in combined:
        indicators.append("bootstrap")
    if "permutation" in combined:
        indicators.append("permutation")
    if "random walk" in combined or "random_walk" in combined:
        indicators.append("random_walk")
    if "replicate(" in combined:
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
