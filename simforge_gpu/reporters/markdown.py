"""Markdown report renderers for MVP artifacts."""

from __future__ import annotations

from simforge_gpu.ir.schema import UnsupportedFeature
from simforge_gpu.suggestions.review import SuggestionReview


def render_unsupported_report(features: list[UnsupportedFeature] | tuple[UnsupportedFeature, ...]) -> str:
    lines = ["# Unsupported Report", ""]
    if not features:
        lines.extend(["No unsupported features detected.", ""])
        return "\n".join(lines)
    for index, feature in enumerate(features, start=1):
        lines.extend(
            [
                f"## {index}. `{feature.code}`",
                "",
                f"Category: {feature.category}",
                "",
                f"Source: {feature.source}",
                "",
                f"Reason: {feature.reason}",
                "",
                f"Action: {feature.action}",
                "",
            ]
        )
    return "\n".join(lines)


def render_suggestion_review_report(review: SuggestionReview) -> str:
    lines = [
        "# Suggestion Review",
        "",
        f"Review status: {review.status}",
        f"Source file: {review.source_file}",
        f"Suggestion file: {review.suggestion_file}",
        "",
        "## Accepted Fields",
        "",
    ]
    if review.accepted_fields:
        lines.extend(
            f"- {entry.field}: {entry.reason}" for entry in review.accepted_fields
        )
    else:
        lines.append("- No accepted fields.")
    lines.extend(["", "## Warnings", ""])
    if review.warnings:
        lines.extend(f"- {warning.field}: {warning.message}" for warning in review.warnings)
    else:
        lines.append("- No warnings.")
    lines.extend(["", "## Rejected Fields", ""])
    if review.rejected_fields:
        lines.extend(
            f"- {entry.field}: {entry.reason}" for entry in review.rejected_fields
        )
    else:
        lines.append("- No rejected fields.")
    lines.extend(["", "## Unsupported", ""])
    if review.unsupported_features:
        for feature in review.unsupported_features:
            lines.append(
                f"- {feature.get('code', 'unknown')}: {feature.get('reason', '')}"
            )
    else:
        lines.append("- No unsupported model suggestions.")
    lines.extend(
        [
            "",
            "Note: Model advisory input is not treated as a trusted transformation.",
            "",
        ]
    )
    return "\n".join(lines)


def render_benchmark_report(result: dict[str, object]) -> str:
    status = result.get("status", "UNKNOWN")
    lines = ["# Benchmark Report", "", f"Benchmark status: {status}", ""]
    if "repeat" in result:
        lines.append(f"Repeat count: {result['repeat']}")
    if "warmup" in result:
        lines.append(f"Warmup runs: {result['warmup']}")
    if "repeat" in result or "warmup" in result:
        lines.append("")
    if "measurement_method" in result:
        lines.append(f"Measurement method: {result['measurement_method']}")
    if "trust_level" in result:
        lines.append(f"Trust level: {result['trust_level']}")
    if "includes_transfer_overhead" in result:
        lines.append(f"Includes transfer overhead: {result['includes_transfer_overhead']}")
    if "measurement_method" in result or "trust_level" in result or "includes_transfer_overhead" in result:
        lines.append("")
    if status == "SKIPPED":
        lines.append(f"Reason: {result.get('reason', 'No reason provided.')}")
        _append_limitations(lines, result)
        lines.append("")
        return "\n".join(lines)
    if "cpu_runtime" in result:
        lines.append(f"CPU runtime: {result['cpu_runtime']}")
    if "gpu_runtime" in result:
        lines.append(f"GPU runtime: {result['gpu_runtime']}")
    if "cpu_median_runtime" in result:
        lines.append(f"CPU median runtime: {result['cpu_median_runtime']}")
    if "gpu_median_runtime" in result:
        lines.append(f"GPU median runtime: {result['gpu_median_runtime']}")
    if "cpu_min_runtime" in result:
        lines.append(f"CPU min runtime: {result['cpu_min_runtime']}")
    if "gpu_min_runtime" in result:
        lines.append(f"GPU min runtime: {result['gpu_min_runtime']}")
    if "cpu_mean_runtime" in result:
        lines.append(f"CPU mean runtime: {result['cpu_mean_runtime']}")
    if "gpu_mean_runtime" in result:
        lines.append(f"GPU mean runtime: {result['gpu_mean_runtime']}")
    if "speedup" in result:
        lines.append(f"Speedup: {result['speedup']}")
    if "reason" in result:
        lines.append(f"Reason: {result['reason']}")
    _append_limitations(lines, result)
    lines.append("")
    return "\n".join(lines)


def _append_limitations(lines: list[str], result: dict[str, object]) -> None:
    limitations = result.get("limitations", [])
    if not limitations:
        return
    lines.extend(["", "Limitations:"])
    if isinstance(limitations, list):
        lines.extend(f"- {limitation}" for limitation in limitations)


def render_validation_report(result: dict[str, object]) -> str:
    status = result.get("status", "UNKNOWN")
    validation_type = result.get("validation_type", "unknown")
    equivalence_level = result.get("equivalence_level", "unknown")
    lines = [
        "# Validation Report",
        "",
        f"Validation status: {status}",
        f"Validation type: {validation_type}",
        f"Equivalence level: {equivalence_level}",
        "",
    ]
    if status == "SKIPPED":
        if "repeat" in result:
            lines.append(f"Repeat count: {result['repeat']}")
        if "tolerance" in result:
            lines.append(f"Tolerance: {result['tolerance']}")
        lines.append(f"Reason: {result.get('reason', 'No reason provided.')}")
        lines.append("")
    else:
        if "repeat" in result:
            lines.append(f"Repeat count: {result['repeat']}")
        if "output_kind" in result:
            lines.append(f"Output kind: {result['output_kind']}")
        if "element_count" in result:
            lines.append(f"Element count: {result['element_count']}")
        if "cpu_output" in result:
            lines.append(f"CPU output: {result['cpu_output']}")
        if "gpu_output" in result:
            lines.append(f"GPU output: {result['gpu_output']}")
        if "absolute_difference" in result:
            lines.append(f"Absolute difference: {result['absolute_difference']}")
        if "max_absolute_difference" in result:
            lines.append(f"Max absolute difference: {result['max_absolute_difference']}")
        if "mean_absolute_difference" in result:
            lines.append(f"Mean absolute difference: {result['mean_absolute_difference']}")
        if "tolerance" in result:
            lines.append(f"Tolerance: {result['tolerance']}")
        if "reason" in result:
            lines.append(f"Reason: {result['reason']}")
        lines.append("")
    return "\n".join(lines)


def render_syntax_report(result: dict[str, object]) -> str:
    status = result.get("status", "UNKNOWN")
    lines = ["# Syntax Report", "", f"Syntax status: {status}", ""]
    if "source" in result:
        lines.append(f"Source: {result['source']}")
    if "reason" in result:
        lines.append(f"Reason: {result['reason']}")
    if "message" in result:
        lines.append(str(result["message"]))
    lines.append("")
    return "\n".join(lines)


def render_quality_report(result: dict[str, object]) -> str:
    lines = [
        "# Quality Report",
        "",
        f"Quality gate: {result.get('quality_gate', 'UNKNOWN')}",
        "",
        f"Syntax status: {result.get('syntax_status', 'UNKNOWN')}",
        f"Unsupported features: {result.get('unsupported_count', 'unknown')}",
        f"Validation status: {result.get('validation_status', 'UNKNOWN')}",
        f"Benchmark status: {result.get('benchmark_status', 'UNKNOWN')}",
        "",
    ]
    if "suggestion_review_status" in result:
        lines.extend([f"Suggestion review: {result['suggestion_review_status']}", ""])
    notes = result.get("notes", [])
    if notes:
        lines.append("Notes:")
        lines.extend(f"- {note}" for note in notes)
        lines.append("")
    return "\n".join(lines)


def render_explanation_report(summary: dict[str, object]) -> str:
    lines = ["# Explanation Report", "", str(summary.get("summary", "")), ""]
    for key in ("validation_status", "benchmark_status"):
        if key in summary:
            lines.append(f"{key}: {summary[key]}")
    artifacts = summary.get("artifacts", [])
    if artifacts:
        lines.extend(["", "Artifacts:"])
        lines.extend(f"- {artifact}" for artifact in artifacts)
    lines.append("")
    return "\n".join(lines)
