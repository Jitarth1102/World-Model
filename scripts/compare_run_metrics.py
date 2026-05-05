#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_RUNS = OrderedDict(
    [
        ("no_memory", Path("outputs/train_nomemory_real_movia_subset50_v4")),
        ("memory_baseline", Path("outputs/train_memory_real_movia_subset50_v1")),
        ("memory_strengthened", Path("outputs/train_memory_real_movia_subset50_v3")),
        ("memory_uncertainty_convgru", Path("outputs/train_memory_uncertainty_real_movia_subset50_v1")),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics across ConvGRU no-memory, memory, and uncertainty variants.")
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Override default runs with LABEL=PATH. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/model_comparison_current"),
        help="Directory where comparison markdown/json will be written.",
    )
    parser.add_argument(
        "--rollout-name",
        default="rollout_00000",
        help="Rollout subdirectory to summarize when present.",
    )
    return parser.parse_args()


def parse_run_mapping(overrides: list[str]) -> OrderedDict[str, Path]:
    if not overrides:
        return DEFAULT_RUNS.copy()
    mapping: OrderedDict[str, Path] = OrderedDict()
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Expected LABEL=PATH, got: {override}")
        label, raw_path = override.split("=", 1)
        mapping[label.strip()] = Path(raw_path).expanduser()
    return mapping


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_value(value: Any, precision: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def delta(model_value: Any, baseline_value: Any) -> float | None:
    model_float = to_float(model_value)
    baseline_float = to_float(baseline_value)
    if model_float is None or baseline_float is None:
        return None
    return model_float - baseline_float


def collect_run_summary(label: str, run_dir: Path, rollout_name: str) -> dict[str, Any]:
    metrics_path = run_dir / "metrics.json"
    rollout_metrics_path = run_dir / rollout_name / "metrics.json"
    summary: dict[str, Any] = {
        "label": label,
        "run_dir": str(run_dir),
        "status": "missing",
        "metrics_path": str(metrics_path),
        "rollout_metrics_path": str(rollout_metrics_path),
    }
    if not metrics_path.exists():
        return summary

    metrics = load_json(metrics_path)
    final_val = metrics.get("final_val", {})
    rollout = load_json(rollout_metrics_path) if rollout_metrics_path.exists() else {}

    summary.update(
        {
            "status": "ok",
            "metrics": metrics,
            "final_val": final_val,
            "rollout": rollout,
            "best_val_l1": to_float(metrics.get("best_val_l1")),
            "final_model_l1": to_float(get_nested(metrics, "final_val", "model_l1")),
            "final_model_dynamic_l1": to_float(get_nested(metrics, "final_val", "model_dynamic_l1")),
            "final_model_memory_covered_l1": to_float(get_nested(metrics, "final_val", "model_memory_covered_l1")),
            "final_model_psnr": to_float(get_nested(metrics, "final_val", "model_psnr")),
            "final_uncertainty_error_corr": to_float(get_nested(metrics, "final_val", "uncertainty_error_corr")),
            "final_write_coverage": to_float(get_nested(metrics, "final_val", "write_coverage")),
            "final_baseline_l1": to_float(get_nested(metrics, "final_val", "baseline_l1")),
            "final_baseline_dynamic_l1": to_float(get_nested(metrics, "final_val", "baseline_dynamic_l1")),
            "final_baseline_memory_covered_l1": to_float(get_nested(metrics, "final_val", "baseline_memory_covered_l1")),
            "final_memory_coverage": to_float(get_nested(metrics, "final_val", "memory_coverage")),
            "final_motion_fraction": to_float(get_nested(metrics, "final_val", "motion_fraction")),
            "final_model_depth_l1": to_float(get_nested(metrics, "final_val", "model_depth_l1")),
            "beats_baseline_l1": metrics.get("model_beats_baseline_on_val_l1"),
            "beats_baseline_dynamic_l1": metrics.get("model_beats_baseline_on_val_dynamic_l1"),
            "beats_baseline_memory_covered_l1": metrics.get("model_beats_baseline_on_val_memory_covered_l1"),
            "rollout_model_l1": to_float(rollout.get("model_l1")),
            "rollout_model_dynamic_l1": to_float(rollout.get("model_dynamic_l1")),
            "rollout_model_memory_covered_l1": to_float(rollout.get("model_memory_covered_l1")),
            "rollout_model_psnr": to_float(rollout.get("model_psnr")),
            "rollout_uncertainty_error_corr": to_float(rollout.get("uncertainty_error_corr")),
            "rollout_write_coverage": to_float(rollout.get("write_coverage")),
            "rollout_baseline_l1": to_float(rollout.get("baseline_l1")),
            "rollout_baseline_dynamic_l1": to_float(rollout.get("baseline_dynamic_l1")),
            "rollout_baseline_memory_covered_l1": to_float(rollout.get("baseline_memory_covered_l1")),
            "rollout_memory_coverage": to_float(rollout.get("memory_coverage")),
        }
    )
    return summary


def build_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    table.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(table)


def build_validation_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for summary in summaries:
        rows.append(
            [
                summary["label"],
                summary["status"],
                format_value(summary.get("best_val_l1")),
                format_value(summary.get("final_model_l1")),
                format_value(delta(summary.get("final_model_l1"), summary.get("final_baseline_l1"))),
                format_value(summary.get("final_model_dynamic_l1")),
                format_value(delta(summary.get("final_model_dynamic_l1"), summary.get("final_baseline_dynamic_l1"))),
                format_value(summary.get("final_model_memory_covered_l1")),
                format_value(delta(summary.get("final_model_memory_covered_l1"), summary.get("final_baseline_memory_covered_l1"))),
                format_value(summary.get("final_model_psnr")),
                format_value(summary.get("final_memory_coverage")),
                format_value(summary.get("final_uncertainty_error_corr")),
                format_value(summary.get("final_write_coverage")),
                format_value(summary.get("beats_baseline_dynamic_l1")),
                format_value(summary.get("beats_baseline_memory_covered_l1")),
            ]
        )
    return rows


def build_rollout_rows(summaries: list[dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for summary in summaries:
        rows.append(
            [
                summary["label"],
                summary["status"],
                format_value(summary.get("rollout_model_l1")),
                format_value(delta(summary.get("rollout_model_l1"), summary.get("rollout_baseline_l1"))),
                format_value(summary.get("rollout_model_dynamic_l1")),
                format_value(delta(summary.get("rollout_model_dynamic_l1"), summary.get("rollout_baseline_dynamic_l1"))),
                format_value(summary.get("rollout_model_memory_covered_l1")),
                format_value(delta(summary.get("rollout_model_memory_covered_l1"), summary.get("rollout_baseline_memory_covered_l1"))),
                format_value(summary.get("rollout_model_psnr")),
                format_value(summary.get("rollout_memory_coverage")),
                format_value(summary.get("rollout_uncertainty_error_corr")),
                format_value(summary.get("rollout_write_coverage")),
            ]
        )
    return rows


def build_reference_rows(summaries: list[dict[str, Any]], reference_label: str = "no_memory") -> list[list[str]]:
    reference = next((summary for summary in summaries if summary["label"] == reference_label and summary["status"] == "ok"), None)
    if reference is None:
        return []
    rows: list[list[str]] = []
    for summary in summaries:
        if summary["label"] == reference_label or summary["status"] != "ok":
            continue
        rows.append(
            [
                f"{summary['label']} vs {reference_label}",
                format_value(delta(summary.get("final_model_l1"), reference.get("final_model_l1"))),
                format_value(delta(summary.get("final_model_dynamic_l1"), reference.get("final_model_dynamic_l1"))),
                format_value(delta(summary.get("final_model_memory_covered_l1"), reference.get("final_model_memory_covered_l1"))),
                format_value(delta(summary.get("rollout_model_dynamic_l1"), reference.get("rollout_model_dynamic_l1"))),
            ]
        )
    return rows


def write_outputs(output_dir: Path, summaries: list[dict[str, Any]], rollout_name: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_headers = [
        "run",
        "status",
        "best_val_l1",
        "val_l1",
        "delta_vs_copy_last",
        "val_dyn_l1",
        "dyn_delta_vs_copy_last",
        "val_mem_cov_l1",
        "mem_cov_delta_vs_copy_last",
        "val_psnr",
        "memory_cov",
        "unc_corr",
        "write_cov",
        "beats_dyn",
        "beats_mem_cov",
    ]
    rollout_headers = [
        "run",
        "status",
        "rollout_l1",
        "delta_vs_copy_last",
        "rollout_dyn_l1",
        "dyn_delta_vs_copy_last",
        "rollout_mem_cov_l1",
        "mem_cov_delta_vs_copy_last",
        "rollout_psnr",
        "memory_cov",
        "unc_corr",
        "write_cov",
    ]
    reference_headers = [
        "comparison",
        "delta_val_l1",
        "delta_val_dyn_l1",
        "delta_val_mem_cov_l1",
        "delta_rollout_dyn_l1",
    ]

    markdown = "\n".join(
        [
            "# Experiment Comparison",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "## Validation",
            "",
            build_markdown_table(validation_headers, build_validation_rows(summaries)),
            "",
            f"## Rollout ({rollout_name})",
            "",
            build_markdown_table(rollout_headers, build_rollout_rows(summaries)),
            "",
            "## Delta Vs No-Memory",
            "",
            build_markdown_table(reference_headers, build_reference_rows(summaries)) if build_reference_rows(summaries) else "_No reference run found._",
            "",
        ]
    )

    comparison_json = {
        "runs": summaries,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "rollout_name": rollout_name,
    }
    markdown_path = output_dir / "comparison.md"
    json_path = output_dir / "comparison.json"
    markdown_path.write_text(markdown, encoding="utf-8")
    json_path.write_text(json.dumps(comparison_json, indent=2), encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    args = parse_args()
    run_mapping = parse_run_mapping(args.run)
    summaries = [collect_run_summary(label, run_dir, args.rollout_name) for label, run_dir in run_mapping.items()]
    markdown_path, json_path = write_outputs(args.output_dir, summaries, args.rollout_name)
    print(f"wrote {markdown_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
