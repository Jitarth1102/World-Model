from __future__ import annotations

import json
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset, load_manifest, split_clip_paths
from world_model.eval.benchmark_slices import (
    WindowStats,
    build_default_slices,
    collect_window_stats,
    compute_slice_memberships,
    serialize_slice_definitions,
    serialize_window_stats,
)
from world_model.eval.metrics_image import masked_l1, motion_mask_from_last_context, psnr
from world_model.eval.metrics_memory import baseline_advantage, memory_covered_l1, oracle_alignment_l1
from world_model.models.convgru_predictor import NoMemoryPredictor
from world_model.models.world_model import MemoryConditionedWorldModel


@dataclass(frozen=True)
class EvalRunSpec:
    label: str
    variant: str
    run_dir: Path
    checkpoint_path: Path
    hidden_channels: int
    status: str = "ok"


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _move_sample_to_device(sample: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                moved[key] = value.unsqueeze(0).to(device=device, dtype=torch.float32)
            else:
                moved[key] = value.unsqueeze(0).to(device)
        else:
            moved[key] = value
    return moved


def _infer_hidden_channels(run_dir: Path, default: int = 96) -> int:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return default
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return int(payload.get("hidden_channels", default))


def resolve_run_specs(
    default_runs: OrderedDict[str, tuple[str, Path]],
    checkpoint_kind: str = "last",
) -> list[EvalRunSpec]:
    specs: list[EvalRunSpec] = []
    for label, (variant, run_dir) in default_runs.items():
        if variant == "no_memory":
            preferred = run_dir / f"nomemory_model_{checkpoint_kind}.pt"
            fallback = run_dir / "nomemory_model_best.pt"
        else:
            preferred = run_dir / f"memory_model_{checkpoint_kind}.pt"
            fallback = run_dir / "memory_model_best.pt"
        checkpoint_path = preferred if preferred.exists() else fallback
        status = "ok" if checkpoint_path.exists() else "missing"
        specs.append(
            EvalRunSpec(
                label=label,
                variant=variant,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                hidden_channels=_infer_hidden_channels(run_dir),
                status=status,
            )
        )
    return specs


def build_eval_dataset(
    manifest: Path,
    context_frames: int,
    predict_frames: int,
    image_size: int,
    val_ratio: float,
    seed: int,
    max_val_windows_per_clip: int | None,
    memory_grid_resolution: tuple[int, int, int],
    memory_stride: int,
    memory_splat_radius: int,
) -> tuple[MemoryConditionedClipWindowDataset, list[Path]]:
    clip_paths = load_manifest(manifest)
    _, val_paths = split_clip_paths(clip_paths, val_ratio=val_ratio, seed=seed)
    dataset = MemoryConditionedClipWindowDataset(
        clip_paths=val_paths,
        context_frames=context_frames,
        predict_frames=predict_frames,
        image_size=image_size,
        max_windows_per_clip=max_val_windows_per_clip,
        memory_grid_resolution=memory_grid_resolution,
        memory_stride=memory_stride,
        memory_splat_radius=memory_splat_radius,
    )
    return dataset, val_paths


def _load_model(spec: EvalRunSpec, device: torch.device) -> torch.nn.Module | None:
    if spec.status != "ok":
        return None
    if spec.variant == "no_memory":
        model: torch.nn.Module = NoMemoryPredictor(hidden_channels=spec.hidden_channels).to(device)
    else:
        model = MemoryConditionedWorldModel(hidden_channels=spec.hidden_channels).to(device)
    state_dict = torch.load(spec.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _predict(spec: EvalRunSpec, model: torch.nn.Module, sample: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
    context_rgb = sample["context_rgb"]
    context_poses = sample["context_poses"]
    target_poses = sample["target_poses"]
    if spec.variant == "no_memory":
        pred_rgb = model(context_rgb, context_poses, target_poses)
        return pred_rgb, None
    pred_rgb, pred_depth = model(context_rgb, context_poses, target_poses, sample["memory_condition"])
    return pred_rgb, pred_depth


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    numeric_totals: dict[str, float] = defaultdict(float)
    count = len(rows)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool):
                numeric_totals[key] += float(value)
            elif isinstance(value, (int, float)):
                numeric_totals[key] += float(value)
    aggregate = {"count": count}
    for key, total in numeric_totals.items():
        aggregate[key] = total / count
    return aggregate


def evaluate_run(
    spec: EvalRunSpec,
    dataset: MemoryConditionedClipWindowDataset,
    window_stats: list[WindowStats],
    slice_memberships: dict[str, list[int]],
    device: torch.device,
    motion_threshold: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "label": spec.label,
        "variant": spec.variant,
        "run_dir": str(spec.run_dir),
        "checkpoint_path": str(spec.checkpoint_path),
        "status": spec.status,
    }
    if spec.status != "ok":
        return result

    model = _load_model(spec, device)
    if model is None:
        result["status"] = "missing"
        return result

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for idx, stats in enumerate(window_stats):
            raw_sample = dataset[idx]
            sample = _move_sample_to_device(raw_sample, device)
            pred_rgb, pred_depth = _predict(spec, model, sample)
            target_rgb = sample["target_rgb"]
            baseline_rgb = sample["context_rgb"][:, -1:].repeat(1, target_rgb.shape[1], 1, 1, 1)
            dynamic_mask = motion_mask_from_last_context(sample["context_rgb"], target_rgb, threshold=motion_threshold)
            memory_mask = sample["memory_render_mask"]
            row = {
                "index": idx,
                "clip_path": stats.clip_path,
                "start_frame": stats.start_frame,
                "model_l1": float(masked_l1(pred_rgb, target_rgb)),
                "baseline_l1": float(masked_l1(baseline_rgb, target_rgb)),
                "model_psnr": float(psnr(pred_rgb, target_rgb)),
                "baseline_psnr": float(psnr(baseline_rgb, target_rgb)),
                "model_dynamic_l1": float(masked_l1(pred_rgb, target_rgb, dynamic_mask)),
                "baseline_dynamic_l1": float(masked_l1(baseline_rgb, target_rgb, dynamic_mask)),
                "dynamic_advantage": float(baseline_advantage(masked_l1(pred_rgb, target_rgb, dynamic_mask), masked_l1(baseline_rgb, target_rgb, dynamic_mask))),
                "model_memory_covered_l1": float(memory_covered_l1(pred_rgb, target_rgb, memory_mask)),
                "baseline_memory_covered_l1": float(memory_covered_l1(baseline_rgb, target_rgb, memory_mask)),
                "memory_covered_advantage": float(
                    baseline_advantage(
                        memory_covered_l1(pred_rgb, target_rgb, memory_mask),
                        memory_covered_l1(baseline_rgb, target_rgb, memory_mask),
                    )
                ),
                "oracle_alignment_l1": float(oracle_alignment_l1(pred_rgb, sample["memory_render_rgb"], memory_mask)),
                "memory_render_coverage": float(raw_sample["memory_render_coverage"]),
                "memory_occupancy_fraction": float(raw_sample["memory_occupancy_fraction"]),
                "oracle_memory_render_l1_covered": float(raw_sample["memory_render_l1_covered"]),
                "motion_fraction": stats.motion_fraction,
                "camera_translation": stats.camera_translation,
                "camera_rotation_deg": stats.camera_rotation_deg,
                "depth_edge_fraction": stats.depth_edge_fraction,
                "occlusion_recovery": stats.occlusion_recovery,
            }
            if pred_depth is not None:
                depth_mask = (sample["target_depth"] > 0.0).to(dtype=sample["target_depth"].dtype)
                row["model_depth_l1"] = float(masked_l1(pred_depth, sample["target_depth"], depth_mask))
            rows.append(row)

    overall = _aggregate_rows(rows)
    slices = {}
    for slice_name, indices in slice_memberships.items():
        slice_rows = [rows[index] for index in indices]
        slices[slice_name] = _aggregate_rows(slice_rows)

    result.update(
        {
            "overall": overall,
            "slices": slices,
            "per_window": rows,
        }
    )
    return result


def build_markdown_summary(
    results: list[dict[str, Any]],
    slice_info: dict[str, dict[str, Any]],
) -> str:
    headers = [
        "run",
        "status",
        "val_l1",
        "val_dyn_l1",
        "val_mem_cov_l1",
        "dyn_adv",
        "mem_cov_adv",
        "val_psnr",
    ]
    rows = []
    for result in results:
        overall = result.get("overall", {})
        rows.append(
            [
                result["label"],
                result["status"],
                "-" if "model_l1" not in overall else f"{overall['model_l1']:.4f}",
                "-" if "model_dynamic_l1" not in overall else f"{overall['model_dynamic_l1']:.4f}",
                "-" if "model_memory_covered_l1" not in overall else f"{overall['model_memory_covered_l1']:.4f}",
                "-" if "dynamic_advantage" not in overall else f"{overall['dynamic_advantage']:.4f}",
                "-" if "memory_covered_advantage" not in overall else f"{overall['memory_covered_advantage']:.4f}",
                "-" if "model_psnr" not in overall else f"{overall['model_psnr']:.4f}",
            ]
        )
    lines = [
        "# Phase 5 Evaluation",
        "",
        "## Overall",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    lines.append("")
    lines.append("## Slices")
    lines.append("")

    for slice_name, info in slice_info.items():
        lines.append(f"### {slice_name}")
        lines.append("")
        lines.append(info["description"])
        lines.append("")
        lines.append(f"Count: {info['count']}")
        lines.append("")
        lines.append("| run | status | l1 | dyn_l1 | mem_cov_l1 | dyn_adv | mem_cov_adv |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for result in results:
            aggregate = result.get("slices", {}).get(slice_name, {})
            lines.append(
                "| "
                + " | ".join(
                    [
                        result["label"],
                        result["status"],
                        "-" if "model_l1" not in aggregate else f"{aggregate['model_l1']:.4f}",
                        "-" if "model_dynamic_l1" not in aggregate else f"{aggregate['model_dynamic_l1']:.4f}",
                        "-" if "model_memory_covered_l1" not in aggregate else f"{aggregate['model_memory_covered_l1']:.4f}",
                        "-" if "dynamic_advantage" not in aggregate else f"{aggregate['dynamic_advantage']:.4f}",
                        "-" if "memory_covered_advantage" not in aggregate else f"{aggregate['memory_covered_advantage']:.4f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def write_evaluation_outputs(
    output_dir: Path,
    results: list[dict[str, Any]],
    window_stats: list[WindowStats],
    slice_info: dict[str, dict[str, Any]],
    run_specs: list[EvalRunSpec],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_run_specs = [
        {
            "label": spec.label,
            "variant": spec.variant,
            "run_dir": str(spec.run_dir),
            "checkpoint_path": str(spec.checkpoint_path),
            "hidden_channels": spec.hidden_channels,
            "status": spec.status,
        }
        for spec in run_specs
    ]
    payload = {
        "runs": results,
        "window_stats": serialize_window_stats(window_stats),
        "slice_info": slice_info,
        "run_specs": serialized_run_specs,
    }
    json_path = output_dir / "evaluation.json"
    md_path = output_dir / "evaluation.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown_summary(results, slice_info), encoding="utf-8")
    return json_path, md_path


def run_full_evaluation(
    manifest: Path,
    default_runs: OrderedDict[str, tuple[str, Path]],
    output_dir: Path,
    context_frames: int,
    predict_frames: int,
    image_size: int,
    val_ratio: float,
    seed: int,
    max_val_windows_per_clip: int | None,
    memory_grid_resolution: tuple[int, int, int],
    memory_stride: int,
    memory_splat_radius: int,
    motion_threshold: float,
    checkpoint_kind: str,
    device: torch.device,
) -> tuple[Path, Path]:
    dataset, _ = build_eval_dataset(
        manifest=manifest,
        context_frames=context_frames,
        predict_frames=predict_frames,
        image_size=image_size,
        val_ratio=val_ratio,
        seed=seed,
        max_val_windows_per_clip=max_val_windows_per_clip,
        memory_grid_resolution=memory_grid_resolution,
        memory_stride=memory_stride,
        memory_splat_radius=memory_splat_radius,
    )
    window_stats = collect_window_stats(dataset)
    slice_definitions = build_default_slices(window_stats)
    slice_memberships = compute_slice_memberships(window_stats, slice_definitions)
    slice_info = serialize_slice_definitions(slice_definitions, slice_memberships)
    run_specs = resolve_run_specs(default_runs, checkpoint_kind=checkpoint_kind)
    results = [evaluate_run(spec, dataset, window_stats, slice_memberships, device, motion_threshold) for spec in run_specs]
    return write_evaluation_outputs(output_dir, results, window_stats, slice_info, run_specs)
