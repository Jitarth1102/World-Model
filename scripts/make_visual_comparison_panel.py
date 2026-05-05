#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset
from world_model.eval.benchmark_slices import build_default_slices, collect_window_stats, compute_slice_memberships
from world_model.eval.evaluator import EvalRunSpec, _load_model, _load_run_metrics_payload, _move_sample_to_device, _predict, pick_device
from world_model.inference.uncertainty_rollout import rollout_convgru_uncertainty, rollout_diffusion_uncertainty


@dataclass(frozen=True)
class SelectedCase:
    name: str
    index: int
    clip_path: str
    start_frame: int
    reason: str


DEFAULT_RUNS = OrderedDict(
    [
        ("convgru_no_memory", ("no_memory", Path("outputs/scaled_movia100_or_200_convgru_nomemory_v1"))),
        ("convgru_memory", ("memory", Path("outputs/scaled_movia100_or_200_convgru_memory_v1"))),
        ("convgru_uncertainty", ("memory_uncertainty_convgru", Path("outputs/scaled_movia100_or_200_convgru_uncertainty_t099_g4_v1"))),
        ("diffusion_no_memory", ("diffusion_no_memory", Path("outputs/scaled_movia100_or_200_diffusion_nomemory_v1"))),
        ("diffusion_memory", ("diffusion_memory", Path("outputs/scaled_movia100_or_200_diffusion_memory_v1"))),
        ("diffusion_uncertainty", ("diffusion_memory_uncertainty", Path("outputs/scaled_movia100_or_200_diffusion_uncertainty_v1"))),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-ready visual comparison panels across scaled model variants.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/movi_a_128_subset100/manifest.json"))
    parser.add_argument("--evaluation-json", type=Path, required=True)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Override default runs with LABEL=VARIANT:PATH. Can be passed multiple times.",
    )
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--predict-frames", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-val-windows-per-clip", type=int, default=6)
    parser.add_argument("--memory-grid-resolution", type=int, nargs=3, default=(48, 40, 48))
    parser.add_argument("--memory-stride", type=int, default=1)
    parser.add_argument("--memory-splat-radius", type=int, default=1)
    parser.add_argument("--uncertainty-label", default="convgru_uncertainty")
    parser.add_argument("--failure-label", default="diffusion_memory")
    parser.add_argument(
        "--skip-failure-case",
        action="store_true",
        help="Do not include the highest-error window (often ugly); better for slides/report aesthetics.",
    )
    parser.add_argument(
        "--strip-scale",
        type=int,
        default=4,
        help="Integer upscale for strips before layout (LANCZOS). Training is still at --image-size; this only affects exported PNGs.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_visual_comparisons_scaled_v1"))
    return parser.parse_args()


def parse_run_mapping(overrides: list[str]) -> OrderedDict[str, tuple[str, Path]]:
    if not overrides:
        return DEFAULT_RUNS.copy()
    mapping: OrderedDict[str, tuple[str, Path]] = OrderedDict()
    for override in overrides:
        if "=" not in override or ":" not in override:
            raise ValueError(f"Expected LABEL=VARIANT:PATH, got: {override}")
        label, raw_value = override.split("=", 1)
        variant, raw_path = raw_value.split(":", 1)
        mapping[label.strip()] = (variant.strip(), Path(raw_path).expanduser())
    return mapping


def load_font(size: int = 16) -> ImageFont.ImageFont:
    for font_name in ["Helvetica.ttc", "Arial.ttf"]:
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def tensor_strip_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = tensor.detach().cpu()
    if tensor.ndim == 4:
        frames = tensor
    elif tensor.ndim == 3:
        frames = tensor.unsqueeze(0)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(tensor.shape)}")
    if frames.shape[1] == 1:
        arrays = []
        for frame in frames:
            array = frame.squeeze(0).numpy().astype("float32")
            min_value = float(array.min())
            max_value = float(array.max())
            if max_value > min_value:
                array = (array - min_value) / (max_value - min_value)
            else:
                array = np.zeros_like(array)
            arrays.append((np.clip(array, 0.0, 1.0) * 255.0).astype("uint8"))
        height, width = arrays[0].shape
        canvas = Image.new("L", (width * len(arrays), height))
        for idx, frame in enumerate(arrays):
            canvas.paste(Image.fromarray(frame), (idx * width, 0))
        return canvas.convert("RGB")
    arrays = [(frame.clamp(0.0, 1.0).numpy() * 255.0).astype("uint8") for frame in frames]
    height, width = arrays[0].shape[1], arrays[0].shape[2]
    canvas = Image.new("RGB", (width * len(arrays), height))
    for idx, frame in enumerate(arrays):
        canvas.paste(Image.fromarray(frame.transpose(1, 2, 0)), (idx * width, 0))
    return canvas


def render_blank_strip(width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), color=(20, 20, 20))
    return canvas


def upscale_strip(image: Image.Image, scale: int) -> Image.Image:
    if scale <= 1:
        return image
    w, h = image.size
    return image.resize((w * scale, h * scale), Image.Resampling.LANCZOS)


def build_dataset(args: argparse.Namespace) -> MemoryConditionedClipWindowDataset:
    from world_model.data.clip_dataset import load_manifest, split_clip_paths

    clip_paths = load_manifest(args.manifest)
    _, val_paths = split_clip_paths(clip_paths, val_ratio=args.val_ratio, seed=args.seed)
    return MemoryConditionedClipWindowDataset(
        clip_paths=val_paths,
        context_frames=args.context_frames,
        predict_frames=args.predict_frames,
        image_size=args.image_size,
        max_windows_per_clip=args.max_val_windows_per_clip,
        memory_grid_resolution=tuple(args.memory_grid_resolution),
        memory_stride=args.memory_stride,
        memory_splat_radius=args.memory_splat_radius,
    )


def select_case_indices(
    *,
    eval_payload: dict[str, Any],
    window_stats: list[Any],
    memberships: dict[str, list[int]],
    uncertainty_label: str,
    failure_label: str,
) -> list[SelectedCase]:
    used: set[int] = set()
    per_run = {run["label"]: run.get("per_window", []) for run in eval_payload["runs"]}
    uncertainty_rows = per_run.get(uncertainty_label, [])
    failure_rows = per_run.get(failure_label, uncertainty_rows)

    def choose_from_indices(name: str, indices: list[int], reason: str, key: str = "model_l1", mode: str = "median") -> SelectedCase | None:
        candidates = [idx for idx in indices if idx not in used]
        if not candidates:
            return None
        rows = failure_rows if name == "failure_case" else uncertainty_rows
        if rows and key in rows[0]:
            scored = sorted(((float(rows[idx][key]), idx) for idx in candidates), key=lambda item: item[0])
            if mode == "max":
                chosen_idx = scored[-1][1]
            else:
                chosen_idx = scored[len(scored) // 2][1]
        else:
            chosen_idx = candidates[0]
        used.add(chosen_idx)
        stats = window_stats[chosen_idx]
        return SelectedCase(name=name, index=chosen_idx, clip_path=stats.clip_path, start_frame=stats.start_frame, reason=reason)

    cases: list[SelectedCase] = []
    all_indices = list(range(len(window_stats)))
    normal = choose_from_indices("normal_case", all_indices, "Median example from all validation windows.", mode="median")
    if normal:
        cases.append(normal)
    high_motion = choose_from_indices("high_motion", memberships.get("high_motion", []), "Representative high-motion example.", mode="median")
    if high_motion:
        cases.append(high_motion)
    high_memory = choose_from_indices("high_memory_coverage", memberships.get("high_memory_coverage", []), "Representative high-memory-coverage example.", mode="median")
    if high_memory:
        cases.append(high_memory)
    occlusion = choose_from_indices("occlusion_recovery", memberships.get("occlusion_recovery", []), "Representative occlusion/reappearance example.", mode="median")
    if occlusion:
        cases.append(occlusion)
    failure = choose_from_indices("failure_case", all_indices, f"Highest-error example according to {failure_label}.", mode="max")
    if failure:
        cases.append(failure)
    return cases[:5]


def build_spec(label: str, variant: str, run_dir: Path) -> EvalRunSpec:
    from world_model.eval.evaluator import _infer_diffusion_config, _infer_hidden_channels

    if variant == "no_memory":
        checkpoint_path = run_dir / "nomemory_model_best.pt"
        return EvalRunSpec(label=label, variant=variant, run_dir=run_dir, checkpoint_path=checkpoint_path, hidden_channels=_infer_hidden_channels(run_dir))
    if variant in {"memory", "memory_uncertainty_convgru"}:
        checkpoint_path = run_dir / "memory_model_best.pt"
        return EvalRunSpec(label=label, variant=variant, run_dir=run_dir, checkpoint_path=checkpoint_path, hidden_channels=_infer_hidden_channels(run_dir))
    if variant in {"diffusion_no_memory", "diffusion_memory", "diffusion_memory_uncertainty"}:
        model_channels, diffusion_steps, sample_steps_eval = _infer_diffusion_config(run_dir)
        checkpoint_path = run_dir / "diffusion_model_best.pt"
        return EvalRunSpec(
            label=label,
            variant=variant,
            run_dir=run_dir,
            checkpoint_path=checkpoint_path,
            hidden_channels=_infer_hidden_channels(run_dir),
            model_channels=model_channels,
            diffusion_steps=diffusion_steps,
            sample_steps_eval=sample_steps_eval,
        )
    raise ValueError(f"Unsupported variant: {variant}")


def infer_predictions(
    *,
    spec: EvalRunSpec,
    model: torch.nn.Module,
    raw_sample: dict[str, Any],
    dataset: MemoryConditionedClipWindowDataset,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sample = _move_sample_to_device(raw_sample, device)
    if spec.variant == "memory_uncertainty_convgru":
        payload = _load_run_metrics_payload(spec.run_dir)
        rollout = rollout_convgru_uncertainty(
            model=model,
            clip_path=raw_sample["clip_path"],
            start_frame=int(raw_sample["start_frame"]),
            context_frames=dataset.context_frames,
            predict_frames=dataset.predict_frames,
            image_size=dataset.image_size or args.image_size,  # type: ignore[name-defined]
            device=device,
            memory_grid_resolution=dataset.memory_grid_resolution,
            memory_stride=dataset.memory_stride,
            memory_splat_radius=dataset.memory_splat_radius,
            confidence_threshold=float(payload.get("write_confidence_threshold", 0.55)),
            confidence_gamma=float(payload.get("confidence_gamma", 1.0)),
        )
        return {
            "prediction": rollout["prediction"],
            "uncertainty": rollout["uncertainty"],
            "write_mask": rollout["write_mask"],
        }
    if spec.variant == "diffusion_memory_uncertainty":
        payload = _load_run_metrics_payload(spec.run_dir)
        rollout = rollout_diffusion_uncertainty(
            model=model,
            clip_path=raw_sample["clip_path"],
            start_frame=int(raw_sample["start_frame"]),
            context_frames=dataset.context_frames,
            predict_frames=dataset.predict_frames,
            image_size=dataset.image_size or args.image_size,  # type: ignore[name-defined]
            device=device,
            memory_grid_resolution=dataset.memory_grid_resolution,
            memory_stride=dataset.memory_stride,
            memory_splat_radius=dataset.memory_splat_radius,
            confidence_threshold=float(payload.get("write_confidence_threshold", 0.55)),
            sample_steps=int(payload.get("sample_steps_eval", spec.sample_steps_eval or 50)),
            uncertainty_samples=int(payload.get("uncertainty_samples", 4)),
            confidence_gamma=float(payload.get("confidence_gamma", 1.0)),
        )
        return {
            "prediction": rollout["prediction"],
            "uncertainty": rollout["uncertainty"],
            "write_mask": rollout["write_mask"],
        }
    pred_rgb, _ = _predict(spec, model, sample)
    return {"prediction": pred_rgb[0].detach().cpu()}


def build_panel(
    *,
    case: SelectedCase,
    raw_sample: dict[str, Any],
    predictions: dict[str, dict[str, torch.Tensor]],
    output_path: Path,
    strip_scale: int = 1,
) -> None:
    title_size = min(12 + 3 * strip_scale, 36)
    subtitle_size = min(10 + 2 * strip_scale, 22)
    row_label_size = min(14 + 2 * strip_scale, 28)
    font = load_font(title_size)
    rows: list[tuple[str, Image.Image]] = []
    context_strip = upscale_strip(tensor_strip_to_image(raw_sample["context_rgb"]), strip_scale)
    target_strip = upscale_strip(tensor_strip_to_image(raw_sample["target_rgb"]), strip_scale)
    strip_width, strip_height = context_strip.size
    rows.append(("Context", context_strip))
    rows.append(("Ground Truth", target_strip))

    ordered_rows = [
        ("ConvGRU No Memory", "convgru_no_memory"),
        ("ConvGRU Memory", "convgru_memory"),
        ("ConvGRU Uncertainty", "convgru_uncertainty"),
        ("Diffusion No Memory", "diffusion_no_memory"),
        ("Diffusion Memory", "diffusion_memory"),
        ("Diffusion Uncertainty", "diffusion_uncertainty"),
    ]
    for label, key in ordered_rows:
        if key in predictions:
            rows.append((label, upscale_strip(tensor_strip_to_image(predictions[key]["prediction"]), strip_scale)))
    if "convgru_uncertainty" in predictions:
        rows.append(
            ("Uncertainty Map", upscale_strip(tensor_strip_to_image(predictions["convgru_uncertainty"]["uncertainty"]), strip_scale))
        )
        rows.append(
            ("Write Mask", upscale_strip(tensor_strip_to_image(predictions["convgru_uncertainty"]["write_mask"]), strip_scale))
        )

    label_width = max(190, 44 + 14 * strip_scale)
    header_height = max(42, 18 + title_size + subtitle_size)
    gap = max(6, strip_scale * 2)
    panel_width = label_width + strip_width
    panel_height = header_height + len(rows) * (strip_height + gap)
    canvas = Image.new("RGB", (panel_width, panel_height), color=(248, 246, 240))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), f"{case.name}: {Path(case.clip_path).name} @ frame {case.start_frame}", fill=(20, 20, 20), font=font)
    draw.text((12, 8 + title_size), case.reason, fill=(90, 90, 90), font=load_font(subtitle_size))

    y = header_height
    row_font = load_font(row_label_size)
    for label, row_image in rows:
        row_canvas = Image.new("RGB", (panel_width, strip_height), color=(248, 246, 240))
        row_draw = ImageDraw.Draw(row_canvas)
        row_draw.text((12, max(0, strip_height // 2 - row_label_size // 2)), label, fill=(20, 20, 20), font=row_font)
        row_canvas.paste(row_image, (label_width, 0))
        canvas.paste(row_canvas, (0, y))
        y += strip_height + gap

    canvas.save(output_path)


def main() -> None:
    global args
    args = parse_args()
    device = pick_device(args.device)
    eval_payload = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    run_mapping = parse_run_mapping(args.run)
    dataset = build_dataset(args)
    window_stats = collect_window_stats(dataset)
    slice_memberships = compute_slice_memberships(window_stats, build_default_slices(window_stats))
    cases = select_case_indices(
        eval_payload=eval_payload,
        window_stats=window_stats,
        memberships=slice_memberships,
        uncertainty_label=args.uncertainty_label,
        failure_label=args.failure_label,
    )
    if args.skip_failure_case:
        cases = [c for c in cases if c.name != "failure_case"]

    specs = OrderedDict((label, build_spec(label, variant, run_dir)) for label, (variant, run_dir) in run_mapping.items())
    models = OrderedDict((label, _load_model(spec, device)) for label, spec in specs.items())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for case in cases:
        raw_sample = dataset[case.index]
        predictions: dict[str, dict[str, torch.Tensor]] = {}
        for label, spec in specs.items():
            model = models[label]
            if model is None:
                continue
            predictions[label] = infer_predictions(spec=spec, model=model, raw_sample=raw_sample, dataset=dataset, device=device)
        case_dir = args.output_dir / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        build_panel(
            case=case,
            raw_sample=raw_sample,
            predictions=predictions,
            output_path=case_dir / "panel.png",
            strip_scale=max(1, args.strip_scale),
        )
        (case_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "case": case.name,
                    "clip_path": case.clip_path,
                    "start_frame": case.start_frame,
                    "reason": case.reason,
                    "index": case.index,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved {case_dir / 'panel.png'}")


if __name__ == "__main__":
    main()
