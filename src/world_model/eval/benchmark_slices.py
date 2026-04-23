from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from world_model.data.clip_dataset import MemoryConditionedClipWindowDataset
from world_model.eval.metrics_image import motion_mask_from_last_context
from world_model.types import ClipSample


@dataclass(frozen=True)
class WindowStats:
    index: int
    clip_path: str
    start_frame: int
    motion_fraction: float
    camera_translation: float
    camera_rotation_deg: float
    memory_render_coverage: float
    depth_edge_fraction: float
    occlusion_recovery: bool


@dataclass(frozen=True)
class SliceDefinition:
    name: str
    description: str
    selector: Callable[[WindowStats], bool]


@lru_cache(maxsize=128)
def _load_clip(path_str: str) -> ClipSample:
    return ClipSample.load_npz(path_str)


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    trace = float(np.trace(rotation))
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _camera_motion(context_poses: torch.Tensor, target_poses: torch.Tensor) -> tuple[float, float]:
    anchor = context_poses[-1].cpu().numpy()
    anchor_inv = np.linalg.inv(anchor)
    max_translation = 0.0
    max_rotation = 0.0
    for pose in target_poses.cpu().numpy():
        relative = anchor_inv @ pose
        max_translation = max(max_translation, float(np.linalg.norm(relative[:3, 3])))
        max_rotation = max(max_rotation, _rotation_angle_deg(relative[:3, :3]))
    return max_translation, max_rotation


def _depth_edge_fraction(target_depth: torch.Tensor, threshold: float = 0.015) -> float:
    depth = target_depth.squeeze(1)
    if depth.ndim != 3:
        return 0.0
    dx = torch.abs(depth[:, :, 1:] - depth[:, :, :-1])
    dy = torch.abs(depth[:, 1:, :] - depth[:, :-1, :])
    valid_x = (depth[:, :, 1:] > 0.0) & (depth[:, :, :-1] > 0.0)
    valid_y = (depth[:, 1:, :] > 0.0) & (depth[:, :-1, :] > 0.0)
    edge_x = (dx > threshold) & valid_x
    edge_y = (dy > threshold) & valid_y
    total_valid = int(valid_x.sum().item() + valid_y.sum().item())
    if total_valid == 0:
        return 0.0
    total_edges = int(edge_x.sum().item() + edge_y.sum().item())
    return float(total_edges / total_valid)


def _detect_occlusion_recovery(
    clip: ClipSample,
    start_frame: int,
    context_frames: int,
    predict_frames: int,
    visible_threshold: int = 64,
    hidden_threshold: int = 8,
) -> bool:
    if clip.visibility is None or clip.visibility.ndim != 2:
        return False
    context_end = start_frame + context_frames
    target_end = min(context_end + predict_frames, clip.visibility.shape[1])
    visibility = clip.visibility[:, start_frame:target_end]
    context_vis = visibility[:, :context_frames]
    target_vis = visibility[:, context_frames:]
    if target_vis.size == 0:
        return False
    for obj_idx in range(visibility.shape[0]):
        if int(context_vis[obj_idx].max()) <= visible_threshold:
            continue
        hidden_steps = np.where(target_vis[obj_idx] <= hidden_threshold)[0]
        if len(hidden_steps) == 0:
            continue
        first_hidden = int(hidden_steps[0])
        if int(target_vis[obj_idx, first_hidden + 1 :].max(initial=0)) > visible_threshold:
            return True
    return False


def collect_window_stats(dataset: MemoryConditionedClipWindowDataset) -> list[WindowStats]:
    stats: list[WindowStats] = []
    for index, window in enumerate(dataset.windows):
        sample = dataset[index]
        motion_fraction = float(
            motion_mask_from_last_context(
                sample["context_rgb"].unsqueeze(0),
                sample["target_rgb"].unsqueeze(0),
            ).mean()
        )
        camera_translation, camera_rotation = _camera_motion(sample["context_poses"], sample["target_poses"])
        clip = _load_clip(str(window.clip_path))
        stats.append(
            WindowStats(
                index=index,
                clip_path=str(window.clip_path),
                start_frame=window.start_frame,
                motion_fraction=motion_fraction,
                camera_translation=camera_translation,
                camera_rotation_deg=camera_rotation,
                memory_render_coverage=float(sample["memory_render_coverage"]),
                depth_edge_fraction=_depth_edge_fraction(sample["target_depth"]),
                occlusion_recovery=_detect_occlusion_recovery(
                    clip=clip,
                    start_frame=window.start_frame,
                    context_frames=dataset.context_frames,
                    predict_frames=dataset.predict_frames,
                ),
            )
        )
    return stats


def build_default_slices(window_stats: list[WindowStats]) -> OrderedDict[str, SliceDefinition]:
    if not window_stats:
        return OrderedDict()

    motion_values = np.array([stats.motion_fraction for stats in window_stats], dtype=np.float32)
    translation_values = np.array([stats.camera_translation for stats in window_stats], dtype=np.float32)
    rotation_values = np.array([stats.camera_rotation_deg for stats in window_stats], dtype=np.float32)
    coverage_values = np.array([stats.memory_render_coverage for stats in window_stats], dtype=np.float32)
    edge_values = np.array([stats.depth_edge_fraction for stats in window_stats], dtype=np.float32)

    motion_threshold = float(np.quantile(motion_values, 0.75))
    translation_threshold = float(np.quantile(translation_values, 0.75))
    rotation_threshold = float(np.quantile(rotation_values, 0.75))
    coverage_threshold = float(np.quantile(coverage_values, 0.75))
    edge_threshold = float(np.quantile(edge_values, 0.75))
    translation_cutoff = max(translation_threshold, 1e-5)
    rotation_cutoff = max(rotation_threshold, 1e-3)

    return OrderedDict(
        [
            (
                "all",
                SliceDefinition(
                    name="all",
                    description="All validation windows.",
                    selector=lambda stats: True,
                ),
            ),
            (
                "high_motion",
                SliceDefinition(
                    name="high_motion",
                    description=f"Windows above the 75th percentile of GT motion fraction ({motion_threshold:.4f}).",
                    selector=lambda stats, threshold=motion_threshold: stats.motion_fraction >= threshold,
                ),
            ),
            (
                "high_camera_motion",
                SliceDefinition(
                    name="high_camera_motion",
                    description=(
                        "Windows above the 75th percentile of camera translation "
                        f"({translation_cutoff:.4f}) or rotation ({rotation_cutoff:.2f} deg)."
                    ),
                    selector=lambda stats, t_thr=translation_cutoff, r_thr=rotation_cutoff: (
                        stats.camera_translation > t_thr or stats.camera_rotation_deg > r_thr
                    ),
                ),
            ),
            (
                "occlusion_recovery",
                SliceDefinition(
                    name="occlusion_recovery",
                    description="Windows with a visible-hidden-visible object according to MOVi visibility metadata.",
                    selector=lambda stats: stats.occlusion_recovery,
                ),
            ),
            (
                "high_memory_coverage",
                SliceDefinition(
                    name="high_memory_coverage",
                    description=f"Windows above the 75th percentile of oracle memory render coverage ({coverage_threshold:.4f}).",
                    selector=lambda stats, threshold=coverage_threshold: stats.memory_render_coverage >= threshold,
                ),
            ),
            (
                "depth_edge_heavy",
                SliceDefinition(
                    name="depth_edge_heavy",
                    description=f"Windows above the 75th percentile of normalized depth-edge fraction ({edge_threshold:.4f}).",
                    selector=lambda stats, threshold=edge_threshold: stats.depth_edge_fraction >= threshold,
                ),
            ),
        ]
    )


def compute_slice_memberships(
    window_stats: list[WindowStats],
    slice_definitions: OrderedDict[str, SliceDefinition],
) -> dict[str, list[int]]:
    memberships: dict[str, list[int]] = {}
    for slice_name, definition in slice_definitions.items():
        memberships[slice_name] = [stats.index for stats in window_stats if definition.selector(stats)]
    return memberships


def serialize_window_stats(window_stats: list[WindowStats]) -> list[dict[str, float | int | str | bool]]:
    return [asdict(stats) for stats in window_stats]


def serialize_slice_definitions(
    slice_definitions: OrderedDict[str, SliceDefinition],
    memberships: dict[str, list[int]],
) -> dict[str, dict[str, str | int]]:
    payload: dict[str, dict[str, str | int]] = {}
    for slice_name, definition in slice_definitions.items():
        payload[slice_name] = {
            "description": definition.description,
            "count": len(memberships.get(slice_name, [])),
        }
    return payload
