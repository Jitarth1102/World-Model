from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model.geometry.camera import project_points
from world_model.memory.voxel_grid import VoxelGrid
from world_model.types import CameraIntrinsics


@dataclass
class RenderedView:
    rgb: np.ndarray
    mask: np.ndarray
    forward_depth: np.ndarray
    radial_depth: np.ndarray


def render_memory_view(
    memory: VoxelGrid,
    target_pose: np.ndarray,
    intrinsics: CameraIntrinsics,
    splat_radius: int = 0,
) -> RenderedView:
    rgb = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.float32)
    mask = np.zeros((intrinsics.height, intrinsics.width), dtype=bool)
    forward_depth = np.full((intrinsics.height, intrinsics.width), np.inf, dtype=np.float32)
    radial_depth = np.zeros((intrinsics.height, intrinsics.width), dtype=np.float32)

    world_points, colors, weights = memory.occupied_centers_and_colors()
    if len(world_points) == 0:
        return RenderedView(rgb=rgb, mask=mask, forward_depth=forward_depth, radial_depth=radial_depth)

    pixels, z_forward, z_radial = project_points(world_points, target_pose, intrinsics)
    valid = (
        (z_forward > 1e-6)
        & np.isfinite(z_forward)
        & np.isfinite(z_radial)
        & (pixels[:, 0] >= -splat_radius)
        & (pixels[:, 0] < intrinsics.width + splat_radius)
        & (pixels[:, 1] >= -splat_radius)
        & (pixels[:, 1] < intrinsics.height + splat_radius)
    )
    if not np.any(valid):
        return RenderedView(rgb=rgb, mask=mask, forward_depth=forward_depth, radial_depth=radial_depth)

    pixels = pixels[valid]
    z_forward = z_forward[valid]
    z_radial = z_radial[valid]
    colors = colors[valid]
    weights = weights[valid]
    order = np.argsort(z_forward)

    for idx in order:
        px = int(np.round(pixels[idx, 0]))
        py = int(np.round(pixels[idx, 1]))
        for dx in range(-splat_radius, splat_radius + 1):
            for dy in range(-splat_radius, splat_radius + 1):
                x = px + dx
                y = py + dy
                if x < 0 or x >= intrinsics.width or y < 0 or y >= intrinsics.height:
                    continue
                if z_forward[idx] < forward_depth[y, x]:
                    forward_depth[y, x] = z_forward[idx]
                    radial_depth[y, x] = z_radial[idx]
                    rgb[y, x] = colors[idx]
                    mask[y, x] = True

    return RenderedView(rgb=rgb, mask=mask, forward_depth=forward_depth, radial_depth=radial_depth)
