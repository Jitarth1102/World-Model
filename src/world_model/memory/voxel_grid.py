from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VoxelGridSpec:
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    resolution: tuple[int, int, int]

    @property
    def voxel_size(self) -> np.ndarray:
        return (np.asarray(self.bounds_max, dtype=np.float32) - np.asarray(self.bounds_min, dtype=np.float32)) / np.asarray(
            self.resolution, dtype=np.float32
        )

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.resolution


class VoxelGrid:
    def __init__(self, spec: VoxelGridSpec):
        self.spec = spec
        shape = spec.shape
        self.color_sum = np.zeros(shape + (3,), dtype=np.float32)
        self.weight = np.zeros(shape, dtype=np.float32)
        self.occupancy = np.zeros(shape, dtype=np.uint16)
        self.confidence = np.zeros(shape, dtype=np.float32)

    def clone_empty(self) -> "VoxelGrid":
        return VoxelGrid(self.spec)

    def world_to_index(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bounds_min = np.asarray(self.spec.bounds_min, dtype=np.float32)
        bounds_max = np.asarray(self.spec.bounds_max, dtype=np.float32)
        resolution = np.asarray(self.spec.resolution, dtype=np.int32)
        voxel_size = self.spec.voxel_size
        scaled = (points_world - bounds_min) / voxel_size
        indices = np.floor(scaled).astype(np.int32)
        valid = np.all(points_world >= bounds_min, axis=-1) & np.all(points_world < bounds_max, axis=-1)
        valid &= np.all(indices >= 0, axis=-1) & np.all(indices < resolution, axis=-1)
        return indices, valid

    def index_to_world(self, indices: np.ndarray) -> np.ndarray:
        bounds_min = np.asarray(self.spec.bounds_min, dtype=np.float32)
        return bounds_min + (indices.astype(np.float32) + 0.5) * self.spec.voxel_size

    def splat_rgb(self, points_world: np.ndarray, colors: np.ndarray, weights: np.ndarray | None = None) -> int:
        if weights is None:
            weights = np.ones((len(points_world),), dtype=np.float32)
        indices, valid = self.world_to_index(points_world)
        if not np.any(valid):
            return 0
        indices = indices[valid]
        colors = colors[valid]
        weights = weights[valid]
        x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
        np.add.at(self.color_sum[..., 0], (x, y, z), colors[:, 0] * weights)
        np.add.at(self.color_sum[..., 1], (x, y, z), colors[:, 1] * weights)
        np.add.at(self.color_sum[..., 2], (x, y, z), colors[:, 2] * weights)
        np.add.at(self.weight, (x, y, z), weights)
        np.add.at(self.occupancy, (x, y, z), 1)
        np.add.at(self.confidence, (x, y, z), weights)
        return int(len(indices))

    def mean_color(self) -> np.ndarray:
        weight = np.clip(self.weight[..., None], 1e-6, None)
        return self.color_sum / weight

    def occupied_mask(self) -> np.ndarray:
        return self.weight > 0.0

    def occupied_centers_and_colors(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        occupied = np.argwhere(self.occupied_mask())
        if len(occupied) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        centers = self.index_to_world(occupied)
        mean_color = self.mean_color()
        colors = mean_color[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        weights = self.weight[occupied[:, 0], occupied[:, 1], occupied[:, 2]]
        return centers, colors, weights

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            bounds_min=np.asarray(self.spec.bounds_min, dtype=np.float32),
            bounds_max=np.asarray(self.spec.bounds_max, dtype=np.float32),
            resolution=np.asarray(self.spec.resolution, dtype=np.int32),
            color_sum=self.color_sum,
            weight=self.weight,
            occupancy=self.occupancy,
            confidence=self.confidence,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "VoxelGrid":
        with np.load(path) as data:
            spec = VoxelGridSpec(
                bounds_min=tuple(data["bounds_min"].tolist()),
                bounds_max=tuple(data["bounds_max"].tolist()),
                resolution=tuple(int(v) for v in data["resolution"].tolist()),
            )
            grid = cls(spec)
            grid.color_sum = data["color_sum"]
            grid.weight = data["weight"]
            grid.occupancy = data["occupancy"]
            grid.confidence = data["confidence"]
            return grid
