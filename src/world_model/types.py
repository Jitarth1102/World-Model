from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    depth_is_radial: bool = True

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "width": self.width,
            "height": self.height,
            "depth_is_radial": self.depth_is_radial,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraIntrinsics":
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            width=int(data["width"]),
            height=int(data["height"]),
            depth_is_radial=bool(data.get("depth_is_radial", True)),
        )


@dataclass
class ClipSample:
    video: np.ndarray
    depth: np.ndarray
    poses: np.ndarray
    intrinsics: CameraIntrinsics
    segmentations: np.ndarray | None = None
    visibility: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.video.ndim != 4 or self.video.shape[-1] != 3:
            raise ValueError(f"video must have shape [T,H,W,3], got {self.video.shape}")
        if self.depth.ndim != 3:
            raise ValueError(f"depth must have shape [T,H,W], got {self.depth.shape}")
        if self.video.shape[:3] != self.depth.shape:
            raise ValueError("video and depth shapes do not align")
        if self.poses.shape != (self.video.shape[0], 4, 4):
            raise ValueError(f"poses must have shape [T,4,4], got {self.poses.shape}")
        if self.segmentations is not None and self.segmentations.shape != self.depth.shape:
            raise ValueError("segmentations must match depth shape [T,H,W]")

    @property
    def num_frames(self) -> int:
        return int(self.video.shape[0])

    @property
    def image_size(self) -> tuple[int, int]:
        return int(self.video.shape[1]), int(self.video.shape[2])

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "video": self.video,
            "depth": self.depth.astype(np.float32),
            "poses": self.poses.astype(np.float32),
            "intrinsics": np.array(self.intrinsics.as_dict(), dtype=object),
            "metadata": np.array(self.metadata, dtype=object),
        }
        if self.segmentations is not None:
            payload["segmentations"] = self.segmentations
        if self.visibility is not None:
            payload["visibility"] = self.visibility
        np.savez_compressed(path, **payload)

    @classmethod
    def load_npz(cls, path: str | Path) -> "ClipSample":
        with np.load(path, allow_pickle=True) as data:
            intrinsics = CameraIntrinsics.from_dict(data["intrinsics"].item())
            metadata = data["metadata"].item()
            segmentations = data["segmentations"] if "segmentations" in data else None
            visibility = data["visibility"] if "visibility" in data else None
            return cls(
                video=data["video"],
                depth=data["depth"],
                poses=data["poses"],
                intrinsics=intrinsics,
                segmentations=segmentations,
                visibility=visibility,
                metadata=metadata,
            )
