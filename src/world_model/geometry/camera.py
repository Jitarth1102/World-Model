from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world_model.types import CameraIntrinsics


def normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / np.clip(norm, eps, None)


def quaternion_wxyz_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(value) for value in quaternion]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def pose_from_kubric_camera(position: np.ndarray, quaternion_wxyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = quaternion_wxyz_to_rotation_matrix(quaternion_wxyz)
    pose[:3, 3] = np.asarray(position, dtype=np.float32)
    return pose


def intrinsics_from_kubric(
    focal_length: float,
    sensor_width: float,
    width: int,
    height: int,
) -> CameraIntrinsics:
    sensor_height = sensor_width * (height / width)
    fx = focal_length / sensor_width * width
    fy = focal_length / sensor_height * height
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height, depth_is_radial=True)


def pixel_centers(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    return np.meshgrid(xs, ys, indexing="xy")


def ray_directions_camera(intrinsics: CameraIntrinsics) -> np.ndarray:
    u, v = pixel_centers(intrinsics.height, intrinsics.width)
    x = (u - intrinsics.cx) / intrinsics.fx
    y = -(v - intrinsics.cy) / intrinsics.fy
    z = -np.ones_like(x)
    rays = np.stack([x, y, z], axis=-1)
    return normalize(rays)


def depth_to_camera_points(depth: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    if depth.shape != (intrinsics.height, intrinsics.width):
        raise ValueError(f"depth shape {depth.shape} does not match intrinsics {(intrinsics.height, intrinsics.width)}")
    rays = ray_directions_camera(intrinsics)
    return rays * depth[..., None]


def camera_to_world_points(points_camera: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return points_camera @ rotation.T + translation


def world_to_camera_points(points_world: np.ndarray, camera_to_world: np.ndarray) -> np.ndarray:
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return (points_world - translation) @ rotation


def depth_to_world_points(depth: np.ndarray, camera_to_world: np.ndarray, intrinsics: CameraIntrinsics) -> np.ndarray:
    camera_points = depth_to_camera_points(depth, intrinsics)
    return camera_to_world_points(camera_points.reshape(-1, 3), camera_to_world).reshape(camera_points.shape)


def project_points(points_world: np.ndarray, camera_to_world: np.ndarray, intrinsics: CameraIntrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera_points = world_to_camera_points(points_world, camera_to_world)
    forward_depth = -camera_points[:, 2]
    valid = forward_depth > 1e-6
    u = intrinsics.cx + intrinsics.fx * (camera_points[:, 0] / np.clip(forward_depth, 1e-6, None))
    v = intrinsics.cy - intrinsics.fy * (camera_points[:, 1] / np.clip(forward_depth, 1e-6, None))
    pixels = np.stack([u, v], axis=-1)
    radial_depth = np.linalg.norm(camera_points, axis=-1)
    return pixels, forward_depth, radial_depth * valid


@dataclass(frozen=True)
class CameraPose:
    matrix: np.ndarray


def look_at_pose(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray | None = None) -> CameraPose:
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    forward = normalize(target - eye)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    right = normalize(right)
    up = normalize(np.cross(right, forward))

    # Kubric cameras use local front = -Z and up = +Y.
    rotation = np.stack([right, up, -forward], axis=1).astype(np.float32)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = rotation
    pose[:3, 3] = eye
    return CameraPose(matrix=pose)
