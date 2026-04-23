from __future__ import annotations

import unittest

import numpy as np

from world_model.geometry.camera import CameraPose, depth_to_world_points, look_at_pose, project_points
from world_model.types import CameraIntrinsics


class CameraGeometryTest(unittest.TestCase):
    def test_project_and_unproject_center_pixel(self) -> None:
        intrinsics = CameraIntrinsics(fx=80.0, fy=80.0, cx=31.5, cy=31.5, width=64, height=64)
        pose = look_at_pose(np.array([0.0, 0.0, 2.0], dtype=np.float32), np.array([0.0, 0.0, 0.0], dtype=np.float32)).matrix
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[32, 32] = 2.0
        world_points = depth_to_world_points(depth, pose, intrinsics)
        center_point = world_points[32, 32][None, :]
        pixels, z_forward, _ = project_points(center_point, pose, intrinsics)
        self.assertTrue(z_forward[0] > 0.0)
        self.assertAlmostEqual(float(pixels[0, 0]), 32.0, delta=1.0)
        self.assertAlmostEqual(float(pixels[0, 1]), 32.0, delta=1.0)


if __name__ == "__main__":
    unittest.main()
