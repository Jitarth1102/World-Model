from __future__ import annotations

import unittest

import torch

from world_model.models.convgru_predictor import NoMemoryPredictor
from world_model.models.world_model import MemoryConditionedWorldModel


class ModelShapeTest(unittest.TestCase):
    def test_nomemory_forward_shape(self) -> None:
        model = NoMemoryPredictor()
        context_rgb = torch.rand(2, 4, 3, 64, 64)
        context_poses = torch.eye(4).repeat(2, 4, 1, 1)
        target_poses = torch.eye(4).repeat(2, 2, 1, 1)
        output = model(context_rgb, context_poses, target_poses)
        self.assertEqual(tuple(output.shape), (2, 2, 3, 64, 64))

    def test_memory_forward_shape(self) -> None:
        model = MemoryConditionedWorldModel()
        context_rgb = torch.rand(2, 4, 3, 64, 64)
        context_poses = torch.eye(4).repeat(2, 4, 1, 1)
        target_poses = torch.eye(4).repeat(2, 2, 1, 1)
        memory_condition = torch.rand(2, 2, 5, 64, 64)
        rgb, depth = model(context_rgb, context_poses, target_poses, memory_condition)
        self.assertEqual(tuple(rgb.shape), (2, 2, 3, 64, 64))
        self.assertEqual(tuple(depth.shape), (2, 2, 1, 64, 64))


if __name__ == "__main__":
    unittest.main()
