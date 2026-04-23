from __future__ import annotations

import unittest

import torch

from world_model.models.diffusion import ConditionalVideoDiffusion


class DiffusionModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.context_rgb = torch.rand(2, 4, 3, 32, 32)
        self.target_rgb = torch.rand(2, 2, 3, 32, 32)
        self.context_poses = torch.eye(4).repeat(2, 4, 1, 1)
        self.target_poses = torch.eye(4).repeat(2, 2, 1, 1)
        self.memory_condition = torch.rand(2, 2, 5, 32, 32)

    def test_no_memory_loss_shape(self) -> None:
        model = ConditionalVideoDiffusion(
            context_frames=4,
            predict_frames=2,
            variant="no_memory",
            model_channels=16,
            diffusion_steps=16,
        )
        loss, metrics = model.training_loss(
            context_rgb=self.context_rgb,
            target_rgb=self.target_rgb,
            context_poses=self.context_poses,
            target_poses=self.target_poses,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertIn("diffusion_loss", metrics)

    def test_sampler_output_shape(self) -> None:
        model = ConditionalVideoDiffusion(
            context_frames=4,
            predict_frames=2,
            variant="no_memory",
            model_channels=16,
            diffusion_steps=16,
        )
        sample, intermediates = model.sample(
            context_rgb=self.context_rgb,
            context_poses=self.context_poses,
            target_poses=self.target_poses,
            sample_steps=4,
            return_intermediates=True,
        )
        self.assertEqual(tuple(sample.shape), (2, 2, 3, 32, 32))
        self.assertGreater(len(intermediates), 0)

    def test_memory_conditioned_path(self) -> None:
        model = ConditionalVideoDiffusion(
            context_frames=4,
            predict_frames=2,
            variant="memory",
            model_channels=16,
            diffusion_steps=16,
        )
        loss, _ = model.training_loss(
            context_rgb=self.context_rgb,
            target_rgb=self.target_rgb,
            context_poses=self.context_poses,
            target_poses=self.target_poses,
            memory_condition=self.memory_condition,
        )
        self.assertGreaterEqual(float(loss.detach()), 0.0)

    def test_tiny_training_step(self) -> None:
        model = ConditionalVideoDiffusion(
            context_frames=4,
            predict_frames=2,
            variant="memory",
            model_channels=16,
            diffusion_steps=16,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss, _ = model.training_loss(
            context_rgb=self.context_rgb,
            target_rgb=self.target_rgb,
            context_poses=self.context_poses,
            target_poses=self.target_poses,
            memory_condition=self.memory_condition,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    unittest.main()
