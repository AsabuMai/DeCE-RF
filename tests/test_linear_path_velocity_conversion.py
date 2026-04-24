from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from energies import clean_delta_to_velocity  # noqa: E402


class LinearPathVelocityConversionTest(unittest.TestCase):
    def test_clean_delta_conversion(self):
        x_t = torch.randn(2, 4, 8, 8)
        v = torch.randn_like(x_t)
        t = torch.tensor(0.5)
        x0_hat = x_t - t * v

        delta_x0 = torch.randn_like(x_t) * 0.1
        u = clean_delta_to_velocity(delta_x0, t)

        x0_hat_new = x_t - t * (v + u)
        self.assertTrue(torch.allclose(x0_hat_new - x0_hat, delta_x0, atol=1e-5))

    def test_reconstruction_target(self):
        x_t = torch.randn(2, 4, 8, 8)
        v = torch.randn_like(x_t)
        t = torch.tensor(0.5)
        x0_hat = x_t - t * v

        x_src = torch.randn_like(x_t)
        delta_x0 = x_src - x0_hat
        u = clean_delta_to_velocity(delta_x0, t)
        x0_hat_new = x_t - t * (v + u)
        self.assertTrue(torch.allclose(x0_hat_new, x_src, atol=1e-5))

    def test_target_anchor(self):
        x_t = torch.randn(2, 4, 8, 8)
        x0_src = torch.randn_like(x_t)
        x0_tar = torch.randn_like(x_t)
        t = torch.tensor(0.5)

        delta_x0 = x0_tar - x0_src
        u = clean_delta_to_velocity(delta_x0, t)
        x0_new = x0_src - t * u
        self.assertTrue(torch.allclose(x0_new, x0_tar, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
