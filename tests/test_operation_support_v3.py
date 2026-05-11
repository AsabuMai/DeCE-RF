from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from operation_support_v3 import (  # noqa: E402
    build_above_host_region,
    build_operation_support_v3,
    build_surface_region,
    default_candidate_for_operation,
)


class OperationSupportV3Test(unittest.TestCase):
    def test_above_host_region_builds_relation_box(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 10:16, 7:13] = 1.0

        relation = build_above_host_region(host, expand_x=0.25, above_height=0.5, overlap_height=0.0)

        self.assertGreater(float(relation[0, 0, 8, 10]), 0.9)
        self.assertEqual(float(relation[0, 0, 13, 10]), 0.0)
        self.assertLess(float(relation.mean()), 0.20)

    def test_surface_region_keeps_host_center(self):
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 5:17, 4:16] = 1.0

        surface = build_surface_region(host, erode_kernel=1, center_width=0.5, center_height=0.5)

        self.assertGreater(float(surface[0, 0, 11, 10]), 0.9)
        self.assertEqual(float(surface[0, 0, 5, 4]), 0.0)
        self.assertLess(float(surface.mean()), float(host.mean()))

    def test_default_candidate_is_operation_aware(self):
        self.assertEqual(
            default_candidate_for_operation("add_object", relation="above_host", has_relation=True),
            "relation_x_response",
        )
        self.assertEqual(
            default_candidate_for_operation("add_decal", relation="on_surface", has_grounding=True),
            "host_surface_x_response",
        )
        self.assertEqual(
            default_candidate_for_operation("remove_object", has_grounding=True),
            "seg_x_response",
        )

    def test_build_operation_support_v3_selects_relation_candidate(self):
        x_t = torch.zeros(1, 4, 20, 20)
        source_v = torch.zeros_like(x_t)
        target_v = torch.zeros_like(x_t)
        target_v[:, :, 7:11, 6:14] = 2.0
        attention = torch.zeros(1, 1, 20, 20)
        attention[:, :, 7:11, 8:12] = 1.0
        host = torch.zeros(1, 1, 20, 20)
        host[:, :, 11:17, 7:13] = 1.0

        result = build_operation_support_v3(
            attention_map=attention,
            x_t=x_t,
            t=torch.tensor(0.5),
            source_velocity=source_v,
            target_velocity=target_v,
            host_attention_map=host,
            edit_operation="add_object",
            relation="above_host",
            candidate="operation_default",
            top_percentile=80,
            keep_components=1,
            dilate_radius=1,
            blur_kernel=1,
        )

        self.assertEqual(result.stats["support_mode"], "operation_v3")
        self.assertEqual(result.stats["support_score"], "relation_x_response")
        self.assertIsNotNone(result.relation_map)
        self.assertGreater(float(result.edit_mask[0, 0, 8, 10]), 0.0)


if __name__ == "__main__":
    unittest.main()
