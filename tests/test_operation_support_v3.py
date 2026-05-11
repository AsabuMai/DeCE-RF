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
    build_core_ring_preserve_masks,
    build_operation_support_v3,
    build_surface_region,
    compute_clean_disagreement,
    compute_velocity_disagreement,
    default_candidate_for_operation,
    parse_edit_operation,
    save_support_debug,
    support_overlap_metrics,
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
        self.assertEqual(parse_edit_operation("decal"), "add_decal")
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

    def test_clean_velocity_wrappers_and_core_ring_masks(self):
        x_t = torch.zeros(1, 4, 8, 8)
        source_v = torch.zeros_like(x_t)
        target_v = torch.ones_like(x_t)
        clean = compute_clean_disagreement(x_t, torch.tensor(0.5), source_v, target_v)
        velocity = compute_velocity_disagreement(source_v, target_v)
        self.assertEqual(clean.shape, (1, 1, 8, 8))
        self.assertEqual(velocity.shape, (1, 1, 8, 8))

        core_input = torch.zeros(1, 1, 8, 8)
        core_input[:, :, 3:5, 3:5] = 1.0
        core, ring, preserve = build_core_ring_preserve_masks(core_input, ring_dilate_radius=3)
        self.assertGreater(float(core.sum()), 0.0)
        self.assertGreater(float(ring.sum()), 0.0)
        self.assertLess(float(preserve[0, 0, 4, 4]), 0.1)

    def test_overlap_metrics_reports_iou_coverage_leakage(self):
        pred = torch.zeros(1, 1, 8, 8)
        ref = torch.zeros(1, 1, 8, 8)
        pred[:, :, 2:6, 2:6] = 1.0
        ref[:, :, 3:7, 3:7] = 1.0

        metrics = support_overlap_metrics(pred, ref)

        self.assertGreater(metrics["support_iou"], 0.0)
        self.assertLess(metrics["support_iou"], 1.0)
        self.assertGreater(metrics["support_coverage"], 0.0)
        self.assertGreater(metrics["support_leakage"], 0.0)

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

    def test_save_support_debug_writes_candidate_maps(self):
        with self.subTest("debug output"):
            import tempfile

            x_t = torch.zeros(1, 4, 8, 8)
            source_v = torch.zeros_like(x_t)
            target_v = torch.ones_like(x_t)
            attention = torch.ones(1, 1, 8, 8)
            result = build_operation_support_v3(
                attention_map=attention,
                x_t=x_t,
                t=torch.tensor(0.5),
                source_velocity=source_v,
                target_velocity=target_v,
                candidate="attention_x_clean",
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                save_support_debug(result, tmpdir, max_candidates=2)
                self.assertTrue((Path(tmpdir) / "operation_v3_support_score.png").exists())


if __name__ == "__main__":
    unittest.main()
