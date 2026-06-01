from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from run_edit_sd3 import (  # noqa: E402
    apply_y_highpass_texture_restore,
    build_proposal_diff_mask,
    clamp_normalized_box,
    estimate_foreground_head_structure_boxes,
    expand_normalized_box,
    parse_normalized_box,
    parse_word_list,
    save_structure_glasses_mask,
)
from attention_mask import _changed_words, _content_edit_words  # noqa: E402
from scripts.make_semantic_mask import (  # noqa: E402
    _infer_support_plan,
    _should_fallback_large_eyes_anchor,
    support_from_anchor_mask,
)
from sd3_hrec import (  # noqa: E402
    _attention_object_mask_from_map,
    _attention_velocity_object_mask,
    _box_from_mask,
    _build_recolor_clean_projection_image,
    _conservative_attention_box,
    _estimate_recolor_closed_form_alpha,
    _estimate_recolor_boundary_alpha,
    _largest_component_box_from_mask,
    _largest_component_mask_from_mask,
    _mask_binary_area_ratio,
    _top_components_mask_from_mask,
    _velocity_diff_object_mask,
    build_object_contact_masks,
    filter_spatial_mask_components,
    latent_structure_edge_mask,
    masked_chroma_luma_loss,
    masked_recolor_texture_boundary_loss,
    normalized_box_mask_like,
    spatial_mask_stats,
    source_color_similarity_mask,
    translate_spatial_mask,
)
from energies import editing_velocity_surrogate_total  # noqa: E402


class RunEditBoxHelperTest(unittest.TestCase):
    def test_parse_normalized_box_uses_argument_name_in_errors(self):
        self.assertEqual(parse_normalized_box("0.1, 0.2, 0.7, 0.8"), (0.1, 0.2, 0.7, 0.8))
        with self.assertRaisesRegex(ValueError, "--source-inject-mask-box coordinates"):
            parse_normalized_box("0.0,0.0,1.2,1.0", "--source-inject-mask-box")

    def test_parse_word_list_normalizes_and_drops_empty_items(self):
        self.assertEqual(parse_word_list(" Sunglasses, eyes, ,PANDA "), ["sunglasses", "eyes", "panda"])
        self.assertIsNone(parse_word_list(" , "))

    def test_content_edit_words_prioritizes_object_over_modifiers(self):
        _, target_words = _changed_words(
            "A panda is walking in a forest.",
            "A panda wearing small black sunglasses is walking in a forest.",
        )

        self.assertEqual(_content_edit_words(target_words), ["sunglasses"])

    def test_expand_normalized_box_clamps_and_enforces_min_size(self):
        self.assertEqual(clamp_normalized_box((1.2, 0.8, -0.2, 0.1)), (0.0, 0.1, 1.0, 0.8))
        expanded = expand_normalized_box((0.45, 0.45, 0.55, 0.55), min_width=0.4, min_height=0.2)
        self.assertAlmostEqual(expanded[0], 0.3)
        self.assertAlmostEqual(expanded[1], 0.4)
        self.assertAlmostEqual(expanded[2], 0.7)
        self.assertAlmostEqual(expanded[3], 0.6)

    def test_foreground_head_structure_fallback_finds_left_head(self):
        image = Image.new("RGB", (128, 128), (70, 150, 45))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 32, 54, 70), fill=(20, 18, 15))
        draw.ellipse((42, 58, 100, 104), fill=(24, 22, 18))
        draw.rectangle((20, 46, 42, 58), fill=(170, 90, 45))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dog.png"
            image.save(path)
            boxes, meta = estimate_foreground_head_structure_boxes(str(path), max_image_size=128)

        self.assertTrue(meta["structure_found"])
        self.assertEqual(meta["structure_mode"], "foreground_head")
        self.assertEqual(meta["structure_head_side"], "left")
        self.assertIsNotNone(boxes["edit"])
        self.assertLess(boxes["edit"][0], 0.5)

    def test_structure_glasses_mask_supports_rotation(self):
        image = Image.new("RGB", (128, 128), (180, 180, 180))
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.png"
            out = Path(tmpdir) / "mask.png"
            image.save(src)
            save_structure_glasses_mask(
                str(src),
                max_image_size=128,
                edit_box=(0.25, 0.35, 0.75, 0.55),
                output_path=str(out),
                angle_deg=7.0,
            )
            mask = Image.open(out).convert("L")

        self.assertEqual(mask.size, (128, 128))
        self.assertGreater(max(mask.getdata()), 0)

    def test_proposal_diff_mask_extracts_changed_component(self):
        source = Image.new("RGB", (64, 64), (180, 180, 180))
        proposal = source.copy()
        draw = ImageDraw.Draw(proposal)
        draw.rectangle((24, 20, 42, 30), fill=(5, 5, 5))
        proposal_2 = source.copy()
        draw_2 = ImageDraw.Draw(proposal_2)
        draw_2.rectangle((24, 20, 42, 30), fill=(10, 10, 10))
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "source.png"
            prop = Path(tmpdir) / "proposal.png"
            prop_2 = Path(tmpdir) / "proposal_2.png"
            out = Path(tmpdir) / "mask.png"
            source.save(src)
            proposal.save(prop)
            proposal_2.save(prop_2)
            _, meta = build_proposal_diff_mask(
                str(src),
                f"{prop},{prop_2}",
                max_image_size=64,
                output_path=str(out),
                threshold=0.2,
                keep_components=1,
                dilate=1,
                blur=1,
            )
            mask = Image.open(out).convert("L")

        self.assertTrue(meta["proposal_diff_found"])
        self.assertEqual(meta["proposal_diff_consensus_count"], 2)
        self.assertEqual(mask.size, (64, 64))
        self.assertGreater(mask.getpixel((32, 25)), 200)
        self.assertLess(mask.getpixel((5, 5)), 10)

    def test_support_from_anchor_inside_keeps_anchor_mask(self):
        anchor = torch.zeros(10, 12).numpy()
        anchor[4:7, 5:9] = 1.0

        support, meta = support_from_anchor_mask(anchor, "inside")

        self.assertEqual(meta["support_relation"], "inside")
        self.assertEqual(meta["anchor_box_xyxy"], [5, 4, 9, 7])
        self.assertTrue((support == anchor).all())

    def test_support_from_anchor_above_builds_insertion_band(self):
        anchor = torch.zeros(20, 20).numpy()
        anchor[10:16, 7:13] = 1.0

        support, meta = support_from_anchor_mask(
            anchor,
            "above",
            expand_x=0.25,
            band_ratio=0.5,
            overlap_ratio=0.0,
        )

        self.assertEqual(meta["support_relation"], "above")
        self.assertEqual(meta["anchor_box_xyxy"], [7, 10, 13, 16])
        self.assertEqual(meta["support_box_xyxy"], [5, 7, 15, 10])
        self.assertGreater(float(support[8, 10]), 0.9)
        self.assertEqual(float(support[12, 10]), 0.0)

    def test_support_from_anchor_profile_eye_builds_single_lens_on_visible_side(self):
        anchor = torch.zeros(40, 60).numpy()
        anchor[10:30, 20:50] = 0.4
        anchor[14:24, 36:50] = 1.0

        support, meta = support_from_anchor_mask(anchor, "profile_eye", threshold=0.2)

        self.assertEqual(meta["support_relation"], "profile_eye")
        self.assertEqual(meta["profile_eye_side"], "right")
        self.assertGreater(float(support.max()), 0.9)
        self.assertGreater(float(support[:, 35:].sum()), float(support[:, :25].sum()))
        self.assertLess(float((support > 0.2).mean()), 0.08)

    def test_infer_support_plan_maps_headwear_to_subject_head_top_center(self):
        phrase, relation, meta = _infer_support_plan(
            "A cat sitting in grass.",
            "A cat wearing a small golden crown sitting in grass.",
            explicit_phrase=None,
            requested_relation="auto",
        )

        self.assertEqual(phrase, "cat head")
        self.assertEqual(relation, "top_center")
        self.assertIn("crown", meta["semantic_edit_words"])

    def test_infer_support_plan_keeps_manual_phrase_but_auto_relation(self):
        phrase, relation, _ = _infer_support_plan(
            "A panda is walking in a forest.",
            "A panda wearing small black sunglasses is walking in a forest.",
            explicit_phrase="eyes",
            requested_relation="auto",
        )

        self.assertEqual(phrase, "eyes")
        self.assertEqual(relation, "box")

    def test_infer_support_plan_gives_glasses_expanded_box_params(self):
        _, relation, meta = _infer_support_plan(
            "A panda is walking in a forest.",
            "A panda wearing small black sunglasses is walking in a forest.",
            explicit_phrase=None,
            requested_relation="auto",
        )

        self.assertEqual(relation, "box")
        self.assertEqual(meta["semantic_phrase_inferred"], "panda eyes")
        self.assertGreater(meta["support_plan_params"]["expand_x"], 0.0)
        self.assertGreater(meta["support_plan_params"]["expand_y"], 0.0)

    def test_infer_support_plan_maps_color_attribute_to_subject_inside(self):
        phrase, relation, meta = _infer_support_plan(
            "A yellow bus parked in a driveway.",
            "A blue bus parked in a driveway.",
            explicit_phrase=None,
            requested_relation="auto",
        )

        self.assertEqual(phrase, "bus")
        self.assertEqual(relation, "inside")
        self.assertEqual(meta["support_plan_rule"]["type"], "color_attribute")

    def test_large_eyes_anchor_triggers_quality_gate_fallback(self):
        anchor = torch.ones(20, 20).numpy()

        should_fallback, meta = _should_fallback_large_eyes_anchor(
            anchor,
            "rabbit eyes",
            "box",
            area_threshold=0.16,
            box_area_threshold=0.34,
            support_threshold=0.2,
        )

        self.assertTrue(should_fallback)
        self.assertGreater(meta["eyes_anchor_area_ratio"], 0.16)

    def test_small_eyes_anchor_keeps_box_support(self):
        anchor = torch.zeros(20, 20).numpy()
        anchor[8:10, 7:9] = 1.0
        anchor[8:10, 12:14] = 1.0

        should_fallback, meta = _should_fallback_large_eyes_anchor(
            anchor,
            "panda eyes",
            "box",
            area_threshold=0.16,
            box_area_threshold=0.34,
            support_threshold=0.2,
        )

        self.assertFalse(should_fallback)
        self.assertLess(meta["eyes_anchor_area_ratio"], 0.16)


class SD3MaskHelperTest(unittest.TestCase):
    def test_translate_spatial_mask_has_no_wraparound(self):
        mask = torch.zeros(1, 1, 4, 5)
        mask[..., 1, 1] = 1.0

        shifted = translate_spatial_mask(mask, shift_y=0.25, shift_x=0.4)

        self.assertEqual(float(shifted.sum().item()), 1.0)
        self.assertEqual(float(shifted[..., 2, 3].item()), 1.0)
        self.assertEqual(float(shifted[..., 1, 1].item()), 0.0)

    def test_chroma_luma_loss_can_preserve_luma_gradients(self):
        source = torch.zeros(1, 3, 4, 4)
        source[..., :, 2:] = 1.0
        current = torch.full_like(source, 0.5)
        mask = torch.ones(1, 1, 4, 4)
        target_rgb = torch.tensor([0.5, 0.5, 0.5])

        without_edges = masked_chroma_luma_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            luma_gradient_preserve_scale=0.0,
        )
        with_edges = masked_chroma_luma_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            luma_gradient_preserve_scale=1.0,
        )

        self.assertGreater(float(with_edges.item()), float(without_edges.item()))

    def test_chroma_luma_loss_target_chroma_scale_softens_target(self):
        source = torch.full((1, 3, 2, 2), 0.5)
        current = torch.tensor([0.25, 0.33, 0.75], dtype=torch.float32).view(1, 3, 1, 1).expand_as(source)
        mask = torch.ones(1, 1, 2, 2)
        target_rgb = torch.tensor([0.05, 0.16, 0.9])

        full_target = masked_chroma_luma_loss(
            current,
            source,
            target_rgb,
            mask,
            target_chroma_scale=1.0,
            luma_preserve_scale=0.0,
        )
        softened_target = masked_chroma_luma_loss(
            current,
            source,
            target_rgb,
            mask,
            target_chroma_scale=0.5,
            luma_preserve_scale=0.0,
        )

        self.assertLess(float(softened_target.item()), float(full_target.item()))

    def test_recolor_texture_boundary_loss_preserves_highpass_texture(self):
        yy, xx = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
        checker = ((xx + yy) % 2).float().view(1, 1, 8, 8)
        source = checker.expand(1, 3, 8, 8)
        current = torch.full_like(source, 0.5)
        mask = torch.ones(1, 1, 8, 8)
        target_rgb = torch.tensor([0.5, 0.5, 0.5])

        without_texture = masked_recolor_texture_boundary_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            luma_gradient_preserve_scale=0.0,
            texture_preserve_scale=0.0,
        )
        with_texture = masked_recolor_texture_boundary_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            luma_gradient_preserve_scale=0.0,
            texture_preserve_scale=1.0,
            texture_kernel_size=3,
        )

        self.assertGreater(float(with_texture.item()), float(without_texture.item()))

    def test_recolor_texture_boundary_loss_boundary_scale_adds_edge_chroma_pressure(self):
        source = torch.full((1, 3, 6, 6), 0.5)
        target_rgb = torch.tensor([0.9, 0.05, 0.04])
        current = target_rgb.view(1, 3, 1, 1).expand_as(source).clone()
        current[..., 1, 1:5] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
        current[..., 4, 1:5] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
        current[..., 1:5, 1] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
        current[..., 1:5, 4] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1)
        mask = torch.zeros(1, 1, 6, 6)
        mask[..., 1:5, 1:5] = 1.0

        without_boundary = masked_recolor_texture_boundary_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            boundary_chroma_scale=0.0,
            boundary_kernel_size=3,
        )
        with_boundary = masked_recolor_texture_boundary_loss(
            current,
            source,
            target_rgb,
            mask,
            luma_preserve_scale=0.0,
            boundary_chroma_scale=2.0,
            boundary_kernel_size=3,
        )

        self.assertGreater(float(with_boundary.item()), float(without_boundary.item()))

    def test_recolor_boundary_alpha_uses_color_only_near_mask_edge(self):
        source = torch.zeros(1, 3, 7, 7)
        source[:] = torch.tensor([0.0, 0.0, 1.0]).view(1, 3, 1, 1)
        source[..., 2:5, 2:5] = torch.tensor([0.9, 0.05, 0.04]).view(3, 1, 1)
        mask = torch.zeros(1, 1, 7, 7)
        mask[..., 1:6, 1:6] = 1.0

        alpha = _estimate_recolor_boundary_alpha(
            source,
            mask,
            torch.tensor([0.9, 0.05, 0.04]),
            threshold=0.2,
            softness=0.02,
            boundary_kernel_size=3,
        )

        self.assertGreater(float(alpha[..., 3, 3].item()), 0.95)
        self.assertLess(float(alpha[..., 1, 1].item()), 0.05)

    def test_recolor_closed_form_alpha_keeps_known_foreground_and_background(self):
        source = torch.zeros(1, 3, 12, 12)
        source[..., :, :6] = torch.tensor([0.9, 0.05, 0.04]).view(3, 1, 1)
        source[..., :, 6:] = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        mask = torch.zeros(1, 1, 12, 12)
        mask[..., 2:10, 2:10] = 1.0

        alpha = _estimate_recolor_closed_form_alpha(
            source,
            mask,
            boundary_kernel_size=3,
            max_size=32,
            epsilon=1e-7,
            constraint_scale=100.0,
        )

        self.assertGreater(float(alpha[..., 5, 5].item()), 0.95)
        self.assertLess(float(alpha[..., 0, 0].item()), 0.05)

    def test_yuv_texture_projection_matches_soft_without_chroma_texture(self):
        source = torch.rand(1, 3, 8, 8, generator=torch.Generator().manual_seed(0))
        alpha = torch.ones(1, 1, 8, 8)
        target_rgb = torch.tensor([0.0, 0.0, 1.0])

        soft = _build_recolor_clean_projection_image(
            source,
            source,
            None,
            target_rgb,
            alpha,
            mode="soft",
            texture_kernel_size=3,
            luma_texture_scale=1.0,
            chroma_texture_scale=0.0,
            composite_mode="blend",
            background_kernel_size=3,
        )
        yuv_texture = _build_recolor_clean_projection_image(
            source,
            source,
            None,
            target_rgb,
            alpha,
            mode="yuv_texture",
            texture_kernel_size=3,
            luma_texture_scale=1.0,
            chroma_texture_scale=0.0,
            composite_mode="blend",
            background_kernel_size=3,
        )

        self.assertLess(float((soft - yuv_texture).abs().max().item()), 1e-6)

    def test_yuv_texture_projection_keeps_source_chroma_highpass(self):
        source = torch.full((1, 3, 8, 8), 0.5)
        source[..., 2::2, 2::2] = torch.tensor([0.9, 0.05, 0.04]).view(3, 1, 1)
        source[..., 1::2, 1::2] = torch.tensor([0.4, 0.2, 0.9]).view(3, 1, 1)
        alpha = torch.ones(1, 1, 8, 8)
        target_rgb = torch.tensor([0.0, 0.0, 1.0])

        soft = _build_recolor_clean_projection_image(
            source,
            source,
            None,
            target_rgb,
            alpha,
            mode="soft",
            texture_kernel_size=3,
            luma_texture_scale=1.0,
            chroma_texture_scale=0.0,
            composite_mode="blend",
            background_kernel_size=3,
        )
        yuv_texture = _build_recolor_clean_projection_image(
            source,
            source,
            None,
            target_rgb,
            alpha,
            mode="yuv_texture",
            texture_kernel_size=3,
            luma_texture_scale=1.0,
            chroma_texture_scale=0.75,
            composite_mode="blend",
            background_kernel_size=3,
        )

        self.assertGreater(float((soft - yuv_texture).abs().mean().item()), 1e-3)

    def test_y_highpass_texture_restore_preserves_result_chroma(self):
        source = Image.new("RGB", (8, 8), (128, 128, 128))
        draw = ImageDraw.Draw(source)
        for x in range(0, 8, 2):
            draw.line((x, 0, x, 7), fill=(230, 230, 230))
        result = Image.new("RGB", (8, 8), (20, 60, 210))
        mask = Image.new("L", (8, 8), 255)

        restored = apply_y_highpass_texture_restore(
            result,
            source,
            mask,
            strength=1.0,
            kernel_size=3,
        )
        restored_tensor = torch.frombuffer(bytearray(restored.tobytes()), dtype=torch.uint8).view(8, 8, 3).float()

        self.assertGreater(float(restored_tensor[..., 2].mean().item()), float(restored_tensor[..., 0].mean().item()))
        self.assertGreater(float(restored_tensor[..., 0].std().item()), 1.0)

    def test_source_color_mask_luma_gate_suppresses_dark_details(self):
        source = torch.zeros(1, 3, 2, 2)
        source[..., 0, 0] = torch.tensor([0.9, 0.8, 0.05])
        source[..., 0, 1] = torch.tensor([0.09, 0.08, 0.02])
        source_rgb = torch.tensor([0.9, 0.8, 0.05])

        ungated = source_color_similarity_mask(
            source,
            source_rgb,
            object_mask=None,
            threshold=1.0,
            softness=0.2,
        )
        gated = source_color_similarity_mask(
            source,
            source_rgb,
            object_mask=None,
            threshold=1.0,
            softness=0.2,
            luma_gate_min=0.25,
            luma_gate_softness=0.04,
        )

        self.assertGreater(float(gated[..., 0, 0].item()), 0.8 * float(ungated[..., 0, 0].item()))
        self.assertLess(float(gated[..., 0, 1].item()), 0.2 * float(ungated[..., 0, 1].item()))

    def test_source_color_mask_detail_protect_suppresses_edges(self):
        source = torch.full((1, 3, 4, 4), 0.8)
        source[..., :, 2:] = 0.2
        source_rgb = torch.tensor([0.8, 0.8, 0.8])

        unprotected = source_color_similarity_mask(
            source,
            source_rgb,
            object_mask=None,
            threshold=1.0,
            softness=0.2,
        )
        protected = source_color_similarity_mask(
            source,
            source_rgb,
            object_mask=None,
            threshold=1.0,
            softness=0.2,
            detail_protect_scale=0.9,
            detail_protect_threshold=0.2,
            detail_protect_softness=0.04,
        )

        self.assertLess(float(protected[..., 1, 2].item()), 0.5 * float(unprotected[..., 1, 2].item()))
        self.assertGreater(float(protected[..., 1, 0].item()), 0.8 * float(unprotected[..., 1, 0].item()))

    def test_object_contact_masks_split_edit_support(self):
        edit = torch.zeros(1, 1, 7, 7)
        edit[..., 3, 3] = 1.0

        subject, obj, contact, preserve, edge = build_object_contact_masks(
            edit,
            object_threshold=0.45,
            contact_dilate_kernel=3,
            contact_scale=0.25,
        )

        self.assertIsNone(edge)
        self.assertEqual(float(obj[..., 3, 3].item()), 1.0)
        self.assertEqual(float(contact[..., 3, 3].item()), 0.0)
        self.assertEqual(float(subject[..., 3, 3].item()), 1.0)
        self.assertEqual(float(subject[..., 2, 3].item()), 0.25)
        self.assertEqual(float(preserve[..., 0, 0].item()), 1.0)
        self.assertEqual(float(preserve[..., 2, 3].item()), 0.0)

    def test_contact_mask_respects_source_structure_edges(self):
        edit = torch.zeros(1, 1, 7, 7)
        edit[..., 3, 3] = 1.0
        reference = torch.zeros(1, 2, 7, 7)
        reference[..., :, 3:] = 10.0

        edge = latent_structure_edge_mask(reference, threshold=0.2, soften_kernel=1)
        subject, obj, contact, preserve, edge_from_layering = build_object_contact_masks(
            edit,
            structure_reference=reference,
            object_threshold=0.45,
            contact_dilate_kernel=3,
            contact_scale=0.25,
            contact_edge_threshold=0.2,
            contact_edge_protect_scale=1.0,
        )

        self.assertIsNotNone(edge_from_layering)
        self.assertGreater(float(edge[..., 3, 3].item()), 0.5)
        self.assertLess(float(contact[..., 2, 3].item()), 1.0)
        self.assertGreater(float(preserve[..., 2, 3].item()), 0.3)
        self.assertEqual(float(subject[..., 3, 3].item()), 1.0)

    def test_filter_spatial_mask_components_keeps_largest_matching_component(self):
        mask = torch.zeros(1, 1, 6, 6)
        mask[..., 0, 0] = 0.9
        mask[..., 3:5, 3:5] = 0.8

        filtered = filter_spatial_mask_components(mask, threshold=0.5, keep_components=1, center_y_min=0.4)

        self.assertEqual(float(filtered[..., 0, 0].item()), 0.0)
        self.assertAlmostEqual(float(filtered[..., 3:5, 3:5].sum().item()), 3.2, places=5)

    def test_attention_object_mask_uses_largest_high_response_component(self):
        attention = torch.zeros(1, 1, 6, 6)
        attention[..., 0, 0] = 0.95
        attention[..., 3:5, 3:5] = 0.8

        component = _largest_component_mask_from_mask(attention, threshold=0.7)
        object_mask = _attention_object_mask_from_map(attention, threshold=0.7, quantile=0.0)

        self.assertEqual(float(component[..., 0, 0].item()), 0.0)
        self.assertAlmostEqual(float(component[..., 3:5, 3:5].sum().item()), 3.2, places=5)
        self.assertEqual(float(object_mask[..., 0, 0].item()), 0.0)
        self.assertGreater(float(object_mask[..., 3:5, 3:5].min().item()), 0.99)

    def test_top_components_mask_keeps_multiple_components(self):
        attention = torch.zeros(1, 1, 6, 6)
        attention[..., 0, 0] = 0.95
        attention[..., 2, 2] = 0.85
        attention[..., 4, 4] = 0.75

        top_two = _top_components_mask_from_mask(attention, threshold=0.7, keep_components=2)

        self.assertAlmostEqual(float(top_two[..., 0, 0].item()), 0.95, places=5)
        self.assertAlmostEqual(float(top_two[..., 2, 2].item()), 0.85, places=5)
        self.assertEqual(float(top_two[..., 4, 4].item()), 0.0)

    def test_attention_object_mask_can_keep_multiple_components(self):
        attention = torch.zeros(1, 1, 6, 6)
        attention[..., 1, 1] = 0.9
        attention[..., 4, 4] = 0.8

        object_mask = _attention_object_mask_from_map(
            attention,
            threshold=0.7,
            quantile=0.0,
            keep_components=2,
        )

        self.assertGreater(float(object_mask[..., 1, 1].item()), 0.99)
        self.assertGreater(float(object_mask[..., 4, 4].item()), 0.85)

    def test_velocity_diff_object_mask_uses_internal_velocity_change(self):
        src_v = torch.zeros(1, 2, 6, 6)
        tar_v = torch.zeros(1, 2, 6, 6)
        tar_v[..., 0, 0] = 3.0
        tar_v[..., 3:5, 3:5] = 2.0

        object_mask = _velocity_diff_object_mask(src_v, tar_v, threshold=0.5, quantile=0.0)

        self.assertEqual(float(object_mask[..., 0, 0].item()), 0.0)
        self.assertGreater(float(object_mask[..., 3:5, 3:5].min().item()), 0.99)

    def test_attention_velocity_object_mask_uses_velocity_when_attention_is_broad(self):
        attention = torch.ones(1, 1, 6, 6)
        velocity = torch.zeros(1, 1, 6, 6)
        velocity[..., 3:5, 3:5] = 1.0

        fused = _attention_velocity_object_mask(attention, velocity, max_attention_area=0.2)

        self.assertLess(float(fused[..., 0, 0].item()), 1e-4)
        self.assertGreater(float(fused[..., 3:5, 3:5].min().item()), 0.7)

    def test_attention_velocity_object_mask_unions_compact_supports(self):
        attention = torch.zeros(1, 1, 6, 6)
        velocity = torch.zeros(1, 1, 6, 6)
        attention[..., 2, 2] = 1.0
        velocity[..., 3, 3] = 1.0

        fused = _attention_velocity_object_mask(attention, velocity)

        self.assertGreater(float(fused[..., 2, 2].item()), 0.7)
        self.assertGreater(float(fused[..., 3, 3].item()), 0.7)
        self.assertGreater(float(fused[..., 2:4, 2:4].min().item()), 0.7)

    def test_attention_velocity_object_mask_can_keep_sparse_pixels_for_ablation(self):
        attention = torch.zeros(1, 1, 6, 6)
        velocity = torch.zeros(1, 1, 6, 6)
        attention[..., 2, 2] = 1.0
        velocity[..., 3, 3] = 1.0

        fused = _attention_velocity_object_mask(attention, velocity, continuous_support=False)

        self.assertEqual(float(fused[..., 2, 2].item()), 1.0)
        self.assertEqual(float(fused[..., 3, 3].item()), 1.0)
        self.assertEqual(float(fused[..., 2, 3].item()), 0.0)

    def test_normalized_box_mask_like_builds_soft_bchw_mask(self):
        reference = torch.zeros(2, 4, 5, 5)
        mask = normalized_box_mask_like(reference, (0.25, 0.25, 0.75, 0.75), feather=0.01)

        self.assertEqual(mask.shape, (2, 1, 5, 5))
        self.assertGreater(float(mask[0, 0, 2, 2].item()), 0.9)
        self.assertLess(float(mask[0, 0, 0, 0].item()), 0.01)

    def test_box_from_mask_and_stats_use_normalized_coordinates(self):
        mask = torch.zeros(1, 1, 5, 5)
        mask[..., 1:4, 2:5] = 1.0

        self.assertEqual(_box_from_mask(mask, threshold=0.5), (0.5, 0.25, 1.0, 0.75))

        stats = spatial_mask_stats(mask, prefix="edit")
        self.assertAlmostEqual(stats["edit_area_ratio"], 9 / 25)
        self.assertAlmostEqual(stats["edit_center_x"], 0.75)
        self.assertAlmostEqual(stats["edit_center_y"], 0.5)
        self.assertEqual(stats["edit_bbox_x0"], 0.5)
        self.assertEqual(stats["edit_bbox_y0"], 0.25)

    def test_box_from_mask_returns_fallback_when_empty(self):
        fallback = (0.1, 0.2, 0.3, 0.4)
        self.assertEqual(_box_from_mask(torch.zeros(1, 1, 3, 3), fallback=fallback), fallback)

    def test_largest_component_box_ignores_smaller_components(self):
        mask = torch.zeros(1, 1, 6, 6)
        mask[..., 0, 0] = 1.0
        mask[..., 3:5, 2:5] = 0.8

        self.assertAlmostEqual(_mask_binary_area_ratio(mask), 7 / 36)
        self.assertEqual(_largest_component_box_from_mask(mask, threshold=0.5), (0.4, 0.6, 0.8, 0.8))

    def test_conservative_attention_box_uses_high_response_component(self):
        attention = torch.zeros(1, 1, 8, 8)
        attention[..., 1:7, 1:7] = 0.45
        attention[..., 4:6, 5:7] = 1.0

        box = _conservative_attention_box(attention, threshold=0.7, quantile=0.8)

        self.assertEqual(box, (5 / 7, 4 / 7, 6 / 7, 5 / 7))

    def test_editing_base_velocity_is_masked(self):
        base = torch.ones(1, 2, 4, 4)
        x0_tar = torch.zeros_like(base)
        x0_src = torch.zeros_like(base)
        mask = torch.zeros(1, 1, 4, 4)
        mask[..., 1:3, 1:3] = 1.0

        terms = editing_velocity_surrogate_total(
            base_edit_velocity=base,
            x0_tar=x0_tar,
            x0_src=x0_src,
            t_scalar=torch.tensor(0.5),
            M_edit=mask,
            lambda_base=1.0,
            lambda_anchor=0.0,
            lambda_region=0.0,
        )

        self.assertEqual(float(terms["base"][..., 0, 0].sum().item()), 0.0)
        self.assertEqual(float(terms["base"][..., 1, 1].sum().item()), 2.0)


if __name__ == "__main__":
    unittest.main()
