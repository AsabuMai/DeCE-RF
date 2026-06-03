# Revised Strict Phase 1 Human Quick Visual Audit

Reviewer: Codex quick human-facing screen pass from revised strict task audit grids. Scores are coarse 1-5 and intended for filtering obvious failures before paper figure selection.

## Keep / Use With Confidence

- cat_crown / support_v3_controller_rmsgap: clear crown, stable cat and background.
- bowl_apple_inside / support_v3_controller_rmsgap: apple centered inside bowl; stable scene.
- tshirt_star / support_v3_controller_rmsgap: large red star centered on shirt; stable folds, pose, jeans, and background.
- red_chair_blue / support_v3_controller_rmsgap: blue chair; stable room.
- pillow_vertical_fabric_strip / support_v3_controller_rmsgap: vertical blue silk strip follows pillow perspective, with softened top seam and clean bottom edge; all seeds 10/11/12 pass.
- backpack_remove_toy_charm / support_v3_controller_rmsgap: toy removed; surface/background stable.

## Borderline

- bowl_apple_inside / adaptive_full_generic_support: apple appears, but placement is weak/rim-like.
- pillow_vertical_fabric_strip / adaptive_full_generic_support: scene is stable and a blue vertical seam appears, but coverage is too thin/dark for the canonical T5 result.
- backpack_remove_toy_charm / direct_target: toy mostly removed but deformation/artifacts make it weak.

## Removed From Strict

- dog_sunglasses: DeCE-RF preservation is good, but eyewear placement is too high for a strong main row.
- mug_heart: visually clean, but the edit is too small/weak for the strict main grid; tshirt_star is stronger.
- pillow_blue_stripes: superseded by pillow_vertical_fabric_strip after human review; the new silk-strip task has cleaner coverage, perspective, and boundaries.

## Reject For Paper Figures

- direct_target on cat, bowl, tshirt, chair, pillow_vertical_fabric_strip: semantic miss, severe over-edit, or scene/crop drift.
- adaptive_full_generic_support on cat, tshirt, backpack: target edit/removal missing.

## Metric Files

- Fixed manual/diff masks: experiments/support_v3_2026-06-02/eval_masks/
- Metrics CSV: experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
- Metrics summary: experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
- Visual audit CSV: experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
- Visual audit grid for new T5: experiments/support_v3_2026-06-02/visual_audit/pillow_vertical_fabric_strip_audit_grid.png
