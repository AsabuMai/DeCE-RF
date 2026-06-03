# WACV Phase 1 Code Map

Current paper source of truth:

- `paper/wacv_experiment_design.md`
- `paper/core6_phase1_images_prompts.md`

## Minimal Paper Runner

Use this wrapper instead of calling `scripts/run_pretty_matrix.sh` directly for WACV Phase 1 work:

```bash
SCOPE=implemented bash scripts/run_wacv_phase1.sh
```

Supported scopes:

- `implemented`: runnable updated-task subset only.
- `strict`: revised strict Core-6 with `cat_crown` for T1 and `tshirt_star` for T3; T2/T5 are implemented and human-reviewed as passing.
- `old_server_evidence`: previous server-evidence task set; diagnostic/supplement only.

## Paper-To-Code Task Alignment

| WACV row | Paper task | Runner status | Runner task | Notes |
| --- | --- | --- | --- | --- |
| T1 attached accessory | `cat_crown` | implemented; visual pass | `cat_crown` / `P1` | Canonical T1 after dog sunglasses placement failed quick audit. |
| T2 container insertion | `bowl_apple_inside` | implemented; seed-10 pass | `bowl_apple_inside` | Uses `inside_container` relation and an empty blue ceramic bowl source for a cleaner interior-placement gate. |
| T3 surface decal | `tshirt_star` | implemented; visual pass | `tshirt_star` / `P5` | Canonical T3 after mug heart was too small/weak for the main grid. |
| T4 local recolor | `red_chair_blue` | implemented | `red_chair_blue` / `P4` | Keep after visual audit confirms local recolor. |
| T5 surface material strip | `pillow_vertical_fabric_strip` | implemented; seeds 10/11/12 human pass | `pillow_vertical_fabric_strip` | Uses a perspective-aligned blue silk strip reference on the pillow surface; supersedes the earlier `pillow_blue_stripes` probe. |
| T6 exposed removal | `backpack_remove_toy_charm` | implemented | `backpack_remove_toy_charm` / `P7` | Canonical T6 with zipper/fabric caveat. |

## E2 Baseline Upgrade Note

Current E2 artifacts cover E2-A, the SD3-matched RF-native / target-mode comparison against FlowEdit, FlowAlign, SplitFlow, and DeCE-RF. The redesigned E2-B pool is a native-backbone contextual RF / FLUX comparison covering `rf_solver_edit`, `reflex`, `fireflow`, `stable_flow`, `ot_rf_otip`, and `dvrf`; at least one contextual baseline must become runnable, or its blocker must be disclosed, before making the stronger practical claim that existing RF editors do not replace localized edit-preserve control.

## Minimal Method Set

Paper-facing E1 methods:

- `base_only`
- `direct_target`
- `adaptive_full_generic_support`
- `support_v3_controller_rmsgap`

E4 ablation methods:

- `support_v3_fixed`
- `support_v3_controller_rmsgap`
- selected controller variants only after the E1 signal is stable.

## Active Code Surface

Keep these as active for Phase 1:

- `run_edit_sd3.py`: CLI entrypoint.
- `edit_cli_args.py`: CLI argument definition separated from runtime logic.
- `sd3_hrec.py`: controller/runtime implementation.
- `recolor_projection.py`: recolor projection and clean-estimate debug helpers used by the controller.
- `sd3_model_ops.py`: SD3 velocity and RF helpers.
- `operation_support_v3.py`: operation-conditioned geometry.
- `spatial_masks.py`, `guidance_fields.py`, `edit_preprocess.py`: mask/guidance/preprocess support.
- `scripts/run_wacv_phase1.sh`: paper-facing runner wrapper.
- `scripts/run_pretty_matrix.sh`: slim Core-6 matrix executor used by the paper wrapper.
- `scripts/core6_tasks.sh`, `scripts/core6_methods.sh`: separated task and method configuration for the revised strict Core-6.
- `scripts/evaluate_paper_metrics.py`, `scripts/make_paper_grids.py`, `scripts/summarize_fixed_mask_audit.py`: current evaluation/grids.

## Parked Or Diagnostic Surface

Do not delete yet, but keep out of the Phase 1 path:

- Replacement routes: `support_v3_controller_rmsgap_replace_*` and replacement tasks.
- Add-editor probes: `support_v3_controller_rmsgap_add_editor_*`, `web_*` insertion tasks.
- Completion-prior removal scripts and whiteboard replacement probes.
- Non-RF baseline setup/wrapper scripts; E2 active code keeps only reduced-comparison preparation and summary.
- Archived paper files under `paper/archive_old_core6_20260602/`.

## Simplification Plan

1. Use `scripts/run_wacv_phase1.sh` for all Phase 1 runs.
2. Use the revised strict Core-6 task list as the frozen Phase 1 starting point: `cat_crown`, `bowl_apple_inside`, `tshirt_star`, `red_chair_blue`, `pillow_vertical_fabric_strip`, and `backpack_remove_toy_charm`.
3. Keep exploratory task/method variants in `legacy/cleanup_20260603/scripts/run_pretty_matrix.full_legacy.sh`; active Core-6 config lives in `scripts/core6_tasks.sh` and `scripts/core6_methods.sh`.
4. Parked diagnostic scripts were moved to `legacy/cleanup_20260603/scripts/`; restore from the cleanup backup only for supplementary experiments.
5. Keep metrics/grids pointed at the strict six-task list in `experiments/support_v3_2026-06-02/`, and move older T2/T5 probes into diagnostic/supplement-only language.
