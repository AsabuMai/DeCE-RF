# RF h-Edit Project Map

This file is the current entry point for the project after the 2026-05-11
cleanup.

## Current Main Line

The active controller branch is:

```text
support_v3_controller_rmsgap
```

The active research direction is:

```text
decoupled clean-estimate edit-preserve control for localized Rectified Flow editing
```

Support-v3 is not the full method. In paper-facing language, it should be
treated as an operation-conditioned control geometry estimator that supplies
the spatial geometry of the decoupled clean displacement:

```text
M_edit, M_core, M_contact/M_ring, M_preserve
```

The main controller keeps the source-conditioned trajectory and applies
\(v_{\mathrm{DeCE}} = v_{\mathrm{src}} - t^{-1}\Delta_t^0\), where the clean
displacement is decomposed into edit and preserve components. The
rmsgap/adaptive path updates the edit and preserve displacement weights online.
`support_v3_fixed` is the baseline, and M22/core-target-transport is currently a
side branch, not the mainline.

## Active Code

- `run_edit_sd3.py`: main SD3 RF editing CLI.
- `edit_cli_args.py`: CLI argument definition separated from runtime logic.
- `edit_preprocess.py`: CLI-side image preprocessing, normalized box helpers,
  structure-mask builders, and proposal-diff masks.
- `sd3_hrec.py`: main RF editing orchestration and ODE controller.
- `recolor_projection.py`: recolor clean-projection, alpha refinement, and clean-estimate debug helpers.
- `sd3_model_ops.py`: SD3 velocity calls, Q/K/V injection, feature extraction,
  linear RF x0 prediction, and source inversion.
- `spatial_masks.py`: mask morphology, component filtering, support fusion,
  mask statistics, and external mask/image loading.
- `guidance_fields.py`: decode-space color/reference losses and guidance-field
  smoothing/projection/RMS limiting.
- `operation_support_v3.py`: operation-aware support proposal.
- `generic_support.py`: generic support utilities used by v1/v2 paths.
- `attention_mask.py`, `energies.py`, `schedules.py`, `clip_text_reward.py`:
  support utilities.
- `scripts/run_wacv_phase1.sh`: paper-facing WACV Phase 1 runner wrapper. Use this first for current paper runs.
- `scripts/run_wacv_phase1_batch.sh`, `scripts/run_sd3_batch.py`: batch execution path that queues matrix commands and reuses one loaded SD3 pipeline for large runs.
- `scripts/run_pretty_matrix.sh`: slim Core-6 matrix executor; task and method definitions are sourced from `scripts/core6_tasks.sh` and `scripts/core6_methods.sh`.
- `scripts/core6_tasks.sh`, `scripts/core6_methods.sh`: paper-facing Core-6 task and method configuration.
- `scripts/evaluate_paper_metrics.py`: current metric evaluator.
- `scripts/make_semantic_mask.py` and reference-mask builders: current support
  and reference preparation utilities.
- `tests/test_operation_support_v3.py`: support-v3 unit coverage.

## Active Documents

- `docs/rf_h_edit_updated_direction.md`: main clean-estimate-control direction.
- `docs/research_direction_assessment_2026-05-11.md`: direction assessment.
- `docs/rf_h_edit_support_v3_math_plan.md`: support-v3 implementation plan.
- `docs/rf_h_edit_support_v3_refinement_and_removal_controller.md`:
  refinement/removal-controller plan.
- `docs/operation_conditioned_support_interface.md`: paper-facing abstraction
  that reframes support-v3 as operation-conditioned support rather than a
  task-specific heuristic.
- `docs/worklog_2026-06-03.md`: E2 baseline/environment worklog and experiment readout.
- `docs/worklog_2026-06-10.md`: T5 cable-knit redesign, decal-reference code fixes, and E1-E5 consolidated report pointer.
- `docs/rmsgap_mainline_mechanism_audit.md`: current rmsgap mainline mechanism
  map and next-step implications.
- `docs/wacv_phase1_code_map.md`: current paper-to-code alignment and cleanup boundary for the redesigned WACV Phase 1 plan.
- `docs/batch_runner_usage.md`: command recipes for SD3 batch generation on Slurm/A100.

Older notes were moved to:

```text
docs/archive_legacy_2026-05-11/
```

## Active Experiments

Current support-v3 experiment pack:

```text
experiments/support_v3_2026-06-02/
```

Important files:

- `strict_fixed_mask_metrics.csv`
- `strict_fixed_mask_metrics_summary.md`
- `strict_visual_human_quick_audit.csv`
- `e2_reduced_rf_comparison_summary.csv`
- `e2_reduced_rf_comparison_summary.md`
- `e2_native_flux_contextual_table.md`
- `e2_support_matched_contextual_table.md`
- `e4_fixed_dece_component_ablation_compact.md`

Older experiment summaries were moved to:

```text
experiments/archive_legacy_2026-05-11/
```

## E2 Baseline Upgrade Note

`experiments/support_v3_2026-06-02/` currently contains completed E2-A SD3-matched RF-native comparison artifacts. E2-B is now a native-backbone contextual RF / FLUX comparison: `rf_solver_edit` (RF-Solver-Edit / RF-Edit, FLUX), `reflex` (ReFlex, FLUX), `fireflow` and `stable_flow` (FLUX rows), plus planned `ot_rf_otip` (OT-RF / OTIP-style) and `dvrf` (DVRF / Delta Velocity RF). Broad cross-backbone language should wait until at least one E2-B contextual candidate is runnable or the blockers are explicitly disclosed.

## Generated Outputs

The current paper plan lives under paper/, with paper/wacv_experiment_design.md and paper/core6_phase1_images_prompts.md as source of truth. The old server-evidence paper material is archived under paper/archive_old_core6_20260602/.

The current seed-10 support-v3 images live under:

```text
outputs/pretty_matrix/<task>/support_v3_controller_rmsgap/seed_10/
```

Current WACV Phase 1 strict tasks:

- cat_crown: implemented T1 attached accessory; replaces dog_sunglasses as the canonical row.
- bowl_apple_inside: implemented T2 container-constrained insertion; seed-10 gate passed.
- tshirt_star: implemented T3 surface decal; replaces mug_heart as the canonical row.
- red_chair_blue: implemented T4 local recolor; strict audit passed.
- pillow_same_color_cable_knit: implemented T5 localized same-color material replacement (entire pillow surface to white cable-knit, model-driven recipe, 2026-06-10); replaces pillow_same_color_corduroy_panel, which stays as a diagnostic candidate.
- backpack_remove_toy_charm: implemented T6 exposed-object removal.

Previous server-evidence tasks such as dog_sunglasses, mug_heart, pillow_blue_stripes, and pillow_vertical_fabric_strip are supplement/diagnostic rows, not replacements for the revised strict Core-6.
`outputs/` is large and generated. Do not treat it as the project entry point.

## Cleanup Rules

- Keep active code and current docs at repo root / `docs/`.
- Put old planning notes in `docs/archive_legacy_2026-05-11/`.
- Put old metric summaries in `experiments/archive_legacy_2026-05-11/`.
- Put old table runners, baseline wrappers, audit helpers, visual-audit initializers, and workshop
  builders in `legacy/cleanup_20260603/scripts/` or `scripts/archive_legacy_2026-05-11/`.
- Keep new experiment packs as dated subdirectories under `experiments/`.
- Do not delete output runs unless they are known failed scratch runs.
