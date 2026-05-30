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
- `edit_preprocess.py`: CLI-side image preprocessing, normalized box helpers,
  structure-mask builders, and proposal-diff masks.
- `sd3_hrec.py`: main RF editing orchestration and ODE controller.
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
- `scripts/run_pretty_matrix.sh`: current task runner for support-v3 tests.
- `scripts/evaluate_paper_metrics.py`: current metric evaluator.
- `scripts/make_semantic_mask.py` and reference-mask builders: current support
  and reference preparation utilities.
- `scripts/init_pretty_visual_audit.py`,
  `scripts/init_baseline_parity_manifest.py`: retained because the current
  metric tests import them directly.
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
- `docs/worklog_2026-05-11.md`: current worklog and experiment readout.
- `docs/rmsgap_mainline_mechanism_audit.md`: current rmsgap mainline mechanism
  map and next-step implications.

Older notes were moved to:

```text
docs/archive_legacy_2026-05-11/
```

## Active Experiments

Current support-v3 experiment pack:

```text
experiments/support_v3_2026-05-11/
```

Important files:

- `support_v3_refinement_metrics_clip.csv`
- `mug_candidate_compare_metrics_clip.csv`
- `removal_controller_compare_metrics_clip.csv`
- `README.md`

Older experiment summaries were moved to:

```text
experiments/archive_legacy_2026-05-11/
```

## Generated Outputs

The current seed-10 support-v3 images live under:

```text
outputs/pretty_matrix/<task>/adaptive_full_support_v3/seed_10/
```

Main tasks:

- `cat_crown`
- `dog_sunglasses`
- `mug_heart`
- `backpack_remove_toy_charm`

`outputs/` is large and generated. Do not treat it as the project entry point.

## Cleanup Rules

- Keep active code and current docs at repo root / `docs/`.
- Put old planning notes in `docs/archive_legacy_2026-05-11/`.
- Put old metric summaries in `experiments/archive_legacy_2026-05-11/`.
- Put old table runners, baseline wrappers, audit helpers, and workshop
  builders in `scripts/archive_legacy_2026-05-11/`.
- Keep new experiment packs as dated subdirectories under `experiments/`.
- Do not delete output runs unless they are known failed scratch runs.
