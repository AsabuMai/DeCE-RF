# RF h-Edit Project

This repository is a research sandbox for Rectified Flow image editing.

Current working paper title:

```text
Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing
```

Current method name:

```text
DeCE-RF
```

The current direction is no longer a broad claim that a reconstruction/editing
velocity split is sufficient. The active research question is narrower:

```text
support-aware clean-estimate control for Rectified Flow image editing
```

The working hypothesis is that local RF editing is limited by both velocity
coupling and spatial support quality. The method separates:

- edit and reconstruction velocity corrections;
- an operation-conditioned support interface that estimates where each
  correction should act;
- a clean-estimate-space adaptive controller that measures edit progress and
  preservation drift along the ODE trajectory.

## Active Code

- `run_edit_sd3.py`  
  Main CLI entrypoint for SD3 RF editing experiments.

- `edit_preprocess.py`  
  Image preprocessing, normalized box helpers, structure-mask builders, and
  proposal-diff mask construction used by the CLI.

- `sd3_hrec.py`  
  Main SD3 RF editing orchestration and ODE controller.

- `sd3_model_ops.py`  
  SD3-specific velocity evaluation, Q/K/V injection, feature extraction,
  linear RF x0 prediction, and source inversion helpers.

- `spatial_masks.py`  
  Spatial mask morphology, component filtering, mask statistics, external
  mask/image loading, and attention/velocity support fusion.

- `guidance_fields.py`  
  Decode-space color/reference losses, guidance smoothing/projection, and RMS
  limiting helpers.

- `generic_support.py`  
  Generic / operation-aware support proposal utilities.

- `attention_mask.py`, `energies.py`, `schedules.py`, `clip_text_reward.py`  
  Supporting utilities for masks, guidance terms, schedules, and reward
  gradients.

## Project Map

Start here for the current file layout:

- `PROJECT_MAP.md`

## Active Research Notes

- `docs/research_direction_assessment_2026-05-11.md`  
  Current paper-direction assessment and recommended narrowing.

- `docs/rf_h_edit_updated_direction.md`  
  Reframes the project around clean-estimate-space local control.

- `docs/rf_h_edit_support_v3_math_plan.md`  
  Mathematical implementation plan for operation-aware support-v3.

- `docs/rf_h_edit_support_v3_refinement_and_removal_controller.md`  
  Latest support-v3 refinement and optional removal-controller plan.

- `docs/operation_conditioned_support_interface.md`  
  Paper-facing reframing of support-v3 as an operation-conditioned control
  geometry interface rather than a task-specific heuristic.

- `docs/worklog_2026-05-11.md`  
  Latest worklog.

## Active Experiments

- `experiments/support_v3_2026-05-11/`  
  Current support-v3 experiment pack, including refinement metrics, mug
  variants, removal-controller variants, and visual/debug artifacts.

Older experiment summaries were moved to
`experiments/archive_legacy_2026-05-11/`.

## Main Scripts

Run or reproduce the current support-v3 matrix:

```bash
scripts/run_pretty_matrix.sh
```

Initialize visual-audit templates used by tests / legacy comparisons:

```bash
scripts/init_pretty_visual_audit.py
scripts/init_baseline_parity_manifest.py
```

Evaluate current outputs:

```bash
scripts/evaluate_paper_metrics.py
```

Build current comparison artifacts:

```bash
scripts/make_comparison_grid.py
```

Build support and reference masks:

```bash
scripts/make_semantic_mask.py
scripts/make_source_color_mask.py
scripts/make_surface_recolor_reference.py
scripts/make_decal_reference.py
scripts/make_glasses_reference.py
scripts/make_mask_badge_reference.py
scripts/refine_surface_mask.py
scripts/subtract_masks.py
```

## Current Research Position

The current evidence supports a conservative claim:

```text
Clean-estimate-space adaptive control improves preservation when the support
region is reasonable, but automatic support proposal remains the bottleneck.
```

It does not yet support a broad claim of general automatic RF image editing.

The support-v3 seed-10 gate has been run. Current readout:

- `cat_crown`: support-v3 is a strong relation-support positive case.
- `dog_sunglasses`: support-v3 is a non-regression positive control.
- `mug_heart`: support is usable, but preservation/controller leakage remains.
- `backpack_remove_toy_charm`: support is compact, but removal dynamics remain
  weak.

## Paper Materials

The current manuscript materials live in `paper/`:

- `paper/outline.md`
- `paper/manuscript.md`
- `paper/draft.md`
- `paper/results.md`
- `paper/tables.md`
- `paper/figures.md`
- `paper/limitations.md`
- `paper/references.bib`
- `paper/experiment_plan.md`

These materials should be updated after the support-v3 gate. The paper claim
should stay narrow until automatic support is stable across seeds.

## Data and Outputs

- `data/` contains tracked source images used by the experiment scripts.
- `outputs/`, `results/`, and `workshop_materials/` are generated artifacts and
  are ignored by Git.

## Legacy Material

Older panda / sunglasses probes, early CLI paths, and previous planning notes
were moved out of the active path:

- `scripts/archive_legacy_2026-05-11/`
- `docs/archive_legacy_2026-05-11/`
- `experiments/archive_legacy_2026-05-11/`
- `legacy/2026-05-11/`

These files are kept for traceability but are not the current project entry
points.
