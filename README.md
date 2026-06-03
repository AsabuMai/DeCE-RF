# RF h-Edit Project

This repository is a research sandbox for Rectified Flow image editing.

Current working paper title:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Current method name:

```text
DeCE-RF
```

The current direction is no longer a broad claim that a reconstruction/editing
velocity split is sufficient. The active research question is narrower and
more unified:

```text
decoupled clean-estimate edit-preserve control for localized Rectified Flow editing
```

The working hypothesis is that local RF editing is limited by both velocity
coupling and spatial support quality. The method keeps the source-conditioned
trajectory and designs a decoupled clean-estimate displacement at each ODE step:

- `Delta_edit` moves edit regions toward the target clean estimate;
- `Delta_pres` pulls preserve regions back toward the source latent;
- the total clean displacement is mapped back to RF velocity with
  `v_DeCE = v_src - t^-1 Delta_0`;
- operation-conditioned support estimates the displacement geometry;
- clean-estimate feedback updates the edit and preserve displacement weights
  along the ODE trajectory.

## Active Code

- `run_edit_sd3.py`  
  Main CLI entrypoint for SD3 RF editing experiments.

- `edit_cli_args.py`  
  CLI argument definition kept separate from the runtime entrypoint.

- `edit_preprocess.py`  
  Image preprocessing, normalized box helpers, structure-mask builders, and
  proposal-diff mask construction used by the CLI.

- `recolor_projection.py`  
  Recolor clean-projection, alpha refinement, and clean-estimate debug helpers
  used by the SD3 controller.

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

- `experiments/support_v3_2026-06-02/`  
  Current strict Core-6 paper experiment pack and reduced RF-comparison artifacts.

Older experiment summaries were moved to
`experiments/archive_legacy_2026-05-11/`.

## Main Scripts

Run or reproduce the current strict Core-6 matrix:

```bash
SCOPE=implemented bash scripts/run_wacv_phase1.sh
```

The underlying Core-6 executor is `scripts/run_pretty_matrix.sh`; paper task and method definitions live in `scripts/core6_tasks.sh` and `scripts/core6_methods.sh`.

Initialize visual-audit templates used by tests / current comparisons:

```bash
scripts/init_pretty_visual_audit.py
```

Evaluate current outputs:

```bash
scripts/evaluate_paper_metrics.py
```

Build current paper grids:

```bash
scripts/make_paper_grids.py
```

Build support and reference masks:

```bash
scripts/make_semantic_mask.py
scripts/make_source_color_mask.py
scripts/make_surface_recolor_reference.py
scripts/make_decal_reference.py
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

The strict Core-6 seed-10 gate has been run. Current paper rows are:

- `cat_crown`
- `bowl_apple_inside`
- `tshirt_star`
- `red_chair_blue`
- `pillow_vertical_fabric_strip`
- `backpack_remove_toy_charm`

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
- `legacy/cleanup_20260603/`

These files are kept for traceability but are not the current project entry
points.
