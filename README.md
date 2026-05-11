# RF h-Edit Project

This repository is a research sandbox for Rectified Flow image editing.

The current direction is no longer a broad claim that a reconstruction/editing
velocity split is sufficient. The active research question is narrower:

```text
support-aware clean-estimate control for Rectified Flow image editing
```

The working hypothesis is that local RF editing is limited by spatial support
quality. The method separates:

- a support proposer that estimates where editing should happen;
- a clean-estimate-space adaptive controller that measures edit progress and
  preservation drift;
- RF velocity corrections that execute edit and preserve updates along the ODE
  trajectory.

## Active Code

- `run_edit_sd3.py`  
  Main CLI entrypoint for SD3 RF editing experiments.

- `sd3_hrec.py`  
  Core SD3 RF editing implementation.

- `generic_support.py`  
  Generic / operation-aware support proposal utilities.

- `attention_mask.py`, `energies.py`, `schedules.py`, `clip_text_reward.py`  
  Supporting utilities for masks, guidance terms, schedules, and reward
  gradients.

## Active Research Notes

- `docs/research_direction_assessment_2026-05-11.md`  
  Current paper-direction assessment and recommended narrowing.

- `docs/rf_h_edit_updated_direction.md`  
  Reframes the project around clean-estimate-space local control.

- `docs/rf_h_edit_generic_support_todo.md`  
  Generic support proposal plan.

- `docs/rf_h_edit_support_v2_direction.md`  
  Operation-aware support direction and support-v2 readout.

- `docs/rf_h_edit_support_v3_math_plan.md`  
  Mathematical implementation plan for operation-aware support-v3.

- `docs/worklog_2026-05-11.md`  
  Latest worklog.

## Active Experiments

- `experiments/main_matrix.md`  
  Submission-oriented method/task matrix.

- `experiments/pretty_matrix.md`  
  Pretty-task matrix used for support proposal evaluation.

- `experiments/adaptive_v0_v1_summary.md`  
  Adaptive controller v0/v1 comparison.

- `experiments/generic_support_summary.md`  
  Generic support v1 and support ablation summary.

- `experiments/support_v2_minimal_summary.md`  
  Operation-aware support-v2 minimal gate result.

## Main Scripts

Run or reproduce current matrices:

```bash
scripts/run_main_table.sh
scripts/run_pretty_matrix.sh
scripts/run_ablation_table.sh
scripts/run_missing_main_matrix.sh
```

Evaluate and audit records:

```bash
scripts/evaluate_paper_metrics.py
scripts/audit_experiment_records.py
scripts/audit_main_matrix_coverage.py
```

Build paper / evidence artifacts:

```bash
scripts/make_paper_figures.sh
scripts/make_baseline_evidence_pack.py
scripts/make_comparison_grid.py
```

Build support and reference masks:

```bash
scripts/make_semantic_mask.py
scripts/make_source_color_mask.py
scripts/make_vehicle_paint_mask.py
scripts/make_surface_recolor_reference.py
scripts/make_decal_reference.py
scripts/make_glasses_reference.py
scripts/make_mask_badge_reference.py
scripts/refine_surface_mask.py
scripts/subtract_masks.py
```

External baseline wrappers:

```bash
scripts/run_flowedit_baseline.py
scripts/run_splitflow_baseline.py
scripts/run_fireflow_baseline.py
scripts/run_rf_solver_edit_baseline.py
scripts/run_reflex_baseline.py
```

## Current Research Position

The current evidence supports a conservative claim:

```text
Clean-estimate-space adaptive control improves preservation when the support
region is reasonable, but automatic support proposal remains the bottleneck.
```

It does not yet support a broad claim of general automatic RF image editing.

The next planned experiment is a small `support_v3_gate`, not a full matrix:

```text
tasks: cat_crown, dog_sunglasses, mug_heart, backpack_remove_toy_charm
seed: 10
compare: manual support, generic v1, support v2 minimal, support v3 candidate
```

Gate criteria:

- at least two tasks improve visibly over generic v1;
- `dog_sunglasses` does not regress;
- `cat_crown` or `mug_heart` becomes displayable;
- `backpack_remove_toy_charm` improves object coverage / residue;
- preservation drift does not obviously worsen.

## Paper Materials

The current manuscript materials live in `paper/`:

- `paper/outline.md`
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
- `legacy/2026-05-11/`

These files are kept for traceability but are not the current project entry
points.
