# Paper Workspace

Current source of truth for the WACV paper package.

## Active Experiment Design

- `wacv_experiment_design.md`: canonical 2026-06-05 experiment design and claim boundary.
- `core6_phase1_images_prompts.md`: frozen strict Core-6 Phase 1 task/image plan.
- `phase2_core6_images_prompts.md`: planned Phase 2 Core-6 expansion manifest.
- `e2_status_lock.md`: E2 method/backbone status and paper-use lock.
- `results.md`: current result readout and safe wording.
- `figures.md`: current main/supplement figure plan.
- `tables.md`: current main/supplement table plan.

## Active Manuscript Files

- `manuscript.md`: main manuscript draft to keep synchronized with the current experiment design.
- `draft.md`: working draft/notes; useful for section material, not the final authority.
- `outline.md`: current paper structure sketch.
- `argument_blueprint.md`: claim/evidence map.
- `limitations.md`: limitation language aligned to strict Core-6.
- `references.bib`: bibliography stub/source file.

## Current Experimental Scope

The active strict Core-6 tasks are:

```text
cat_crown
bowl_apple_inside
tshirt_star
red_chair_blue
pillow_same_color_cable_knit
backpack_remove_toy_charm
```

The active E2 design is a layered fairness experiment:

```text
E2.1: backbone calibration for reconstruction/direct-target floors.
E2.2: same-backbone SD3 algorithm comparison against RF-native and preservation-control rows.
E2.3: native preservation-aware RF / FLUX implementation comparison, reported separately.
E2.4: support-matched diagnostic to separate localization input from controller design.
Non-RF supplement: InstructPix2Pix and H-Edit / P2P-style only for positioning.
```

Do not promote archived Core-5, old Core-6, or server-snapshot rows into the
main claim unless they are rerun under the current strict protocol.

## Archives

- `archive_superseded_2026-06-03/`: superseded drafts and handoff/precheck files that conflict with the current experiment design.
- `archive_old_core6_20260602/`: old Core-6/server-evidence materials.
- `archive_superseded_2026-06-02/`: earlier experiment-plan drafts.
- `server_snapshot_2026-06-02/`: raw snapshot from the server before the current cleanup.

Use archived material only for history, diagnostics, or wording recovery. The
active files listed above are the paper-facing source of truth.
