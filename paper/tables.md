# Table Plan

Current source of truth: `paper/archive_old_core6_20260602/old_core6_server_results.md` for completed server evidence and
`paper/wacv_experiment_design.md` for the updated WACV Core-6 taxonomy.

## Main Comparison

Rows:

```text
revised strict Phase 1: 6 task instances x 4 paper-facing methods x seeds 10,11,12 = 72 complete runs
```

Server task instances:

```text
cat_crown
bowl_apple_inside
tshirt_star
red_chair_blue
pillow_vertical_fabric_strip
backpack_remove_toy_charm
```

Mapping into the updated Core-6 taxonomy:

```text
T1 attached accessory: cat_crown canonical; dog_sunglasses diagnostic only
T2 container-constrained insertion: bowl_apple_inside canonical
T3 surface decal: tshirt_star canonical; mug_heart diagnostic only
T4 object-level recolor: red_chair_blue canonical
T5 surface material strip editing: pillow_vertical_fabric_strip canonical; DeCE-RF passes human review with a perspective-aligned blue silk strip
T6 exposed-object removal: backpack_remove_toy_charm canonical
```

Paper-facing methods:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
DeCE-RF
```

Current artifacts:

```text
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.csv
experiments/support_v3_2026-06-02/strict_fixed_mask_metrics_summary.md
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.csv
experiments/support_v3_2026-06-02/strict_visual_human_quick_audit.md
```

Headline columns:

- outside-mask L1 / RMSE
- inside-mask change
- source SSIM
- DINO/source similarity
- CLIP target-source delta, labeled carefully for removal and recolor
- internal visual audit: edit success, preservation, locality, artifact, overall

`support_v3_fixed` should not appear as a headline main-table method. It belongs in the component ablation table.

## Ablation Table

Rows:

```text
6 server task instances x support_v3_fixed x seeds 10,11,12 = 18 complete
fixed-control runs; compare against the matching DeCE-RF rows in
`core6_fixed_control_metrics.csv`
```

Current use:

```text
support_v3_fixed vs DeCE-RF
```

Interpretation:

- isolates feedback-updated displacement weights from fixed displacement weights;
- supports component evidence, not the headline claim;
- is not yet the final E4 evidence for the revised strict Core-6; use it as a
  component baseline until the matched controller/Pareto runs are complete.

## Expansion Tables

The completed server evidence table includes one recolor task gated by seed-10
visual inspection and a corrected fixed evaluation mask:

```text
selected server expansion: red_chair_blue
updated T2: bowl_apple_inside, now included in strict Phase 1
updated T5: pillow_vertical_fabric_strip, now included in strict Phase 1 after replacing the earlier pillow_blue_stripes probe
```

Weak replacement/removal candidates should move to a limitation/stress table
rather than being tuned into the main table. `backpack_remove_toy_charm` remains
usable as T6 target-removal evidence with a local zipper/fabric preservation
caveat.

## Extension Probe Tables

Report these separately from the main comparison:

```text
laptop_remove_sticker: high-confidence completion-prior removal extension
whiteboard_probe_red_star_sticker: non-glyph replacement target-formation probe
```

Do not aggregate their metrics into the base DeCE-RF main-table mean unless the method column explicitly names the extra route.

## Limitation / Diagnostic Table

Rows:

```text
whiteboard_remove_yellow_letter
dog_remove_tennis_ball
dog_replace_tennis_ball_star
fridge_remove_yellow_magnet
fridge_remove_peach_magnet
whiteboard_probe_blank / blue T / red A
```

Purpose:

```text
show accurate support can still fail when completion, occluded host synthesis, precise glyph control, or replacement target formation is the bottleneck.
```

## Same-Support Removal Diagnostic

A removal-only diagnostic has been generated for `backpack_remove_toy_charm`
using Telea and Navier-Stokes OpenCV inpainting with the DeCE-RF support mask.
Report it separately from the main comparison because it receives the same
support mask and only applies to removal/fill cases.

Artifact status: this is a legacy diagnostic row. The active repository no
longer tracks summary CSV/JSON files for this run; generated images remain under
ignored generated same-support inpaint output directories and should
not be cited as active paper evidence unless regenerated and summarized.

Readout: same-support inpainting gives lower outside drift on the backpack case but produces visible fill artifacts around the strap/zipper region, while DeCE-RF removes the target charm but locally smooths the occluded zipper/fabric.

## External Baseline Table

Baseline gap audit and external-baseline protocol:

```text
experiments/support_v3_2026-06-02/e2_baseline_download_registry.csv
experiments/support_v3_2026-06-02/e2_baseline_runnable_validation.csv
experiments/support_v3_2026-06-02/e2_baseline_audit.md
```

Current server check: external baseline source repositories have been downloaded
under `/workspace/baselines/src`, with pinned clone status in
`/workspace/baselines/download_status.tsv`. Separate per-baseline environments
were created under `/workspace/baselines/envs`; after quota pressure, the active
server keeps the RF-family environments needed for E2 target-mode validation.
FlowEdit, FlowAlign, and SplitFlow now have revised strict Core-6 target-mode
generation, fixed-mask metrics, and internal visual audit over 6 tasks x seeds
10/11/12, so they enter the SD3-matched E2-A comparison. E2-B is now a
native-backbone contextual RF / FLUX comparison: `rf_solver_edit`, `reflex`,
`fireflow`, and `stable_flow` are FLUX.1-dev rows blocked by gated checkpoint
access or adapter gaps; `ot_rf_otip` and `dvrf` are planned entries that still
need verified repos, environments, smoke tests, and Core-6 adapters. LEDITS++
has legacy Core-6 artifacts only and must be rerun on the revised strict T2/T5
task set before paper-facing supplement use.

Downloaded baseline repositories:

```text
FlowEdit
SplitFlow
FireFlow
RF-Solver-Edit
ReFlex
FlowAlign
stable-flow
ZONE
instruct-pix2pix
pix2pix-zero
MasaCtrl
prompt-to-prompt
h-edit
ledits_pp
```

Registered but not-yet-downloaded E2-B contextual candidates:

```text
OT-RF / OTIP-style, backbone TBD
DVRF / Delta Velocity RF, backbone TBD
```

Existing baseline artifacts are older core-4 or legacy Core-6 evidence and
should be reported as availability/qualitative context unless rerun under the
current revised strict evidence protocol.

Do not use this table to claim that DeCE-RF beats all RF or FLUX baselines.
Report it as a SD3-matched RF comparison against FlowEdit, FlowAlign, and
SplitFlow, paired with an E2-B native-backbone contextual status/audit table
covering FLUX rows and planned RF candidates. The main algorithmic claim comes
from E2-A, not cross-backbone rows.

SD3-matched E2-A comparison artifacts:

```text
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.json
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed10_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed11_grid.png
experiments/support_v3_2026-06-02/visual_audit/e2_flowedit_seed12_grid.png
```

Readout: FlowEdit, FlowAlign, and SplitFlow are runnable SD3 target-mode RF
baselines and frequently form the requested target, but human visual audit
rejects all 54 external strict outputs because the source identity, object
geometry, crop, or background is substantially redrawn. Quantitatively, the
SD3-matched target-mode RF baselines show higher outside-mask change or weaker
source preservation than DeCE-RF under the same fixed masks. Phrase this as a
SD3-matched E2-A comparison, not a broad claim over all RF or FLUX baselines.

Legacy baseline artifacts, for historical audit only:

```text
experiments/archive_legacy_2026-05-11/baseline_parity_manifest.csv
experiments/archive_legacy_2026-05-11/baseline_summary.csv
experiments/archive_legacy_2026-05-11/baseline_summary.md
paper/archive_old_core6_20260602/old_stage2_5_integrity_precheck.md
```

Recommended disclosure columns:

- runnable status
- complete rows
- failed rows and concrete reason
- model family / backbone
- seed-matching caveat
- whether the baseline used text-only, automatic masks, manual masks, or same-support masks

LEDITS++ status: legacy outputs exist only as ignored/generated review assets
and are not active strict Core-6 evidence. Do not list LEDITS++ as a completed
paper-facing baseline unless it is rerun and summarized under the current
`experiments/support_v3_2026-06-02/` protocol.
