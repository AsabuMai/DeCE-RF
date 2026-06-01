# Table Plan

Current source of truth: `paper/results.md` for completed server evidence and
`paper/wacv_experiment_design.md` for the updated WACV Core-6 taxonomy.

## Main Comparison

Rows:

```text
current server evidence: 6 task instances x 4 paper-facing methods x seeds 10,11,12 = 72 complete runs
```

Server task instances:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
red_chair_blue
```

Mapping into the updated Core-6 taxonomy:

```text
T1 attached accessory: dog_sunglasses canonical; cat_crown teaser/supplement
T2 container-constrained insertion: not completed; bowl/apple probes remain diagnostic
T3 small surface decal: mug_heart canonical
T4 object-level recolor: red_chair_blue canonical
T5 surface pattern editing: not completed; tshirt_star is clothing-decal evidence only
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
experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.csv
experiments/support_v3_2026-05-11/core6_seed10_12_fixed_mask_metrics.json
experiments/support_v3_2026-05-11/core6_fixed_mask_audit_summary.csv
experiments/support_v3_2026-05-11/core6_fixed_mask_audit_summary.md
experiments/support_v3_2026-05-11/core6_visual_audit_template.csv
experiments/support_v3_2026-05-11/core6_visual_audit_filled.csv
experiments/support_v3_2026-05-11/core6_visual_audit_summary.md
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
- is now complete for the promoted server evidence set, but updated strict T2/T5
  still need their own runs if they enter the main table.

## Expansion Tables

The completed server evidence table includes one recolor task gated by seed-10
visual inspection and a corrected fixed evaluation mask:

```text
selected server expansion: red_chair_blue
pending updated T2: bowl_apple_inside or equivalent after container-interior support
pending updated T5: pillow_blue_stripes or equivalent surface-pattern task
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

Artifacts:

```text
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_metrics.csv
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_metrics.json
experiments/support_v3_2026-05-11/backpack_same_support_inpaint_summary.md
```

Readout: same-support inpainting gives lower outside drift on the backpack case but produces visible fill artifacts around the strap/zipper region, while DeCE-RF removes the target charm but locally smooths the occluded zipper/fabric.

## External Baseline Table

Baseline gap audit and external-baseline protocol:

```text
experiments/support_v3_2026-05-11/core6_baseline_gap_audit.md
experiments/support_v3_2026-05-11/core6_external_baseline_plan.md
```

Current server check: external baseline source repositories have been downloaded
under `/workspace/baselines/src`, with pinned clone status in
`/workspace/baselines/download_status.tsv`. LEDITS++ has completed the server
evidence task set over seeds 10/11/12 as an external diffusion-editing baseline;
other external baseline environments, model weights, and output images are not
validated yet. The old FlowEdit root and Python environment from the archived
2026-05-10 baseline scripts are missing, and no current `outputs/baselines`
result images were found for the task set.

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

Existing baseline artifacts are older core-4 evidence and should be reported as
availability/qualitative context unless rerun under the current evidence
protocol.

Artifacts:

```text
experiments/baseline_parity_manifest.csv
experiments/baseline_summary.csv
experiments/baseline_summary.md
experiments/baseline_visual_scores_seed10_12.csv
paper/stage2_5_integrity_precheck.md
```

Recommended disclosure columns:

- runnable status
- complete rows
- failed rows and concrete reason
- model family / backbone
- seed-matching caveat
- whether the baseline used text-only, automatic masks, manual masks, or same-support masks

LEDITS++ completed baseline artifacts:

```text
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.json
experiments/support_v3_2026-05-11/leditspp_core6_baseline_summary.md
experiments/support_v3_2026-05-11/visual_gates/leditspp_core6_seed10_12_grid.png
```
