# WACV Experiment Design for DeCE-RF

Current streamlined design: `paper/wacv_experiment_design_v2.md`.

This file keeps the broader baseline/task inventory. Use the v2 document as
the current source of truth for WACV main-paper experiments, figures, and run
order.

Working title:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

## Target Claim

For WACV, keep the claim conservative:

```text
DeCE-RF improves the edit-preserve tradeoff for localized Rectified Flow
editing by decoupling target-directed edit displacement from source-preserving
displacement, using operation-conditioned support and feedback-updated
displacement weights.
```

Do not claim general-purpose image editing or broad state-of-the-art image
editing. The paper should be framed as a controlled method study for localized
addition/accessory insertion, decal insertion, exposed-object removal, and one
localized attribute/recolor task. Completion-prior removal and replacement
target formation are separate operation-conditioned probes.

## Experimental Questions

Q1. Does decoupled clean-estimate edit-preserve displacement reduce unintended
outside-region drift compared with direct target guidance?

Q2. Does operation-conditioned support give better control geometry than generic
support or single-signal support?

Q3. Does feedback-updated displacement weighting stabilize the edit-preserve
tradeoff relative to fixed displacement weights, especially under edit-strength
or support perturbations?

Q4. Where does the method fail on removal/replacement cases, and are those
failures consistent with the claimed limitations?

Q5. Can the same edit-preserve control support a localized recoloring case
without requiring a separate completion or replacement route?

## Task Set

### Core WACV Task Set

Use six main tasks for the main quantitative table.

| Task | Operation | Relation | Role |
| --- | --- | --- | --- |
| `cat_crown` | `add_object` | `above_host` | compact insertion, main teaser case |
| `dog_sunglasses` | `add_object` | `on_face` | strong positive face/accessory case |
| `mug_heart` | `add_decal` | `on_surface` | surface/decal localization |
| `tshirt_star` | `add_decal` | `on_clothing_surface` | second surface/decal generalization |
| `backpack_remove_toy_charm` | `remove_object` | `remove_source_object` | removal of a small attached object |
| `red_chair_blue` or `red_office_chair_to_blue_office_chair` | `recolor` | `object_attribute` | localized attribute/color-change probe |

The current completed evidence base is core-5 without recolor:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
```

Run a seed-10 visual gate for the recolor candidate before promoting it to the
main six-task matrix.

### Operation-Conditioned Extension Probes

Report these separately from the main task set because they use additional
operation-conditioned routes.

| Task | Operation | Relation | Role |
| --- | --- | --- | --- |
| `laptop_remove_sticker` | `remove_object` + completion prior | `remove_source_object` | high-confidence planar removal where completion clean-delta is reliable |
| `whiteboard_probe_red_star_sticker` | `replace` | `replace_source_object` | non-glyph replacement in a semantic letter field |

Do not include these rows in the main Core-6 table unless the method column
clearly labels the additional route, e.g. `DeCE-RF + high-confidence completion
prior` or `DeCE-RF + replacement target route`.

### Optional Limit/Stress Tasks

Use these only for qualitative limits or supplementary material.

| Task | Operation | Role |
| --- | --- | --- |
| `tote_leaf` | `add_decal` | extra logo/decal case |
| `rabbit_sunglasses` | `add_object` | difficult profile relation |
| `backpack_replace_patch_blue` | `replace` | replacement ambiguity |
| `dog_remove_tennis_ball` | `remove_object` | occluded mouth/host completion failure |
| `dog_replace_tennis_ball_star` | `replace` | partial replacement; target pressure fires but the object is not clean |
| `whiteboard_remove_yellow_letter` | `remove_object` | glyph-field hallucination under blank removal |
| `fridge_remove_yellow_magnet` / `fridge_remove_peach_magnet` | `remove_object` | cluttered-surface completion limitations |

Do not use weak replacement or stylization tasks in the main WACV claim.

## Methods

### Main Comparison

Use the main table to compare DeCE-RF against general and RF editing baselines.
Do not make `fixed DeCE displacement` a headline baseline in this table; it is
an internal ablation control for isolating the feedback controller.

| Paper name | Runner / implementation name | Purpose |
| --- | --- | --- |
| Source image | n/a | visual reference only; not a method row |
| RF reconstruction / base reconstruction | `base_only` | reconstruction-only RF pass; may drift from the source |
| Direct target guidance | `direct_target` | coupled target guidance baseline |
| Generic support control | `adaptive_full_generic_support` | weak automatic support baseline |
| DeCE-RF | `support_v3_controller_rmsgap` | full method |

If available, add manual/external support only as an upper-bound diagnostic in
supplementary material, not as the primary automatic comparison.

Report `Fixed DeCE displacement` only in the component ablation and feedback
stress analysis:

| Paper name | Runner / implementation name | Purpose |
| --- | --- | --- |
| Fixed DeCE displacement | `support_v3_fixed` | decoupled displacement with operation-conditioned support, but without feedback |

This separation avoids overclaiming a small mean gap between fixed DeCE and
DeCE-RF while still answering the reviewer question about which component
causes the improvement.

### External Baselines

For WACV, include a small number of external baselines that are directly related
to localized text/instruction-guided image editing. Do not overload the paper
with many weak or poorly tuned baselines.

Because the paper is explicitly about Rectified Flow editing, at least one
RF-native editing baseline must appear in the main comparison. Diffusion-only
baselines are not sufficient for a WACV submission with this title and claim.

#### Required RF / Flow Baseline Suite

Because the contribution is RF-native, include the RF family as its own
comparison block rather than treating RF methods as isolated optional baselines.
Prioritize baselines that can run on the same or closely matched flow backbone.

| Baseline | Category | Why it matters | Use in paper |
| --- | --- | --- | --- |
| Vanilla RF target guidance | internal RF | simplest coupled target-conditioning baseline | main RF table |
| RF inversion + target resampling | internal RF | exposes reconstruction/editability tradeoff under inversion | main RF table |
| FlowEdit | inversion-free RF | direct source-target ODE editing for pre-trained flow models | main RF table |
| FlowAlign | inversion-free RF | trajectory-regularized extension of FlowEdit-style editing | supplementary RF audit |
| SplitFlow | inversion-free RF | flow decomposition/aggregation baseline for prompt disentanglement | supplementary RF audit |
| RF-Inversion | inversion-based RF | stochastic rectified differential equation inversion/editing baseline | main RF table if backbone-compatible |
| RF-Edit / RF-Solver-Edit | inversion-based RF | improved RF ODE solver and structural preservation baseline | main RF table if runnable |
| FireFlow | fast inversion RF | fast RF inversion/editing baseline, useful for runtime comparison | supplementary RF audit |
| Stable Flow | training-free flow editing | layer/feature control baseline for flow-matching DiT editing | supplement if runnable |
| OTIP / transport-guided RF inversion | transport/inversion RF | optimal-transport-guided RF inversion/editing baseline | supplement if implementation is available |

The RF suite should be reported separately from diffusion-only baselines, because
its purpose is to answer the reviewer question: "Why not use an existing RF
editing method?"

Fairness rule: do not compare a FLUX-only baseline against an SD3-based DeCE-RF
as if the method difference were isolated. If the backbone differs, report the
result as cross-backbone qualitative evidence or supplementary context.

Minimum RF comparison for WACV:

```text
Direct target / vanilla RF edit
FlowEdit
RF-Inversion or RF-Edit
DeCE-RF
```

Main RF comparison:

```text
Direct target / vanilla RF edit
RF inversion + target resampling
FlowEdit
RF-Inversion or RF-Edit / RF-Solver-Edit
DeCE-RF
```

Supplementary RF audit:

```text
FlowAlign
SplitFlow
FireFlow
Stable Flow
OTIP, if code is available
```

If some public implementations are unavailable, fail to run, or require a
different backbone, keep them in a "not run / incompatible" row in the
supplementary baseline audit rather than silently dropping them.

#### Required External Baselines

Run these on the six-task main set if implementation time allows. Since RF
baselines are handled in the RF suite, choose two or three non-RF baselines from
this list.

| Baseline | Why it matters | Use in paper |
| --- | --- | --- |
| Masked inpainting / masked img2img | strongest simple locality baseline when an edit region is available | main table or supplementary table |
| InstructPix2Pix | standard instruction-based editing baseline | main external table |
| DiffEdit | automatic mask-based semantic editing baseline | main external table |
| Prompt-to-Prompt + inversion or Null-text inversion | classic attention-control editing baseline | main external table or supplementary |

Masked inpainting should use the same source image and the same textual edit.
If it receives manual masks, label it clearly as `Manual-mask inpainting`; if it
uses the DeCE-RF support mask, label it as `Same-support inpainting`.

#### Strong Optional Baselines

Run these only if setup is stable.

| Baseline | Why it matters | Priority |
| --- | --- | --- |
| ZONE | recent zero-shot instruction-guided local editing method | high |
| MasaCtrl | tuning-free self-attention control for consistent editing | medium |
| Pix2Pix-Zero | zero-shot image-to-image translation/editing | medium |
| Plug-and-Play Diffusion Features | feature-injection image-to-image translation baseline | low-medium |

ZONE is the most relevant optional baseline because it explicitly targets local
instruction-guided editing and preservation of non-edited regions.

#### Baselines To Avoid As Main Comparisons

Avoid making these primary baselines unless they are carefully controlled:

```text
closed commercial editors
large proprietary instruction-editing APIs
unmasked text-to-image regeneration
methods requiring different user inputs that make the comparison unfair
methods that cannot be run reproducibly or anonymously
```

Closed or API-only models can be shown qualitatively in supplementary material
only if allowed by venue policy and if prompts, dates, versions, and settings
are reported.

### Recommended Baseline Tiers

Use a tiered plan so the paper remains feasible.

#### Tier 0: Internal Baselines

These are mandatory.

```text
RF reconstruction / base only
Direct target guidance
Generic support control
DeCE-RF
```

Run `Fixed DeCE displacement` as the ablation-only internal control, not as a
paper-facing external baseline.

#### Tier 1: WACV Main Non-RF External Baselines

These are the best balance of relevance and feasibility.

```text
Masked inpainting / masked img2img
InstructPix2Pix
```

#### Tier 2: Add If Time Allows

```text
DiffEdit
Prompt-to-Prompt + inversion
ZONE
MasaCtrl
Pix2Pix-Zero
```

If only one Tier 2 baseline can be added, choose `DiffEdit`; if two can be
added, choose `DiffEdit` and `ZONE`.

### External Baseline Matrix

Do not run every external baseline on every ablation. Use them only for the main
task comparison.

### RF Baseline Matrix

Run the main RF baselines as a dedicated RF comparison. This is separate from
the general non-RF external baseline table.

Main RF suite:

```text
6 tasks x 4 RF baselines x 3 seeds = 72 runs
```

RF baselines counted here:

```text
Vanilla RF target guidance
RF inversion + target resampling
FlowEdit
RF-Inversion
```

If `RF-Inversion` is not runnable on the chosen backbone, replace it with:

```text
RF-Edit / RF-Solver-Edit
```

Supplementary RF audit, run only on four core tasks and seed 10 unless setup is
already stable:

```text
4 core tasks x up to 5 supplementary RF baselines x 1 seed = up to 20 runs
```

Supplementary RF audit candidates:

```text
FlowAlign
SplitFlow
FireFlow
Stable Flow
OTIP
```

DeCE-RF appears in this table as the proposed method and is not counted as an
external baseline.

Recommended external baseline matrix:

```text
6 tasks x 2 non-RF external baselines x 3 seeds = 36 runs
```

Recommended non-RF external baselines:

```text
Masked inpainting / masked img2img
InstructPix2Pix
```

If compute or implementation time is tight:

```text
4 core tasks x 2 non-RF external baselines x 3 seeds = 24 runs
```

Add if time allows:

```text
DiffEdit
ZONE
```

### Recolor Core-6 Matrix

Run the recolor case as the sixth main task after a seed-10 visual gate passes.

Internal recolor matrix:

```text
1 recolor task x 4 paper-facing internal methods x 3 seeds = 12 runs
```

Recommended methods:

```text
RF reconstruction / base only
Direct target guidance
Generic support control
DeCE-RF
```

Optionally add `Fixed DeCE displacement` for the recolor ablation cache:

```text
1 recolor task x 1 fixed-control method x 3 seeds = 3 runs
```

RF recolor comparison:

```text
1 recolor task x 4 RF baselines x 3 seeds = 12 runs
```

Recommended RF baselines:

```text
Vanilla RF target guidance
RF inversion + target resampling
FlowEdit
RF-Inversion or RF-Edit / RF-Solver-Edit
```

Non-RF recolor comparison is optional:

```text
1 recolor task x 2 non-RF baselines x 3 seeds = 6 runs
```

Recommended non-RF baselines:

```text
Masked inpainting / masked img2img
InstructPix2Pix
```

If time is tight, run only the internal recolor matrix and keep RF/non-RF
recolor baselines as supplementary context.

If only one more non-RF baseline can be added, use `DiffEdit`. If two can be
added, use `DiffEdit` and `ZONE`.

### Fairness Rules For Baselines

Use the same source image, same intended edit, same output resolution, and same
seed set whenever possible.

Report different input assumptions explicitly:

| Baseline type | Required disclosure |
| --- | --- |
| text-only / instruction-only | no mask or region input |
| automatic-mask method | mask is generated by the method |
| manual-mask method | manual mask is extra user input |
| same-support method | uses DeCE-RF support but not DeCE-RF displacement control |

Do not compare a manual-mask baseline against automatic DeCE-RF as if both had
the same user input. Treat manual support as an upper bound or diagnostic.

## Main Matrix

Primary paper-facing WACV internal matrix:

```text
6 tasks x 4 methods x 3 seeds = 72 runs
```

Seeds:

```text
10, 11, 12
```

Use seed 10 for visual gate checks before expanding to all seeds.

Recommended execution bundle:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 runs
6 tasks x fixed DeCE ablation-control x 3 seeds = 18 runs
total internal execution bundle = 90 runs
```

The 18 fixed-DeCE runs are reused by the component ablation and feedback stress
analysis. They should not be presented as a main external baseline.

## Ablation Design

Run ablations on three representative tasks:

```text
cat_crown
mug_heart
backpack_remove_toy_charm
```

These cover object insertion, surface decal localization, and deletion. Do not
run every ablation on every task unless time remains.

### WACV Compact Component Ablation

Use one compact ablation table in the main paper:

```text
direct target guidance
generic support control
fixed DeCE displacement
DeCE-RF
```

Matrix:

```text
3 tasks x 4 variants x 3 seeds = 36 runs
```

This compact table jointly tests whether the gain comes from decoupled
displacement, operation-conditioned support, and feedback-updated control.

If compute is tight, use the 27-run fallback:

```text
generic support control
fixed DeCE displacement
DeCE-RF
```

```text
3 tasks x 3 variants x 3 seeds = 27 runs
```

### Optional A1: Decoupled Clean-Estimate Displacement

Purpose: show that the clean-estimate displacement decomposition matters.

Compare:

```text
direct_target
support_v3_fixed_no_preserve
support_v3_fixed_no_edit
support_v3_fixed
DeCE-RF
```

Matrix:

```text
3 tasks x 5 variants x 3 seeds = 45 runs
```

If time is tight, drop `support_v3_fixed_no_edit` and report:

```text
3 tasks x 4 variants x 3 seeds = 36 runs
```

Run this optional ablation only if the compact table leaves an unresolved
reviewer concern.

### Optional A2: Operation-Conditioned Support Geometry

Purpose: show that support is not just a mask trick.

Compare:

```text
generic support
attention-only support
clean-disagreement-only support
operation-conditioned support
manual/external support upper bound, if available
```

Matrix:

```text
3 tasks x 4 automatic variants x 3 seeds = 36 runs
```

Manual support can be single-seed or supplementary if setup time is high.

Run this optional ablation only if support quality becomes a central reviewer
concern.

### Optional A3: Feedback-Updated Displacement Weights

Purpose: show that feedback stabilizes the edit-preserve tradeoff relative to
fixed weights. This is the correct place to compare `support_v3_fixed` against
DeCE-RF.

Compare:

```text
support_v3_fixed
DeCE-RF
support_v3_fixed under edit-strength/support perturbation
DeCE-RF under edit-strength/support perturbation
```

Recommended stress cases:

```text
cat_crown: edit-strength sweep
dog_sunglasses: support perturbation
mug_heart: both edit-strength and support perturbation
```

Target size:

```text
30-45 runs, depending on sweep density
```

Run a small seed-10 version before large expansion if the feedback contribution
needs stronger evidence. Expand to three seeds only if DeCE-RF separates from
fixed DeCE under stress.

## Metrics

Report metrics as mean +- std over seeds.

### Fixed Evaluation Mask Protocol

For each task, build one task-level evaluation mask and reuse it for every
method and every seed. This prevents methods from receiving different
inside/outside regions for preservation metrics.

Mask source priority:

```text
manual mask
source-target diff mask
corrected DeCE/support mask frozen as an evaluation mask
```

Use the fixed mask for inside/outside L1/RMSE, luma SSIM, DINO/source
similarity, CLIP-local scores where applicable, and the composite edit score.
Report each method's own support mask only as a support diagnostic, not as the
main evaluation region.

### Preservation Metrics

```text
outside-mask L1
outside-mask RMSE
luma SSIM loss
DINO source similarity
```

### Edit Metrics

```text
inside-mask change
CLIP target score
CLIP target-source delta
manual edit success score
```

### Recolor Metrics

Use these for the promoted Core-6 recolor task:

```text
target-region color distance to requested color
outside-region color drift
outside-mask L1
DINO source similarity
manual color accuracy
manual preservation
```

Report recolor-specific metrics separately from add/remove/decal metrics, but
include the task in the Core-6 coverage summary as localized attribute editing.

### Controller Diagnostics

```text
edit_target_gap_rms
preserve_drift
adaptive_edit_weight
adaptive_preserve_weight
projection_norm_or_ratio
```

### Support Diagnostics

```text
support area
support leakage
support overlap with manual mask, if available
edit/preserve overlap
```

## Tables

### Table 1: Main Edit-Preserve Comparison

Rows:

```text
RF reconstruction / base reconstruction
Direct target guidance
Generic support control
RF and non-RF external baselines
DeCE-RF
```

Columns:

```text
CLIP target-source delta
outside-mask L1
DINO source similarity
luma SSIM
manual edit success
manual preservation
```

This is the most important table. Fixed DeCE is intentionally excluded from the
headline ranking and moved to Table 2.

### Table 2: Component Ablation

Group rows by component:

```text
clean displacement decomposition
support geometry
feedback-updated control
```

Include `Fixed DeCE displacement` here. Keep this compact; WACV main paper
space is limited.

### Table 3: RF Baseline Comparison

Rows:

```text
Vanilla RF target guidance
RF inversion + target resampling
FlowEdit
RF-Inversion or RF-Edit / RF-Solver-Edit
DeCE-RF
```

Columns:

```text
CLIP target-source delta
outside-mask L1
DINO source similarity
manual edit success
manual preservation
runtime
```

This table answers the main RF-reviewer concern and should be in the main paper
if space allows. If space is tight, merge it with Table 1 by adding the RF rows.

### Table 4: Runtime and Reproducibility

Report:

```text
steps
resolution
seeds
runtime per image
GPU memory
code/artifact availability
```

This table can go to supplementary material if space is tight.

## Figures

### Figure 1: Method Overview

Show:

```text
source latent
target/source clean estimates
Delta_edit
Delta_pres
operation-conditioned support
feedback-updated weights
final edit
```

### Figure 2: Main Qualitative Comparison

Use four rows:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

Columns:

```text
Source
Direct target
Generic support
DeCE-RF
Support overlay
```

Use seed 10 only if it is representative of the 3-seed average. Avoid selecting
a visually lucky seed.

### Figure 3: Support Geometry

Use `mug_heart` as the cleanest support visualization:

```text
attention-only
clean disagreement
velocity disagreement
operation-conditioned support
edit mask
preserve mask
```

### Figure 4: Feedback Controller

Use `cat_crown` or `dog_sunglasses`.

Show:

```text
Fixed DeCE displacement
DeCE-RF
edit gap curve
preserve drift curve
adaptive edit weight
adaptive preserve weight
final visual comparison
```

### Figure 5: Extension Probes

Use two rows:

```text
laptop_remove_sticker: high-confidence completion prior for planar removal
whiteboard_probe_red_star_sticker: non-glyph replacement target formation
```

Label the method route explicitly, e.g. `DeCE-RF + completion prior` and
`DeCE-RF + replacement route`.

### Figure 6: Limitations

Use two or three cases:

```text
whiteboard_remove_yellow_letter: glyph-field hallucination under blank removal
dog_remove_tennis_ball: occluded mouth/host completion failure
dog_replace_tennis_ball_star: partial replacement with mixed residual object
```

The limitation figure should be honest and tied to the scope of the claim.

### Figure 7: Recolor Core Task

Use the promoted recolor candidate:

```text
red_chair_blue or red_office_chair_to_blue_office_chair
```

Columns:

```text
Source
Direct target
Generic support
DeCE-RF
Support overlay
```

Place this in the main figure if it becomes Core-6; otherwise move it to the
supplement.

## Execution Order

### Phase 0: Visual Gate

Run only seed 10.

```text
recolor candidate x 4 paper-facing methods x 1 seed = 4 runs
optional fixed-DeCE ablation cache = 1 run
```

Pass criteria:

```text
all methods produce result.png
all methods produce stats.json
support masks are saved
DeCE-RF changes chair color locally without major geometry/background drift
```

### Phase 1: Main Matrix

After Phase 0 passes:

```text
6 tasks x 4 paper-facing methods x 3 seeds = 72 runs
fixed-DeCE ablation cache over the same tasks/seeds = 18 runs
```

This 90-run internal execution bundle is the core WACV evidence, but the
paper-facing main table should emphasize the 72-run comparison against general
and RF baselines.

### Phase 2: Ablations

Run the compact component ablation first, because it defends the method
components with the smallest run budget.

Priority:

```text
compact component ablation
optional A1 clean displacement decomposition
optional A2 operation-conditioned support geometry
optional A3 feedback/stress
```

### Phase 2b: Recolor Generalization

Run after the seed-10 recolor gate passes.

Priority:

```text
internal recolor matrix
RF recolor comparison
optional non-RF recolor comparison
```

### Phase 3: Figures and Manual Audit

Before writing final results:

```text
select representative seed
create qualitative grids
create support overlays
score manual edit success and preservation
write failure taxonomy
```

### Phase 4: Supplementary Material

Include:

```text
all seed grids
additional masks
additional curves
commands
environment
anonymous code instructions
```

## WACV Minimum Submission Package

The paper is not ready until these exist:

```text
main comparison table over 6 tasks and 3 seeds
RF baseline table over at least 5 RF methods and 3 seeds
component ablation table with fixed DeCE over 3 tasks and 3 seeds
Core-6 recolor row over 1 task and 3 seeds with recolor-specific metrics
extension-probe figure/table for laptop highconf removal and whiteboard red-star replacement
main qualitative grid
support visualization
controller curve figure
failure figure
supplementary all-seed grids
anonymous reproduction instructions
```

## Run Budget

Recommended WACV budget:

```text
Main internal bundle: 90 runs (72 paper-facing + 18 fixed-DeCE ablation cache)
RF suite:           72 runs
Non-RF baselines:   36 runs
Compact ablation:   reused from internal bundle, plus 0-9 extra if needed
Extension probes:   already available / targeted supplementary runs only
Limit/failure:      6-12 runs
```

Expected total:

```text
216-252 runs, depending on how many ablation rows are reused
```

If compute is tight, the fallback budget is:

```text
Main internal bundle: 90 runs
RF suite:           72 runs
Non-RF baselines:   24 runs
Compact ablation:   reused from internal bundle
Recolor probe:      12 runs
Limit/failure:      6 runs
```

Fallback total:

```text
204 runs
```

Do not go below the 72-run paper-facing internal matrix plus the fixed-DeCE
ablation cache for a WACV main-conference submission unless the paper is
reframed as a short exploratory method study. If all RF baselines are included,
reduce non-RF external baselines before reducing the RF suite.

Optional A1/A2/A3 stress ablations can add another 30-100 runs, but they should
not block the WACV execution plan.
