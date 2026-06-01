# WACV Experiment Design

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-06-01
- Verification Status: UNVERIFIED until final reruns and visual audits are locked
- Version Label: wacv_experiment_design_phase_plan

## One-Sentence Experimental Thesis

DeCE-RF should be evaluated as a localized Rectified Flow control method: it is
not trying to win every image-editing setting, but to improve the edit-preserve
tradeoff when an operation label and relation cue specify where a local edit is
intended.

Suggested opening sentence for the Experiments section:

```text
We evaluate DeCE-RF by testing whether clean-estimate edit-preserve control
improves localized edit success and preserve-region fidelity under matched
support, whether the advantage remains against RF-native editing baselines, and
which components--operation-conditioned geometry and feedback control--are
responsible for the gain.
```

This version replaces the previous "more baselines and more figures" plan with
a tighter evidence chain:

```text
Main effect -> RF-specific comparison -> mechanism ablation -> boundary cases
```

The paper should have enough figures to make the argument legible, but each
figure must answer a different reviewer question. More examples are better only
when they reduce ambiguity; repeated success grids with the same evidence do
not strengthen the claim much under an 8-page WACV budget.

Separate two quantities throughout planning:

```text
main-paper display: about 50-70 image cells across 5 figures, 6 max
actual generated outputs: minimum 250-300, recommended 550-650, heavy 900+
```

The main paper should not look like a gallery. The underlying experiment cache,
however, must be large enough that the selected examples do not look
cherry-picked.

## Reviewer Questions To Answer

| Reviewer question | Experiment that answers it | Main-paper artifact |
| --- | --- | --- |
| Does DeCE-RF actually improve localized edit-preserve behavior? | E1 controlled Core-6 edit-preserve benchmark | Table 1 + qualitative Figure 3 |
| Why not use an existing RF editing method? | E2 RF-native baseline comparison | Table 2 |
| Is the support geometry a real component rather than a hand-picked mask? | E3 support geometry ablation | Figure 4 + small table |
| Does feedback control matter beyond fixed decoupled displacement? | E4 controller/stress ablation | Figure 5 |
| Where does the method stop working? | E5 boundary and extension probes | Figure 6 + limitation paragraph |

## Experiment E1: Controlled Core-6 Edit-Preserve Benchmark

### Purpose

This is the headline experiment. It tests whether DeCE-RF improves the
edit-preserve tradeoff over coupled target guidance and weak generic support
under matched source images, prompts, seeds, and fixed evaluation masks.

Concrete Phase 1 image and prompt choices are tracked in:

```text
paper/core6_phase1_images_prompts.md
```

### Task Set

Core-6 should be defined as six localized-edit categories, not merely six
nice-looking examples. Each category corresponds to a different edit-preserve
difficulty:

| ID | Core-6 category | Operation / relation | Primary difficulty | Main component tested |
| --- | --- | --- | --- | --- |
| T1 | Attached accessory addition | `add_object` + `above_host` / `on_face` | add a small attached object while preserving host identity, pose, and background | `M_core + M_contact`, local object formation |
| T2 | Container-constrained spatial insertion | `add_object` + `inside` | insert a new object inside an existing host/container without redrawing the host | inside-region prior and operation-conditioned support |
| T3 | Surface decal / logo addition | `add_decal` + `on_surface` | place a mark on an existing surface without changing object boundary or background | surface-local support and preserve geometry |
| T4 | Object-level recoloring | `recolor` + `inside` | change color while preserving shape, layout, and identity | low-drift appearance displacement |
| T5 | Surface pattern editing | `add_decal` + `on_surface` | add a medium-size surface pattern while preserving host silhouette and background | edit-preserve Pareto under stronger surface appearance change |
| T6 | Simple exposed-object removal | `remove_object` + `remove_source_object` | remove a visible small object without requiring hard occlusion completion | removal support and preserve-region correction |

The Core-6 claim should be:

```text
DeCE-RF improves localized edit-preserve control under reasonable support across
insertion, surface editing, appearance editing, and simple exposed-removal
scenarios.
```

Full object replacement is not part of E1. Replacement requires source-object
suppression, target-object generation, and attribute inheritance at the same
time, so it can obscure whether the controller works. Put replacement success,
failure, and route-specific extensions in E5.

### Phase 1 Canonical Examples

Use one example per category for the first sanity-check run:

| Category | Phase 1 candidate | Status / note |
| --- | --- | --- |
| T1 Attached accessory addition | `dog_sunglasses` primary; `cat_crown` secondary/teaser | existing strong accessory/contact cases |
| T2 Container-constrained spatial insertion | `bowl_apple_inside` | new task needed; uses the currently supported `inside` relation |
| T3 Surface decal / logo addition | `mug_heart` | existing, clean surface-local decal |
| T4 Local recoloring | `red_chair_blue` | existing candidate; keep only after visual audit confirms local recolor |
| T5 Surface pattern editing | `pillow_blue_stripes` | new task needed; treat as `add_decal`-like pattern overlay, not full material transfer |
| T6 Simple exposed-object removal | `backpack_remove_toy_charm` | existing, exposed small-object removal |

`tshirt_star` remains a strong T3 expansion example and should be used in Phase
2 or supplementary grids. It is not the Phase 1 canonical T3 example if
`mug_heart` is available, because `mug_heart` is the cleaner surface diagnostic.

`red_chair_blue` has candidate outputs and metrics for the internal methods.
Keep it in the main table only after a visual audit confirms that the result is
a local chair-color change rather than a scene-level style drift. The paper
should call it a "localized recolor probe", not evidence of general recoloring.

The unified Core-6 implementation claim is:

```text
All Core-6 tasks use the same DeCE-RF controller. The task category only
selects an operation-conditioned support constructor that maps source/target
tokens and a supported relation into M_core, M_edit, M_contact, and M_preserve.
We freeze operation-level area budgets and post-processing parameters before
evaluation. Thus, task variation changes the geometry of where editing is
allowed, not the editing algorithm itself.
```

Do not use `next_to`, `beside`, `near`, or `on_desk` tasks in the main Core-6
suite unless the corresponding relation constructors are added to
`operation_support_v3.py`. Keep laptop/cactus-style next-to insertion for E5
extension or failure analysis until then. Likewise, do not label T5 as full
material transfer. In the current implementation it is a local surface-pattern
edit implemented through `add_decal + on_surface`.

Code-level caveat for T2: the current `inside` relation is supported, but it is
closer to a normalized host mask than a true container-interior/free-space mask.
For `bowl_apple_inside`, the intended support is the bowl interior, not the
entire bowl surface or rim. If time allows, add a container-interior refinement
before the final run; otherwise report T2 as a first-pass inside-relation probe
and inspect support masks carefully.

Recommended operation-level area order:

```text
T3 decal < T1 accessory ~= T6 removal < T2 inside insertion
< T5 surface pattern < T4 recolor
```

Important wording: Core-6 is a controlled diagnostic suite, not a large-scale
benchmark. If the final submission uses only the six canonical source images,
write "controlled Core-6 suite" throughout the paper and do not imply broad
dataset coverage. If compute allows, expand Core-6 into task categories with
multiple source images:

```text
minimum main-paper suite: 6 canonical source images x 3 seeds
preferred conference suite: 8-10 source images x 3 seeds
stronger supplement: 6 operation categories x 5-10 source examples
```

The preferred next expansion is not to add new operations blindly. Add one or
two extra images for already-supported operations: another accessory insertion,
another decal/surface mark, another local recolor, and one additional exposed
small-object removal if it passes the same frozen protocol.

### Methods

Use four paper-facing methods:

| Paper name | Runner name | Role |
| --- | --- | --- |
| RF reconstruction | `base_only` | reconstruction floor |
| Direct target guidance | `direct_target` | coupled edit baseline |
| Generic support control | `adaptive_full_generic_support` | weak automatic support baseline |
| DeCE-RF | `support_v3_controller_rmsgap` | full method |

Do not put `support_v3_fixed` in Table 1. It belongs to E4 as a component
ablation.

### Matrix

```text
6 tasks x 4 methods x 3 seeds = 72 headline runs
seeds: 10, 11, 12
```

Current completed evidence:

```text
Server results from 2026-06-01 cover the previous "Core-6" task set:
cat_crown, dog_sunglasses, mug_heart, tshirt_star,
backpack_remove_toy_charm, and red_chair_blue.

Map these into the updated design as:
- T1 evidence: dog_sunglasses canonical; cat_crown teaser/supplement.
- T3 evidence: mug_heart canonical; tshirt_star as clothing-decal expansion.
- T4 evidence: red_chair_blue.
- T6 evidence: backpack_remove_toy_charm with a zipper/fabric smoothing caveat.

They do not yet complete updated T2 or T5. Server add-editor probes show
bowl/apple insertion is still pulled toward the bowl rim or external shadow, so
T2 needs a geometry-aware container-interior placement prior. `pillow_blue_stripes`
is not implemented; `tshirt_star` should not be silently renamed as final T5.
```

Before writing final numbers, choose whether the main paper reports the completed
server evidence package as a controlled evidence suite or waits for updated T2
and T5 runs. Do not mix the two without a mapping note.

### Metrics

Primary metrics must be split by region:

| Metric family | Metric | Region | Interpretation |
| --- | --- | --- | --- |
| preservation | outside-mask L1 / RMSE | outside fixed edit mask | lower is better |
| preservation | SSIM / luma SSIM | full image and outside region if available | higher is better |
| edit magnitude | inside-mask L1 / RMSE | inside fixed edit mask | should increase when an edit occurs |
| semantic edit | CLIP target-source delta | full or local crop | useful for add/decal; weak for removal |
| perceptual source fidelity | DINO or LPIPS-source | full image / outside mask | report with caveats |
| recolor-specific | inside blue ratio | fixed chair mask | higher is better for `red_chair_blue` |
| audit | edit success, preservation, locality, artifact, overall | internal visual audit | 1-5 rubric; not a user study |

Every metric table must state that the evaluation mask is fixed per task and is
not each method's own support mask.

Do not collapse the result into a single composite score. Report three metric
families:

| Metric group | Examples | Reviewer question |
| --- | --- | --- |
| Edit success | masked CLIP-T, directional CLIP, local CLIP crop, VQA/binary audit | did the requested edit happen? |
| Preserve fidelity | out-of-mask LPIPS/DINO/SSIM/L1 | did the method avoid unrelated changes? |
| Leakage/locality | out-of-mask change energy, support leakage, preserve drift | is the edit actually localized? |

### Main Table Layout

Table 1 should aggregate over all Core-6 tasks and also show per-operation
breakdowns:

```text
Method | Edit success audit | Outside L1 down | Source SSIM up | Locality audit up | Artifact down | Task wins
```

Then add a compact appendix/supplement table with per-task rows. Do not spend
half a WACV page on a giant per-task table in the main paper.

## Experiment E2: RF-Native Baseline Comparison

### Purpose

This answers the important WACV reviewer question: if the paper is about
Rectified Flow editing, why is DeCE-RF needed over existing RF editing routes?

The claim should not be "existing RF methods are bad." The claim is narrower:
existing RF methods optimize inversion, transport, trajectory quality, or
global source-target editing, while DeCE-RF explicitly targets localized
edit-preserve decomposition under an operation-conditioned support interface.

### Baseline Policy

Separate RF-native baselines from diffusion-only baselines. Mixing all baselines
in one table makes the input/backbone story muddy.

### Minimum Main-Paper RF Suite

Run on the Core-6 tasks if feasible; otherwise run on the four most diagnostic
tasks (`cat_crown`, `mug_heart`, `backpack_remove_toy_charm`, `red_chair_blue`)
and clearly label it as a reduced RF suite.

| Baseline | Condition | Paper role |
| --- | --- | --- |
| Direct target guidance | already internal | simplest RF edit baseline |
| RF inversion + target resampling | matched or closest backbone | reconstruction/editability tradeoff |
| FlowEdit | matched or closest RF backbone | existing inversion-free RF editing |
| RF-Inversion or RF-Edit/RF-Solver-Edit | runnable implementation | inversion-based RF comparator |
| OT-RF / FlowEdit+OTC if runnable | matched or closest backbone | recent WACV RF tradeoff comparator |
| Same-mask FlowEdit variant if implementable | receives the same fixed/support mask | strongest locality fairness probe |
| DeCE-RF | same task protocol | proposed method |

### Matrix Options

Preferred:

```text
6 tasks x 5 RF methods x 3 seeds = 90 rows
```

Feasible main-paper fallback:

```text
4 tasks x 5 RF methods x 3 seeds = 60 rows
```

If an RF baseline cannot run under matched conditions, include it in a
supplementary baseline-audit table with status, failure reason, backbone, and
input condition. Do not silently omit it and do not claim superiority over a
baseline that was not run.

E2 should stay in the main paper, even if compact. A reduced RF suite with
fewer tasks is preferable to moving RF baselines entirely to supplementary
material.

## Experiment E3: Support Geometry Ablation

### Purpose

This isolates the operation-conditioned control geometry. The goal is not to
prove the mask is perfect; it is to show that support estimation changes the
edit-preserve tradeoff in predictable ways.

### Tasks

Use three representative cases:

```text
cat_crown
mug_heart
backpack_remove_toy_charm
```

These cover compact insertion, surface decal, and exposed removal.

### Variants

| Variant | Runner / source | What it tests |
| --- | --- | --- |
| attention-only support | available generic attention variant if runnable | semantic localization only |
| clean-disagreement support | clean-only diagnostic if runnable | source-target clean estimate gap |
| velocity-disagreement support | velocity-only diagnostic if runnable | RF response without relation prior |
| grounding/SAM-only support | external segmentation where available | segmentation is not edit geometry |
| generic support | `adaptive_full_generic_support` | weak automatic support |
| operation-conditioned support | DeCE-RF support path | full geometry estimator |
| manual/same external support | optional upper bound | support quality ceiling, not a fair automatic baseline |

If attention-only or clean-only variants are not stable as runnable methods,
show their support maps as diagnostics rather than adding weak method rows.

All support hyperparameters must be frozen before evaluation:

```text
thresholds, percentiles, min/max area, component count, dilation radius,
blur kernel, relation presets, and fallback policy
```

Do not tune support thresholds per case. If a task needs a different preset,
that preset must be named by operation/relation before seeing final metrics.

### Main Evidence

Figure 4 should show one case, preferably `mug_heart` or `tshirt_star`:

```text
attention evidence | clean disagreement | velocity disagreement |
operation-conditioned support | M_edit / M_preserve | final result
```

The small table should report:

| Metric | Purpose |
| --- | --- |
| support IoU with fixed evaluation mask | support accuracy |
| support precision | whether the support avoids unrelated areas |
| support recall | whether it covers the intended edit region |
| area ratio | whether the method wins by enlarging the mask |
| outside-mask drift | downstream preservation |
| edit success | downstream edit quality |
| support-quality vs edit/preserve correlation | whether support explains final behavior |

This directly borrows the useful lesson from localized editing papers: the
support/mask is part of the experiment, not just a hidden implementation detail.

## Experiment E4: Controller And Robustness Ablation

### Purpose

This answers whether feedback-updated clean-estimate control contributes beyond
fixed DeCE displacement. Because the fixed-vs-feedback gap can be small in
ordinary cases, the strongest evidence should be under perturbation or stronger
edit pressure.

### Methods

| Variant | Runner name | Role |
| --- | --- | --- |
| Fixed DeCE displacement | `support_v3_fixed` | decoupling + operation support, no feedback |
| Adaptive edit only | implement if available | edit-deficit feedback only |
| Adaptive preserve only | implement if available | preserve-drift feedback only |
| Projection only | implement if available | conflict projection without full feedback |
| DeCE-RF | `support_v3_controller_rmsgap` | feedback-updated weights/projection |

### Tasks

Use:

```text
cat_crown
dog_sunglasses
mug_heart
```

These already have controller/stress artifacts and give a clean add/accessory/
surface spread. Add `red_chair_blue` only if recolor feedback curves are
already saved.

### Stress Axes

Use existing stress artifacts where possible:

| Axis | Levels | Evidence |
| --- | --- | --- |
| edit strength | 0.5, 0.75, 1.0, 1.25, 1.5, 2.0 | edit-preserve Pareto curve |
| support perturbation | shrink/base/dilate or support budget variants | locality robustness |
| lambda sweep | edit/preserve weight changes | whether the tradeoff surface improves |
| target prompt strength | weak/strong target conditioning if available | controller response to target pressure |
| late timestep cutoff | cutoff variants if available | checks that gains are not accidental tuning |
| controller trajectory | per-step edit gap, preserve drift, adaptive weights, projection norm | mechanism curve |

Figure 5 should be a Pareto/curve figure rather than another qualitative grid:

```text
x-axis: outside-mask drift
y-axis: edit success or CLIP/local edit score
markers: fixed DeCE vs DeCE-RF across stress levels
```

The text should say: feedback is a stabilizer and robustness component, not the
sole source of the headline gain.

The strongest E4 claim is Pareto-based:

```text
At comparable edit success, DeCE-RF should show lower preserve drift; at
comparable preservation budget, it should keep higher edit success.
```

Single-point fixed-vs-feedback numbers are useful but not enough by themselves,
because a reviewer can dismiss them as hyperparameter choice.

## Experiment E5: Boundary, Extension, And Failure Cases

### Purpose

This keeps the claim honest and turns failures into evidence about the method's
scope.

### Positive Extension Probes

Report separately from the Core-6 main table:

| Probe | Route | Why separate |
| --- | --- | --- |
| `laptop_remove_sticker` | DeCE-RF + high-confidence completion prior | uses an additional completion-clean-delta route |
| `whiteboard_probe_red_star_sticker` | DeCE-RF + replacement target route | uses replacement target formation |

These are useful because papers such as localized editing and operation-centric
editing papers often include targeted case studies, but the method column must
name the extra route.

Do not aggregate these extension routes into the base DeCE-RF mean. They can be
shown as "DeCE-RF + completion prior" or "DeCE-RF + replacement route", but not
as evidence for the base controller alone.

### Failure/Limit Set

Use three or four examples only:

| Task | Failure type | Paper message |
| --- | --- | --- |
| `dog_remove_tennis_ball` | occluded host completion | localization is not enough when the host must be synthesized |
| `whiteboard_remove_yellow_letter` | semantic glyph hallucination | blank removal in text-like fields is outside the claim |
| `fridge_remove_yellow_magnet` or `fridge_remove_peach_magnet` | cluttered-surface damage | support may be accurate while completion fails |
| `dog_replace_tennis_ball_star` | replacement ambiguity | target pressure can fire without clean replacement |

Figure 6 should have one positive extension row and two limitation rows. More
failure examples can go to supplementary material.

Failure labels should use a controlled vocabulary:

```text
support wrong
target response diffuse
target formation weak
removal completion failure
replacement ambiguity
over-preservation
preserve-region drift
```

## Non-RF External Baselines

Non-RF baselines should be used to position the paper, not to become the main
technical contest.

### Main Non-RF Baselines

Run two if time allows:

| Baseline | Input condition | Use |
| --- | --- | --- |
| Same-support inpainting / masked img2img | receives DeCE support or fixed eval mask | locality upper-bound style baseline |
| InstructPix2Pix or equivalent instruction editor | text/instruction only | standard instruction baseline |

Add `DiffEdit` or `ZONE` only if setup is stable. If only one extra can be run,
choose `DiffEdit`; if two, choose `DiffEdit` and `ZONE`.

### Fairness Table

Every external baseline table needs these columns:

```text
method | model family/backbone | user input | mask source | seeds | runnable status | caveat
```

This avoids the unfair comparison trap where a manual-mask method, a text-only
method, and an operation-conditioned method are averaged as if they used the
same information.

## Phased Experiment Execution Plan

Run the experiments as a three-stage decision funnel. Do not start with the
900-output plan. First test whether the method has enough signal to justify the
larger runs.

### Phase 1: Minimum Sanity Check

Purpose: decide whether DeCE-RF has a real edit-preserve signal before spending
days on full evidence generation.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main internal benchmark | 6 categories x 1 example x 4 methods x 3 seeds | 72 |
| E2 RF-native compact baselines | 6 categories x 1 example x 3 RF baselines x 2 seeds | 36 |
| E4 controller variants | 6 categories x 1 example x 5 variants x 2 seeds | 60 |

Total:

```text
about 168 output images, rounded to about 170
```

Runtime estimate:

```text
170 outputs x 2 min/output = 340 min = about 5.7 hours pure generation
half-day wall-clock target with queueing, metrics, and visual checks
```

Phase 1 intentionally omits full E3 and E5. Support/failure evidence is not
worth expanding until the main effect, RF-baseline comparison, and controller
signal are plausible.

Phase 1 Go criteria:

| Gate | Pass signal |
| --- | --- |
| E1 edit signal | DeCE-RF has clear visual edit success on at least 4/6 canonical tasks |
| E1 preservation | DeCE-RF improves outside-mask drift or visual preservation over direct target on most tasks |
| E2 RF comparison | at least one RF baseline shows the expected global/locality weakness that DeCE-RF addresses |
| E4 controller | full DeCE-RF is not worse than fixed/controller variants and shows a plausible Pareto or diagnostic benefit |
| artifacts | each kept run has result image, stats, metadata, and command record |

No-Go criteria:

```text
DeCE-RF fails the requested edit on 3+ of 6 tasks;
direct target is consistently better with no preservation penalty;
RF baselines already solve locality under the same inputs;
controller variants show no interpretable benefit and no stable diagnostics.
```

If Phase 1 is No-Go, stop expansion and revise method/support/target formation
instead of generating more examples.

### Phase 2: Recommended WACV Working Set

Purpose: turn the sanity signal into a reviewer-defensible evidence base. This
is the preferred target before writing final experiment results.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main benchmark | 6 categories x 3 examples x 4 methods x 3 seeds | 216 |
| E2 RF-native baselines | 6 categories x 2 examples x 3 RF baselines x 3 seeds | 108 |
| E3 support ablation | 6 categories x 2 examples x 5 support variants x 2 seeds | 120 |
| E4 controller ablation/stress | 6 categories x 2 examples x 5 controller variants x 2 seeds | 120 |
| E5 extension/failure examples | selected probes | 30 |

Total:

```text
about 594 output images, within the 550-650 recommended band
```

Runtime estimate:

```text
600 outputs x 2 min/output = 1200 min = about 20 hours pure generation
2-3 days realistic wall-clock with metrics, failed reruns, grids, and audits
```

Phase 2 Go criteria:

| Gate | Pass signal |
| --- | --- |
| E1 breadth | the positive pattern remains across multiple examples per category |
| E2 fairness | RF baselines are run or explicitly audited with backbone/input caveats |
| E3 support | frozen support variants explain edit/preserve outcomes without per-case tuning |
| E4 Pareto | DeCE-RF improves or stabilizes the edit-preserve frontier |
| E5 boundary | failures can be categorized as scope limits rather than unexplained collapse |

This is the target execution budget for the first serious WACV draft.

### Phase 3: WACV Robustness Completion

Purpose: only after Phase 1 and Phase 2 look good, expand to a stronger
WACV-ready robustness cache.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main benchmark | 6 categories x 5 examples x 4 methods x 3 seeds | 360 |
| E2 RF-native baselines | 6 categories x 3 examples x 3 RF baselines x 3 seeds | 162 |
| E3 support ablation | 6 categories x 3 examples x 5 support variants x 2 seeds | 180 |
| E4 controller ablation/stress | 6 categories x 3 examples x 5 controller variants x 2 seeds | 180 |
| E5 extension/failure examples | selected probes | 30-50 |

Total:

```text
about 900 output images
```

Use Phase 3 to fill supplement tables, full seed grids, support-mask audits,
and additional failure taxonomy examples. Do not run Phase 3 merely because it
is listed here; run it only if Phase 2 supports the paper claim.

## Main-Paper Figure Budget

Use five figures by default and six at most. The main paper should show about
50-70 result image cells total; more starts to look like a gallery rather than
an algorithms paper.

| Figure | Content | Job |
| --- | --- | --- |
| Figure 1 | teaser: 2 examples, Source/Target/Direct/Generic/DeCE-RF | state the problem and result |
| Figure 2 | method overview with clean-estimate decomposition, support, feedback | explain method |
| Figure 3 | E1 Core-6 qualitative grid | show task diversity |
| Figure 4 | E2 RF-native qualitative comparison | show why RF baselines are not enough |
| Figure 5 | E4 Pareto + timestep diagnostics | prove feedback behavior |
| Figure 6 | E5 extension + failure cases, optional if space permits | mark scope boundary |

Do not add more main-paper grids unless one of these six figures fails to answer
its reviewer question.

Approximate main-paper image-cell budget:

| Figure | Image cells |
| --- | ---: |
| Figure 1 teaser | about 10 |
| Figure 2 method overview | 0 result cells |
| Figure 3 E1 qualitative grid | about 24-30 |
| Figure 4 RF baseline comparison | about 15 |
| Figure 5 Pareto/diagnostics | 0 result cells |
| Figure 6 boundary/extension | about 12-18 |

Main-paper target:

```text
tight: 45-55 result image cells
complete: 60-75 result image cells
```

Supplement can contain 150-300 image cells: full Core-6 grids, all seeds,
support masks, RF baseline examples, Pareto sweeps, and failure taxonomy. The
main paper must still be self-contained without the supplement.

## Main-Paper Table Budget

Use three tables by default:

| Table | Content | Job |
| --- | --- | --- |
| Table 1 | E1 main edit/preserve/leakage metrics | headline quantitative evidence |
| Table 2 | E2 RF-native baseline comparison | answer current-alternatives question |
| Table 3 | E3+E4 component ablation | support geometry and controller evidence |

If space allows, split Table 3 into separate support and controller tables. If
space is tight, move detailed per-task rows and full ablation grids to the
supplement.

## Run Order

### Phase 1 Run Order: Half-Day Sanity Check

Run Phase 1 in this order:

```text
E1 internal main effect -> E2 compact RF baselines -> E4 controller variants
-> fixed-mask metrics -> quick visual audit -> Go/No-Go decision
```

#### Phase 1 E1: Internal Main Effect

Six canonical categories, four internal methods, three seeds. For Phase 1, run
the strict six-case sanity set:

```bash
TASKS="dog_sunglasses bowl_apple_inside mug_heart red_chair_blue pillow_blue_stripes backpack_remove_toy_charm" \
METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
SKIP_EXISTING=1 \
bash scripts/run_pretty_matrix.sh
```

`cat_crown` should be run as a T1 teaser/supplement example, not as a seventh
canonical E1 row unless the budget explicitly allows an expanded T1 category.
If `bowl_apple_inside` or `pillow_blue_stripes` is not implemented in time,
label the run as incomplete Core-6 rather than silently replacing T2/T5 with a
different operation.

Current server evidence already provides a completed old-category sanity package:

```text
TASKS="cat_crown dog_sunglasses mug_heart tshirt_star backpack_remove_toy_charm red_chair_blue"
METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap"
SEEDS="10 11 12"
72/72 paper-facing runs complete on server
```

Use this as evidence for T1/T3/T4/T6 and supplement/diagnostics, not as proof
that updated T2/T5 are solved.

Expected output count:

```text
6 task instances x 4 methods x 3 seeds = 72

Optional expanded T1 teaser/supplement run:

7 task instances x 4 methods x 3 seeds = 84
```

#### Phase 1 E2: Compact RF Baselines

Run three RF-native baselines on the same six tasks with two seeds:

```text
FlowEdit
RF-Inversion or RF-Solver/Edit
OT-RF / FlowEdit+OTC if runnable
```

Use seeds:

```text
10, 11
```

Direct target guidance is reused from E1 as the simplest internal RF baseline;
do not double-count it in the 36-output E2 budget.

Expected output count:

```text
6 x 3 x 2 = 36
```

If only one external RF baseline is stable on day one, run FlowEdit first, then
one inversion/OT-style method. Record unavailable baselines in the baseline
audit rather than dropping them silently.

#### Phase 1 E4: Controller Variants

Use the currently runnable controller/fixed variants for the first pass:

```bash
TASKS="dog_sunglasses bowl_apple_inside mug_heart red_chair_blue pillow_blue_stripes backpack_remove_toy_charm" \
METHODS="support_v3_fixed support_v3_controller_rmsgap support_v3_controller_progress support_v3_controller_hybrid support_v3_controller_rmsgap_normgate" \
SEEDS="10 11" \
SKIP_EXISTING=1 \
bash scripts/run_pretty_matrix.sh
```

Expected output count:

```text
6 x 5 x 2 = 60
```

If these five variants are too redundant after inspection, keep
`support_v3_fixed`, `support_v3_controller_rmsgap`, and the two most
diagnostic controller alternatives, then reserve full edit-strength/support
perturbation sweeps for Phase 2.

#### Phase 1 Evaluation

After the three blocks, evaluate with fixed masks and build quick grids:

```bash
python scripts/evaluate_paper_metrics.py
python scripts/summarize_fixed_mask_audit.py
python scripts/make_paper_grids.py
```

Use `SKIP_EXISTING=1` when reusing completed runs, but record that choice in
the experiment log. Phase 1 is a Go/No-Go gate, not final paper evidence.

### Phase 2 Run Order: Recommended WACV Working Set

Only start Phase 2 if Phase 1 passes the Go criteria.

#### Phase 2 E1 Expansion

Expand each of the six categories to three source examples:

```text
6 categories x 3 examples x 4 methods x 3 seeds = 216 outputs
```

Keep the same four methods:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

#### Phase 2 E2 Expansion

Run the compact RF suite on two examples per category:

```text
6 categories x 2 examples x 3 RF baselines x 3 seeds = 108 outputs
```

Prioritize:

```text
FlowEdit
RF-Inversion or RF-Solver/Edit
OT-RF / FlowEdit+OTC if runnable
```

#### Phase 2 E3 Support Ablation

Run support variants on two examples per category:

```bash
TASKS="<phase2_support_task_list>" \
METHODS="adaptive_full_generic_support adaptive_full_attention_only adaptive_full_clean_only adaptive_full_velocity_only adaptive_full_support_v3" \
SEEDS="10 11" \
SKIP_EXISTING=1 \
bash scripts/run_pretty_matrix.sh
```

Expected output count:

```text
6 categories x 2 examples x 5 support variants x 2 seeds = 120
```

Before running E3, freeze all support thresholds, area limits, component
selection, dilation, blur, and relation presets.

#### Phase 2 E4 Pareto / Controller Evidence

Run controller variants on two examples per category:

```text
6 categories x 2 examples x 5 variants x 2 seeds = 120 outputs
```

Then run a targeted stress sweep on the most diagnostic subset. Use the existing
stress script when applicable:

```bash
TASKS="cat_crown dog_sunglasses mug_heart" \
METHODS="support_v3_fixed support_v3_controller_rmsgap" \
SEEDS="10 11" \
SKIP_EXISTING=1 \
bash scripts/run_controller_stress_sweeps.sh
```

The stress sweep is for Pareto and trajectory figures. Do not let it replace
the fixed variant matrix.

#### Phase 2 E5 Boundary / Extension

Select about 30 outputs:

```text
positive extension probes
support failure
diffuse target response
removal completion failure
replacement ambiguity
```

Do not aggregate extension routes into the base DeCE-RF mean.

### Phase 3 Run Order: Robust WACV Completion

Only run Phase 3 if Phase 2 supports the paper claim and the remaining risk is
sample breadth rather than method correctness.

Phase 3 expands to:

```text
E1: 6 categories x 5 examples x 4 methods x 3 seeds = 360
E2: 6 categories x 3 examples x 3 RF baselines x 3 seeds = 162
E3: 6 categories x 3 examples x 5 support variants x 2 seeds = 180
E4: 6 categories x 3 examples x 5 controller variants x 2 seeds = 180
E5: 30-50 selected outputs
total: about 900 outputs
```

Use Phase 3 to complete supplement grids, full per-task tables, and robustness
audits. It should not change the core method story.

### Fixed-Control Cache Note

```bash
TASKS="dog_sunglasses bowl_apple_inside mug_heart red_chair_blue pillow_blue_stripes backpack_remove_toy_charm" \
METHODS="support_v3_fixed" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh
```

This supports E4, not Table 1.

### Non-RF Baselines

Run same-support inpainting first because the repository already has
`scripts/same_support_inpaint_baseline.py`. Then run one instruction baseline.

Non-RF baselines are not part of Phase 1. Add them after Phase 2 if the RF
comparison and main internal evidence are stable.

### Visual Audit

Freeze the audit rubric before scoring final outputs:

```text
edit success: 1-5
source preservation: 1-5
locality: 1-5
artifact severity: 1-5, lower is better
overall: 1-5
failure type: controlled vocabulary
```

Manual scores are internal visual audit scores, not a human study.

## Acceptance Gates

### Phase 1 Gate: Continue Or Stop

After about 170 outputs, decide whether to continue:

| Gate | Continue if |
| --- | --- |
| E1 main effect | DeCE-RF visibly succeeds on at least 4/6 canonical examples |
| preservation | DeCE-RF improves over direct target in outside-mask drift or visual preservation on most examples |
| RF context | compact RF baselines expose a locality or preservation gap that DeCE-RF is designed to address |
| controller signal | full DeCE-RF is competitive with variants and shows interpretable feedback diagnostics |
| reproducibility | results are stable enough across seeds 10 and 11/12 to justify expansion |

If this gate fails, revise the method, target prompts, support policy, or task
set before expanding.

### Phase 2 Gate: Write Main Experiment Section

The experiment section is ready to write when these are true:

| Gate | Required artifact |
| --- | --- |
| Phase 2 E1 locked | combined metrics CSV/JSON and seed grids |
| evaluation masks frozen | one fixed mask per task |
| fixed-control cache | `support_v3_fixed` and controller variants for the declared subset |
| RF baseline minimum | FlowEdit plus one inversion-style RF baseline, or documented failure audit |
| support figure | saved support diagnostic panels |
| controller figure | saved Pareto or trajectory plot |
| failure figure | 2-3 boundary examples with labels |
| visual audit | completed rubric CSV and summary |

If the RF baseline minimum cannot be completed, the paper can still be drafted,
but the experimental claim must be softened to a controlled internal study plus
baseline availability audit.

### Phase 3 Gate: Final Robustness

Phase 3 is complete when the supplement has full seed grids, per-task metric
tables, support masks, RF-baseline audit details, Pareto sweeps, and failure
taxonomy examples. This gate strengthens the submission but should not be
required before drafting the main paper.
