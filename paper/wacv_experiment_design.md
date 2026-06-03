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
| Why not use existing RF editing or preservation-aware RF methods? | E2 RF-native and preservation-aware baseline comparison | Table 2 |
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
| T5 | Surface material strip editing | `add_decal` + `on_surface` | add a medium-size material strip while preserving host silhouette and background | edit-preserve Pareto under stronger surface appearance change |
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
| T1 Attached accessory addition | `cat_crown` | visual pass; replaces dog sunglasses after eyewear-placement audit failure |
| T2 Container-constrained spatial insertion | `bowl_apple_inside` | implemented seed-10 gate; uses `inside_container` relation |
| T3 Surface decal / logo addition | `tshirt_star` | visual pass; clearer surface-local decal than mug heart for the main grid |
| T4 Local recoloring | `red_chair_blue` | existing candidate; keep only after visual audit confirms local recolor |
| T5 Surface material strip editing | `pillow_vertical_fabric_strip` | implemented seeds 10/11/12 gate; perspective-aligned blue silk strip with softened top seam and clean bottom edge |
| T6 Simple exposed-object removal | `backpack_remove_toy_charm` | existing, exposed small-object removal |

`mug_heart` remains a diagnostic T3 example, but it is not the Phase 1 canonical T3 row because the visible edit is too small for the main grid. Use `tshirt_star` for the strict T3 clothing/surface-decal row.

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
< T5 surface material strip < T4 recolor
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
The revised strict Phase 1 matrix is complete for 6 tasks x 4 paper-facing
methods x seeds 10, 11, and 12:
cat_crown, bowl_apple_inside, tshirt_star, red_chair_blue,
pillow_vertical_fabric_strip, and backpack_remove_toy_charm.

Legacy 2026-06-01 server results cover the previous task set and should be
used only as supplementary diagnostics. In that mapping, dog_sunglasses is a
diagnostic T1 example and mug_heart is a diagnostic T3 example, not canonical
strict rows.
```

Before writing final numbers, report the revised strict matrix from
`experiments/support_v3_2026-06-02/` as the active evidence suite. Keep legacy
rows in a separately labeled diagnostic/supplement section.

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

## Experiment E2: RF-Native And Preservation-Aware Baseline Comparison

### Purpose

E2 should answer a stronger reviewer question than the original RF-native
baseline comparison:

```text
Do existing RF-native and preservation-aware RF editing methods already solve
localized edit-preserve control?
```

Keep E2 in the main experimental chain; do not create a separate E6 for
preservation-aware baselines. The point is not to make a broad claim that
DeCE-RF beats every RF method. The point is to test whether methods designed
for RF editing, transport quality, inversion quality, reconstruction fidelity,
or preservation already remove the need for an explicit localized controller.

The paper-facing claim should be:

```text
Existing RF-native and preservation/fidelity-oriented editing methods improve
source-to-target editing or global fidelity, but they do not explicitly model
operation-conditioned edit/preserve geometry. DeCE-RF targets this localized
edit-preserve tradeoff through M_edit, M_core/contact, and M_preserve.
```

### Internal Structure

Organize E2 as one experiment with two required layers and one optional fairness
probe:

```text
E2-A: Native RF editing baselines
E2-B: Preservation-aware / fidelity-oriented RF baselines
E2-C: Optional support-matched compact comparison
```

Main-paper presentation can still be one compact table. The subsection split is
for argument clarity, not for adding another experiment slot.

### E2-A: Native RF Editing Baselines

This is the part already completed in the current workspace. It asks whether
off-the-shelf RF editing methods naturally produce localized edit-preserve
behavior.

Current completed reduced RF-native suite:

```text
6 tasks x 4 methods x 3 seeds = 72 outputs
```

Completed methods:

| Method | Type | Extra support? | Current status |
| --- | --- | --- | --- |
| FlowEdit | RF-native editing | no | completed on strict Core-6 seeds 10/11/12 |
| FlowAlign | RF-native editing | no | completed on strict Core-6 seeds 10/11/12 |
| SplitFlow | RF-native editing | no | completed on strict Core-6 seeds 10/11/12 |
| DeCE-RF | localized RF controller | operation support | completed on strict Core-6 seeds 10/11/12 |

Current E2-A artifacts:

```text
outputs/e2_rf_comparison/
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

This completed table should be described as a reduced target-mode RF-native
comparison. Do not write it as evidence that DeCE-RF beats all RF baselines.

### E2-B: Preservation-Aware / Fidelity-Oriented RF Baselines

This is the missing upgrade. It answers whether RF methods that explicitly care
about fidelity, inversion, transport, or preservation already solve the
non-edit-region protection problem.

Recommended priority order:

| Priority | Baseline slug | Paper-facing candidate | Reason to include | Current baseline status |
| --- | --- | --- | --- | --- |
| 1 | `rf_solver_edit` | RF-Solver-Edit / RF-Edit | strongest direct response to fidelity-aware RF editing | registered; repo/env present; strict run currently blocked by FLUX.1-dev access |
| 2 | `ot_rf_otip` | OT-RF / OTIP-style method | transport/fidelity-aware RF comparator | registered; repo/env/adapter pending |
| 3 | `reflex` | ReFlex | preservation/trajectory-aware RF alternative | registered; repo/env present; help smoke passes; strict run currently blocked by FLUX.1-dev access |
| 4 | `dvrf` | DVRF / Delta Velocity RF | delta-velocity/path-aware RF alternative | registered; repo/env/adapter pending |

Minimum acceptable E2-B:

```text
1 preservation-aware RF baseline x 6 tasks x 3 seeds = 18 new outputs
```

Resource-saving fallback if compute is tight:

```text
1 preservation-aware RF baseline x 6 tasks x 2 seeds = 12 new outputs
```

Preferred stronger E2-B:

```text
2 preservation-aware RF baselines x 6 tasks x 3 seeds = 36 new outputs
```

Use the same strict Core-6 source images, prompts, seeds, fixed evaluation masks,
and metric code. If a preservation-aware baseline cannot be made runnable under
a comparable condition, include it in a baseline-audit disclosure table with
failure reason, required backbone/checkpoint, and input condition. Do not
silently omit it and do not claim superiority over an unrunnable method.

Current audit caveat: RF-Solver-Edit (`rf_solver_edit`) and ReFlex (`reflex`)
are registered E2-B candidates but their strict attempts are blocked by
FLUX.1-dev gated checkpoint access. OT-RF / OTIP (`ot_rf_otip`) and DVRF
(`dvrf`) are registered as planned E2-B candidates but still need exact repo
verification, environment creation, smoke testing, and Core-6 adapters. If no
preservation-aware baseline becomes runnable, report the blocker explicitly in
the audit table instead of claiming superiority over unrunnable methods.

### E2-C: Optional Support-Matched Fairness Probe

If time allows, add a small support-matched probe to address the reviewer
question:

```text
Is DeCE-RF better only because it receives an edit mask/support?
```

Minimum support-matched design:

```text
6 tasks x 1 support condition x 2 seeds
```

Candidate rows:

| Method | Support condition | Role |
| --- | --- | --- |
| Direct target + same M_edit | fixed DeCE/support mask | checks whether target guidance alone benefits from the same locality |
| FlowEdit + same M_edit | fixed DeCE/support mask if implementable | checks RF-native baseline under matched locality |
| Preservation-aware RF + same M_edit | fixed DeCE/support mask if implementable | checks whether preservation-aware RF still lacks edit/preserve decomposition |
| Fixed DeCE | operation support, no feedback | component reference |
| DeCE-RF | operation support + feedback | full method |

This is optional. It should not replace E2-A or E2-B. It can be a small main-text
sanity row or a supplement table.

### Main E2 Table Layout

Use one compact table rather than splitting baselines across multiple unrelated
tables. The table must include method type and support condition so that the
comparison is transparent.

Recommended columns:

| Column | Purpose |
| --- | --- |
| Method | baseline or proposed method |
| Type | RF target guidance / RF-native editing / preservation-aware RF / localized RF controller |
| Extra support? | no, native, same fixed support, or operation support |
| Edit success | masked/local CLIP delta or visual audit success |
| Preserve fidelity | outside-mask LPIPS/SSIM/DINO/source where available |
| Leakage/locality | outside-mask L1 or drift energy |
| Notes | gated checkpoint, adapter caveat, or support condition if needed |

A minimal main-paper table can aggregate over tasks and seeds, with per-task
breakdowns in supplement. Do not collapse edit success and preservation into a
single composite score.

### Updated Matrix Options

Current completed E2-A:

```text
6 tasks x 4 methods x 3 seeds = 72 outputs
```

Minimum upgraded E2:

```text
E2-A completed 72 outputs
+ E2-B one preservation-aware RF baseline x 6 tasks x 3 seeds = 18 outputs
= 90 E2 outputs total
```

Stronger upgraded E2:

```text
E2-A completed 72 outputs
+ E2-B two preservation-aware RF baselines x 6 tasks x 3 seeds = 36 outputs
= 108 E2 outputs total
```

If support-matched E2-C is added, keep it compact and clearly mark it as a
fairness probe rather than a headline baseline table.

### Wording Rules

Use these safe formulations:

```text
E2 evaluates whether existing RF-native and preservation-oriented editing
methods already solve localized edit-preserve control.
```

```text
While these methods can improve source-to-target editing or global fidelity,
they do not explicitly construct operation-conditioned edit and preserve
regions, leading to a weaker localized edit-preserve tradeoff under our fixed
mask evaluation.
```

Avoid these unsafe formulations:

```text
DeCE-RF beats all RF editing methods.
```

```text
Existing RF methods are bad.
```

```text
Preservation-aware RF baselines are only appendix material.
```

## Experiment E3: Support Geometry Ablation

### Purpose

This isolates the operation-conditioned control geometry. The goal is not to
prove the mask is perfect; it is to show that support estimation changes the
edit-preserve tradeoff in predictable ways.

### Tasks

Use three representative cases:

```text
cat_crown
tshirt_star
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
tshirt_star
pillow_vertical_fabric_strip
```

These cover attached accessory, surface decal, and the harder T5 surface
material-strip case. Add `red_chair_blue` only if recolor feedback curves are
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
| E2 RF comparison | at least one external RF baseline is runnable under documented conditions, or the table is explicitly framed as a baseline audit/reduced comparison |
| E4 controller | full DeCE-RF shows a plausible Pareto or trajectory diagnostic benefit beyond fixed/controller variants |
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
| Figure 4 | E2 RF-native qualitative comparison | baseline audit or reduced RF comparison, only after runnable validation |
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
TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm" \
METHODS="base_only direct_target adaptive_full_generic_support support_v3_controller_rmsgap" \
SEEDS="10 11 12" \
SKIP_EXISTING=1 \
bash scripts/run_pretty_matrix.sh
```

`cat_crown` should be run as a T1 teaser/supplement example, not as a seventh
canonical E1 row unless the budget explicitly allows an expanded T1 category.
If `bowl_apple_inside` or `pillow_vertical_fabric_strip` is not implemented in time,
label the run as incomplete Core-6 rather than silently replacing T2/T5 with a
different operation.

Legacy old-category runs live under `paper/archive_old_core6_20260602/` and
should not appear in active reproduction commands.

Expected output count:

```text
6 task instances x 4 methods x 3 seeds = 72

Optional expanded T1 teaser/supplement run:

7 task instances x 4 methods x 3 seeds = 84
```

#### Phase 1 E2: Compact RF Baselines

Planned status: external RF-native baselines are not yet validated for the
revised strict matrix. Run them on the same six tasks with two seeds only after
the baseline wrappers pass a runnable-validation audit:

```text
FlowEdit / FlowAlign / SplitFlow for completed E2-A
rf_solver_edit = RF-Solver-Edit / RF-Edit
ot_rf_otip = OT-RF / OTIP-style
reflex = ReFlex
dvrf = DVRF / Delta Velocity RF
```

Use seeds:

```text
10, 11
```

Direct target guidance is reused from E1 as the simplest internal RF baseline;
do not double-count it in the 36-output E2 budget.

Target output count after validation:

```text
6 x 3 x 2 = 36
```

If only one external RF baseline is stable on day one, run FlowEdit first, then
one inversion/OT-style method. Record unavailable baselines in the baseline
audit rather than dropping them silently. Until this validation is complete,
write the paper claim as a controlled internal comparison plus pending/reduced
RF-baseline evidence, not as superiority over RF baselines.

#### Phase 1 E4: Controller Variants

Use the currently runnable controller/fixed variants for the first pass:

```bash
TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm" \
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
Completed E2-A: FlowEdit, FlowAlign, SplitFlow
E2-B candidates: rf_solver_edit, ot_rf_otip, reflex, dvrf
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
TASKS="cat_crown bowl_apple_inside tshirt_star" \
METHODS="support_v3_fixed support_v3_controller_rmsgap" \
SEEDS="10 11" \
SKIP_EXISTING=1 \
bash scripts/run_controller_stress_sweeps.sh
```

The stress sweep is for Pareto and trajectory figures. Do not let it replace
the fixed variant matrix, and do not support E4 with a single fixed-vs-full
table alone.

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
TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm" \
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
| RF context | compact RF baselines are runnable or explicitly audited; any locality/preservation gap claim is limited to completed matched outputs |
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
| RF baseline minimum | FlowEdit plus one inversion-style RF baseline, or an explicitly labeled documented failure audit/reduced comparison |
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
