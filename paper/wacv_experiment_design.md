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

## Experiment E2: Backbone-Controlled And Preservation-Aware RF Comparison

### Purpose

E2 is redesigned as a layered fairness experiment rather than a flat baseline
leaderboard. It should answer four reviewer questions:

```text
1. What part of the result is due to the RF backbone itself?
2. Under the same SD3 backbone, does DeCE-RF improve localized edit-preserve
   behavior over RF-native and fidelity-oriented editing baselines?
3. Do official/native preservation-aware RF editors already solve the same
   localized edit-preserve tasks in practice, even when they use FLUX or another
   backbone?
4. Is DeCE-RF's advantage merely caused by having an edit mask/support input,
   or by the clean-estimate edit/preserve controller built on top of that
   support?
```

The central fairness rule is:

```text
Same-backbone comparison = algorithmic evidence.
Different-backbone comparison = native implementation / ecosystem context.
```

Do not place `DeCE-RF-SD3` and `ReFlex-FLUX`, `RF-Edit-FLUX`, or
`FlowEdit-FLUX` in a single undifferentiated leaderboard and then claim a pure
algorithmic win. Backbone differences can change prompt following, reconstruction
fidelity, inversion error, image priors, resolution, memory use, and preserve
behavior. Cross-backbone rows are useful, but they must be interpreted as native
implementation context.

The paper-facing E2 claim should be:

```text
Backbone-controlled SD3 results provide the algorithmic evidence for DeCE-RF;
native preservation-aware RF / FLUX rows test whether strong off-the-shelf RF
editors solve the same localized edit-preserve setting, while backbone and input
condition differences are reported explicitly.
```

### E2 Structure

Use four layers:

```text
E2.1 Backbone calibration
E2.2 Same-backbone SD3 algorithm comparison
E2.3 Native preservation-aware RF comparison
E2.4 Support-matched diagnostic
```

Optional extension:

```text
E2.5 Cross-backbone DeCE-RF transfer probe
```

E2.5 is useful only if a FLUX version of DeCE-RF is implemented and smoke-tested.
It should not be allowed to destabilize the main SD3 paper claim.

### E2.1 Backbone Calibration

#### Question

Before comparing methods across SD3 and FLUX, estimate the baseline behavior of
each backbone on the same Core-6 inputs.

This layer asks:

```text
How much reconstruction/preservation error does each backbone already have
before any sophisticated editing method is applied?
```

#### Methods

Use the simplest two controls per backbone:

| Method family | SD3 row | FLUX row | Role |
| --- | --- | --- | --- |
| Source reconstruction / inversion | `base_only` or SD3 reconstruction row | FLUX reconstruction/inversion row if runnable | estimates reconstruction floor |
| Direct target guidance | `direct_target` | FLUX direct target or method-native target guidance if runnable | estimates naive edit/preserve behavior |

For SD3, the strict E1 cache already provides `base_only` and `direct_target`
for the canonical Core-6 tasks. For FLUX, this calibration should run only after
FLUX access and the adapter are validated. If FLUX remains blocked, E2.1 should
report a status/audit row rather than inventing cross-backbone numbers.

#### Matrix

Minimum calibration:

```text
6 strict Core-6 tasks x 2 backbones x 2 base methods x 2 seeds = 48 outputs
seeds: 10, 11
```

Preferred calibration if FLUX access is stable:

```text
6 strict Core-6 tasks x 2 backbones x 2 base methods x 3 seeds = 72 outputs
seeds: 10, 11, 12
```

Phase 2 can reuse existing SD3 `base_only` and `direct_target` outputs; only the
FLUX calibration rows would be new generation if FLUX becomes runnable.

#### Metrics

Report a compact calibration table:

| Backbone | Base method | Reconstruction LPIPS down | Reconstruction SSIM up | Direct-target edit up | Direct-target preserve down | NFE | Resolution | Runtime | Memory |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |

The purpose is not to prove SD3 or FLUX is better. The purpose is to show that
backbone-dependent reconstruction floors are measured before any cross-backbone
interpretation.

### E2.2 Same-Backbone SD3 Algorithm Comparison

#### Question

This is the primary algorithmic comparison:

```text
Under the same SD3 backbone and fixed evaluation masks, does DeCE-RF improve the
localized edit-preserve tradeoff over RF-native and preservation/fidelity-aware
controls?
```

Only this layer should support the main method-level claim.

#### Required Conditions

All rows must use:

```text
same backbone: SD3
same source images
same source / target prompts
same max image size and normalization policy
same fixed evaluation masks, not each method's own support mask
same metric implementation
same seed count per method whenever the method exposes seed control
same no-cherry-pick qualitative selection policy
```

#### Method Rows

Use two groups: existing validated SD3 RF rows, and preservation-aware controls.

| Paper row | Runner / source | Backbone | Input condition | Support condition | Role | Current status |
| --- | --- | --- | --- | --- | --- | --- |
| Direct target guidance | `direct_target` | SD3 | source image + source/target prompts | no support | simplest coupled target RF control | complete from E1 strict cache |
| FlowEdit-SD3 | `flowedit` | SD3 | source image + source/target prompts | no support | RF-native source-to-target editing | complete strict Core-6 seeds 10/11/12 |
| FlowAlign-SD3 | `flowalign` | SD3 | source image + source/target prompts | no support | RF alignment/editing baseline | complete strict Core-6 seeds 10/11/12 |
| SplitFlow-SD3 | `splitflow` | SD3 | source image + source/target prompts | no support | flow decomposition/aggregation baseline | complete strict Core-6 seeds 10/11/12 |
| Fixed DeCE-SD3 | `support_v3_fixed` | SD3 | prompts + operation/relation tokens | operation support, no feedback | isolates fixed decoupled displacement and support geometry | use as E2.2 preservation-control row if strict fixed cache is complete |
| OT-RF / OTIP-SD3 | `ot_rf_otip_sd3` if verified | SD3 | source image + prompts | no/native support | desired fidelity/transport-aware RF baseline | optional; only if repo/backbone/adapter are verified |
| RF-Edit-SD3 / RF-Solver-SD3 | `rf_solver_edit_sd3` if portable | SD3 | source image + prompts | no/native support | desired preservation-aware RF baseline | optional; use only if port is real, not FLUX masquerading as SD3 |
| DeCE-RF-SD3 | `support_v3_controller_rmsgap` | SD3 | prompts + operation/relation tokens | operation support + feedback/projection | full method | complete from E1 strict cache |

Current minimum E2.2 is already strong because `FlowEdit`, `FlowAlign`, and
`SplitFlow` are complete under the strict SD3 protocol. The preservation-aware
upgrade should add `Fixed DeCE` as an internal preservation-control row and then
attempt one true same-backbone external preservation-aware row only if it can be
verified under SD3. Do not relabel a FLUX-only method as an SD3 baseline.

#### Matrix

Current strict canonical E2.2 readout:

```text
6 tasks x 5 core SD3 rows x 3 seeds = 90 analyzed rows
rows: direct_target, FlowEdit, FlowAlign, SplitFlow, DeCE-RF
```

Preservation-control upgrade:

```text
+ 6 tasks x Fixed DeCE x 3 seeds = 18 rows
```

Optional same-backbone preservation-aware external row:

```text
+ 6 tasks x 1 verified SD3 preservation-aware RF baseline x 2-3 seeds = 12-18 rows
```

Phase 2 expansion target:

```text
6 categories x 2 examples x 6 SD3 rows x 3 seeds = 216 rows
rows: direct_target, FlowEdit, FlowAlign, SplitFlow, Fixed DeCE, DeCE-RF
optional verified SD3 preserve-aware external baseline: +36 rows
```

Phase 3 expansion target:

```text
6 categories x 3 examples x 6 SD3 rows x 3 seeds = 324 rows
optional verified SD3 preserve-aware external baseline: +54 rows
```

#### Claim Boundary

Safe E2.2 conclusion:

```text
Under the same SD3 backbone, same source/target prompts, and fixed evaluation
masks, DeCE-RF improves localized edit-preserve behavior over RF-native SD3
editing baselines and fixed decoupled preservation controls.
```

If no verified external SD3 preservation-aware baseline is available, write:

```text
Preservation-aware RF methods whose public implementations are FLUX-bound are
reported in E2.3 native implementation context; the same-backbone algorithmic
claim is restricted to SD3-runnable baselines and internal preservation controls.
```

### E2.3 Native Preservation-Aware RF Comparison

#### Question

This layer asks a practical systems question:

```text
Can strong official/native RF editors, including preservation-aware FLUX methods,
solve the localized edit-preserve tasks out of the box?
```

This is not the main algorithmic fairness table. It is a native implementation
comparison with backbone and input caveats.

#### Candidate Rows

| Baseline slug | Paper-facing label | Native backbone | Preservation mechanism | Input condition | Current status |
| --- | --- | --- | --- | --- | --- |
| `rf_solver_edit` | RF-Solver-Edit / RF-Edit | FLUX.1-dev in current public route | RF solver / inversion accuracy + structural preservation | source image + target/edit prompt, method-native | repo/env present; strict run blocked by FLUX.1-dev access |
| `reflex` | ReFlex | FLUX.1-dev | trajectory/attention or feature adaptation for structure/background preservation | source image + target prompt | repo/env present; help smoke passes; strict run blocked by FLUX.1-dev access |
| `fireflow` | FireFlow | FLUX.1-dev | RF-flow editing/inversion route | source image + prompts | strict run blocked by FLUX.1-dev access |
| `stable_flow` | stable-flow | FLUX.1-dev | flow-layer/feature editing route | source image + prompts | adapter pending |
| `ot_rf_otip` | OT-RF / OTIP-style | SD3/FLUX/TBD after repo verification | optimal transport / inversion fidelity | TBD | registered; repo/env/adapter pending |
| `dvrf` | DVRF / Delta Velocity RF | TBD | delta-velocity/path-aware RF control | TBD | registered; repo/env/adapter pending |
| DeCE-RF-SD3 | `support_v3_controller_rmsgap` | SD3 | operation support + clean edit/preserve controller | source image + prompts + operation/relation tokens | complete; contextual anchor only |

#### Matrix

Minimum if one native baseline becomes runnable:

```text
1 native preservation-aware baseline x 6 strict tasks x 2 seeds = 12 outputs
```

Preferred compact native comparison:

```text
2 native preservation-aware baselines x 6 strict tasks x 2 seeds = 24 outputs
```

Stronger version if FLUX access and adapters are stable:

```text
2 native preservation-aware baselines x 6 strict tasks x 3 seeds = 36 outputs
```

Do not expand E2.3 to Phase 2 multi-example breadth until at least one native
baseline completes the canonical strict Core-6 set. Native rows are expensive to
set up and are not allowed to replace the same-backbone E2.2 evidence.

#### Table Caption Requirement

Any E2.3 table must say:

```text
Backbones differ across methods. This native implementation comparison evaluates
whether off-the-shelf preservation-aware RF editors solve our localized
edit-preserve tasks in practice; algorithm-level conclusions are drawn from the
same-backbone SD3 comparison.
```

### E2.4 Support-Matched Diagnostic

#### Question

This diagnostic answers:

```text
Is DeCE-RF better only because it receives localization/support information?
```

Run this as a compact diagnostic, not as a headline external-baseline table.

#### Design

Give selected baselines the same binary edit support `M_edit` when possible, but
not the full DeCE-RF geometry or controller internals.

Allowed for support-matched diagnostic rows:

```text
M_edit or binary edit mask
```

Not allowed for baseline rows:

```text
M_core
M_contact
M_preserve
adaptive feedback weights
clean-estimate projection
operation-conditioned preserve controller
```

Rows:

| Diagnostic row | Support condition | Role |
| --- | --- | --- |
| Direct target + same edit mask | same binary `M_edit`; image/output blending if needed | tests whether target guidance plus localization is enough |
| FlowEdit + same edit mask | same binary `M_edit`; only if wrapper is stable | tests RF-native baseline under matched locality |
| Preserve-aware RF + same edit mask | same binary `M_edit`; only if method supports it | optional native/locality diagnostic |
| Fixed DeCE | full operation support, no feedback | component reference |
| DeCE-RF | operation support + feedback/projection | full method |

If a baseline does not support masks, output blending may be used only as a
clearly labeled diagnostic:

```text
output = M_edit * edited_output + (1 - M_edit) * source_image
```

Do not present post-hoc blending as a fair main baseline. It can create boundary
artifacts and gives a different input/processing condition.

#### Matrix

Recommended compact subset:

```text
3 representative tasks x 1 case x 4-5 diagnostic rows x 2 seeds = 24-30 outputs
```

Recommended tasks:

```text
cat_crown
tshirt_star
backpack_remove_toy_charm
```

These cover attached accessory addition, surface decal, and exposed-object
removal. If recolor fairness becomes reviewer-critical, add `red_chair_blue` as
a fourth task.

### E2.5 Optional Cross-Backbone Transfer Probe

This is optional and should be attempted only after the SD3 story is stable.

Clean factorial probe:

```text
3 tasks x 1 case x 2 methods x 2 backbones x 2 seeds = 24 outputs
methods: direct target, DeCE-RF
backbones: SD3, FLUX
```

If DeCE-RF-FLUX is not implemented, do not run this probe. Instead state:

```text
The current DeCE-RF implementation is SD3-specific. Cross-backbone transfer to
FLUX requires a separate implementation and validation.
```

### E2 Metrics And Normalization

All E2 metrics must be computed in image space or backbone-independent feature
space. Do not use backbone-specific latent norms as primary cross-backbone
metrics.

Use fixed evaluation masks:

```text
M_eval_edit
M_eval_preserve
```

Every method is evaluated with the same masks. DeCE-RF may use its own support
for control, and baselines may use no support, but scoring uses fixed external
evaluation masks.

Metric families:

| Family | Metrics | Notes |
| --- | --- | --- |
| Edit success | masked/local CLIP-T, directional CLIP, VQA/binary audit where available | report removal/recolor caveats |
| Preserve fidelity | out-of-mask LPIPS, DINO distance, SSIM, L1/RMSE | primary preserve evidence |
| Leakage/locality | outside-mask change energy, boundary leakage, preserve-region drift | shows whether edits leak |
| Efficiency | NFE, runtime, peak memory, resolution | needed when comparing native systems |

Add a backbone-normalized preservation score for any cross-backbone table:

```text
Excess Preserve Error(method, backbone)
= E_pres(method, backbone) - E_pres(reconstruction, backbone)
```

Use the difference form by default because it has a simple interpretation:

```text
After subtracting the reconstruction floor of that backbone, how much additional
preserve-region damage did the editing method introduce?
```

If ratio form is reported in supplement, define epsilon explicitly:

```text
Relative Preserve Drift = E_pres(method, backbone) / (E_pres(reconstruction, backbone) + epsilon)
```

### E2 Table Layout

Use explicit columns for backbone and input condition in every E2 table.

Main same-backbone table:

```text
Table 2a: Same-backbone SD3 algorithm comparison
```

Columns:

| Column | Purpose |
| --- | --- |
| Method | paper-facing row name |
| Backbone | SD3 for all Table 2a rows |
| Input condition | prompts only, prompts + operation/relation, support-matched, etc. |
| Support used for control | none, binary edit support, operation support, native support |
| Edit success | local/masked edit score or audit |
| Preserve fidelity | out-of-mask LPIPS/DINO/SSIM/L1 |
| Excess preserve error | subtracts SD3 reconstruction floor |
| Leakage/locality | outside-mask drift or boundary leakage |
| NFE/runtime | efficiency context |

Native implementation table:

```text
Table 2b: Native preservation-aware RF implementation comparison
```

Columns:

| Column | Purpose |
| --- | --- |
| Method | native baseline or DeCE-RF contextual anchor |
| Native backbone | SD3, FLUX.1-dev, FLUX Kontext, TBD |
| Input condition | source/target prompts, image+instruction, operation labels, etc. |
| Preservation mechanism | solver/inversion, feature injection, OT/fidelity, DeCE support/controller |
| Runnable status | complete, blocked, pending adapter |
| Edit / preserve / leakage metrics | only if strict generation completed |
| Caveat | gated checkpoint, adapter gap, seed mismatch, interface mismatch |

Support-matched diagnostic table:

```text
Table S-E2: Support-matched diagnostic
```

This table should be supplement or a compact subtable. It separates localization
input from DeCE-RF's controller.

### E2 Figure Layout

Figure 4 should remain compact. Use two or three representative tasks and label
backbones in method names.

Possible columns:

```text
Source | FlowEdit-SD3 | preservation-aware native RF row if runnable | Fixed DeCE | DeCE-RF
```

If native rows are blocked, use:

```text
Source | FlowEdit-SD3 | FlowAlign-SD3 or SplitFlow-SD3 | Fixed DeCE | DeCE-RF
```

Caption requirement:

```text
Backbone is shown in each method label. Same-backbone SD3 rows support the
algorithmic comparison; native-backbone rows, when included, are contextual.
```

### Current Artifact Mapping

Current completed E2 evidence:

```text
outputs/e2_rf_comparison/
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

These artifacts belong to E2.2 same-backbone SD3 comparison. The old label
`reduced_rf_comparison` should be interpreted as the completed strict SD3 RF
comparison cache, not as a reduced or weaker experimental design.

Current native-baseline state:

```text
RF-Solver-Edit / ReFlex / FireFlow / stable-flow: FLUX.1-dev access or adapter blocked
OT-RF / OTIP and DVRF: registered planned entries, repo/env/adapter pending
```

Until one native method becomes runnable, E2.3 is a status/audit table rather
than a quantitative claim.

### E2 Wording Rules

Allowed:

```text
Under the same SD3 backbone and fixed evaluation masks, DeCE-RF improves the
localized edit-preserve tradeoff over RF-native SD3 baselines.
```

```text
Native preservation-aware RF editors are reported as implementation-context
baselines because their public routes use different backbones or interfaces.
```

```text
Excess preserve error normalizes preserve-region damage by each backbone's
reconstruction floor.
```

Avoid:

```text
DeCE-RF beats FLUX.
DeCE-RF beats all RF editors.
SD3-DeCE is directly superior to ReFlex-FLUX or RF-Edit-FLUX as an algorithm.
Support-matched blending is a fair main baseline.
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

## Non-RF Supplement Baselines

Non-RF baselines are supplement-only. They should position the paper against
recognizable image-editing families, but they must not become the main technical
contest or support the claim that DeCE-RF beats RF baselines.

### Selected Supplement Baselines

Use exactly two non-RF supplement baselines unless a reviewer specifically asks
for more:

| Baseline slug | Paper-facing label | Model family / role | Input condition | Use |
| --- | --- | --- | --- | --- |
| `instruct_pix2pix` | InstructPix2Pix | instruction-guided diffusion editing | source image + instruction/text prompt, no DeCE support mask | representative text-instruction editor |
| `h_edit_r_p2p` | H-Edit / P2P-style | diffusion bridge / Prompt-to-Prompt-style editing | source image + source/target prompts, no DeCE support mask | representative attention/path editing comparator |

Do not add MasaCtrl, ZONE, Pix2Pix-Zero, Prompt-to-Prompt, LEDITS++, or
same-support inpainting to the main E2 plan. Add them only after the layered RF
fairness evidence is stable, and keep them as supplement/transparency rows.

### Supplement Matrix

Minimum supplement run:

```text
2 non-RF supplement baselines x 6 Core-6 tasks x 2 seeds = 24 outputs
seeds: 10, 11
```

Preferred if setup is stable:

```text
2 non-RF supplement baselines x 6 Core-6 tasks x 3 seeds = 36 outputs
seeds: 10, 11, 12
```

Use the same source images, prompts, max image size, fixed evaluation masks, and
visual-audit rubric as E1/E2. Do not give these baselines DeCE operation support
or hand-tuned masks. If a method cannot express a source/target prompt pair
cleanly, record the exact instruction/prompt translation in its command file and
metadata.

### Fairness Table

Every supplement baseline table needs these columns:

```text
method | model family/backbone | user input | mask source | seeds | runnable status | caveat
```

The supplement table should be labeled:

```text
Non-RF supplement baselines, not matched RF baselines.
```

This avoids the unfair comparison trap where a text-only editor, a P2P-style
method, and an operation-conditioned RF controller are averaged as if they used
the same information.

## Phased Experiment Execution Plan

Run the experiments as a three-stage decision funnel. Do not start with the
full Phase 2/3 cache. First test whether the method has enough signal to justify
the larger runs, then add fairness controls, then expand breadth.

### Phase 1: Minimum Sanity Check

Purpose: decide whether DeCE-RF has a real edit-preserve signal before spending
days on full evidence generation.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main internal benchmark | 6 categories x 1 example x 4 methods x 3 seeds | 72 |
| E2.2 same-backbone SD3 RF cache | completed strict FlowEdit/FlowAlign/SplitFlow + DeCE-RF rows; reuse, do not rerun | 72 analyzed rows |
| E4 controller variants | 6 categories x 1 example x 5 variants x 2 seeds | 60 |

Total:

```text
about 204 analyzed/generated rows, with E2.2 mostly reused from completed cache
```

If Phase 1 is being rerun on a new server, do not rerun all E2.2 baselines by
default. First verify the current environment with DeCE-RF migration checks and
reuse the completed E2.2 SD3 artifacts unless prompts, image normalization, or
metric code changed.

Phase 1 decision gate:

| Question | Go signal |
| --- | --- |
| E1 main effect | DeCE-RF remains visually usable on all strict Core-6 tasks |
| E2.2 SD3 comparison | completed SD3 RF rows remain valid under current fixed masks/metrics |
| E4 controller | fixed-vs-feedback evidence is at least plausible enough to justify Pareto/stress runs |
| Engineering | batch runner and environment produce deterministic artifacts with command/stats/metadata files |

### Phase 2: Recommended WACV Working Set

Purpose: turn the sanity signal into a reviewer-defensible evidence base while
making the E2 fairness structure explicit. Phase 2 should not merely add more
baseline names; it should add the missing calibration, preservation-aware, and
support-matched controls that make the comparison hard to attack.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main benchmark | 6 categories x 3 examples x 4 internal methods x 3 seeds | 216 |
| E2.1 backbone calibration | 6 canonical tasks x 2 backbones x 2 base methods x 2 seeds | 48 |
| E2.2 same-backbone SD3 algorithm comparison | 6 categories x 2 examples x 6 SD3 rows x 3 seeds | 216 |
| E2.2 optional verified SD3 preserve-aware external row | 6 categories x 2 examples x 1 row x 3 seeds | +36 |
| E2.3 native preservation-aware RF comparison | 6 canonical tasks x 1-2 native rows x 2 seeds | 12-24 |
| E2.4 support-matched diagnostic | 3 representative tasks x 4-5 rows x 2 seeds | 24-30 |
| E3 support geometry ablation | 6 categories x 2 examples x 5 support variants x 2 seeds | 120 |
| E4 controller ablation/stress | 6 categories x 2 examples x 5 controller variants x 2 seeds | 120 |
| E5 extension/failure examples | selected probes | 30 |

Phase 2 total target:

```text
about 786-804 analyzed/generated rows without optional SD3 preserve-aware external row
about 822-840 rows if one verified SD3 preserve-aware external row is added
```

If FLUX access remains blocked, E2.1 FLUX rows and E2.3 native rows become
status/audit rows rather than generated outputs. In that case, do not replace
them with unrelated non-RF baselines; keep the fairness limitation explicit.

Runtime estimate:

```text
~800 outputs x 2 min/output = about 26-27 hours pure generation
3-4 days realistic wall-clock with metrics, failed reruns, grids, and audits
```

Phase 2 Go criteria:

| Gate | Pass signal |
| --- | --- |
| E1 breadth | the positive pattern remains across multiple examples per category |
| E2.1 calibration | reconstruction/direct-target floors are measured or blocked status is recorded for each backbone |
| E2.2 same-backbone | SD3 comparison includes RF-native rows plus a preservation-control row under fixed masks |
| E2.3 native preservation-aware | at least one native RF row is complete, or blockers are documented as Table 2b status rows |
| E2.4 support diagnostic | support-matched rows show whether localization alone explains the gain |
| E3 support | frozen support variants explain edit/preserve outcomes without per-case tuning |
| E4 Pareto | DeCE-RF improves or stabilizes the edit-preserve frontier |
| E5 boundary | failures can be categorized as scope limits rather than unexplained collapse |

This is the target execution budget for the first serious WACV draft.

### Phase 3: WACV Robustness Completion

Purpose: only after Phase 1 and Phase 2 look good, expand to a stronger
WACV-ready robustness cache. Phase 3 should increase breadth; it should not
change the E2 logic or turn cross-backbone rows into algorithmic evidence.

| Experiment | Matrix | Outputs |
| --- | --- | ---: |
| E1 main benchmark | 6 categories x 5 examples x 4 internal methods x 3 seeds | 360 |
| E2.1 backbone calibration | 6 canonical tasks x 2 backbones x 2 base methods x 3 seeds | 72 |
| E2.2 same-backbone SD3 algorithm comparison | 6 categories x 3 examples x 6 SD3 rows x 3 seeds | 324 |
| E2.2 optional verified SD3 preserve-aware external row | 6 categories x 3 examples x 1 row x 3 seeds | +54 |
| E2.3 native preservation-aware RF comparison | 6 categories x 2 examples x 2-3 native rows x 2 seeds | 48-72 |
| E2.4 support-matched diagnostic | 3 categories x 2 examples x 4-5 rows x 2 seeds | 48-60 |
| E3 support ablation | 6 categories x 3 examples x 5 support variants x 2 seeds | 180 |
| E4 controller ablation/stress | 6 categories x 3 examples x 5 controller variants x 2 seeds | 180 |
| E5 extension/failure examples | selected probes | 30-50 |

Phase 3 total target:

```text
about 1242-1298 rows without optional SD3 preserve-aware external row
about 1296-1352 rows if one verified SD3 preserve-aware external row is added
```

Use Phase 3 to complete supplement grids, full per-task tables, robustness
audits, support-matched diagnostics, and any native E2.3 contextual rows that
became runnable after Phase 2. Do not run Phase 3 merely because it is listed
here; run it only if Phase 2 supports the paper claim and the remaining risk is
sample breadth rather than method correctness.

## Main-Paper Figure Budget

Use five figures by default and six at most. The main paper should show about
50-70 result image cells total; more starts to look like a gallery rather than
an algorithms paper.

| Figure | Content | Job |
| --- | --- | --- |
| Figure 1 | teaser: 2 examples, Source/Target/Direct/Generic/DeCE-RF | state the problem and result |
| Figure 2 | method overview with clean-estimate decomposition, support, feedback | explain method |
| Figure 3 | E1 Core-6 qualitative grid | show task diversity |
| Figure 4 | E2 fairness comparison: same-backbone SD3 rows, plus one native row only if clearly labeled | visualize RF alternatives without hiding backbone differences |
| Figure 5 | E4 Pareto + timestep diagnostics | prove feedback behavior |
| Figure 6 | E5 extension + failure cases, optional | mark scope boundary |

Do not add more main-paper grids unless one of these six figures fails to answer
its reviewer question.

Approximate main-paper image-cell budget:

| Figure | Image cells |
| --- | ---: |
| Figure 1 teaser | about 10 |
| Figure 2 method overview | 0 result cells |
| Figure 3 E1 qualitative grid | about 24-30 |
| Figure 4 E2 fairness comparison | about 12-18 |
| Figure 5 Pareto/diagnostics | 0 result cells |
| Figure 6 boundary/extension | about 12-18 |

Main-paper target:

```text
tight: 45-55 result image cells
complete: 60-75 result image cells
```

Supplement can contain 150-300 image cells: full Core-6 grids, all seeds,
support masks, RF baseline examples, calibration rows, support-matched
diagnostics, Pareto sweeps, and failure taxonomy. The main paper must still be
self-contained without the supplement.

## Main-Paper Table Budget

Use one main E2 algorithmic table plus one compact native/status table by
default. If space is tight, the native/status table can be a subtable within
Table 2, but the rows must remain visually separated.

| Table | Content | Job |
| --- | --- | --- |
| Table 1 | E1 main edit/preserve/leakage metrics | headline quantitative evidence |
| Table 2a | E2.2 same-backbone SD3 algorithm comparison | primary RF algorithmic evidence |
| Table 2b | E2.1/E2.3 calibration and native preservation-aware status/metrics | disclose backbone floors and cross-backbone context |
| Table 3 | E3+E4 component ablation | support geometry and controller evidence |

Supplement tables:

| Table | Content | Job |
| --- | --- | --- |
| Table S-E2-cal | full E2.1 backbone calibration | reconstruction/direct-target floors, NFE, runtime, memory |
| Table S-E2-native | full E2.3 native preservation-aware RF comparison | native method status, metrics if complete, blockers if not |
| Table S-E2-support | E2.4 support-matched diagnostic | separates localization input from DeCE-RF controller |

Required E2 columns:

```text
method | backbone | input condition | support used for control | edit success |
preserve fidelity | excess preserve error | leakage/locality | NFE/runtime | caveat
```

If space allows, split Table 3 into separate support and controller tables. If
space is tight, move detailed per-task rows and full ablation grids to the
supplement.

## Run Order

### Phase 1 Run Order: Half-Day Sanity Check

Run Phase 1 in this order:

```text
E1 internal main effect -> reuse completed E2.2 SD3 cache -> E4 controller variants
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

Expected output count:

```text
6 task instances x 4 methods x 3 seeds = 72
```

Legacy old-category runs live under `paper/archive_old_core6_20260602/` and
should not appear in active reproduction commands.

#### Phase 1 E2.2: Reuse Completed Same-Backbone SD3 Cache

Status: the strict same-backbone SD3 RF cache is complete. Do not rerun this
block unless prompts, image normalization, or metric code change.

Completed rows:

```text
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
DeCE-RF-SD3
```

Join these with E1 `direct_target` and, when needed, `support_v3_fixed` for the
same-backbone preservation-control readout.

Artifacts:

```text
outputs/e2_rf_comparison/
experiments/support_v3_2026-06-02/e2_strict_rf_baseline_manifest.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_fixed_mask_metrics.csv
experiments/support_v3_2026-06-02/e2_reduced_rf_comparison_summary.md
experiments/support_v3_2026-06-02/e2_reduced_rf_visual_audit.md
```

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
`support_v3_fixed`, `support_v3_controller_rmsgap`, and the two most diagnostic
controller alternatives, then reserve full edit-strength/support perturbation
sweeps for Phase 2.

#### Phase 1 Evaluation

After the three blocks, evaluate with fixed masks and build quick grids:

```bash
python scripts/evaluate_paper_metrics.py
python scripts/make_paper_grids.py
```

Use `SKIP_EXISTING=1` when reusing completed runs, but record that choice in the
experiment log. Phase 1 is a Go/No-Go gate, not final paper evidence.

### Phase 2 Run Order: Recommended WACV Working Set

Only start Phase 2 if Phase 1 passes the Go criteria.

#### Phase 2 E1 Expansion

Expand each of the six categories to three source examples:

```text
6 categories x 3 examples x 4 methods x 3 seeds = 216 outputs
```

Keep the same four internal methods:

```text
base_only
direct_target
adaptive_full_generic_support
support_v3_controller_rmsgap
```

#### Phase 2 E2.1 Backbone Calibration

Run only after the target backbone adapter is validated. SD3 calibration rows
can reuse E1 `base_only` and `direct_target`; FLUX rows are new only if FLUX
access is available.

```text
6 canonical tasks x 2 backbones x 2 base methods x 2 seeds = 48 outputs
```

If FLUX is blocked, write the blocker into Table 2b and proceed with same-
backbone SD3 E2.2. Do not replace FLUX calibration with unrelated diffusion
baselines.

#### Phase 2 E2.2 Same-Backbone SD3 Expansion

Expand the validated SD3 RF comparison to two source examples per category:

```text
6 categories x 2 examples x 6 SD3 rows x 3 seeds = 216 rows
```

Rows:

```text
direct_target
FlowEdit-SD3
FlowAlign-SD3
SplitFlow-SD3
Fixed DeCE-SD3
DeCE-RF-SD3
```

Optional addition only if real same-backbone support exists:

```text
OT-RF/OTIP-SD3 or RF-Edit-SD3: +36 rows
```

Do not move `rf_solver_edit`, `reflex`, `fireflow`, or `stable_flow` into this
layer unless they are genuinely ported to the matched SD3 protocol.

#### Phase 2 E2.3 Native Preservation-Aware RF Rows

Run after a native method passes strict Core-6 smoke and access checks:

```text
6 canonical tasks x 1-2 native rows x 2 seeds = 12-24 outputs
```

Candidate rows:

```text
rf_solver_edit
reflex
fireflow
stable_flow
ot_rf_otip
dvrf
```

If all remain blocked, Table 2b should be a status/audit table with exact
failure reasons.

#### Phase 2 E2.4 Support-Matched Diagnostic

Run a compact diagnostic:

```text
3 representative tasks x 4-5 rows x 2 seeds = 24-30 outputs
```

Recommended tasks:

```text
cat_crown
tshirt_star
backpack_remove_toy_charm
```

Rows:

```text
direct_target + same M_edit diagnostic
FlowEdit + same M_edit diagnostic if wrapper is stable
Fixed DeCE
DeCE-RF
optional native preserve-aware + same M_edit if method supports it
```

Keep output blending clearly labeled as a diagnostic, not a fair main baseline.

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

Then run a targeted stress sweep on the most diagnostic subset. Use the legacy
stress script when applicable; restore it to active `scripts/` only if the sweep
becomes paper-critical:

```bash
TASKS="cat_crown bowl_apple_inside tshirt_star" \
METHODS="support_v3_fixed support_v3_controller_rmsgap" \
SEEDS="10 11" \
SKIP_EXISTING=1 \
bash legacy/cleanup_20260603/scripts/run_controller_stress_sweeps.sh
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

Phase 3 expands the same layered E2 design rather than introducing new baseline
families:

```text
E1: 6 categories x 5 examples x 4 methods x 3 seeds = 360
E2.1: 6 canonical tasks x 2 backbones x 2 base methods x 3 seeds = 72
E2.2: 6 categories x 3 examples x 6 SD3 rows x 3 seeds = 324
E2.2 optional SD3 preserve-aware external row = +54
E2.3: 6 categories x 2 examples x 2-3 native rows x 2 seeds = 48-72
E2.4: 3 categories x 2 examples x 4-5 rows x 2 seeds = 48-60
E3: 6 categories x 3 examples x 5 support variants x 2 seeds = 180
E4: 6 categories x 3 examples x 5 controller variants x 2 seeds = 180
E5: 30-50 selected outputs
```

Use Phase 3 to complete supplement grids, full per-task tables, robustness
audits, and any native E2.3 contextual rows that became runnable after Phase 2.
It should not change the core method story or turn non-RF supplement baselines
into E2 evidence.

### Fixed-Control Cache Note

```bash
TASKS="cat_crown bowl_apple_inside tshirt_star red_chair_blue pillow_vertical_fabric_strip backpack_remove_toy_charm" \
METHODS="support_v3_fixed" \
SEEDS="10 11 12" \
bash scripts/run_pretty_matrix.sh
```

This supports E4, not Table 1.

### Non-RF Supplement Baselines

Run only the two selected supplement baselines after the RF evidence is stable:
`instruct_pix2pix` and `h_edit_r_p2p`. They are not part of Phase 1, not part of
E2.1-E2.4, and not evidence for the RF-specific claim. Use them as a supplement
positioning table with the same Core-6 sources, prompts, fixed masks, and visual
audit rubric.

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
