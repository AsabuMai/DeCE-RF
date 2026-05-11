# RF h-Edit: Support v2 Direction Notes

## 0. Motivation

The current `adaptive_full` controller is becoming more general, but the **support / mask generation** is still the main bottleneck.

The current generic support v1 is roughly:

\[
S = A_{\mathrm{changed}} \odot D_{\mathrm{clean}}
\]

where:

- \(A_{\mathrm{changed}}\): changed target-token attention
- \(D_{\mathrm{clean}}\): source-target clean-estimate disagreement

This works well for tasks where the changed token has a strong and localized visual region, such as:

```text
dog_sunglasses
```

However, it fails or becomes weak for tasks that require:

- relative placement,
- surface/decal localization,
- source-object removal,
- host-object reasoning.

Examples:

```text
cat_crown
mug_heart
backpack_remove_toy_charm
```

Therefore, the next step should be:

> Replace the single generic support formula with **operation-aware generic support**.

---

## 1. Current Observation

From the current seed10 comparison:

| Task | full | v1_manual | v1_generic | Observation |
|---|---|---|---|---|
| cat_crown | crown appears | crown appears | crown mostly missing | generic support fails to localize head/top region |
| dog_sunglasses | sunglasses appear | sunglasses appear | strongest result | generic support works well |
| mug_heart | heart appears | heart appears | heart missing | generic support fails to localize mug surface |
| backpack_remove_toy_charm | charm removed better | charm removed better | charm remains / residue | generic support fails for source-object removal |

Main conclusion:

> Generic support v1 is not useless, but it only works well for strongly localized target tokens. It is insufficient for relation, decal, and removal edits.

So the next bottleneck is:

\[
\boxed{\text{support quality}}
\]

not the adaptive controller itself.

---

## 2. Related Ideas to Borrow

### 2.1 DiffEdit: source-target prediction difference

DiffEdit generates an edit mask by contrasting model predictions under different text prompts.

Relevant idea:

\[
D_{\mathrm{clean}}
=
\left\|
\hat{x}_0^{tar} - \hat{x}_0^{src}
\right\|
\]

should be treated as a generic response map, not as the final mask by itself.

---

### 2.2 VeloEdit: velocity discrepancy

VeloEdit uses discrepancy between source-preserving and desired-edit velocities to locate editing regions.

Relevant idea:

\[
D_{\mathrm{vel}}
=
\left\|
v^{tar} - v^{src}
\right\|
\]

can be used as another support cue.

---

### 2.3 SteerFlow: adaptive masking + source-target velocity difference

SteerFlow uses source-target velocity differences and adaptive masking to constrain editing signals.

Relevant idea:

Support should combine:

\[
A_{\mathrm{token}},\quad D_{\mathrm{clean}},\quad D_{\mathrm{vel}},\quad M_{\mathrm{seg}}
\]

rather than relying on attention alone.

---

### 2.4 InstructEdit: instruction parsing + Grounded SAM

InstructEdit uses language parsing and Grounded Segment Anything to obtain segmentation prompts and masks.

Relevant idea:

Support generation should first parse the edit operation:

- add object
- add decal / surface pattern
- remove object
- replace object / attribute

Then choose suitable support cues.

---

### 2.5 Uniform Attention Maps

Uniform Attention Maps use source-target branch differences and timestep-adaptive masks.

Relevant idea:

We can start with static support aggregated over early / middle steps, and later explore dynamic / EMA masks.

---

### 2.6 PowerPaint: task-type prompts

PowerPaint uses different task prompts for different editing modes.

Relevant idea:

The support proposer should be aware of edit operation type.

---

## 3. New Direction: Operation-Aware Generic Support

The new support should not be:

```text
one formula for all tasks
```

Instead, it should be:

\[
\boxed{
M_{\mathrm{edit}}
=
\mathrm{Postprocess}
\left(
\mathcal{S}_{op}
(
A_{\mathrm{changed}},
A_{\mathrm{host}},
A_{\mathrm{removed}},
D_{\mathrm{clean}},
D_{\mathrm{vel}},
M_{\mathrm{seg}}
)
\right)
}
\]

where:

- \(A_{\mathrm{changed}}\): target changed-token attention
- \(A_{\mathrm{host}}\): host object attention / segmentation
- \(A_{\mathrm{removed}}\): source removed-object attention / segmentation
- \(D_{\mathrm{clean}}\): clean-estimate disagreement
- \(D_{\mathrm{vel}}\): velocity disagreement
- \(M_{\mathrm{seg}}\): optional segmentation mask
- \(\mathcal{S}_{op}\): operation-aware support score

The key idea is:

> Use a generic operation type, not task-specific object templates.

---

## 4. Avoid Task Templates

Do not write object-specific rules such as:

```text
if sunglasses -> front_glasses_auto
if crown -> top_of_head
if heart -> decal_box
```

Instead, write operation-level rules such as:

```text
operation = add_object
operation = add_decal
operation = remove_object
operation = replace
```

This keeps the method generic.

---

## 5. Operation Types and Support Formulas

### 5.1 Add Object / Add Accessory

Examples:

```text
cat_crown
dog_sunglasses
```

Basic support:

\[
S =
A_{\mathrm{new}}
\odot
D_{\mathrm{clean}}
\]

where:

- \(A_{\mathrm{new}}\): attention for new target token such as `crown` or `sunglasses`

For tasks needing host relation, add host support:

\[
S =
A_{\mathrm{new}}
\odot
D_{\mathrm{clean}}
+
\lambda_h
A_{\mathrm{host}}
\odot
D_{\mathrm{clean}}
\]

where:

- \(A_{\mathrm{host}}\): attention / segmentation for host object such as `cat` or `dog`

Do not directly encode crown-specific top-of-head geometry in the first version.

---

### 5.2 Add Decal / Surface Pattern

Example:

```text
mug_heart
```

Here the new token is not an independent object. It should appear on a host surface.

Recommended support:

\[
S =
A_{\mathrm{new}}
\odot
A_{\mathrm{host}}
\odot
D_{\mathrm{clean}}
\]

If \(A_{\mathrm{new}}\) is weak:

\[
S =
A_{\mathrm{host}}
\odot
D_{\mathrm{clean}}
\]

where:

- \(A_{\mathrm{new}}\): attention for `heart`
- \(A_{\mathrm{host}}\): attention / segmentation for `mug`

This prevents the heart support from drifting to background.

---

### 5.3 Remove Object

Example:

```text
backpack_remove_toy_charm
```

Removal is different because the removed object may not appear in the target prompt.

Do not rely on target changed-token attention.

Use source-side support:

\[
S =
A_{\mathrm{removed,src}}
\odot
D_{\mathrm{clean}}
\]

or with segmentation:

\[
S =
M_{\mathrm{seg}}(\mathrm{removed\ object})
\odot
D_{\mathrm{clean}}
\]

where:

- \(A_{\mathrm{removed,src}}\): source attention for `toy charm`
- \(M_{\mathrm{seg}}\): optional segmentation mask for removed object

---

### 5.4 Replace Object / Replace Attribute

For future object replacement tasks:

\[
S =
(A_{\mathrm{src\ object}} + A_{\mathrm{tar\ object}})
\odot
D_{\mathrm{clean}}
\]

For subject replacement:

\[
S =
M_{\mathrm{seg}}(\mathrm{source\ object})
\odot
D_{\mathrm{clean}}
\]

However, object replacement can cause overlay-like failure, so it should not be the first main task.

---

## 6. Support Candidate Bank

Instead of producing only one support map, generate multiple candidates.

Candidate examples:

```text
attention_only
clean_disagreement_only
velocity_disagreement_only
attention_x_clean
attention_x_velocity
host_x_clean
new_x_host_x_clean
removed_src_x_clean
segmentation_x_clean
```

Code sketch:

```python
candidates = {
    "attention_only": A_new,
    "clean_only": D_clean,
    "velocity_only": D_vel,
    "attn_x_clean": A_new * D_clean,
    "attn_x_velocity": A_new * D_vel,
    "host_x_clean": A_host * D_clean,
    "new_x_host_x_clean": A_new * A_host * D_clean,
    "removed_src_x_clean": A_removed_src * D_clean,
    "seg_x_clean": M_seg * D_clean,
}
```

All candidates should go through the same postprocessing pipeline.

---

## 7. Candidate Selection

### 7.1 First stage: manual candidate selection by operation type

Do this first for stability.

Suggested mapping:

| Task | Operation | Candidate |
|---|---|---|
| cat_crown | add_object | host_x_clean or new_plus_host_x_clean |
| dog_sunglasses | add_object | attn_x_clean |
| mug_heart | add_decal | new_x_host_x_clean or host_x_clean |
| backpack_remove_toy_charm | remove_object | removed_src_x_clean or seg_x_clean |

This is still operation-level, not object-template-level.

---

### 7.2 Second stage: automatic candidate selection

After manual selection works, add automatic selection.

For each candidate mask \(M_k\), evaluate:

\[
J(M_k)
=
p_{\mathrm{edit}}(M_k)
-
\lambda d_{\mathrm{pres}}(M_k)
-
\mu \mathrm{area}(M_k)
-
\nu \mathrm{leakage}(M_k)
\]

Choose:

\[
M^*
=
\arg\max_k J(M_k)
\]

This would make the support proposer more aligned with the clean-estimate controller.

---

## 8. Mask Postprocessing

Every candidate should use the same postprocessing.

Recommended steps:

1. normalize score to \([0,1]\)
2. threshold by top percentile / top-k
3. connected component filtering
4. keep top \(N\) components
5. enforce min / max area ratio
6. dilate
7. blur / smooth
8. build core / ring / preserve regions

Suggested parameters:

```text
top_percentile = 85 or 90
min_area_ratio = 0.02
max_area_ratio = 0.30
dilate_radius = 3 to 8
blur_sigma = 2 to 5
top_components = 1 or 2
```

These should remain generic.

---

## 9. Core / Ring / Preserve Regions

Do not use only one hard binary mask.

Generate three regions:

### Edit core

\[
M_{\mathrm{core}}
\]

Strong edit region.

### Transition ring

\[
M_{\mathrm{ring}}
=
\mathrm{Dilate}(M_{\mathrm{core}})
-
M_{\mathrm{core}}
\]

Weak edit / weak preserve transition region.

### Preserve region

\[
M_{\mathrm{preserve}}
=
1-\mathrm{Dilate}(M_{\mathrm{core}})
\]

Strong preservation region.

The adaptive controller should compute:

- directional edit progress in \(M_{\mathrm{core}}\)
- preserve drift in \(M_{\mathrm{preserve}}\)
- conflict projection mainly in \(M_{\mathrm{preserve}}\)

---

## 10. Static vs Dynamic Support

### First version: static support

Compute support over early / middle ODE steps and freeze it.

For support steps:

\[
t\in\mathcal{T}_{support}
\]

compute:

\[
S_t = A_t \odot D_t
\]

Aggregate:

\[
S = \frac{1}{K}\sum_{t\in\mathcal{T}_{support}} S_t
\]

or:

\[
S = \max_{t\in\mathcal{T}_{support}} S_t
\]

This is easier to debug.

### Later version: dynamic support

Use EMA:

\[
M_t = \rho M_{t-1} + (1-\rho)\tilde{M}_t
\]

Only try this after static support v2 is stable.

---

## 11. Proposed New File

Create:

```text
operation_support.py
```

or:

```text
generic_support_v2.py
```

Suggested functions:

```python
parse_edit_operation(...)
compute_changed_token_attention(...)
compute_source_removed_attention(...)
compute_host_attention(...)
compute_clean_disagreement(...)
compute_velocity_disagreement(...)
build_support_candidates(...)
select_support_candidate_by_operation(...)
select_support_candidate_by_score(...)
postprocess_support_score(...)
build_core_ring_preserve_masks(...)
save_support_debug_maps(...)
```

---

## 12. Suggested CLI Flags

```bash
--support-mode operation_aware
--edit-operation add_object
--new-tokens crown
--host-tokens cat
--removed-tokens toy,charm
--support-candidate host_x_clean
--support-score attention_x_clean
--support-top-percentile 90
--support-min-area-ratio 0.02
--support-max-area-ratio 0.30
--support-dilate-radius 5
--support-blur-sigma 3
--support-aggregation mean
--support-steps early_mid
--save-support-debug
```

For ablation:

```bash
--support-candidate attention_only
--support-candidate clean_only
--support-candidate velocity_only
--support-candidate attn_x_clean
--support-candidate host_x_clean
--support-candidate new_x_host_x_clean
--support-candidate removed_src_x_clean
--support-candidate seg_x_clean
```

---

## 13. Debug Outputs

For each support run, save:

```text
attention_new.png
attention_host.png
attention_removed_src.png
clean_disagreement.png
velocity_disagreement.png
support_score.png
M_core.png
M_ring.png
M_preserve.png
final_used_mask.png
```

Save JSON:

```json
{
  "operation": "add_decal",
  "candidate": "new_x_host_x_clean",
  "mask_area_core": 0.0,
  "mask_area_ring": 0.0,
  "mask_area_preserve": 0.0,
  "num_components": 0,
  "top_component_area": 0.0
}
```

---

## 14. Experiment Plan

Compare:

```text
manual_support
generic_support_v1
operation_aware_support_v2
```

on:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

At least use:

```text
seed10, seed11, seed12
```

if compute is available.

---

## 15. Support Quality Metrics

If manual support is treated as an upper-bound pseudo-reference, compute:

### IoU

\[
\mathrm{IoU}
=
\frac{
|M_{\mathrm{generic}}\cap M_{\mathrm{manual}}|
}{
|M_{\mathrm{generic}}\cup M_{\mathrm{manual}}|
}
\]

### Coverage

\[
\mathrm{coverage}
=
\frac{
|M_{\mathrm{generic}}\cap M_{\mathrm{manual}}|
}{
|M_{\mathrm{manual}}|
}
\]

### Leakage

\[
\mathrm{leakage}
=
\frac{
|M_{\mathrm{generic}}\setminus M_{\mathrm{manual}}|
}{
|M_{\mathrm{generic}}|
}
\]

These metrics can show how close generic support is to manual support.

---

## 16. Expected Improvements

### cat_crown

Problem in v1:

- generic support misses crown / top region.

Expected v2 improvement:

- host-aware support should cover cat head / nearby crown response region.

---

### dog_sunglasses

Problem in v1:

- already works well.

Expected v2 improvement:

- should maintain or slightly improve performance.
- should not degrade.

---

### mug_heart

Problem in v1:

- heart disappears because support does not localize mug surface.

Expected v2 improvement:

- decal support using host mug attention / segmentation should localize mug surface.

---

### backpack_remove_toy_charm

Problem in v1:

- charm remains because target changed-token attention does not identify removed source object.

Expected v2 improvement:

- source removed-object attention or segmentation should localize toy charm.

---

## 17. Updated Research Narrative

Current narrative should be updated from:

> generic support = changed-token attention × clean disagreement

to:

> generic support v1 used changed-token attention × clean disagreement. It works for strongly localized target tokens but fails for relation, decal, and removal cases. Therefore, we introduce operation-aware generic support, which selects support cues according to edit operation type while avoiding object-specific templates.

The new claim:

> The method combines operation-aware generic support with clean-estimate-space adaptive control.

---

## 18. Immediate To-Do List

Current implementation note, 2026-05-11:

The first `support_v2_minimal` pass has been implemented and run on seed 10 for
the four pretty tasks. It adds operation-aware candidates and CLI flags, but the
seed-10 visual and mask metrics show that it does not yet improve the failed
tasks over generic support v1. The three-seed expansion should wait until the
seed-10 gate improves.

Evidence:

```text
outputs/pretty_matrix/support_v2_minimal_seed10_overview.png
outputs/pretty_matrix/support_v2_minimal_seed10_masks.png
experiments/support_v2_minimal_metrics_seed10.csv
experiments/support_v2_minimal_summary.md
```

### Support v2 implementation

- [ ] Create `operation_support.py` or `generic_support_v2.py`.
- [x] Add support v2 minimal path in existing `generic_support.py`.
- [x] Add edit operation parser.
- [x] Add operation types:
  - `add_object`
  - `add_decal`
  - `remove_object`
  - `replace`
- [x] Add support candidate bank.
- [x] Add host attention support.
- [x] Add removed-source attention support.
- [x] Add clean disagreement support.
- [x] Add velocity disagreement support.
- [ ] Add optional segmentation support.
- [x] Add common postprocessing pipeline.
- [x] Add core / ring / preserve mask construction.

### Experiments

- [x] Run manual support.
- [x] Run generic v1.
- [x] Run operation-aware v2.
- [x] Compare on four pretty tasks.
- [x] Save support visualization maps.
- [x] Compute support IoU / coverage / leakage against manual upper bound.
- [x] Generate comparison panels.

### Documentation

- [ ] Mark generic support v1 as limited.
- [ ] Add operation-aware support v2 to method notes.
- [ ] State manual support is upper bound / diagnostic, not core method.
- [ ] Update paper narrative to focus on operation-aware generic support + clean-estimate controller.

---

## 19. One-Sentence Summary

The next support update should move from a single formula, `changed-token attention × clean disagreement`, to an **operation-aware generic support candidate bank** that uses changed-token attention, host-object support, removed-source-object support, clean disagreement, velocity disagreement, and optional segmentation to generate more reliable edit regions for the clean-estimate-space adaptive controller.
