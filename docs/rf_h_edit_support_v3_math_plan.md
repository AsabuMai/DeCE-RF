# RF h-Edit: Support-v3 Mathematical Implementation Plan

## 0. Goal

The current generic support v1 is too weak:

\[
S = A_{\mathrm{changed}} \odot D_{\mathrm{clean}}
\]

It only works when the changed target token itself produces a strong, local, clean visual response, such as `dog_sunglasses`.

However, it fails for:

- `cat_crown`: needs relation / layout reasoning.
- `mug_heart`: needs host surface / decal localization.
- `backpack_remove_toy_charm`: needs source-side removed-object localization.

Therefore, support-v3 should introduce **operation-aware support proposal** with stronger evidence sources:

\[
\boxed{
\text{support-v3}
=
\text{token attention}
+
\text{clean / velocity response}
+
\text{grounding / segmentation}
+
\text{relation proposal}
}
\]

The main goal is to generate a better edit support mask:

\[
M_{\mathrm{edit}}
\]

and then derive:

\[
M_{\mathrm{core}},\quad M_{\mathrm{ring}},\quad M_{\mathrm{preserve}}.
\]

These masks are then used by the clean-estimate-space adaptive controller.

---

## 1. Core RF Clean Estimate

We use the linear Rectified Flow path:

\[
x_t = (1-t)x_0 + t x_1.
\]

The RF model provides a velocity field:

\[
v_\theta(x_t,t).
\]

The clean estimate is:

\[
\boxed{
\hat{x}_0(x_t,t)
=
x_t - t\,v_\theta(x_t,t)
}
\]

For source and target conditions:

\[
\hat{x}_{0,t}^{src}
=
x_t - t\,v_{\theta}^{src}(x_t,t)
\]

\[
\hat{x}_{0,t}^{tar}
=
x_t - t\,v_{\theta}^{tar}(x_t,t)
\]

The clean-estimate disagreement is:

\[
\boxed{
D_{\mathrm{clean},t}
=
\left\|
\hat{x}_{0,t}^{tar}
-
\hat{x}_{0,t}^{src}
\right\|_2
}
\]

If \(x_t\in \mathbb{R}^{C\times H\times W}\), compute it per spatial location:

\[
D_{\mathrm{clean},t}(i,j)
=
\sqrt{
\sum_{c=1}^{C}
\left(
\hat{x}_{0,t,cij}^{tar}
-
\hat{x}_{0,t,cij}^{src}
\right)^2
}.
\]

---

## 2. Velocity Disagreement

Besides clean-estimate disagreement, use velocity disagreement:

\[
\boxed{
D_{\mathrm{vel},t}
=
\left\|
v_{\theta}^{tar}(x_t,t)
-
v_{\theta}^{src}(x_t,t)
\right\|_2
}
\]

Per spatial location:

\[
D_{\mathrm{vel},t}(i,j)
=
\sqrt{
\sum_{c=1}^{C}
\left(
v_{\theta,cij}^{tar}
-
v_{\theta,cij}^{src}
\right)^2
}.
\]

This shows where source and target dynamics differ.

---

## 3. Token Attention Evidence

Let \(A_q(t)\) be the normalized cross-attention map for token query \(q\).

Examples:

- \(A_{\mathrm{new}}\): attention map for a new target token, e.g. `crown`, `sunglasses`, `heart`.
- \(A_{\mathrm{host}}\): attention map for host object, e.g. `cat`, `dog`, `mug`, `backpack`.
- \(A_{\mathrm{removed,src}}\): source-side attention for removed object, e.g. `toy charm`.

All attention maps should be resized to latent spatial resolution:

\[
A_q(t) \in [0,1]^{H\times W}.
\]

Normalize each attention map:

\[
\mathrm{Norm}(A)
=
\frac{A-\min(A)}{\max(A)-\min(A)+\epsilon}.
\]

For multiple tokens:

\[
A_{\mathcal{Q}}(t)
=
\frac{1}{|\mathcal{Q}|}
\sum_{q\in \mathcal{Q}}
A_q(t).
\]

---

## 4. Grounding / Segmentation Evidence

Use an optional grounding or segmentation model such as GroundedSAM, SAM/SAM2, or CLIPSeg.

For a text query \(q\):

\[
M_{\mathrm{ground}}(q)
=
\mathrm{Segment}(x_{\mathrm{src}}, q).
\]

Examples:

\[
M_{\mathrm{cat}}
=
\mathrm{Segment}(x_{\mathrm{src}}, \text{``cat''})
\]

\[
M_{\mathrm{mug}}
=
\mathrm{Segment}(x_{\mathrm{src}}, \text{``mug''})
\]

\[
M_{\mathrm{removed}}
=
\mathrm{Segment}(x_{\mathrm{src}}, \text{``toy charm''})
\]

Grounding evidence is important for:

- host object localization,
- removed-object localization,
- surface support,
- relation proposal.

---

## 5. Operation Types

Do not use object-specific rules such as:

```text
if sunglasses -> front_glasses_auto
if heart -> decal_box
if crown -> top_of_head
```

Use operation-level rules:

```text
add_object
add_decal
remove_object
replace
```

Each operation chooses different evidence sources.

---

## 6. Operation-Aware Support Formulas

### 6.1 Add Object / Add Accessory

Examples:

```text
cat_crown
dog_sunglasses
```

Inputs:

- new token: \(q_{\mathrm{new}}\), e.g. `crown`, `sunglasses`
- host token: \(q_{\mathrm{host}}\), e.g. `cat`, `dog`
- optional relation: `above_host`, `on_face`, `near_host`

Evidence:

\[
A_{\mathrm{new}}
=
A_{q_{\mathrm{new}}}
\]

\[
A_{\mathrm{host}}
=
A_{q_{\mathrm{host}}}
\]

\[
M_{\mathrm{host}}
=
M_{\mathrm{ground}}(q_{\mathrm{host}})
\]

Default add-object support:

\[
S_{\mathrm{add}}
=
\mathrm{Norm}(A_{\mathrm{new}})^{\gamma}
\odot
\mathrm{Norm}(D_{\mathrm{clean}})^{\eta}.
\]

Host-aware support:

\[
S_{\mathrm{add-host}}
=
\left(
\lambda_n \mathrm{Norm}(A_{\mathrm{new}})
+
\lambda_h \mathrm{Norm}(A_{\mathrm{host}})
\right)
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

If segmentation is available:

\[
S_{\mathrm{add-seg}}
=
\left(
\lambda_n \mathrm{Norm}(A_{\mathrm{new}})
+
\lambda_h M_{\mathrm{host}}
\right)
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

If relation proposal is available:

\[
S_{\mathrm{add-rel}}
=
R_{\mathrm{relation}}
\odot
\left(
\lambda_n \mathrm{Norm}(A_{\mathrm{new}})
+
\lambda_d \mathrm{Norm}(D_{\mathrm{clean}})
+
\lambda_v \mathrm{Norm}(D_{\mathrm{vel}})
\right).
\]

Candidate set:

\[
\mathcal{C}_{add}
=
\{
S_{\mathrm{add}},
S_{\mathrm{add-host}},
S_{\mathrm{add-seg}},
S_{\mathrm{add-rel}}
\}.
\]

---

### 6.2 Add Decal / Surface Pattern

Example:

```text
mug_heart
```

This is not independent object insertion. The new element appears on a host surface.

Inputs:

- new token: \(q_{\mathrm{new}}\), e.g. `heart`
- host token: \(q_{\mathrm{host}}\), e.g. `mug`
- relation: `on_surface`

Evidence:

\[
A_{\mathrm{new}}=A_{q_{\mathrm{new}}},\quad
A_{\mathrm{host}}=A_{q_{\mathrm{host}}},\quad
M_{\mathrm{host}}=M_{\mathrm{ground}}(q_{\mathrm{host}})
\]

Create a surface proposal:

\[
M_{\mathrm{surface}}
=
\mathrm{SurfaceProposal}(M_{\mathrm{host}}).
\]

Simple first versions:

\[
M_{\mathrm{surface}}
=
\mathrm{Erode}(M_{\mathrm{host}})
\]

or

\[
M_{\mathrm{surface}}
=
\mathrm{CenterRegion}(M_{\mathrm{host}}).
\]

Default decal support:

\[
\boxed{
S_{\mathrm{decal}}
=
\mathrm{Norm}(A_{\mathrm{new}})
\odot
M_{\mathrm{surface}}
\odot
\mathrm{Norm}(D_{\mathrm{clean}})
}
\]

If \(A_{\mathrm{new}}\) is weak:

\[
S_{\mathrm{decal-host}}
=
M_{\mathrm{surface}}
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

Velocity variant:

\[
S_{\mathrm{decal-vel}}
=
M_{\mathrm{surface}}
\odot
\mathrm{Norm}(D_{\mathrm{vel}}).
\]

Candidate set:

\[
\mathcal{C}_{decal}
=
\{
S_{\mathrm{decal}},
S_{\mathrm{decal-host}},
S_{\mathrm{decal-vel}}
\}.
\]

---

### 6.3 Remove Object

Example:

```text
backpack_remove_toy_charm
```

For removal, target prompt attention is often weak or absent because the removed object should disappear.

Support must come from source-side evidence.

Inputs:

- removed token: \(q_{\mathrm{removed}}\), e.g. `toy charm`
- host token: \(q_{\mathrm{host}}\), e.g. `backpack`

Evidence:

\[
A_{\mathrm{removed,src}}
=
A_{q_{\mathrm{removed}}}^{src}
\]

\[
M_{\mathrm{removed}}
=
M_{\mathrm{ground}}(q_{\mathrm{removed}})
\]

Default remove support:

\[
\boxed{
S_{\mathrm{remove}}
=
M_{\mathrm{removed}}
\odot
\mathrm{Norm}(D_{\mathrm{clean}})
}
\]

Attention fallback:

\[
S_{\mathrm{remove-attn}}
=
\mathrm{Norm}(A_{\mathrm{removed,src}})
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

Velocity variant:

\[
S_{\mathrm{remove-vel}}
=
M_{\mathrm{removed}}
\odot
\mathrm{Norm}(D_{\mathrm{vel}}).
\]

Candidate set:

\[
\mathcal{C}_{remove}
=
\{
S_{\mathrm{remove}},
S_{\mathrm{remove-attn}},
S_{\mathrm{remove-vel}}
\}.
\]

---

### 6.4 Replace Object / Replace Attribute

For future replacement tasks:

Inputs:

- source object: \(q_{\mathrm{src}}\)
- target object: \(q_{\mathrm{tar}}\)

Support:

\[
S_{\mathrm{replace}}
=
\left(
\mathrm{Norm}(A_{\mathrm{src}})
+
\mathrm{Norm}(A_{\mathrm{tar}})
\right)
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

If source segmentation is available:

\[
S_{\mathrm{replace-seg}}
=
M_{\mathrm{ground}}(q_{\mathrm{src}})
\odot
\mathrm{Norm}(D_{\mathrm{clean}}).
\]

Candidate set:

\[
\mathcal{C}_{replace}
=
\{
S_{\mathrm{replace}},
S_{\mathrm{replace-seg}}
\}.
\]

Object replacement is harder and more likely to produce overlay-like artifacts, so it should not be the first main task.

---

## 7. Relation / Layout Proposal

### 7.1 Above Host

For `cat_crown`, relation is:

```text
above_host
```

Given host mask \(M_{\mathrm{host}}\), compute its bounding box:

\[
B_{\mathrm{host}} = (x_{\min}, y_{\min}, x_{\max}, y_{\max})
\]

Let:

\[
h = y_{\max}-y_{\min},\quad w=x_{\max}-x_{\min}.
\]

Define above-host box:

\[
B_{\mathrm{above}}
=
\left[
x_{\min}-\alpha_w w,\;
y_{\min}-\alpha_h h,\;
x_{\max}+\alpha_w w,\;
y_{\min}+\beta_h h
\right].
\]

Typical values:

```text
alpha_w = 0.1 to 0.3
alpha_h = 0.4 to 0.8
beta_h  = 0.1 to 0.3
```

Convert to mask:

\[
R_{\mathrm{above}} = \mathrm{BoxMask}(B_{\mathrm{above}}).
\]

Then:

\[
S_{\mathrm{above}}
=
R_{\mathrm{above}}
\odot
\left(
\lambda_n A_{\mathrm{new}}
+
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
\right).
\]

This is a relation-level rule, not a crown-specific rule.

---

### 7.2 On Surface

For `mug_heart`, relation is:

```text
on_surface
```

Given host mask \(M_{\mathrm{host}}\), define:

\[
R_{\mathrm{surface}}
=
\mathrm{Erode}(M_{\mathrm{host}})
\]

or:

\[
R_{\mathrm{surface}}
=
\mathrm{CenterRegion}(M_{\mathrm{host}}).
\]

Then:

\[
S_{\mathrm{surface}}
=
R_{\mathrm{surface}}
\odot
\left(
\lambda_n A_{\mathrm{new}}
+
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
\right).
\]

---

### 7.3 Remove Source Object

For removal:

```text
remove_source_object
```

Use:

\[
R_{\mathrm{remove}}
=
M_{\mathrm{ground}}(q_{\mathrm{removed}})
\]

or source attention:

\[
R_{\mathrm{remove}}
=
A_{\mathrm{removed,src}}.
\]

Then:

\[
S_{\mathrm{remove}}
=
R_{\mathrm{remove}}
\odot
\left(
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
ight).
\]

---

## 8. Temporal Aggregation

Support can be computed over early / middle ODE steps.

For support steps:

\[
\mathcal{T}_{support}
=
\{t_1,t_2,\dots,t_K\}
\]

compute \(S_t\) and aggregate:

\[
S_{\mathrm{mean}}
=
\frac{1}{K}\sum_{t\in\mathcal{T}_{support}} S_t
\]

or:

\[
S_{\mathrm{max}}
=
\max_{t\in\mathcal{T}_{support}} S_t.
\]

Recommended first version:

\[
S = S_{\mathrm{mean}}.
\]

Dynamic support can be added later:

\[
M_t
=
\rho M_{t-1}
+
(1-\rho)\tilde{M}_t.
\]

---

## 9. Postprocessing

Each support score \(S\) should go through the same postprocessing pipeline.

### 9.1 Normalize

\[
S' = \frac{S-\min(S)}{\max(S)-\min(S)+\epsilon}.
\]

### 9.2 Threshold

Use percentile threshold:

\[
\tau = \mathrm{Percentile}(S', p)
\]

\[
M_0 = \mathbf{1}[S' > \tau].
\]

Suggested:

```text
p = 85 or 90
```

### 9.3 Connected Components

Let connected components be:

\[
\{C_1,C_2,\dots,C_N\}.
\]

Keep top \(K\) components by area or score:

\[
\mathrm{score}(C_i)=\sum_{(x,y)\in C_i} S'(x,y).
\]

\[
M_1 = \bigcup_{i\in \mathrm{TopK}} C_i.
\]

### 9.4 Area Clamp

Let:

\[
r = \frac{|M_1|}{HW}.
\]

If \(r<r_{\min}\), lower threshold or dilate.

If \(r>r_{\max}\), increase threshold or keep fewer components.

Suggested:

```text
r_min = 0.02
r_max = 0.30
```

### 9.5 Dilation and Blur

\[
M_{\mathrm{dilate}}
=
\mathrm{Dilate}(M_1, r_d)
\]

\[
M_{\mathrm{soft}}
=
\mathrm{GaussianBlur}(M_{\mathrm{dilate}}, \sigma).
\]

Suggested:

```text
r_d = 3 to 8
sigma = 2 to 5
```

---

## 10. Core / Ring / Preserve Masks

Define core mask:

\[
M_{\mathrm{core}} = M_{\mathrm{soft}}.
\]

Define a larger dilated mask:

\[
M_{\mathrm{wide}}
=
\mathrm{Dilate}(M_{\mathrm{core}}, r_{\mathrm{ring}}).
\]

Define ring:

\[
M_{\mathrm{ring}}
=
M_{\mathrm{wide}} - M_{\mathrm{core}}.
\]

Define preserve:

\[
M_{\mathrm{preserve}}
=
1 - M_{\mathrm{wide}}.
\]

Use masks in controller:

\[
u_{\mathrm{edit}}^{masked}
=
M_{\mathrm{core}}\odot u_{\mathrm{edit}}
+
\lambda_{\mathrm{ring}}M_{\mathrm{ring}}\odot u_{\mathrm{edit}}
\]

\[
u_{\mathrm{rec}}^{masked}
=
M_{\mathrm{preserve}}\odot u_{\mathrm{rec}}
+
\lambda_{\mathrm{ring}}^{rec}M_{\mathrm{ring}}\odot u_{\mathrm{rec}}.
\]

---

## 11. Candidate Selection

### 11.1 Candidate Bank

For each operation, produce multiple candidates:

```python
candidates = {
    "attention_only": A_new,
    "clean_only": D_clean,
    "velocity_only": D_vel,
    "attn_x_clean": A_new * D_clean,
    "host_x_clean": A_host * D_clean,
    "new_x_host_x_clean": A_new * A_host * D_clean,
    "removed_src_x_clean": A_removed_src * D_clean,
    "seg_x_clean": M_seg * D_clean,
    "relation_x_clean": R_relation * D_clean,
}
```

### 11.2 Manual Candidate Selection by Operation

First version should use operation-level defaults:

| Operation | Default Candidate |
|---|---|
| add_object | `relation_x_clean` if relation exists, else `attn_x_clean` |
| add_decal | `new_x_host_x_clean` or `host_surface_x_clean` |
| remove_object | `seg_x_clean` or `removed_src_x_clean` |
| replace | `seg_x_clean` or `src_tar_attn_x_clean` |

This is not task-specific. It is operation-specific.

### 11.3 Automatic Candidate Selection

Later, choose candidate by score.

For each candidate mask \(M_k\):

\[
J(M_k)
=
\alpha \mathrm{EditResponse}(M_k)
-
\beta \mathrm{PreserveLeakage}(M_k)
-
\gamma \mathrm{AreaPenalty}(M_k)
+
\delta \mathrm{GroundingConfidence}(M_k)
\]

where:

\[
\mathrm{EditResponse}(M_k)
=
\|M_k\odot D_{\mathrm{clean}}\|
\]

\[
\mathrm{PreserveLeakage}(M_k)
=
\|(1-M_k)\odot D_{\mathrm{clean}}\|
\]

\[
\mathrm{AreaPenalty}(M_k)
=
\left|
\frac{|M_k|}{HW}
-
r_{\mathrm{target}}
\right|.
\]

Select:

\[
M^*
=
\arg\max_k J(M_k).
\]

---

## 12. Task-Specific Operation Settings

These settings are operation-level, not object-template-level.

### 12.1 cat_crown

```text
operation = add_object
new_tokens = crown
host_tokens = cat
relation = above_host
```

Evidence:

\[
M_{\mathrm{cat}},\quad R_{\mathrm{above}},\quad A_{\mathrm{crown}},\quad D_{\mathrm{clean}},\quad D_{\mathrm{vel}}
\]

Recommended support:

\[
S =
R_{\mathrm{above}}
\odot
\left(
\lambda_n A_{\mathrm{crown}}
+
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
\right)
\]

### 12.2 dog_sunglasses

```text
operation = add_object
new_tokens = sunglasses
host_tokens = dog
relation = on_face or none
```

Recommended support:

\[
S =
A_{\mathrm{sunglasses}}
\odot
D_{\mathrm{clean}}
\]

Optional restriction:

\[
S =
S \odot \mathrm{Dilate}(M_{\mathrm{dog}})
\]

Do not degrade this task.

### 12.3 mug_heart

```text
operation = add_decal
new_tokens = heart
host_tokens = mug
relation = on_surface
```

Recommended support:

\[
S =
M_{\mathrm{surface}}
\odot
\left(
\lambda_n A_{\mathrm{heart}}
+
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
\right)
\]

where:

\[
M_{\mathrm{surface}}
=
\mathrm{SurfaceProposal}(M_{\mathrm{mug}}).
\]

### 12.4 backpack_remove_toy_charm

```text
operation = remove_object
removed_tokens = toy charm
host_tokens = backpack
relation = remove_source_object
```

Recommended support:

\[
S =
M_{\mathrm{removed}}
\odot
\left(
\lambda_d D_{\mathrm{clean}}
+
\lambda_v D_{\mathrm{vel}}
+
\lambda_a A_{\mathrm{removed,src}}
\right)
\]

where:

\[
M_{\mathrm{removed}}
=
\mathrm{Segment}(x_{\mathrm{src}}, \text{``toy charm''}).
\]

---

## 13. Evaluation Metrics for Support

If manual support is treated as pseudo upper bound:

### 13.1 IoU

\[
\mathrm{IoU}
=
\frac{
|M_{\mathrm{pred}}\cap M_{\mathrm{manual}}|
}{
|M_{\mathrm{pred}}\cup M_{\mathrm{manual}}|
}
\]

### 13.2 Coverage

\[
\mathrm{Coverage}
=
\frac{
|M_{\mathrm{pred}}\cap M_{\mathrm{manual}}|
}{
|M_{\mathrm{manual}}|
}
\]

### 13.3 Leakage

\[
\mathrm{Leakage}
=
\frac{
|M_{\mathrm{pred}}\setminus M_{\mathrm{manual}}|
}{
|M_{\mathrm{pred}}|+\epsilon
}
\]

### 13.4 Component Count

\[
N_{\mathrm{comp}}
=
\text{number of connected components in } M_{\mathrm{pred}}.
\]

Useful diagnostics:

```text
support_iou
support_coverage
support_leakage
support_num_components
support_area_ratio
```

---

## 14. Experiment Plan

Compare:

```text
manual_support
generic_support_v1
operation_aware_support_v2
support_v3_grounded
```

on:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

For each:

- generate support maps
- save mask metrics
- run adaptive controller
- compare final editing result

---

## 15. Expected Outcomes

### cat_crown

Expected improvement:

- IoU / coverage should increase from 0.
- Crown should appear more reliably.
- Support should shift toward above-head region.

### dog_sunglasses

Expected improvement:

- should remain strong.
- v3 should not degrade the current success.

### mug_heart

Expected improvement:

- support should cover mug surface.
- heart should appear more reliably.

### backpack_remove_toy_charm

Expected improvement:

- source-object coverage should increase beyond current 0.39–0.41.
- removal should become cleaner.

---

## 16. Implementation To-Do

### New module

Create:

```text
operation_support_v3.py
```

### Functions

```python
parse_edit_operation(...)
ground_object_mask(...)
compute_token_attention(...)
compute_clean_disagreement(...)
compute_velocity_disagreement(...)
build_relation_region(...)
build_surface_region(...)
build_support_candidates(...)
postprocess_support(...)
build_core_ring_preserve_masks(...)
score_support_candidate(...)
save_support_debug(...)
```

### CLI flags

```bash
--support-mode operation_v3
--edit-operation add_object
--new-tokens crown
--host-tokens cat
--removed-tokens toy,charm
--relation above_host
--grounding-method grounded_sam
--support-candidate auto
--support-top-percentile 90
--support-min-area-ratio 0.02
--support-max-area-ratio 0.30
--save-support-debug
```

---

## 17. Final Summary

The next support update should not continue tuning:

\[
A_{\mathrm{changed}}\odot D_{\mathrm{clean}}.
\]

Instead, support-v3 should use:

\[
\boxed{
\text{operation-aware support}
=
\text{token evidence}
+
\text{clean / velocity response}
+
\text{grounding / segmentation}
+
\text{relation proposal}
}
\]

This directly addresses the current failures:

- `cat_crown`: needs relation/layout support.
- `mug_heart`: needs host surface/decal support.
- `backpack_remove_toy_charm`: needs source-object segmentation.
- `dog_sunglasses`: should remain as a positive case.

Once support-v3 improves support quality, the adaptive clean-estimate controller can be evaluated more fairly.
