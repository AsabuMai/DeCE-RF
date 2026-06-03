# RF h-Edit: Support-v3 Refinement and Removal Controller Plan

## 0. Current Diagnosis

The current system can be summarized as:

> support-v3 has improved localization evidence, but correct support is not yet reliably converted into correct editing results.

More specifically:

- `cat_crown`: relation support works, but the mask is too large / too low / too coarse.
- `dog_sunglasses`: support works well and should be preserved as a positive control.
- `mug_heart`: surface/decal support works, but preservation becomes worse.
- `backpack_remove_toy_charm`: support is already relatively correct, but the removal dynamics are not suitable.

Therefore, the next step should not be another global controller adjustment.  
The next step should separate the problems:

\[
\boxed{\text{support refinement for insertion/decal}}
\]

and

\[
\boxed{\text{removal-specific editing dynamics}}
\]

---

## 1. Keep the Current RF Mathematical Base

The framework still uses the linear Rectified Flow path:

\[
x_t = (1-t)x_0 + t x_1.
\]

The model velocity is:

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
x_t - t\,v_{\theta}^{src}(x_t,t),
\]

\[
\hat{x}_{0,t}^{tar}
=
x_t - t\,v_{\theta}^{tar}(x_t,t).
\]

The clean-space displacement to RF velocity correction remains:

\[
\boxed{
u
=
-\frac{\Delta x_0}{t}
}
\]

This relation should remain the central mathematical interface.

---

## 2. Updated Task Categories

| Task | Type | Main Problem | Next Action |
|---|---|---|---|
| `cat_crown` | add object with relation | support too large / too low | refine relation support |
| `dog_sunglasses` | local accessory insertion | already works | keep as positive control |
| `mug_heart` | decal / surface pattern | edit works but preserve worsens | shrink decal support + strengthen preserve |
| `backpack_remove_toy_charm` | removal | support is decent but removal fails | add removal-specific controller |

The main research focus should temporarily be:

\[
\text{local insertion + decal editing}
\]

while removal should be treated as:

\[
\text{extension / separate module / failure analysis}
\]

until a dedicated removal controller works.

---

# 3. `cat_crown`: Tighter Relation Support

## 3.1 Current Problem

Observed support statistics:

```text
manual support area: 0.0201
v3 support score area: 0.0324
v3 final mask area: 0.0595
v3 vs manual IoU: 0.1463
v3 support bbox: y=0.000-0.214
manual bbox: y=0.000-0.143
```

This means the relation support knows the rough direction, but:

- it is too large,
- it extends too low,
- it includes too much of the cat head / face region,
- the final mask is roughly 3x the manual support area.

## 3.2 Above-Host Relation Proposal

For `cat_crown`, the operation is:

```text
operation = add_object
relation = above_host
host = cat
new_token = crown
```

Let the host mask be:

\[
M_{\mathrm{host}}.
\]

Compute the host bounding box:

\[
B_{\mathrm{host}}
=
(x_{\min},y_{\min},x_{\max},y_{\max}).
\]

Let:

\[
w = x_{\max}-x_{\min},
\]

\[
h = y_{\max}-y_{\min}.
\]

Define a tighter above-host box:

\[
\boxed{
B_{\mathrm{above}}
=
[
x_{\min}-\alpha_w w,\;
y_{\min}-\alpha_h h,\;
x_{\max}+\alpha_w w,\;
y_{\min}+\beta_h h
]
}
\]

Suggested values:

```text
alpha_w = 0.10 to 0.20
alpha_h = 0.35 to 0.55
beta_h  = 0.05 to 0.15
```

The important part is:

\[
y_{\max}^{above}
=
y_{\min}+\beta_h h
\]

should stay close to the host top.  
It should not extend too far down into the head / face.

Convert the box into a relation mask:

\[
R_{\mathrm{above}}
=
\mathrm{BoxMask}(B_{\mathrm{above}}).
\]

## 3.3 Crown Support Score

The support score should be:

\[
S_{\mathrm{crown}}
=
R_{\mathrm{above}}
\odot
\left(
\lambda_n \,\mathrm{Norm}(A_{\mathrm{crown}})
+
\lambda_d \,\mathrm{Norm}(D_{\mathrm{clean}})
+
\lambda_v \,\mathrm{Norm}(D_{\mathrm{vel}})
\right).
\]

where:

- \(A_{\mathrm{crown}}\): crown-token attention map
- \(D_{\mathrm{clean}}\): clean-estimate disagreement
- \(D_{\mathrm{vel}}\): source-target velocity disagreement

Suggested first weights:

```text
lambda_n = 0.4
lambda_d = 0.4
lambda_v = 0.2
```

## 3.4 Crown-Specific Operation Bounds

This is not object-template-specific.  
It is operation/relation-specific.

For:

```text
add_object + above_host
```

use smaller support bounds:

```text
max_area_ratio = 0.025 to 0.035
min_area_ratio = 0.008 to 0.015
dilate_radius = 2 to 4
blur_sigma = 1.5 to 3
top_components = 1
```

The goal is:

\[
0.02 \lesssim |M_{\mathrm{edit}}|/HW \lesssim 0.035.
\]

## 3.5 Cat Crown Success Criteria

Next version should aim for:

\[
\mathrm{IoU}(M_{\mathrm{v3}},M_{\mathrm{manual}})
>
0.25
\]

and:

\[
\mathrm{area}(M_{\mathrm{v3}})
\approx
0.02\sim 0.035.
\]

The bottom of the support bbox should move closer to manual:

```text
manual bbox bottom y ≈ 0.143
current v3 bottom y ≈ 0.214
target next bottom y < 0.16
```

---

# 4. `dog_sunglasses`: Positive Control / Non-Regression

## 4.1 Current Status

`dog_sunglasses` works well because the token `sunglasses` has:

- strong local attention,
- strong edit response,
- natural alignment with the face / eyes region.

This task should not be degraded while improving other tasks.

## 4.2 Confidence-Gated Support Selection

Use attention localization confidence:

\[
c_{\mathrm{attn}}
=
\frac{
\sum_{(i,j)\in \mathrm{TopK}(A)} A(i,j)
}{
\sum_{i,j} A(i,j)+\epsilon
}.
\]

If:

\[
c_{\mathrm{attn}} > \tau_{\mathrm{local}},
\]

then use the simple support:

\[
S_{\mathrm{dog}}
=
A_{\mathrm{sunglasses}}
\odot
D_{\mathrm{clean}}.
\]

Suggested:

```text
tau_local = 0.45 to 0.60
```

If attention is not concentrated, fallback to operation-aware support.

## 4.3 Optional Host Restriction

If the support leaks into background, restrict by the host:

\[
S_{\mathrm{dog}}
=
S_{\mathrm{dog}}
\odot
\mathrm{Dilate}(M_{\mathrm{dog}}).
\]

This should only be used if leakage is observed.

## 4.4 Dog Sunglasses Success Criteria

The dog sunglasses result should remain at least as good as current generic support.

This task is a **non-regression test**:

```text
Do not sacrifice dog_sunglasses while improving cat_crown and mug_heart.
```

---

# 5. `mug_heart`: Decal Support and Stronger Preservation

## 5.1 Current Problem

Observed:

```text
v1 CLIPΔ: -0.0273
v2 CLIPΔ: -0.0254
v3 CLIPΔ: 0.0438
v3 a_in_grounding: 1.0
v3 vs relation IoU: 0.8586
outside-mask L1 increases from 0.0078 to 0.0284
support_threshold = 0.0
```

Interpretation:

- v3 support finds the mug surface.
- heart editing works.
- but outside-mask drift becomes worse.
- support score is sparse and unstable.
- edit velocity leaks into preserve regions.

## 5.2 Surface-Local Normalization

For decal tasks, normalize disagreement only inside the surface mask.

Let:

\[
M_{\mathrm{surface}}
\]

be the mug surface proposal.

Define local min/max:

\[
D_{\min}^{surf}
=
\min_{(i,j)\in M_{\mathrm{surface}}} D(i,j),
\]

\[
D_{\max}^{surf}
=
\max_{(i,j)\in M_{\mathrm{surface}}} D(i,j).
\]

Then:

\[
\boxed{
D_{\mathrm{surface}}(i,j)
=
\frac{
D(i,j)-D_{\min}^{surf}
}{
D_{\max}^{surf}-D_{\min}^{surf}+\epsilon
}
}
\]

inside \(M_{\mathrm{surface}}\), and zero outside:

\[
D_{\mathrm{surface}}
=
M_{\mathrm{surface}}
\odot
D_{\mathrm{surface}}.
\]

This prevents global normalization from making surface response collapse to zero.

## 5.3 Decal Support Score

The decal support should be:

\[
\boxed{
S_{\mathrm{heart}}
=
M_{\mathrm{surface}}
\odot
\left(
\lambda_a \,\mathrm{Norm}_{surf}(A_{\mathrm{heart}})
+
\lambda_d \,D_{\mathrm{surface}}
+
\lambda_v \,\mathrm{Norm}_{surf}(D_{\mathrm{vel}})
\right)
}
\]

Suggested first weights:

```text
lambda_a = 0.3
lambda_d = 0.5
lambda_v = 0.2
```

If heart attention is unreliable, use:

\[
S_{\mathrm{heart}}
=
M_{\mathrm{surface}}
\odot
\left(
\lambda_d D_{\mathrm{surface}}
+
\lambda_v \mathrm{Norm}_{surf}(D_{\mathrm{vel}})
\right).
\]

## 5.4 Smaller Decal Mask

For:

```text
operation = add_decal
```

use smaller mask settings:

```text
max_area_ratio = 0.015 to 0.035
min_area_ratio = 0.005 to 0.012
top_components = 1
dilate_radius = 1 to 3
blur_sigma = 1.0 to 2.5
```

The decal core should be small:

\[
M_{\mathrm{core}}
\subset
M_{\mathrm{surface}}.
\]

## 5.5 Stronger Preserve Outside the Decal

For decal operations, edit velocity should be strongly masked:

\[
u_{\mathrm{edit}}^{masked}
=
M_{\mathrm{core}}\odot u_{\mathrm{edit}}
+
\lambda_{\mathrm{ring}} M_{\mathrm{ring}}\odot u_{\mathrm{edit}}.
\]

Suggested:

```text
lambda_ring = 0.2 to 0.4
```

In preserve region:

\[
M_{\mathrm{preserve}}\odot u_{\mathrm{edit}}
\approx 0.
\]

Reconstruction should be stronger outside the decal:

\[
u_{\mathrm{rec}}^{masked}
=
\lambda_{\mathrm{pres}}
M_{\mathrm{preserve}}\odot u_{\mathrm{rec}}
+
\lambda_{\mathrm{ring}}^{rec}
M_{\mathrm{ring}}\odot u_{\mathrm{rec}}.
\]

Suggested:

```text
lambda_preserve higher than add_object
trajectory_preserve higher than add_object
edit leakage suppression enabled
clean-effect projection enabled
```

## 5.6 Clean-Effect Projection for Decal

For edit velocity:

\[
u_{\mathrm{edit}},
\]

its clean-estimate effect is:

\[
\Delta \hat{x}_0^{edit}
=
-t\,u_{\mathrm{edit}}.
\]

Preserve error:

\[
e_{\mathrm{pres}}
=
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{\mathrm{src}}).
\]

Preserve effect of edit:

\[
\Delta_{\mathrm{pres}}^{edit}
=
M_{\mathrm{preserve}}
\odot
\Delta \hat{x}_0^{edit}.
\]

If:

\[
\left\langle
e_{\mathrm{pres}},
\Delta_{\mathrm{pres}}^{edit}
\right\rangle
>0,
\]

then edit velocity increases preserve drift.

Project or suppress this component:

\[
\Delta_{\mathrm{pres}}^{edit,\perp}
=
\Delta_{\mathrm{pres}}^{edit}
-
\frac{
\left\langle
\Delta_{\mathrm{pres}}^{edit},
e_{\mathrm{pres}}
\right\rangle
}{
\|e_{\mathrm{pres}}\|^2+\epsilon
}
e_{\mathrm{pres}}.
\]

Then map back to velocity:

\[
u_{\mathrm{edit}}^{\perp}
=
-\frac{\Delta \hat{x}_0^{edit,\perp}}{t}.
\]

For decal tasks, this projection should be enabled by default.

## 5.7 Mug Heart Success Criteria

Next version should preserve the positive CLIP improvement while reducing outside-mask drift:

\[
\Delta_{\mathrm{CLIP}} > 0
\]

and:

\[
\mathrm{outside\text{-}mask\ L1}
<
0.015.
\]

The heart should remain visible, but mug/background should not become gray or globally altered.

---

# 6. `backpack_remove_toy_charm`: Removal-Specific Controller

## 6.1 Current Problem

Observed:

```text
v3 support vs manual IoU: 0.6588
v3 a_in_manual: 1.0
CLIPΔ: -0.0082
object remains / becomes distorted
```

Interpretation:

- support is already reasonably good.
- the problem is not primarily localization.
- current edit dynamics do not know how to erase an object and inpaint the local region.

Therefore, removal needs a separate controller.

## 6.2 Why Removal Is Different

Addition/decal editing tries to introduce new visual content.

Removal requires:

1. suppressing the removed object,
2. filling the region with plausible background / host texture,
3. preserving boundary consistency,
4. avoiding deformation of nearby objects.

Thus removal should not use the same edit dynamics as add_object.

## 6.3 Removal Support

Let:

\[
M_{\mathrm{remove}}
\]

be the support mask for the removed object.

Prefer:

\[
M_{\mathrm{remove}}
=
\mathrm{Segment}(x_{\mathrm{src}}, q_{\mathrm{removed}})
\]

or:

\[
M_{\mathrm{remove}}
=
A_{\mathrm{removed,src}}\odot D_{\mathrm{clean}}.
\]

For current task:

```text
q_removed = "toy charm"
```

## 6.4 No-Object Clean Estimate

Define a no-object prompt:

```text
c_noobj = source prompt without removed object
```

or:

```text
c_noobj = target prompt
```

Compute no-object velocity:

\[
v_{\theta}^{noobj}(x_t,t).
\]

No-object clean estimate:

\[
\hat{x}_{0,t}^{noobj}
=
x_t - t\,v_{\theta}^{noobj}(x_t,t).
\]

Current clean estimate:

\[
\hat{x}_{0,t}^{cur}
=
x_t - t\,v_{\theta}^{cur}(x_t,t).
\]

## 6.5 Local Fill Correction

Define local clean-space fill displacement:

\[
\Delta x_0^{fill}
=
M_{\mathrm{remove}}
\odot
\left(
\hat{x}_{0,t}^{noobj}
-
\hat{x}_{0,t}^{cur}
\right).
\]

Convert to RF velocity:

\[
\boxed{
u_{\mathrm{fill}}
=
-\frac{
\Delta x_0^{fill}
}{t}
}
\]

This directly uses the clean-estimate-space control interface.

## 6.6 Removed-Token Suppression

If a removed-token direction is available, define:

\[
u_{\mathrm{sup}}
=
-\lambda_{\mathrm{sup}}
M_{\mathrm{remove}}
\odot
u_{\mathrm{removed-token}}.
\]

If not available, use local target-source velocity:

\[
u_{\mathrm{sup}}
=
\lambda_{\mathrm{sup}}
M_{\mathrm{remove}}
\odot
\left(
v_{\theta}^{noobj}
-
v_{\theta}^{src}
\right).
\]

This suppresses the source-object direction inside the removal mask.

## 6.7 Boundary Ring Preservation

Define ring:

\[
M_{\mathrm{ring}}
=
\mathrm{Dilate}(M_{\mathrm{remove}})
-
M_{\mathrm{remove}}.
\]

Boundary preserve velocity:

\[
u_{\mathrm{ring-rec}}
=
\lambda_{\mathrm{ring}}
M_{\mathrm{ring}}
\odot
u_{\mathrm{rec}}.
\]

This preserves local continuity between removed region and surrounding background / host object.

## 6.8 Removal Total Velocity

The removal-specific correction is:

\[
\boxed{
u_{\mathrm{remove}}
=
\lambda_f u_{\mathrm{fill}}
+
\lambda_s u_{\mathrm{sup}}
+
\lambda_r u_{\mathrm{ring-rec}}
}
\]

Total dynamics:

\[
\dot{x}_t
=
v_{\mathrm{src}}
+
u_{\mathrm{remove}}
+
u_{\mathrm{preserve}}.
\]

where:

\[
u_{\mathrm{preserve}}
=
M_{\mathrm{preserve}}\odot u_{\mathrm{rec}}.
\]

Suggested first weights:

```text
lambda_f = 1.0
lambda_s = 0.5
lambda_r = 0.5
```

Tune carefully.

## 6.9 Removal Success Criteria

The removed object should disappear or become much less visible.

Metrics:

```text
removed-object CLIP / detector score decreases
background / host preservation remains stable
manual-mask region changes
outside-mask drift remains small
```

For now, removal should be treated as:

```text
extension / separate module
```

not the main success case.

---

# 7. Improved Component / Proposal Scoring

The current support maps often have many components.

Instead of only percentile threshold + component filtering, score components.

For each connected component \(C_i\):

\[
\mathrm{score}(C_i)
=
\alpha
\sum_{(x,y)\in C_i} S(x,y)
+
\beta
\sum_{(x,y)\in C_i} D_{\mathrm{clean}}(x,y)
-
\gamma
\mathrm{AreaPenalty}(C_i)
+
\delta
\mathrm{Compactness}(C_i)
+
\eta
\mathrm{RelationOverlap}(C_i).
\]

## 7.1 Area Penalty

Let:

\[
r_i=\frac{|C_i|}{HW}.
\]

Target area ratio:

\[
r_{\mathrm{target}}.
\]

Area penalty:

\[
\mathrm{AreaPenalty}(C_i)
=
|r_i-r_{\mathrm{target}}|.
\]

## 7.2 Compactness

\[
\mathrm{Compactness}(C_i)
=
\frac{
4\pi |C_i|
}{
\mathrm{Perimeter}(C_i)^2+\epsilon
}.
\]

Higher is better.

## 7.3 Relation Overlap

For relation proposal \(R_{\mathrm{relation}}\):

\[
\mathrm{RelationOverlap}(C_i)
=
\frac{
|C_i\cap R_{\mathrm{relation}}|
}{
|C_i|+\epsilon
}.
\]

For surface proposal \(M_{\mathrm{surface}}\):

\[
\mathrm{SurfaceOverlap}(C_i)
=
\frac{
|C_i\cap M_{\mathrm{surface}}|
}{
|C_i|+\epsilon
}.
\]

This helps reject components outside the intended region.

## 7.4 Select Component

Choose:

\[
C^*
=
\arg\max_i \mathrm{score}(C_i).
\]

Then:

\[
M_{\mathrm{core}}=C^*.
\]

---

# 8. Debug Saving Fix

Current debug saving sometimes misses important candidates because it saves the first candidates alphabetically.

Fix this.

Always save:

```text
selected_candidate_raw.png
selected_candidate_postprocessed.png
support_score_selected.png
M_core.png
M_ring.png
M_preserve.png
grounding_mask.png
relation_region.png
surface_region.png
```

Also save metadata:

```json
{
  "selected_candidate": "host_surface_x_response",
  "operation": "add_decal",
  "relation": "on_surface",
  "support_area": 0.0,
  "support_bbox": [0, 0, 0, 0],
  "num_components": 0,
  "component_score": 0.0,
  "iou_to_manual": 0.0,
  "coverage_to_manual": 0.0,
  "leakage_to_manual": 0.0
}
```

---

# 9. Revised Experimental Focus

## Main tasks

Focus on:

```text
cat_crown
dog_sunglasses
mug_heart
```

These represent:

- relation-based insertion,
- positive local insertion,
- decal / surface editing.

## Secondary task

Use:

```text
backpack_remove_toy_charm
```

as:

```text
removal-specific controller test
```

not as the core success case yet.

---

# 10. Immediate To-Do List

## Support refinement

- [ ] Tighten `above_host` region for `cat_crown`.
- [ ] Add smaller area bounds for `add_object + above_host`.
- [ ] Add surface-local normalization for `mug_heart`.
- [ ] Add smaller decal mask settings.
- [ ] Add stronger outside-preserve for decal tasks.
- [ ] Keep `dog_sunglasses` as non-regression.

## Removal controller

- [ ] Add no-object clean estimate.
- [ ] Add local fill correction \(u_{\mathrm{fill}}\).
- [ ] Add removed-token suppression \(u_{\mathrm{sup}}\).
- [ ] Add boundary ring preservation.
- [ ] Compare removal controller vs normal edit controller.

## Debug

- [ ] Always save selected candidate map.
- [ ] Always save core/ring/preserve masks.
- [ ] Save support metadata JSON.
- [ ] Save component scores.

## Experiments

- [ ] Run cat_crown with tighter relation support.
- [ ] Run mug_heart with decal-local support and stronger preserve.
- [ ] Confirm dog_sunglasses is not degraded.
- [ ] Run backpack_remove with removal controller.

---

# 11. One-Sentence Summary

The next step is to stop treating all edit operations with the same controller: relation-based insertion needs tighter relation support, decal editing needs smaller surface-local support and stronger preservation, and removal needs a dedicated remove-and-fill controller rather than ordinary target-edit velocity.
