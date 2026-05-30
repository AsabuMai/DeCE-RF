# RF h-Edit: Generic Support + Adaptive Controller To-Do Plan

## 0. Current Problem

The current adaptive controller is becoming more general, but the **support / mask generation** still contains many task-specific rules.

Examples of task-template traces:

- `front_glasses_auto`
- hand-designed `DECAL_BOX`
- accessory-specific rules for sunglasses / crown / decals
- explicit phrase + dilation rules for removal tasks
- geometry rules such as `inside`, `on_top`, `front`

This weakens the main research claim.

If the method depends heavily on task-specific support, reviewers may ask:

> Is the improvement caused by the clean-estimate closed-loop controller, or by manually engineered masks?

Therefore, the next important step is to replace task-specific support with a more generic support proposer.

---

## 1. Updated Method Direction

The next version should be framed as:

> **Generic Support Proposal + Clean-Estimate-Space Adaptive Controller**

The support proposer should provide a coarse edit region automatically.

The adaptive controller should then use clean-estimate-space diagnostics to adjust edit and preserve dynamics.

The roles should be separated:

| Component | Role |
|---|---|
| Generic support proposer | Estimate where editing should happen |
| Adaptive controller | Decide whether edit progress is enough and whether preserve drift is too large |
| Velocity correction | Execute edit / preserve updates in RF ODE dynamics |

The support proposer should not be over-engineered for a specific object or task.

---

## 2. Core Idea: Generic Support

The proposed generic support should be based on:

\[
S = \mathrm{Norm}(A_{\mathrm{changed}}) \odot \mathrm{Norm}(D_{\mathrm{clean/velocity}})
\]

where:

- \(A_{\mathrm{changed}}\): attention map of changed target tokens
- \(D_{\mathrm{clean/velocity}}\): clean-estimate or velocity disagreement between source and target dynamics
- \(S\): support score map

Then construct:

\[
M_{\mathrm{edit}} = \mathrm{Postprocess}(S)
\]

\[
M_{\mathrm{preserve}} = 1 - \mathrm{Dilate}(M_{\mathrm{edit}})
\]

This means the edit region should be where:

1. the changed target token attends, and
2. the source-target dynamics actually disagree.

---

## 3. Generic Support Formula

### 3.1 Changed-Token Attention

Let:

\[
A_t
\]

be the attention map for changed target tokens.

Examples of changed target tokens:

| Task | Source | Target | Changed target token |
|---|---|---|---|
| cat_crown | cat | cat wearing a crown | crown |
| dog_sunglasses | dog | dog wearing sunglasses | sunglasses |
| mug_heart | plain mug | mug with a heart | heart |
| backpack_remove_toy_charm | backpack with charm | backpack without charm | toy charm |

The support module may still accept `--changed-target-words`, but it should not use task-specific geometry templates such as `front_glasses_auto`.

Allowed:

```bash
--changed-target-words sunglasses
```

Avoid:

```bash
--front-glasses-auto
--decal-box
--on-top-rule crown
```

---

### 3.2 Clean Disagreement

Use the path-induced clean estimate:

\[
\hat{x}_0 = x_t - t v_\theta(x_t,t)
\]

Then define clean disagreement:

\[
D_t^{clean}
=
\left\|
\hat{x}_{0,t}^{tar}
-
\hat{x}_{0,t}^{src}
\right\|_2
\]

This is preferred because it supports the clean-estimate-space controller narrative.

---

### 3.3 Velocity Disagreement

Optionally define velocity disagreement:

\[
D_t^{vel}
=
\left\|
v_t^{tar}
-
v_t^{src}
\right\|_2
\]

This can be used as an ablation.

---

### 3.4 Support Score

Default support score:

\[
S_t
=
\mathrm{Norm}(A_t)^\gamma
\odot
\mathrm{Norm}(D_t^{clean})^\eta
\]

Suggested first values:

```text
gamma = 1.0
eta = 1.0
```

Optional variants:

```text
attention_only: S = Norm(A)
clean_disagreement_only: S = Norm(D_clean)
velocity_disagreement_only: S = Norm(D_vel)
attention_x_clean: S = Norm(A) * Norm(D_clean)
attention_x_velocity: S = Norm(A) * Norm(D_vel)
```

---

## 4. Static vs Dynamic Support

### First version: Static support

Use early / mid ODE steps to estimate support once, then freeze it.

For selected steps:

\[
t \in \mathcal{T}_{support}
\]

compute:

\[
S_t = A_t \odot D_t
\]

Aggregate by mean or max:

\[
S = \frac{1}{K}\sum_{t\in\mathcal{T}_{support}} S_t
\]

or

\[
S = \max_{t\in\mathcal{T}_{support}} S_t
\]

Then generate fixed masks:

\[
M_{\mathrm{core}}, M_{\mathrm{ring}}, M_{\mathrm{preserve}}
\]

This is preferred for v1 because it is easier to debug.

---

### Optional later version: Dynamic support

Use EMA update:

\[
M_t = \rho M_{t-1} + (1-\rho)\tilde{M}_t
\]

This should only be implemented after static support is stable.

---

## 5. Mask Postprocessing

Generic support must avoid masks that are too small or too large.

Recommended steps:

1. normalize support score to \([0,1]\)
2. threshold by percentile / top-k
3. connected component filtering
4. keep top \(N\) components
5. dilate
6. blur / smooth
7. enforce min / max area ratio

Suggested parameters:

```text
top_percentile = 85 or 90
min_area_ratio = 0.02
max_area_ratio = 0.30
dilate_radius = 3 to 8
blur_sigma = 2 to 5
top_components = 1 or 2
```

These should be generic hyperparameters, not task-specific templates.

---

## 6. Three-Region Support

Do not use only one binary edit mask.

Generate three regions:

### 6.1 Edit Core

\[
M_{\mathrm{core}}
\]

Strong edit region.

### 6.2 Transition Ring

\[
M_{\mathrm{ring}}
=
\mathrm{Dilate}(M_{\mathrm{core}})
-
M_{\mathrm{core}}
\]

Weak edit / weak preserve transition region.

### 6.3 Preserve Region

\[
M_{\mathrm{preserve}}
=
1-\mathrm{Dilate}(M_{\mathrm{core}})
\]

Strong reconstruction / preservation region.

---

## 7. How Controller Uses Support

### 7.1 Editing velocity

\[
u_{\mathrm{edit}}^{masked}
=
M_{\mathrm{core}}\odot u_{\mathrm{edit}}
+
\lambda_{\mathrm{ring}} M_{\mathrm{ring}}\odot u_{\mathrm{edit}}
\]

Suggested:

```text
lambda_ring = 0.3 to 0.6
```

### 7.2 Reconstruction velocity

\[
u_{\mathrm{rec}}^{masked}
=
M_{\mathrm{preserve}}\odot u_{\mathrm{rec}}
+
\lambda_{\mathrm{ring}}^{rec} M_{\mathrm{ring}}\odot u_{\mathrm{rec}}
\]

Suggested:

```text
lambda_ring_rec = 0.3 to 0.8
```

### 7.3 Adaptive controller

The adaptive controller should use these masks to compute:

- directional edit progress in \(M_{\mathrm{core}}\)
- preserve drift in \(M_{\mathrm{preserve}}\)
- conflict projection mainly in \(M_{\mathrm{preserve}}\)

---

## 8. Keep Existing Manual Support as Upper Bound

Do not delete the current hand-engineered support.

Instead, re-label it as:

```text
manual_support / external_support / upper_bound_support
```

It should be used only for:

- upper-bound comparison
- diagnostic comparison
- verifying that the controller works when support is good

It should not be presented as the core method.

---

## 9. Required Support Ablation

For each task, compare:

| Setting | Meaning |
|---|---|
| `full` | current best full method |
| `v1_manual_support` | upper bound with current task-specific support |
| `v1_generic_support` | proposed generic support |
| `v1_attention_only` | changed-token attention only |
| `v1_clean_disagreement_only` | clean-estimate disagreement only |
| `v1_velocity_disagreement_only` | velocity disagreement only |
| `v1_attention_x_clean` | attention × clean disagreement |
| `v1_attention_x_velocity` | attention × velocity disagreement |

The most important comparison is:

\[
v1\_generic\_support
\quad \text{vs.} \quad
v1\_manual\_support
\]

If generic support approaches manual support performance, the method becomes much more convincing.

---

## 10. Main Tasks

Update the task list to the current four pretty tasks.

### Task 1: cat_crown

Purpose:

- local accessory / decoration insertion
- test support above the head
- test whether generic support can localize crown placement

---

### Task 2: dog_sunglasses

Purpose:

- local accessory insertion
- test whether support can localize eyes / face region
- important because current v0 does not clearly improve dog

---

### Task 3: mug_heart

Purpose:

- local symbol / pattern addition
- test whether support can localize mug surface
- useful for distinguishing correct target direction from generic visual change

---

### Task 4: backpack_remove_toy_charm

Purpose:

- removal / preservation-sensitive editing
- test whether support can isolate the removed object
- test whether preserve region remains stable

---

## 11. Integration with adaptive_full v1

The final target system should be:

```text
generic_support + adaptive_full_v1
```

where adaptive_full_v1 includes:

1. directional edit progress
2. preserve drift
3. adaptive edit boost
4. adaptive preserve lock
5. clean-effect projection

The generic support only gives a rough spatial region.

The controller decides how strongly to edit or preserve.

---

## 12. Directional Edit Progress

For v1, use:

\[
\Delta_{\mathrm{cur}}
=
M_{\mathrm{core}}\odot(\hat{x}_{0,t}-x_{0}^{ref})
\]

\[
\Delta_{\mathrm{tar}}
=
M_{\mathrm{core}}\odot(\hat{x}_{0}^{tar}-x_{0}^{ref})
\]

\[
p_{\mathrm{edit}}(t)
=
\frac{
\langle \Delta_{\mathrm{cur}}, \Delta_{\mathrm{tar}}\rangle
}{
\|\Delta_{\mathrm{tar}}\|^2+\epsilon
}
\]

This replaces RMS-only edit progress.

Keep RMS as an auxiliary diagnostic.

---

## 13. Preserve Drift

Use:

\[
d_{\mathrm{pres}}(t)
=
\frac{
\left\|
M_{\mathrm{preserve}}
\odot
(\hat{x}_{0,t}-x_{0}^{ref})
\right\|
}{
\left\|
M_{\mathrm{preserve}}\odot x_{0}^{ref}
\right\|+\epsilon
}
\]

This should trigger adaptive preserve lock.

---

## 14. Clean-Effect Projection

For edit velocity:

\[
u_{\mathrm{edit}}
\]

its clean-estimate effect is:

\[
\Delta \hat{x}_{0}^{edit} = -t u_{\mathrm{edit}}
\]

In preserve region:

\[
e_{\mathrm{pres}}
=
M_{\mathrm{preserve}}\odot(\hat{x}_{0,t}-x_{0}^{ref})
\]

\[
\Delta_{\mathrm{pres}}^{edit}
=
M_{\mathrm{preserve}}\odot(-t u_{\mathrm{edit}})
\]

If:

\[
\langle e_{\mathrm{pres}}, \Delta_{\mathrm{pres}}^{edit}\rangle > 0
\]

then edit velocity increases preserve drift.

Suppress or project that component.

---

## 15. Proposed New File

Create:

```text
generic_support.py
```

Suggested functions:

```python
compute_changed_token_attention(...)
compute_clean_disagreement(...)
compute_velocity_disagreement(...)
build_support_score(...)
aggregate_support_over_steps(...)
postprocess_support_score(...)
build_core_ring_preserve_masks(...)
save_support_debug_maps(...)
```

Optional:

```python
compute_support_area_stats(...)
filter_connected_components(...)
enforce_area_ratio(...)
```

---

## 16. Suggested CLI Flags

Add or standardize:

```bash
--support-mode generic
--support-score attention_x_clean
--changed-target-words sunglasses,crown,heart,toy,charm
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
--support-score attention_only
--support-score clean_disagreement_only
--support-score velocity_disagreement_only
--support-score attention_x_clean
--support-score attention_x_velocity
```

---

## 17. Debug Outputs to Save

For each run, save:

```text
attention_map.png
clean_disagreement_map.png
velocity_disagreement_map.png
support_score.png
M_core.png
M_ring.png
M_preserve.png
final_used_mask.png
```

Also save JSON stats:

```json
{
  "support_mode": "generic",
  "support_score": "attention_x_clean",
  "mask_area_core": 0.0,
  "mask_area_ring": 0.0,
  "mask_area_preserve": 0.0,
  "num_components": 0,
  "top_component_area": 0.0
}
```

These visualizations are important for paper figures and debugging.

---

## 18. Experiment Matrix

For each task:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

Run:

```text
full
v1_manual_support
v1_generic_support
v1_attention_only
v1_clean_disagreement_only
v1_velocity_disagreement_only
v1_attention_x_clean
v1_attention_x_velocity
```

Use at least:

```text
3 seeds
```

if compute resources allow.

---

## 19. Evaluation Questions

For each result, answer:

1. Is the edit localized correctly?
2. Is the target object / pattern visible?
3. Is the non-edit region preserved?
4. Does the result avoid overlay-like artifacts?
5. Is generic support close to manual support?
6. Does attention × clean disagreement outperform attention-only?
7. Does adaptive_full_v1 still work when support is imperfect?

---

## 20. Success Criteria

### Strong success

- `v1_generic_support` is close to `v1_manual_support`
- `attention_x_clean` outperforms attention-only and disagreement-only
- edits are localized and visible
- preserve-region drift is lower than full / direct guidance

### Moderate success

- generic support works for local insertion tasks but not removal
- manual support remains stronger, but generic support is usable

### Weak success

- generic support is unstable or too task-dependent
- manual support is still required for good results

If weak success occurs, the paper claim should be limited to:

> clean-estimate controller with externally provided or high-quality support.

---

## 21. Updated Research Claim

If generic support works, the research claim becomes:

> We propose a generic support proposal and clean-estimate-space closed-loop controller for Rectified Flow editing. The support proposer localizes edit regions using changed-token attention and clean-estimate disagreement, while the controller adaptively balances edit progress and preserve drift through RF velocity corrections.

This is much stronger than:

> We split RF editing into reconstruction and editing terms.

---

## 22. Immediate To-Do List

### Support implementation

- [x] Create `generic_support.py`.
- [x] Implement changed-token attention support.
- [x] Implement clean disagreement map.
- [x] Implement velocity disagreement map.
- [x] Implement support score `attention_x_clean`.
- [x] Implement support postprocessing.
- [x] Implement core / ring / preserve masks.
- [x] Save support debug maps.

### Controller integration

- [x] Connect generic support to adaptive_full_v1.
- [x] Use `M_core` for directional edit progress.
- [x] Use `M_preserve` for preserve drift.
- [x] Use `M_preserve` for clean-effect projection.
- [x] Keep manual support as upper bound.

### Experiments

- [x] Run four pretty tasks with manual support.
- [x] Run four pretty tasks with generic support.
- [x] Run support ablations.
- [x] Generate visual comparison panels.
- [x] Save mask and trajectory diagnostics.

Current status: `generic_support + adaptive_full_v1` is wired and has been run on all four pretty tasks with seeds 10, 11, and 12. Seed-10 support ablations are complete for `attention_only`, `clean_disagreement_only`, `velocity_disagreement_only`, `attention_x_clean`, and `attention_x_velocity`. The generated debug maps include `generic_attention_map.png`, `generic_clean_disagreement_map.png`, `generic_velocity_disagreement_map.png`, and `generic_support_score.png`. The tuned support area is controlled, but the visual result is mixed: `dog_sunglasses` is the strongest case, while `cat_crown`, `mug_heart`, and `backpack_remove_toy_charm` still need stronger support / target evidence.

### Documentation

- [x] Update project direction from task-specific support to generic support.
- [x] Clearly state that manual support is an upper bound, not the core method.
- [x] Update contribution statement around generic support + clean-estimate controller.

---

## 23. One-Sentence Summary

The next step is to replace task-specific support rules with a generic support proposer based on changed-token attention and clean-estimate disagreement, then combine it with adaptive_full_v1 so the project becomes a general clean-estimate-space local closed-loop RF editing method rather than a task-engineered local editing prototype.
