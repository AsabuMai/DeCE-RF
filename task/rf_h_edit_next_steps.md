# RF h-Edit Project: Next-Step Action Plan

This document summarizes what should be done next for the current RF h-Edit / SD3 Rectified Flow editing prototype.

The current project is already following the intended direction:

\[
\dot{x}_t =
\underbrace{v_{\mathrm{src}} + u_{\mathrm{rec}}}_{\text{reconstruction-aware base field}}
+
\underbrace{u_{\mathrm{edit}}}_{\text{editing field}} .
\]

The goal of the next stage is **not** to add more guidance terms blindly.  
The goal is to make the current prototype clean, testable, and suitable for method validation.

---

## 1. Current Status

The current implementation already contains the main ingredients of the intended framework:

- Linear Rectified Flow path:
  \[
  x_t = (1-t)x_0 + t x_1
  \]

- Path-induced clean estimate:
  \[
  \hat{x}_0 = x_t - t v_\theta(x_t,t)
  \]

- Clean-space displacement to velocity correction:
  \[
  u = -\frac{\Delta x_0}{t}
  \]

- Source-conditioned base velocity:
  \[
  v_{\mathrm{src}}
  \]

- Reconstruction correction:
  \[
  u_{\mathrm{rec}}
  \]

- Editing correction:
  \[
  u_{\mathrm{edit}}
  \]

- Baseline scripts for:
  - base only
  - direct target
  - decoupled reconstruction
  - anchor only
  - ablation runs

This means the project is no longer just a theoretical idea. It is now a working research prototype.

---

## 2. Core Research Claim to Validate

The main claim we need to validate is:

> Compared with direct target guidance, a decoupled reconstruction-editing dynamics provides a better trade-off between editing effectiveness and source-image faithfulness in ODE / Rectified Flow image editing.

In practical terms, the experiments should show that:

- direct target flow edits strongly but causes drift,
- reconstruction-aware dynamics preserves structure better,
- the decoupled method can edit while reducing unwanted changes,
- local editing benefits from spatial masks and trajectory preservation.

---

## 3. Preferred Mathematical Convention

All documentation and code comments should use the following convention.

### Base dynamics

\[
v_{\mathrm{base}} = v_{\mathrm{src}}
\]

### Reconstruction-aware base field

\[
v_{\mathrm{rec-base}} = v_{\mathrm{src}} + u_{\mathrm{rec}}
\]

### Editing field

\[
u_{\mathrm{edit}}
=
\lambda_{\mathrm{flow}}(v_{\mathrm{tar}}-v_{\mathrm{src}})
+
\lambda_{\mathrm{anchor}}u_{\mathrm{anchor}}
+
\lambda_{\mathrm{region}}u_{\mathrm{region}}
+
\lambda_{\mathrm{clip}}u_{\mathrm{clip}}
+
\lambda_{\mathrm{text}}u_{\mathrm{text}}
+
\cdots
\]

### Total dynamics

\[
\dot{x}_t =
(v_{\mathrm{src}} + u_{\mathrm{rec}}) + u_{\mathrm{edit}} .
\]

This is the current implementation-level formulation.

Avoid presenting the current implementation as only:

\[
\dot{x}_t = -\alpha\nabla E_{\mathrm{rec}} + \beta\nabla E_{\mathrm{edit}} .
\]

That formula is useful as high-level motivation, but the code currently uses **energy-inspired surrogate velocity fields**, not only exact energy gradients.

---

## 4. Priority 0: Clean Up Documentation Consistency

### Problem

Some documents still contain older formulas such as:

\[
\dot{x}_t = \beta(t)v_{\mathrm{edit}} + \alpha(t)v_{\mathrm{rec}} .
\]

This can make it look like the base RF velocity is missing.

### Required update

Use this formulation everywhere:

\[
\dot{x}_t =
\underbrace{v_{\mathrm{src}} + u_{\mathrm{rec}}}_{\text{reconstruction-aware base field}}
+
\underbrace{u_{\mathrm{edit}}}_{\text{editing field}} .
\]

### Files to check

- `README.md`
- `ode_rf_editing_summary.md`
- `docs/worklog_2026-04-24.md`
- any comments inside `sd3_hrec.py`
- any comments inside `energies.py`

### Expected output

The documentation should clearly state:

> The current implementation uses a source-conditioned RF velocity as the base field, adds a reconstruction-aware correction for faithfulness, and adds editing velocity surrogates for target transformation.

---

## 5. Priority 0: Separate Three Experimental Questions

The current project mixes three different questions. They should be separated.

### Question A: Does the velocity-field formulation work?

Compare:

1. base only
2. direct target flow
3. anchor only

This validates the basic RF editing directions.

### Question B: Does reconstruction decoupling help?

Compare:

1. direct target
2. direct target + reconstruction correction
3. direct target + trajectory preservation

This validates whether the reconstruction branch improves faithfulness.

### Question C: Does spatial support control help local editing?

Compare:

1. broad attention mask
2. target-changed attention mask
3. external diagnostic mask
4. external mask + trajectory preservation

This validates whether local editing depends on accurate spatial support.

---

## 6. Priority 0: Fix Baseline Organization

At minimum, keep the following reproducible scripts.

```text
scripts/
  run_base_only.sh
  run_direct_target.sh
  run_decoupled_rec.sh
  run_anchor_only.sh
  run_ablation_all.sh
  run_sunglasses_local.sh
  run_sunglasses_external_mask.sh
```

Each run should save:

```text
outputs/<experiment_name>/
  result.png
  stats.json
  command.txt
  metadata.json
  masks/
```

The `command.txt` file is important. It makes each result reproducible.

---

## 7. Priority 1: Standardize Two Main Tasks

Do not expand to many prompts yet.  
Use two fixed tasks first.

---

### Task 1: Object Replacement

Example:

```text
source prompt: A panda is walking in a forest.
target prompt: A tiger is walking in a forest.
```

Purpose:

- test semantic replacement,
- expose overlay-like failure,
- compare direct target vs decoupled reconstruction.

Expected observations:

- direct target: stronger semantic shift but more drift,
- anchor-only: may create target texture / symbolic overlay,
- decoupled reconstruction: should preserve more structure but may reduce edit strength.

---

### Task 2: Local Accessory Insertion

Example:

```text
source prompt: A panda is walking in a forest.
target prompt: A panda wearing sunglasses over its eyes is walking in a forest.
```

Purpose:

- test local editing,
- evaluate mask quality,
- evaluate trajectory preservation.

Expected observations:

- broad mask: stronger edit but more face drift,
- target-changed mask: more localized but may be weaker,
- external diagnostic mask: should show whether the dynamics can work when spatial support is good,
- trajectory preservation: should reduce unwanted drift.

---

## 8. Priority 1: Track the Right Statistics

Single final-step values are not enough.  
The method is an ODE process, so the statistics should summarize the whole trajectory.

### Add or verify these trajectory-level metrics

```text
avg_rec_energy
max_rec_energy
avg_rec_norm
max_rec_norm

avg_edit_norm
max_edit_norm

avg_cos_rec_base
avg_cos_rec_edit_total
avg_cos_base_anchor
avg_cos_base_region
avg_cos_anchor_region

avg_mask_area
final_mask_area

avg_beta_t
max_beta_t
```

### Why these matter

- `avg_rec_energy`: whether reconstruction is active across the trajectory.
- `avg_rec_norm`: strength of reconstruction correction.
- `avg_edit_norm`: strength of editing correction.
- `cos_rec_edit_total`: whether reconstruction and editing fight each other.
- `cos_base_anchor`: whether anchor branch agrees with target-source flow.
- `avg_mask_area`: whether the edit mask is too broad.

---

## 9. Priority 1: Diagnose Overlay-Like Failure

The current object replacement results suggest a failure mode:

> The target signal appears visually, but the object is not truly rewritten. Instead, the target may appear as texture, symbol, or overlay on the source object.

This should be treated as a central research bottleneck.

### Hypothesis

Current editing surrogates may capture shallow target cues, such as:

- texture,
- color,
- contour,
- token-level attention,
- target-like visual hints,

but may not enforce object-level semantic replacement.

### Required experiments

Compare these editing fields:

1. target-source velocity difference:
   \[
   v_{\mathrm{tar}} - v_{\mathrm{src}}
   \]

2. clean-space anchor:
   \[
   u_{\mathrm{anchor}} = -\frac{\hat{x}_0^{tar}-\hat{x}_0^{src}}{t}
   \]

3. edit-region-only anchor

4. feature-level attraction / suppression

5. CLIP or text reward guidance

For each, report:

- whether the target appears,
- whether it is realistic,
- whether it looks like overlay,
- whether source structure is preserved.

---

## 10. Priority 1: Clarify Energy vs Surrogate Velocity

The current implementation contains both energy functions and velocity surrogate functions.

This is acceptable, but the method should be described accurately.

### Use this wording

> The current implementation uses energy-inspired surrogate velocity fields. Some branches can be interpreted as gradients of an energy, while others are manually constructed velocity corrections derived from clean-space displacement or feature-space differences.

### Avoid claiming

\[
u_{\mathrm{edit}} = \nabla E_{\mathrm{edit}}
\]

for all branches, unless the code actually uses autograd to compute the gradient.

### Suggested notation

Use:

\[
u_{\mathrm{edit}} = \mathcal{G}_{\mathrm{edit}}(x_t,t)
\]

and optionally write:

\[
\mathcal{G}_{\mathrm{edit}}
=
\sum_i \lambda_i \mathcal{G}_i(x_t,t).
\]

This is more faithful to the current implementation.

---

## 11. Priority 2: Create a Small Experiment Table

For the first internal report, create a table like this.

| Task | Method | Edit Strength | Faithfulness | Drift | Notes |
|---|---|---:|---:|---:|---|
| panda → tiger | base only | low | high | low | source preserved |
| panda → tiger | direct target | high | low | high | semantic shift, drift |
| panda → tiger | anchor only | medium | medium | medium | possible overlay |
| panda → tiger | decoupled rec | medium | higher | lower | better trade-off |
| sunglasses | direct target | high | low | high | face changes |
| sunglasses | external mask + preserve | high | high | low | best local result so far |

This does not need to be a final quantitative result yet.  
It is for organizing the current observations.

---

## 12. Priority 2: Add Visual Comparison Panels

For each fixed task, generate one comparison image.

### Object replacement panel

```text
source | base only | direct target | anchor only | decoupled rec
```

### Local insertion panel

```text
source | direct target | attention mask | external mask | external mask + trajectory preserve
```

Each panel should include the prompt and method names.

These panels will be useful for:

- group meeting,
- thesis notes,
- future paper figures,
- debugging.

---

## 13. Priority 2: Make Mask Evaluation Explicit

For local editing, mask quality is a major factor.

Add mask comparison outputs:

```text
source_subject_mask.png
target_changed_mask.png
combined_mask.png
external_mask.png
final_used_mask.png
```

Also save:

```text
mask_area_ratio
mask_center
mask_bbox
```

This is important because if local editing succeeds only with a good mask, the method should say so honestly.

---

## 14. Priority 2: Define What Counts as Success

### For object replacement

A successful edit should:

- change the object identity,
- avoid pure texture overlay,
- preserve background and global layout,
- avoid unrealistic hybrid objects.

### For local accessory insertion

A successful edit should:

- insert the accessory in the correct region,
- preserve the source identity,
- preserve background,
- avoid full-face regeneration.

This should be written in the project notes before more experiments are run.

---

## 15. Priority 3: Optional Exact-Gradient Branch

If time allows, add one exact-gradient branch for comparison.

Example:

\[
E_{\mathrm{edit}}
=
R_{\mathrm{text}}(\hat{x}_0,c^{edit})
-
\lambda_{\mathrm{src}} R_{\mathrm{text}}(\hat{x}_0,c^{src})
\]

For a branch implemented as a true autograd energy gradient, compute:

\[
u_{\mathrm{edit}} = \nabla_{x_t} E_{\mathrm{edit}}.
\]

This is not the default assumption for every branch. It may be slower, but it
helps answer an important theoretical question:

> How different are true energy gradients from the current hand-designed velocity surrogates?

This is not required immediately, but it would strengthen the method discussion.

---

## 16. Priority 3: Avoid Adding More Guidance Before Cleaning Current Results

Do not add more branches until the current ones are understood.

Before adding new terms, answer:

1. Does the branch agree or conflict with \(v_{\mathrm{tar}}-v_{\mathrm{src}}\)?
2. Does it improve semantic edit or only add texture?
3. Does it preserve non-edited regions?
4. Does it require a precise mask?
5. Does it create overlay-like artifacts?

Adding more terms without these answers will make the method harder to interpret.

---

## 17. Immediate Checklist for Codex

### Current experiment readout

Latest generated panels:

```text
outputs/panda_tiger_ablation_grid.png
outputs/sunglasses_current_compare_grid.png
outputs/panda_tiger_diagnostics_grid.png
outputs/panda_tiger_rewrite_probe_grid.png
outputs/sunglasses_cross_method_grid.png
outputs/sunglasses_cross_method_mask_grid.png
outputs/sunglasses_cross_method_hint_grid.png
```

Panda to tiger ablation:

- `base_only` preserves the source layout but produces a softened panda.
- `direct_target` gives the strongest tiger face signal, but the body/layout
  remains panda-like and the result is a hybrid rather than a full object
  replacement.
- `anchor_only` behaves similarly to direct target; it confirms that the
  clean-space anchor is a strong target cue but can still act like an overlay.
- `decoupled_rec` activates the reconstruction branch and better preserves
  structure/background, but it does not solve the overlay-like replacement
  failure.

Panda to tiger branch diagnostics:

- `flow_only` and `clean_anchor` are both strong enough to create a tiger face,
  but neither rewrites the full panda body.
- `region_anchor` weakens the edit and still keeps the same hybrid structure,
  so simply masking the anchor is not enough for object-level replacement.
- `flow_plus_traj` preserves the source body even more strongly, which is good
  for faithfulness but works against full replacement.
- `target_feature` and `source_suppress` are much weaker than flow/anchor; they
  do not create a realistic tiger replacement by themselves.
- Current evidence points to an object-level rewrite bottleneck, not just a
  reconstruction-vs-edit strength issue.

Panda to tiger rewrite probes:

- mild initial subject noise gives slightly more freedom but remains a hybrid;
- strong initial subject noise destroys coherence instead of producing a clean
  tiger;
- core-only noise leaves the same tiger-face-on-panda-body failure;
- subject noise plus clean anchor strengthens tiger-face cues but still does
  not rewrite the full body.
- Therefore the next algorithmic change should be an object-level replacement
  mechanism, not just more noise or stronger target guidance in the same mask.

Sunglasses ablation:

- the current best result is still the target-changed attention local run:
  `outputs/sunglasses_local/result.png`;
- external-mask diagnostics confirm that spatial support matters, but the
  current external proposal mask is not reliable enough to beat attention local;
- simple eye-box clipping protects the face but pushes the generated glasses
  upward, so box clipping is not the right default.
- cross-method comparison shows RF-Solver and FireFlow place smaller glasses
  closer to the original eye positions with much smaller proposal-source
  difference masks than our broad attention subject mask;
- core-only ODE support is too narrow and fails to generate glasses, while
  subject-generation plus core final blending creates patch artifacts. The next
  local-edit idea should therefore separate contextual generation support from
  precise structural/attention injection, instead of only shrinking `M_edit`.
- inspection of RF-Solver, FireFlow, ReFlex, and h-edit shows that their better
  local placement comes from source-reference feature/attention reuse, not just
  from a smaller mask. RF-Solver reuses source `V`, FireFlow can reuse/add
  `Q/K/V`, ReFlex combines source latent blending with CA/SA/feature injection,
  and h-edit commonly relies on P2P/MasaCtrl/PnP controls around the h-edit
  update.
- a coarse RF-compatible approximation, subject-ring source trajectory
  blending, protects the panda face but suppresses or fragments sunglasses.
  This means the next useful change should be SD3 transformer-level
  source-reference attention/feature injection for the target velocity
  evaluation, while preserving the current total dynamics
  `xdot_t = v_src + u_rec + u_edit`.

Please implement or verify the following:

### Documentation

- [x] Update README formula to:
  \[
  \dot{x}_t = (v_{\mathrm{src}} + u_{\mathrm{rec}}) + u_{\mathrm{edit}}
  \]
- [x] Clarify energy-inspired surrogate velocity terminology.
- [x] Add note that exact energy-gradient guidance is not yet the main implementation.

### Scripts

- [x] Ensure baseline scripts save `command.txt`.
- [x] Ensure all scripts save `metadata.json`.
- [x] Add local sunglasses scripts.
- [x] Add visual comparison script.

### Statistics

- [x] Add average and max trajectory statistics.
- [x] Add cosine similarities between branches.
- [x] Add mask statistics.

### Experiments

- [x] Run panda → tiger ablation.
- [x] Run panda + sunglasses attention-mask ablation.
- [x] Run panda + sunglasses external-mask diagnostic.
- [x] Generate comparison panels.

### Analysis

- [x] Summarize whether each branch causes drift, overlay, or realistic replacement.
- [x] Identify which branch is most useful for object replacement.
- [x] Identify which branch is most useful for local insertion.
- [x] Compare RF-Solver, FireFlow, ReFlex, FlowEdit, and h-edit mechanisms.

### Next Algorithmic Work

- [x] Add an experimental SD3 source-reference attention processor.
- [x] Start with FireFlow-style `V` replacement/addition for selected
      transformer layers and early edit steps.
- [x] Evaluate `v_tar` inside the existing RF h-edit ODE under this injected
      processor, rather than changing the ODE formula.
- [x] Keep the two-mask split: broad generation support for `u_edit`, precise
      local mask only for injection/blending diagnostics.

---

## 18. One-Sentence Next Step

The next step is to keep the RF h-edit dynamics fixed and add SD3
source-reference attention/feature injection, because mask and latent-trajectory
controls alone have now hit the quality/placement trade-off on sunglasses.

---

## 19. Submission Readiness TODO

Status as of 2026-05-08:

- Workshop / prototype presentation: close.
- Workshop short paper: possible after a small fixed matrix and metric pass.
- Full conference submission: not ready yet.

The current blocker is not lack of code. The blocker is that the evidence is
still exploratory rather than paper-table reproducible.

Current shortest path:

- Stop treating the project as an open-ended case pile.
- Freeze 3-4 representative tasks.
- Freeze 5 method variants.
- Run 3 seeds for each task/method.
- Generate automatic metric tables.
- Start the paper outline in parallel so the claim stays narrow.

One-line assessment:

> The project already has research motivation and a method shape, but it is
> not yet a submission-ready paper; the remaining gap is systematic
> experiments, baselines, metrics, ablations, and paper packaging.

### P0: Freeze the paper claim

- [x] Define the main claim in one sentence:
      decoupled RF/ODE editing improves the edit-success vs source-faithfulness
      trade-off over direct target guidance.
- [x] Define what is in the main method:
      `xdot_t = v_src + u_rec + u_edit`.
- [x] Define what is a utility, not the core method:
      mask providers, reference recolor builders, external proposal masks.
- [ ] Decide the first submission target:
      workshop short paper vs full conference paper.

### P0: Build the fixed experiment matrix

Use `experiments/main_matrix.md` as the source of truth.

- [x] Create an image manifest that separates `main`, `supplement_or_replace`,
      and `debug_only` images:
      `experiments/image_manifest.md`.
- [x] Download initial Wikimedia candidates to `data/paper_images/`.
- [x] Replace debug-only bus color edit with a clean licensed clothing/bag/car
      color image before paper-table runs.
- [ ] Replace or approve the human portrait candidate before using it as a main
      local accessory figure.
- [x] T1 local accessory / headwear insertion:
      cat -> cat with crown.
- [x] T2 color / attribute edit:
      backpack -> blue backpack.
- [x] T3 color / attribute edit:
      yellow car -> blue car.
- [x] T4 known failure / robustness:
      rabbit side-profile sunglasses.

For each task, run:

- [x] M0 base only.
- [x] M1 direct target.
- [x] M2 anchor only.
- [x] M3 decoupled reconstruction.
- [x] M4 full method.

Minimum seeds:

- [x] `10`
- [x] `11`
- [x] `12`

Each run must save:

```text
result.png
stats.json
metadata.json
command.txt
masks/
```

### P0: Add paper-grade metrics

Add a separate evaluation script rather than relying only on internal
trajectory stats.

- [x] Edit success:
      CLIP text-image score or VLM-based target check.
- [x] Source preservation:
      LPIPS / DINO / SSIM against source.
- [x] Mask-outside drift:
      outside-mask L1 or LPIPS.
- [x] Region quality:
      mask-inside color/attribute score where applicable.
- [x] Runtime and peak GPU memory.
- [x] Failure flag:
      overlay, semantic miss, background drift, mask artifact.

Expected script:

```text
scripts/evaluate_paper_metrics.py
```

Expected output:

```text
experiments/main_metrics.csv
experiments/main_metrics.json
```

### P0: Make paper runs reproducible

Add one command per paper table or figure.

- [x] `scripts/run_main_table.sh`
- [x] `scripts/run_ablation_table.sh`
- [x] `scripts/make_paper_figures.sh`
- [x] `environment.yml` or `requirements.txt`
- [x] Re-run `scripts/audit_experiment_records.py` after each batch.

Current full-output audit snapshot:

```text
outputs/main_matrix: 82 runs
complete: 82
incomplete: 0
```

Current main-matrix audit snapshot:

```text
outputs/main_matrix planned paper subset:
planned: 60
complete: 60
missing_or_incomplete: 0
```

Important: the first 16-run seed-10 grid was produced mostly with the generic
`attention_velocity` provider. Treat it as baseline evidence, not the final
SAM-supported main method.

Only complete runs should be used in tables or figures.

Missing reproducibility pieces before paper submission:

- [x] One-command ablation table runner.
- [x] One-command figure/table generation.
- [x] Environment lock file:
      `environment.yml` or `requirements.txt`.
- [x] Metric evaluator connected to `outputs/main_matrix`.

### P1: Establish baseline parity

External baselines should be compared only under matching conditions.

- [x] Choose baselines:
      FlowEdit, RF-Solver, FireFlow, h-Edit, ReFlex if runnable.
- [x] Match input image, prompt, resolution, seed, and mask condition where
      possible.
- [x] Do not compare against baselines under weaker prompt, resolution, seed,
      or mask settings; reviewers can reasonably reject that as unfair.
- [x] Record non-runnable baselines as limitations with exact failure reason
      instead of using unfair partial comparisons.
- [ ] Save baseline commands and outputs in the same record format.

### P1: Prepare manuscript files

The `paper/` directory exists, but the manuscript package is not yet active
enough for real academic-pipeline writing/revision stages.

- [x] `paper/outline.md`
- [x] `paper/main.tex` or `paper/draft.md`
- [x] `paper/references.bib`
- [x] `paper/figures.md`
- [x] `paper/tables.md`
- [x] `paper/limitations.md`

Draft structure:

- [x] Introduction: RF/ODE editing needs explicit preservation/edit decoupling.
- [x] Related work: h-Edit, FlowEdit, RF-Solver, FireFlow, ReFlex, PnP/P2P-style
      source feature reuse.
- [x] Method: linear path clean estimate, reconstruction correction, editing
      correction, mask/support interface.
- [x] Experiments: fixed matrix, metrics, baselines, ablations.
- [x] Failure analysis: object replacement overlay and complex reflective color
      surfaces.
- [x] Limitations: dependence on spatial support, SD3-specific implementation,
      baseline resource constraints.

### P1: Add ablations that support the claim

- [x] No reconstruction correction.
- [x] No trajectory preserve.
- [x] Broad mask vs changed-token mask.
- [ ] Attention mask vs external diagnostic mask.
- [ ] Reference guidance off/on for color tasks.
- [ ] `EDIT_REF_CHROMA_MODE=yuv` vs `yuv_direction` only as a surface-color
      ablation, not as the main method.

### P2: Clean outputs for paper use

- [x] Archive or ignore incomplete exploratory folders.
- [x] Regenerate selected qualitative figures from complete runs.
- [x] Ensure every figure panel maps to a run directory and command.
- [x] Keep failure cases visible; do not hide rabbit / object-replacement
      failure.

### Immediate next executable step

Re-run the first small paper subset with the SAM-supported default:

```text
TASKS="T1 T2 T4" METHODS="M0 M1 M3 M4" SEEDS="10" DEVICE=<gpu> scripts/run_main_table.sh
```

Then generate a comparison panel from `outputs/main_matrix` and decide which
tasks need task-specific mask/reference setup before expanding to seeds 11/12.

Suggested converged task set for the next paper pass:

- `crown/headwear`: local accessory insertion with SAM support.
- `bus_color` or replacement publication-safe color task:
  clean object color edit with surface/reference route.
- `sunglasses`: local accessory insertion, if support can be made reliable.
- `rabbit_failure`: explicit robustness/failure case, unless profile-eye
  support makes it reliable enough to move into the main table.

### Immediate support fixes

- [x] Add `top_center` support relation in `scripts/make_semantic_mask.py` for
      crown/hat/headwear tasks.
- [x] Re-run `cat_crown/full` with `top_center`; current `above` support can
      overlap cat ears because the SAM head bbox includes ear tips.
- [x] Add surface-reference mode to `scripts/run_main_table.sh` for color tasks:
      `backpack_blue`, `red_chair_blue`, `yellow_car_blue`.
- [x] Keep `attention_velocity` as a generic baseline method/provider, not as
      the paper main local-support result.
