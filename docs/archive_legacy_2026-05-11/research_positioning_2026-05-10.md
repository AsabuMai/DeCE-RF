# Research Positioning: RF h-Edit Project

Date: 2026-05-10

## Bottom Line

The project is still meaningful as a research prototype, but not as a broad
"new RF image editing method" in its current form. The 2025-2026 literature has
already moved deeply into rectified-flow image editing, including inversion-free
editing, inversion-based faithful editing, trajectory interpolation, adaptive
masking, feature/attention injection, and proximal/geometric formulations.

The project should therefore stop claiming general RF editing novelty. Its
defensible direction is narrower:

> RF-native localized editing via clean-estimate velocity control and automatic
> local support/reference construction.

The strongest current project evidence is localized semantic editing:

- `cat_crown`: local accessory insertion.
- `dog_sunglasses`: accessory insertion with automatic eye support/reference.
- `mug_heart`: local decal edit.
- `backpack_remove_toy_charm`: localized semantic removal.

Pure recoloring and tiny side-profile accessory edits should remain limitation
cases, not success claims.

## Relevant Recent Literature

### Already crowded around our original claim

- FlowEdit, 2024: inversion-free text editing with pretrained flow models using
  an ODE source-to-target construction. This already occupies the broad
  "flow-model text editing" space.
- FlowChef, 2024/2025: vector-field steering for controlled generation, inverse
  problems, and image editing without extra training/inversion/backprop-heavy
  control.
- ReFlex, 2025: real-image editing in rectified flow via mid-step feature
  extraction and attention adaptation. It directly targets RF real-image editing
  and preservation/editability.
- InstantEdit, 2025: few-step text-guided editing with piecewise rectified flow,
  inversion latent injection, disentangled prompt guidance, and ControlNet
  structure cues.
- RK/DDTA, 2025: high-order inversion plus decoupled diffusion transformer
  attention for RF inversion and semantic editing.
- Delta Velocity Rectified Flow, 2025: inversion-free path-aware editing by
  modeling source-target velocity discrepancy.
- SplitFlow, 2025: flow decomposition and aggregation for inversion-free editing
  with semantic sub-prompt decomposition.
- SNR-Edit, 2026: inversion-free flow editing via structure-aware noise
  rectification and segmentation-constrained trajectory anchoring.
- SGPP, 2026: proximal/geometric RF editing framework balancing input fidelity
  and realism; RF inversion appears as a limiting case.
- SteerFlow, 2026: faithful inversion-based RF editing with amortized fixed-point
  inversion, trajectory interpolation between source reconstruction and target
  edit velocities, and adaptive masking using concept segmentation plus
  source-target velocity differences.

SteerFlow is the most direct threat to our current framing because it explicitly
combines:

- source fidelity as the main problem,
- source-reconstruction and target-editing velocity blending,
- adaptive masking,
- FLUX and SD3.5 evaluation,
- multi-turn drift control.

## What Is Still Potentially Novel Here

### 1. Clean-estimate-space velocity control

The project's most defensible technical core is the RF linear-path relation:

```text
x0_hat = x_t - t v_theta(x_t, t)
u = -delta_x0 / t
```

This turns clean-space local constraints into RF velocity corrections. That is a
clearer contribution than simply saying "we add reconstruction and editing
terms." The paper should center this as the implementation-level mechanism.

### 2. Localized edit-support interface

The project already has practical support providers:

- changed-token attention support,
- semantic/SAM support,
- semantic + velocity-difference refinement,
- structure-derived glasses masks,
- decal/reference masks,
- object/contact/preserve layer splitting.

This could become a useful contribution if presented as a unified support
interface rather than a collection of task-specific scripts.

### 3. Reference construction for local edits

The generated local references for glasses, decals, surface recolor, and badge
replacement are practically useful. To be publishable, this must be reframed as
an automatic local reference prior:

```text
prompt diff -> edit type -> anchor/support -> reference prior -> RF velocity correction
```

At present this is too heuristic, but it is the most distinctive engineering
piece in the repository.

## What Is Not Enough Anymore

- "RF h-Edit" alone is not enough. h-Edit already introduced the
  reconstruction/edit decomposition, and RF papers now cover many trajectory and
  velocity-control variants.
- Attention/SAM masks alone are not novel.
- Source Q/K/V injection alone is not novel; ReFlex and related RF/DiT editing
  methods cover feature/attention adaptation more directly.
- A four-image self-selected qualitative set is not enough evidence.
- Pure surface recolor is not a good main task; deterministic color transforms
  can outperform generative RF editing on that operation.

## Recommended Repositioning

Use this claim:

> We propose a clean-estimate-space control interface for localized rectified
> flow editing. The method maps local clean-image constraints and reference
> priors into RF velocity corrections, enabling source-faithful localized
> insertion, decal editing, and object removal.

Avoid this claim:

> We propose a general-purpose state-of-the-art RF image editing method.

## Minimum Work Needed Before a Paper

1. Turn the current scripts into one explicit algorithm:

```text
Edit instruction
  -> edit type / changed concept extraction
  -> anchor object and support mask construction
  -> optional local reference prior
  -> clean-estimate correction
  -> RF velocity update
```

2. Add matched baselines on the same images/prompts:

- FlowEdit
- FireFlow or RF-Solver/RF-Edit
- ReFlex if feasible
- SteerFlow if code is available
- one diffusion attention-control baseline such as Prompt-to-Prompt/MasaCtrl

3. Evaluate on a small benchmark subset rather than only custom examples:

- localized insertion,
- local decal/symbol edit,
- localized object removal,
- small object replacement.

GIE-Bench is especially relevant because it evaluates functional correctness
and preservation instead of relying only on CLIP similarity.

4. Make failure analysis part of the paper:

- recolor failure: deterministic color transforms are stronger for pure
  chroma-only edits;
- side-profile accessory failure: support geometry matters;
- replacement failure: reference prior and mask tightness dominate outcome.

5. Add ablations that map directly to the claimed modules. The current
   go/no-go matrix supports:

- `full_no_ref`: no local reference prior.
- `full_no_rec`: no reconstruction correction.
- `full_no_traj`: no trajectory/source preservation.
- `direct_target` vs `full`: no local support/reference interface versus full
  local support interface.

Future ablations that are still not implemented:

- no clean-estimate correction / non-clean-space velocity correction;
- attention support vs semantic/SAM support inside the pretty-matrix task set.

## Go / No-Go Decision

Go if the project becomes a localized-editing paper centered on
clean-estimate-space RF control and automatic support/reference construction.

No-go if the intended claim remains "general RF image editing," because the
latest literature already covers that space more strongly and with broader
benchmarks.
