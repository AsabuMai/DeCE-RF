# Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing

Working title:

```text
Decoupled Clean-Estimate Control for Localized Rectified Flow Image Editing
```

Method name:

```text
DeCE-RF
```

Expanded form:

```text
Decoupled Clean-Estimate Control for Rectified Flow
```

## Claim

Decoupled RF/ODE image editing improves the edit-success versus
source-faithfulness trade-off over direct target guidance by using:

```text
xdot_t = v_src + u_rec + u_edit
```

The method story has three linked components:

```text
1. Decouple edit and reconstruction velocities.
2. Localize those velocities through an operation-conditioned support
   interface.
3. Adapt their strengths with clean-estimate feedback.
```

## Method Scope

- Core method: source-conditioned RF base velocity, reconstruction correction,
  and editing correction.
- Support interface: operation-conditioned fusion of token attention,
  clean/velocity response, optional grounding, and relation/surface priors into
  `M_edit`, `M_core`, and `M_preserve`.
- Utilities: surface recolor references, external diagnostic masks, and
  source-reference Q/K/V injection.
- Non-goal: adding unrelated guidance terms before the fixed matrix is
  evaluated.

## Draft Structure

1. Introduction: RF/ODE editing needs explicit preservation/edit decoupling.
2. Related Work: h-Edit, FlowEdit, RF-Solver, FireFlow, ReFlex, PnP/P2P-style
   source feature reuse.
3. Method: linear path clean estimate, edit/reconstruction velocity
   decoupling, operation-conditioned support interface, clean-estimate feedback
   controller.
4. Experiments: fixed matrix, metrics, baselines, ablations.
5. Failure Analysis: object replacement overlays and hard local-support cases.
6. Limitations: spatial support dependence, SD3 specificity, and baseline
   resource constraints.

## Experiment Plan

The current server-run plan is tracked in:

```text
paper/experiment_plan.md
```
