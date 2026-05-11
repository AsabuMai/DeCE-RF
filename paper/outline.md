# RF h-Edit Paper Outline

## Claim

Decoupled RF/ODE image editing improves the edit-success versus
source-faithfulness trade-off over direct target guidance by using:

```text
xdot_t = v_src + u_rec + u_edit
```

## Method Scope

- Core method: source-conditioned RF base velocity, reconstruction correction,
  and editing correction.
- Utilities: mask providers, surface recolor references, external diagnostic
  masks, and source-reference Q/K/V injection.
- Non-goal: adding unrelated guidance terms before the fixed matrix is
  evaluated.

## Draft Structure

1. Introduction: RF/ODE editing needs explicit preservation/edit decoupling.
2. Related Work: h-Edit, FlowEdit, RF-Solver, FireFlow, ReFlex, PnP/P2P-style
   source feature reuse.
3. Method: linear path clean estimate, reconstruction correction, editing
   correction, mask/support interface.
4. Experiments: fixed matrix, metrics, baselines, ablations.
5. Failure Analysis: object replacement overlays and hard local-support cases.
6. Limitations: spatial support dependence, SD3 specificity, and baseline
   resource constraints.
