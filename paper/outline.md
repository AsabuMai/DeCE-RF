# Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing

Working title:

```text
DeCE-RF: Decoupled Clean-Estimate Edit-Preserve Control for Localized Rectified Flow Editing
```

Method name:

```text
DeCE-RF
```

Expanded form:

```text
Decoupled Clean-Estimate Edit-Preserve Control for Rectified Flow
```

## Claim

Localized RF/ODE image editing should be formulated as decoupled
clean-estimate displacement control rather than as direct target-velocity
replacement. DeCE-RF keeps the source-conditioned trajectory and maps a clean
displacement back to RF velocity:

```text
v_DeCE = v_src - t^-1 Delta_0
Delta_0 = Delta_edit + Delta_pres
```

The displacement components enforce target clean-estimate matching in edit
regions and source latent reconstruction in preserve regions:

```text
Delta_edit = lambda_e(t) M_e * (x0_hat_tar - x0_hat_src)
Delta_pres = lambda_p(t) M_p * (x_s - x0_hat_src)
```

The implementation uses closed-form correction branches:

```text
-t^-1 Delta_edit maps to the target-source velocity direction
-t^-1 Delta_pres maps to the preserve velocity direction
```

The method story has three linked components:

```text
1. Decompose the desired clean displacement into edit and preserve components.
2. Estimate the displacement geometry through an operation-conditioned support
   interface.
3. Update edit and preserve displacement weights with clean-estimate feedback.
```

## Method Scope

- Core method: per-step decoupled clean-estimate displacement control over a
  source-conditioned RF base trajectory.
- Support interface: operation-conditioned fusion of token attention,
  clean/velocity response, optional grounding, and relation/surface priors into
  `M_edit`, `M_core`, `M_contact`, and `M_preserve`.
- Utilities: surface recolor references, external diagnostic masks, and
  source-reference Q/K/V injection.
- Non-goal: adding unrelated guidance terms before the fixed matrix is
  evaluated.

## Draft Structure

1. Introduction: RF/ODE editing needs clean-estimate edit-preserve control.
2. Related Work: h-Edit, FlowEdit, RF-Solver, FireFlow, ReFlex, PnP/P2P-style
   source feature reuse.
3. Method: local RF editing as clean-estimate displacement control,
   edit-preserve displacement components, operation-conditioned control
   geometry, feedback-updated displacement weights, algorithm and
   instantiations.
4. Experiments: fixed matrix, metrics, baselines, ablations.
5. Failure Analysis: object replacement overlays and hard local-support cases.
6. Limitations: spatial support dependence, SD3 specificity, and baseline
   resource constraints.

## Experiment Plan

The current server-run plan is tracked in:

```text
paper/wacv_experiment_design.md
```
