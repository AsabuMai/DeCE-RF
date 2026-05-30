# Operation-Conditioned Support Interface

This note reframes `support-v3` as a paper-facing method component rather than
an engineering label.

## Position

The support module should be described as an operation-conditioned control
geometry estimator between text-conditioned RF dynamics and the local
decoupled clean-estimate edit-preserve displacement.

It is not the central claim of the paper. The central claim is per-step
decoupled clean-estimate edit-preserve control over a source-conditioned RF
trajectory. The support interface estimates where the two displacement
components should act: edit displacement in target regions and preservation
displacement in source-preserve regions.

## Interface

Given a source image, source prompt, target prompt, current latent state, and an
operation description, the support interface returns:

```text
M_edit
M_core
M_contact
M_preserve
```

These masks are consumed by the ODE controller as displacement geometry:

```text
Delta_0 = Delta_edit(M_edit, M_core, M_contact)
        + Delta_pres(M_preserve)
v_DeCE = v_src - t^-1 Delta_0
```

The support module therefore supplies the geometry of the clean displacement,
not an edit by itself.

## Operation Conditions

The task is reduced to operation-level semantics:

```text
add_object
add_decal
remove_object
replace
recolor
```

Relations specify spatial priors without hard-coding object names:

```text
above_host
on_face
on_surface
remove_source_object
inside
```

Avoid paper language such as "the sunglasses rule" or "the crown heuristic."
Use "face relation", "above-host relation", or "surface relation" instead.

## Evidence Maps

The support interface fuses three classes of evidence.

Text/semantic evidence:

```text
A_new
A_host
A_removed
optional grounded or segmented mask
```

RF response evidence:

```text
D_clean = ||x0_hat_target - x0_hat_source||
D_vel   = ||v_target - v_source||
```

Operation prior:

```text
relation region
surface region
removed-object region
inside-host region
```

Candidate maps combine these evidence sources, for example:

```text
relation_x_response
decal_surface_local_response
removed_src_x_clean
relation_only
```

The operation determines the default candidate family. This is a constrained
selection problem, not a task-specific bag of rules.

## Paper Wording

Preferred:

```text
operation-conditioned support interface
support-aware velocity control
operation-level support proposal
control geometry
```

Avoid:

```text
support-v3 heuristic
task-specific mask rule
manual mask trick
```

`support-v3` can remain the implementation name, but the paper should introduce
the abstraction first and mention the code name only if needed for experiments.

## Experimental Role

The support interface should be evaluated as the localization component of the
control stack:

```text
weak/generic support -> fixed displacement weights
operation-conditioned support -> fixed displacement weights
operation-conditioned support -> feedback-updated DeCE-RF controller
manual/external support -> upper-bound diagnostic
```

This supports the paper story:

```text
1. formulate local RF editing as decoupled clean-estimate displacement control;
2. estimate the displacement geometry from operation-conditioned evidence;
3. adapt edit and preserve displacement weights from clean-estimate feedback.
```

Failures should be reported when operation-conditioned support is wrong or when
the target operation is not handled well, especially removal. These failures
define the boundary of the method rather than invalidating the control claim.
