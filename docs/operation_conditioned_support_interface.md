# Operation-Conditioned Support Interface

This note reframes `support-v3` as a paper-facing method component rather than
an engineering label.

## Position

The support module should be described as an operation-conditioned spatial
interface between text-conditioned RF dynamics and local velocity control.

It is not the central claim of the paper. The central claim is decoupled
edit/reconstruction control in clean-estimate space. The support interface
defines where the decoupled corrections are allowed to act.

## Interface

Given a source image, source prompt, target prompt, current latent state, and an
operation description, the support interface returns:

```text
M_edit
M_core
M_contact
M_preserve
```

These masks are consumed by the ODE controller:

```text
v_total = v_src + u_rec(M_preserve) + u_edit(M_edit, M_core, M_contact)
```

The support module therefore supplies control geometry, not an edit by itself.

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
weak/generic support -> decoupled controller
operation-conditioned support -> fixed decoupled controller
operation-conditioned support -> adaptive RMSGAP controller
manual/external support -> upper-bound diagnostic
```

This supports the paper story:

```text
1. decouple edit and reconstruction;
2. localize the decoupled corrections;
3. adapt their strengths from clean-estimate feedback.
```

Failures should be reported when operation-conditioned support is wrong or when
the target operation is not handled well, especially removal. These failures
define the boundary of the method rather than invalidating the control claim.
