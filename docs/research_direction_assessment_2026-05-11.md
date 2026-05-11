# Research Direction Assessment

Date: 2026-05-11

## Verdict

The current RF h-Edit direction can support a paper direction, but it should not
be framed as a broad, general-purpose RF image editing method yet.

The safer research framing is:

```text
support-aware clean-estimate control for Rectified Flow image editing
```

or, more specifically:

```text
clean-estimate-space support diagnosis and adaptive local control for
Rectified Flow image editing
```

The direction should focus on support reliability, local clean-estimate-space
diagnostics, and the edit/preserve control mechanism. It should not claim
state-of-the-art general editing until the automatic support proposer is much
stronger.

## Why the Broad Framing Is Risky

The broad formulation:

```text
xdot = v_src + u_rec + u_edit
```

is useful as the implementation-level decomposition, but it is not enough as
the main novelty. Recent and nearby work already covers many adjacent ideas:

- FlowEdit: inversion-free ODE editing with pretrained flow models.
- SplitFlow: flow decomposition and aggregation for text-to-image editing.
- FlowAlign: trajectory regularization for source consistency.
- SteerFlow: source-target velocity blending, trajectory anchoring, and
  adaptive masking.
- UniEdit-Flow: flow inversion/editing with region-aware preservation.

Therefore, a paper should not be positioned as:

```text
we split RF velocity into reconstruction and editing terms
```

The more defensible novelty is:

```text
we diagnose edit progress and preservation drift in predicted clean-image space,
then use that diagnosis to control local RF velocity corrections.
```

## Recommended Paper Slices

### A. Support Reliability Paper

Recommended as the safest near-term direction.

Possible title:

```text
Support Reliability in Rectified-Flow Image Editing:
Clean-Estimate Diagnostics for Local Edit Control
```

Core question:

```text
When is a spatial support region reliable for RF local editing, and how does
support error affect the edit/preservation tradeoff?
```

Main contributions:

- Use changed-token attention, clean-estimate disagreement, and velocity
  disagreement as support diagnostics.
- Compare support candidates such as:
  - attention only
  - clean disagreement only
  - velocity disagreement only
  - attention x clean
  - attention x velocity
- Show that the adaptive controller works when support is reasonable, but
  fails gracefully or preserves too much when support is wrong.
- Provide a support failure taxonomy:
  - localized-token success: `dog_sunglasses`
  - relation / placement failure: `cat_crown`
  - surface / decal failure: `mug_heart`
  - source-object removal failure: `backpack_remove_toy_charm`

Status:

```text
This direction is already supported by the current generic-support experiments.
```

### B. Operation-Aware Generic Support Paper

Higher upside, but not ready yet.

Possible title:

```text
Operation-Aware Support Proposal for Training-Free Rectified-Flow Image Editing
```

Core question:

```text
Can a generic support proposer choose different evidence sources depending on
the edit operation?
```

Operation types:

- `add_object`: new-token attention plus host-object evidence.
- `add_decal`: new-token attention plus host surface support.
- `remove_object`: source removed-token evidence and/or segmentation.
- `replace` / `attribute`: source object support plus target attribute
  disagreement.

Current status:

```text
support_v2_minimal is wired, but seed-10 did not pass the gate.
```

Evidence from current metrics:

- `cat_crown`: generic v1 and support v2 both have IoU 0 against manual
  support.
- `mug_heart`: generic v1 and support v2 both have IoU 0 against manual
  support.
- `backpack_remove_toy_charm`: IoU is nonzero, but coverage remains too low and
  support v2 does not improve over v1.
- `dog_sunglasses`: remains the strongest generic-support case.

This direction needs a stronger support-v3 before it can be the main paper.

### C. Clean-Estimate Adaptive Controller Paper

Theoretically attractive, but novelty risk is higher because nearby work also
targets trajectory/source consistency.

Possible title:

```text
Region-Adaptive Clean-Estimate Control for Rectified-Flow Image Editing
```

Core question:

```text
Does measuring edit progress and preservation drift in predicted clean-image
space yield better local RF editing control than fixed guidance or velocity
blending?
```

This direction needs stronger evidence that the adaptive controller itself,
not manually engineered support, causes the improvement.

## Current Best Framing

The most defensible near-term paper story is:

```text
RF local editing is bottlenecked by support quality. We propose a
clean-estimate-space diagnostic/control framework that separates support
proposal from local adaptive control. The controller preserves source content
when support is reliable, while the diagnostics expose relation, decal, and
removal failure modes that require operation-aware support.
```

This is a stronger and more honest story than claiming a finished general
editing method.

## Next Required Experiment

Do not run a large matrix immediately. Build a small `support_v3_gate`.

Tasks:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

Seed:

```text
10 only for the first gate
```

Compare:

```text
manual support
generic v1
support v2 minimal
support v3 candidate
```

Gate criteria:

- At least two of four tasks should visibly improve over generic v1.
- `dog_sunglasses` must not regress.
- `cat_crown` or `mug_heart` should move from clear failure to a displayable
  result.
- `backpack_remove_toy_charm` should improve coverage and reduce residual
  object artifacts.
- Preserve drift should not become obviously worse.
- The method should avoid task-specific hard-coded boxes such as `decal_box`,
  `front_glasses_auto`, or crown-specific top-of-head templates.

## Practical Decision

Proceed with the paper direction, but keep the claim narrow:

```text
support-aware clean-estimate control for RF image editing
```

Do not claim:

```text
general automatic RF editing or state-of-the-art editing across tasks
```

until support-v3 passes the gate and the result is stable across seeds.
