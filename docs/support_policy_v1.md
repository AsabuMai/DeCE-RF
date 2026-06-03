# Support Policy V1

This document freezes the automatic support policy used for the next RF local
editing experiments. The purpose is to keep support estimation generic enough to
defend: rules are tied to edit semantics, not to a specific image, case id, or
observed result.

## Contract

The support estimator may use only:

- source image
- source prompt
- target prompt
- parsed edit operation
- host object tokens
- old/source object tokens
- new/target object tokens
- open-set grounding, SAM masks, text attention maps, and RF disagreement maps

The support estimator must not use:

- task id or image filename
- hand-drawn masks
- per-image thresholds chosen after inspecting results
- case-specific phrases that are not recoverable from the prompts
- different routing rules for dev and held-out images

## Scope Update

As of 2026-05-12, pure recolor tasks are retired from the main rmsgap
evaluation set. They remain supported by the code for diagnostics, but should
not be used as main evidence because deterministic masked recolor is a strong
specialized baseline and the task does not stress the same target-formation
mechanism as local insertion/replacement.

## Operation Routing

| Operation | Allowed semantic role | Support source |
| --- | --- | --- |
| `recolor` | diagnostic only, excluded from main evaluation | grounded existing object/surface mask, then RF disagreement within it |
| `remove_object` | object present in source | grounded source object mask, then RF disagreement within it |
| `replace` | old/source object or attached object | grounded old object mask; if unavailable, host attachment region |
| `add_decal` | logo, symbol, text, printed mark on a host | grounded host visible surface, not the new token location |
| `add_object` with `above_host` | crown or object above a host | area above host/head with small overlap |
| `add_object` with `on_face` | glasses or face accessory | face/eye-band region derived from grounded head/face/host mask |
| `add_object` with `inside_host`/`inside` | object inside a host region | grounded host region |

## Fixed Task Mapping for the Active 11-Case Dev Set

| Task | Operation | Host tokens | Old/source tokens | New tokens | Relation |
| --- | --- | --- | --- | --- | --- |
| `cat_crown` | `add_object` | `cat,head` | - | `crown` | `above_host` |
| `dog_sunglasses` | `add_object` | `dog,eyes` | - | `sunglasses` | `on_face` |
| `mug_heart` | `add_decal` | `mug` | - | `heart` | `on_surface` |
| `tshirt_star` | `add_decal` | `t-shirt,shirt` | - | `star` | `on_surface` |
| `tote_leaf` | `add_decal` | `tote,bag` | - | `leaf,logo` | `on_surface` |
| `backpack_remove_toy_charm` | `remove_object` | `backpack` | `toy,charm` | - | `remove_source_object` |
| `backpack_replace_patch_blue` | `replace` | `backpack` | `patch` | `blue,patch` | `remove_source_object` |
| `cat_replace_bell_heart_tag` | `replace` | `cat,collar` | `bell` | `heart,tag` | `remove_source_object` |
| `dog_replace_tennis_ball_star` | `replace` | `dog,mouth` | `tennis,ball` | `star,toy` | `remove_source_object` |
| `rabbit_sunglasses` | `add_object` | `rabbit,eyes` | - | `sunglasses` | `on_face` |
| `dog_crown` | `add_object` | `dog,head` | - | `crown` | `above_host` |

Retired diagnostic task, excluded from default runs:

| Task | Operation | Host tokens | Old/source tokens | New tokens | Relation |
| --- | --- | --- | --- | --- | --- |
| `red_chair_blue` | `recolor` | `chair` | `chair` | `blue` | `inside` |

## Audit Gates

Support-only previews must be inspected before editing runs. A support mask passes
only if:

- it overlaps the plausible edit location;
- it does not mainly select a distractor object;
- for decals, it lies on the host surface rather than on the new object token;
- for face accessories, it is smaller than the full animal/person body and covers
  the plausible face/eye band;
- for removal/replacement, it covers the source object being removed/replaced.

If a dev-set support mask fails, the rule may be changed only at the operation
level and must be recorded here. Held-out images must not cause new case-specific
rules.

## Claim Boundary

This policy is not a claim that support estimation is solved for arbitrary edits.
The defensible claim is narrower:

> A fixed operation-conditioned support policy provides automatic spatial support
> for evaluating RF clean-estimate controllers, without per-image mask tuning.
