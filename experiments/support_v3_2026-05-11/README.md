# Support-v3 Experiment Pack 2026-05-11

This folder collects the first real support-v3 effect tests and baseline
comparison artifacts.

## Scope

Tasks:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

Seed:

```text
10
```

Compared methods:

```text
manual_support
generic_support_v1
operation_aware_support_v2
support_v3_grounded
```

## Files

Single-method v3 effect test:

```text
support_v3_effect_metrics.csv
support_v3_effect_metrics.json
support_v3_effect_metrics_clip.csv
support_v3_effect_metrics_clip.json
support_v3_visuals/
```

Four-method comparison:

```text
support_v3_compare_metrics_clip.csv
support_v3_compare_metrics_clip.json
support_v3_compare_visuals/
```

Raw run outputs remain under:

```text
outputs/pretty_matrix/<task>/<method>/seed_10/
```

Refinement/removal-controller pass:

```text
support_v3_refinement_metrics_clip.csv
support_v3_refinement_metrics_clip.json
mug_candidate_compare_metrics_clip.csv
mug_candidate_compare_metrics_clip.json
removal_controller_compare_metrics_clip.csv
removal_controller_compare_metrics_clip.json
mug_compare_outputs/
removal_compare_outputs/
```

## Readout

- `cat_crown`: v3 improves relation-aware insertion over generic v1/v2.
- `dog_sunglasses`: v3 does not regress the strong generic-support case.
- `mug_heart`: v3 fixes the surface/decal failure mode seen in v1/v2.
- `backpack_remove_toy_charm`: removal remains unsolved; support localization
  is better than generic support, but removal semantics are still weak.

Refinement readout:

- `cat_crown`: final mask area is now in the desired relation-support range.
- `dog_sunglasses`: remains a non-regression positive control.
- `mug_heart`: stronger preserve is the best tested default, but the
  outside-mask drift target is still not met.
- `backpack_remove_toy_charm`: optional clean-fill removal controller was
  implemented and tested, but the controller-off variant is better for the
  current weights, so removal controller is default-off.

Current conservative claim:

```text
Support-v3 is promising for operation-aware insertion and surface/decal edits.
Removal should be treated as a limitation or a separate follow-up direction.
```
