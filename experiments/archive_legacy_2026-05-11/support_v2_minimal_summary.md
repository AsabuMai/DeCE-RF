# Support v2 Minimal Summary

Date: 2026-05-11

## Objective

Test the next support direction from `rf_h_edit_support_v2_direction.md` with a minimal operation-aware support candidate bank.

The implemented seed-10 test compares:

```text
manual_support
generic_support_v1
support_v2_minimal
```

on the four pretty tasks:

```text
cat_crown
dog_sunglasses
mug_heart
backpack_remove_toy_charm
```

## Implemented Candidates

The existing `generic_support.py` path now supports operation-aware candidates:

```text
host_x_clean
new_x_host_x_clean
new_plus_host_x_clean
removed_src_x_clean
removed_src_x_velocity
```

The command-line path now accepts:

```text
--support-candidate
--edit-operation
--new-tokens
--host-tokens
--removed-tokens
```

`scripts/run_pretty_matrix.sh` adds:

```text
support_v2_minimal
```

with the following task mapping:

| Task | Operation | Candidate |
|---|---|---|
| `cat_crown` | `add_object` | `new_plus_host_x_clean` |
| `dog_sunglasses` | `add_object` | `attention_x_clean` |
| `mug_heart` | `add_decal` | `new_x_host_x_clean` |
| `backpack_remove_toy_charm` | `remove_object` | `removed_src_x_clean` |

## Outputs

Seed-10 result comparison:

```text
outputs/pretty_matrix/support_v2_minimal_seed10_overview.png
```

Seed-10 mask comparison:

```text
outputs/pretty_matrix/support_v2_minimal_seed10_masks.png
```

Support metrics against manual support:

```text
experiments/support_v2_minimal_metrics_seed10.csv
```

## Readout

Support v2 minimal is wired and runs, but it does not yet solve the core failure cases.

- `cat_crown`: v2 still does not create a clear crown. The v2 mask remains too close to the v1 response region and does not recover the manual top-of-head support.
- `dog_sunglasses`: v2 is effectively the same as v1 because the selected candidate is still `attention_x_clean`; this remains the strongest generic case.
- `mug_heart`: v2 does not produce a heart. The host-aware candidate constrains the region, but the edit direction still lacks strong decal evidence.
- `backpack_remove_toy_charm`: v2 does not remove the charm. `removed_src_x_clean` is not enough by itself.

The manual-support IoU metrics confirm that v2 minimal does not improve support alignment over v1 in the failed tasks.

## Decision

The seed-10 gate did not pass. Therefore, the planned three-seed expansion was not run for `support_v2_minimal`.

The next support improvement should add stronger evidence, not only candidate multiplication:

- segmentation-assisted support (`seg_x_clean`) for removal and host surface localization.
- proposal / layout support for relation edits such as `cat_crown`.
- decal/reference evidence for symbol insertion such as `mug_heart`.
- candidate scoring using clean-estimate edit progress and preserve drift.
