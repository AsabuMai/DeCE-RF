# Generic Support Experiment Summary

Date: 2026-05-11

## Scope

This round followed `docs/rf_h_edit_generic_support_todo.md`:

- Implement generic support from changed-token attention and clean / velocity disagreement.
- Integrate it with `adaptive_full_v1`.
- Run the four pretty tasks.
- Run seed-10 support ablations.
- Generate visual panels and diagnostic records.

## Completed Runs

Generic support (`attention_x_clean`) was run on all four tasks with seeds 10, 11, and 12:

- `cat_crown`
- `dog_sunglasses`
- `mug_heart`
- `backpack_remove_toy_charm`

Support ablations were run on seed 10 for all four tasks:

- `attention_only`
- `clean_disagreement_only`
- `velocity_disagreement_only`
- `attention_x_clean`
- `attention_x_velocity`

Metric / mask summary CSV:

- `experiments/generic_support_summary.csv`

Visual panels:

- `outputs/pretty_matrix/manual_vs_generic_support_seed10_overview.png`
- `outputs/pretty_matrix/generic_support_ablation_seed10_overview.png`
- `outputs/pretty_matrix/<task>/manual_vs_generic_support_seeds10_12.png`
- `outputs/pretty_matrix/<task>/generic_support_ablation_seed10.png`

## Current Reading

The implementation is complete enough to support the paper-direction question, but the result is mixed.

`dog_sunglasses` is the strongest case: generic support localizes the face / eye region well enough, and the edit is visible while structure is preserved.

`cat_crown` remains weak: the support is spatially controlled but does not reliably create a crown above the head. This suggests the missing piece is not only mask size, but also a stronger edit target / reference or a better changed-token support signal.

`mug_heart` remains weak without a decal / symbol reference. The support proposer can constrain the region, but the generative direction does not reliably instantiate a clean heart.

`backpack_remove_toy_charm` remains weak for removal. The generic changed-token support is not yet isolating the charm robustly enough to beat manual support.

## Conclusion

The direction is still meaningful, but the claim should be conservative for now:

> The clean-estimate adaptive controller is general once a reasonable support region is available. The current generic support proposal is a promising but not yet sufficient replacement for high-quality manual / external support.

For a stronger publishable story, the next method improvement should focus on support quality, not another controller tweak.
