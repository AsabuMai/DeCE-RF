# Opfield and Replace-v2 Targeted Gate

Date: 2026-05-12

Scope:
- Seed: 10
- Edit scale multiplier: 1.0
- Main comparison: `support_v3_fixed`, `support_v3_controller_rmsgap`, `support_v3_controller_rmsgap_opfield`
- Replace follow-up: `support_v3_controller_rmsgap_replace_v2`
- Preview figures:
  - `experiments/support_v3_2026-05-11/opfield_targeted_gate/policyv1_edit_gate_scale100.png`
  - `experiments/support_v3_2026-05-11/replace_v2_gate/policyv1_edit_gate_scale100.png`

## Coverage

Targeted opfield gate is complete for 9/9 cases:
- `dog_sunglasses`
- `mug_heart`
- `tshirt_star`
- `tote_leaf`
- `backpack_remove_toy_charm`
- `backpack_replace_patch_blue`
- `cat_replace_bell_heart_tag`
- `dog_replace_tennis_ball_star`
- `rabbit_sunglasses`

Replace-v2 gate is complete for 3/3 replace cases:
- `backpack_replace_patch_blue`
- `cat_replace_bell_heart_tag`
- `dog_replace_tennis_ball_star`

## Visual Read

`support_v3_controller_rmsgap_opfield` is a real improvement for local add-style edits:
- `tote_leaf`: fixed and rmsgap mostly preserve the blank bag; opfield produces a clear green leaf/logo on the tote.
- `tshirt_star`: opfield makes the star larger and more legible than fixed/rmsgap.
- `mug_heart`: all three methods succeed; opfield is not worse and keeps the background stable.
- `dog_sunglasses`: all three methods add glasses; opfield is at least comparable to rmsgap/fixed.
- `rabbit_sunglasses`: opfield makes the glasses more recognizable than rmsgap, while fixed still gives the darkest lens.

The remove control does not show an obvious regression:
- `backpack_remove_toy_charm`: opfield removes the toy charm while preserving the backpack structure similarly to fixed/rmsgap.

Replace remains the weak operation family:
- `backpack_replace_patch_blue`: all methods remove or suppress the original cartoon patch, but none reliably creates a plain blue fabric patch.
- `cat_replace_bell_heart_tag`: all methods mostly keep the original collar/bell semantics; the requested red heart tag is not formed.
- `dog_replace_tennis_ball_star`: all methods create a red star-like object at the mouth. opfield is slightly cleaner than rmsgap; replace_v2 is comparable and not a decisive new mechanism.

## Current Decision

Keep `support_v3_controller_rmsgap_opfield` as the main active branch beside `support_v3_controller_rmsgap`.

Do not claim replace is solved. The current replace-v2 policy is only a parameterized opfield variant, and the visual evidence says it does not address the harder "remove source object, synthesize new target object with different semantics" cases. Replace likely needs an operation-specific mechanism, not another small global tuning pass.

Working claim supported by this gate:

> Operation-conditioned clean-estimate control improves the edit-preserve behavior for local add-style edits, especially decal/logo and wearable accessory cases, without breaking the remove control.

Claim not yet supported:

> The same mechanism solves general object replacement.
