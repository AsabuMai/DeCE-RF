# Support Policy v1 Edit Gate Audit

Date: 2026-05-12

Scope:
- Cases: P5, P6, P7, P8, P10, P11, P12
- Methods: `support_v3_fixed`, `support_v3_controller_rmsgap`, `support_v3_core_target_transport` (M22)
- Seed: 10
- Edit scale: 1.0 and 1.5

2026-05-12 update: `P4/red_chair_blue` was removed from the active gate because
pure recolor is not a good fit for the rmsgap local target-formation claim.

Artifacts:
- `policyv1_edit_gate_scale100.png`
- `policyv1_edit_gate_scale150.png`
- `policyv1_edit_gate_scale100.json`
- `policyv1_edit_gate_scale150.json`

## Visual Findings

| case | edit_scale=1.0 | edit_scale=1.5 | gate result |
| --- | --- | --- | --- |
| tshirt_star | Fixed and rmsgap add a small red star on the shirt. M22 only produces a weak partial red mark. | Fixed/rmsgap stronger and still localized. M22 remains weak. | Pass for fixed/rmsgap; M22 fail/weak. |
| tote_leaf | No method produces a visible green leaf logo. | Still no visible target logo. | Fail: support is plausible, but edit formation fails. |
| backpack_remove_toy_charm | Fixed/rmsgap remove the yellow toy while preserving the backpack reasonably. M22 removes more crudely and leaves a larger gray artifact. | Same trend. | Pass for fixed/rmsgap; M22 worse. |
| backpack_replace_patch_blue | Fixed/rmsgap mostly erase the colorful patch but do not create a convincing plain blue replacement patch. M22 creates a beige/gray patch-like artifact, not blue. | Same trend with slightly stronger local change. | Partial fail: source removal works, target replacement does not. |
| cat_replace_bell_heart_tag | No method produces a visible red heart tag; collar/bell area barely changes. | Still no visible heart tag. | Fail: target is too small/weak for current edit model. |
| dog_replace_tennis_ball_star | All methods create a red star-like object near the mouth, with some residual ball/toy ambiguity. Fixed and rmsgap are comparable. | Similar but stronger; target still not cleanly replaces the green ball. | Partial pass: edit occurs, target quality imperfect. |
| rabbit_sunglasses | Fixed produces the clearest black sunglasses. Rmsgap produces a small bluish/weak glasses mark. M22 mostly fails. | Fixed/rmsgap degrade or become less recognizable; M22 still fails. | Scale 1.0 fixed pass; rmsgap weak; scale 1.5 not better. |

## Interpretation

The support policy v1 audit shows that support estimation is now good enough to run controlled comparisons, but this edit gate does not support a strong rmsgap claim yet. The dominant bottleneck is target edit formation, especially for small generated objects and precise replacements:

- Reliable or usable edits: shirt star, backpack toy removal, dog star-like replacement.
- Weak or failed edits: tote leaf logo, cat heart tag, blue replacement patch, rabbit sunglasses at high scale.
- rmsgap is usually comparable to fixed, but not consistently better visually in this seed10 gate.
- M22/core target transport does not improve the weak cases and is often worse than fixed/rmsgap.

Current conclusion: do not run the full edit-strength/AUC sweep yet for M22. The next model work should focus on making target formation stronger under localized support before trying to prove a Pareto-frontier improvement.
