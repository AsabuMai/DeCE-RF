# Support Policy V1 Active 11-Case Audit

2026-05-12 update: `red_chair_blue` / recolor is retired from the active
rmsgap evaluation set. The current active audit excludes pure recolor tasks.

Artifacts:

- Preview: `support_policy_v1_active11_support_audit.png`
- Machine-readable record: `support_policy_v1_active11_support_audit.json`
- Policy: `docs/support_policy_v1.md`

## Visual Gate

| Task | Gate | Notes |
| --- | --- | --- |
| `cat_crown` | pass | Relation region is above the grounded cat/head area. |
| `dog_sunglasses` | pass | Face accessory route narrows support to the eye band instead of the whole head. |
| `mug_heart` | pass | Decal route selects a small front-surface region on the mug. |
| `tshirt_star` | pass | Decal route selects a chest/surface region, not arbitrary new-token attention. |
| `tote_leaf` | pass | Host-surface route now selects the tote panel, not the plant distractor. |
| `backpack_remove_toy_charm` | pass with caution | Removal route covers the toy/charm source object; large support may still require preservation control. |
| `backpack_replace_patch_blue` | pass | Replacement route covers the source patch. |
| `cat_replace_bell_heart_tag` | pass with caution | Bell support is correct but very small; target generation may still fail. |
| `dog_replace_tennis_ball_star` | pass | Replacement route covers the tennis ball. |
| `rabbit_sunglasses` | pass with caution | Face route shrinks from the whole rabbit to a small head/eye-side region; side-profile placement remains hard. |
| `dog_crown` | pass with caution | Above-head route is plausible; source crop leaves limited space near the head. |

## Interpretation

Support Policy V1 is sufficiently coherent for the next small edit gate. The
remaining high-risk cases are not primarily support routing failures:

- `cat_replace_bell_heart_tag`: the support is tiny, so target object formation is likely the bottleneck.
- `rabbit_sunglasses`: the eye region is tiny and side-profile, so local target formation is likely the bottleneck.
- `dog_crown`: support is plausible, but image framing makes the target placement difficult.
