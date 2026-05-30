# Visual audit notes

Artifacts reviewed:
- `cat_crown_edit_sweep_seed10.png`, `cat_crown_perturb_seed10.png`
- `dog_sunglasses_edit_sweep_seed10.png`, `dog_sunglasses_perturb_seed10.png`
- `mug_heart_edit_sweep_seed10.png`, `mug_heart_perturb_seed10.png`
- `*_allseeds.png` detail sheets for selected cases
- `visual_audit_key_cases.png`

## Summary

The visual evidence matches the metric-level story: rmsgap is useful for edit-preserve tradeoff on `cat_crown`, gives small preservation-biased gains on `mug_heart`, and is not visually better for `dog_sunglasses`.

## cat_crown

Visual result supports the strongest claim. Under high edit strength and support perturbations, fixed tends to make the crown larger, brighter, and more expanded beyond the head region. rmsgap usually keeps the crown more compact and better localized while preserving the cat/body/background better. The best qualitative examples are `edit_scale=1.5`, `holes`, and `shift`.

This supports: rmsgap improves the edit-preserve Pareto frontier for this stress case.

## dog_sunglasses

Visual result does not support a broad rmsgap win. Fixed often produces more complete, darker, and more recognizable sunglasses. rmsgap can preserve slightly better, but the glasses are sometimes thinner, more transparent, or show more eye/face texture through the lenses. This matches the CLIP/Pareto results where fixed is stronger for this task.

This should be treated as a counterexample or limitation, not hidden.

## mug_heart

Both methods are visually stable. rmsgap often has slightly lower outside drift and similar heart placement, but the visual difference is subtle. The edit is easy and localized, so this case supports a small preservation gain rather than a dramatic visual improvement.

This supports a weak positive claim only.

## Recommended reporting

Use `cat_crown` as the main visual evidence for the stress claim. Use `mug_heart` as a small-supporting case. Report `dog_sunglasses` honestly as a case where preservation control does not improve the edit-success frontier because it weakens the sunglasses.

Recommended wording:

> rmsgap improves edit-preserve behavior in stress regimes where the edit field tends to overshoot spatial support, especially for compact add-object edits such as `cat_crown`. The benefit is task-dependent: for `dog_sunglasses`, rmsgap preserves slightly better but can under-express the target object, so the Pareto improvement does not hold uniformly.
