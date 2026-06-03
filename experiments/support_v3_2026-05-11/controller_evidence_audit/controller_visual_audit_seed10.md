# Controller Visual Audit, Seed 10

Date: 2026-05-11

## Purpose

This audit checks whether the controller metric gains are visually meaningful.
It is part of the current technical-task cleanup, not paper writing.

## Visual Evidence

Full-image comparison:

```text
experiments/support_v3_2026-05-11/controller_visual_effect_comparison_seed10.png
```

Corrected local crop comparison:

```text
experiments/support_v3_2026-05-11/controller_visual_effect_comparison_seed10_crops_fixed.png
```

Normgate local crop comparison:

```text
experiments/support_v3_2026-05-11/controller_evidence_audit/controller_normgate_seed10_crops.png
```

The earlier crop file:

```text
experiments/support_v3_2026-05-11/controller_visual_effect_comparison_seed10_crops.png
```

should not be used for visual assessment because its result columns were black
from applying a source-image-sized crop box to resized result images.

## Summary

All three seed10 edit tasks are visually successful in the basic sense:

```text
cat_crown:       crown appears and is localized on the head
dog_sunglasses:  sunglasses appear and are localized on the eyes
mug_heart:       heart appears and is localized on the mug front
```

However, the visual gap between `support_v3_fixed` and
`support_v3_controller_rmsgap` is small. The current result is best understood
as a stability/consistency gain rather than a dramatic visual improvement.

## Task-level Findings

### cat_crown

`fixed`, `rmsgap`, `progress`, and `hybrid` all generate a visible crown.

Visual differences:

- `fixed`: strong crown, localized, but stylized and somewhat large.
- `rmsgap`: slightly less aggressive and visually cleaner than fixed/progress.
- `progress`: clear crown, but more aggressive; matches the quantitative
  finding that progress tends to increase inside-region change.
- `hybrid`: close to rmsgap in seed10.

Verdict:

```text
rmsgap is visually acceptable, but improvement over fixed is subtle.
progress should remain a risk branch.
```

### dog_sunglasses

All controller variants generate clear sunglasses on the dog face.

Visual differences are minor. `fixed`, `rmsgap`, and `hybrid` are nearly
equivalent visually. `progress` looks acceptable in seed10, but multi-seed
metrics are worse than rmsgap.

Verdict:

```text
rmsgap is safe, but this task mainly supports stability rather than large
visible improvement.
```

### mug_heart

All controller variants generate a visible heart on the mug.

Visual differences:

- `fixed`: good heart placement and acceptable preservation.
- `rmsgap`: similar visual result, with better quantitative preservation.
- `progress`: heart is visible, but preservation metrics are worse.
- `hybrid`: visible heart, but preservation is slightly weaker than rmsgap.

Verdict:

```text
rmsgap remains the best current trade-off. Progress and hybrid are not main
candidates for decal-like edits.
```

## Overall Visual Verdict

The current project state is technically viable but visually modest:

```text
support_v3_controller_rmsgap produces successful edits and does not visibly
damage non-edit regions, but its improvement over fixed control is subtle.
```

This matches the metric audit:

```text
rmsgap: 9/9 CLIP improved, 9/9 loose success, 6/9 strict success
progress: 0/9 strict success
normgate: negative seed10 result, not expanded
```

## Next Technical Implication

Do not spend the next step on more controller micro-tuning. The bottleneck is
now visual effect strength versus subtle metric gain.

Recommended next step:

```text
build a paper-ready visual evidence pack and inspect failure/risk cases,
especially whether the rmsgap benefit is visually distinguishable enough to
justify the controller claim.
```

If stronger visual gains are required, the next target should be support/edit
field quality, not another small controller variant.

## Normgate Visual Check

The corrected normgate crop comparison shows that `normgate` is visually close
to current `rmsgap` on all three seed10 tasks.

Visual interpretation:

- cat: normgate and rmsgap are almost indistinguishable; normgate does not
  produce a clearly better crown.
- dog: normgate is visually comparable, but its CLIP is lower than current
  rmsgap.
- mug: normgate heart is visible, but preservation metrics are worse than
  current rmsgap.

This supports the metric decision:

```text
Do not expand normgate to seeds 10-12 in this configuration.
```
