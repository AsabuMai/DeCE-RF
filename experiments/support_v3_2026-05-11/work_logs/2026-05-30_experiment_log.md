# 2026-05-30 Experiment Log

## Context

Today focused on finalizing the role of removal and replacement experiments for the DeCE-RF paper, after observing that the whiteboard removal case changes the yellow letter into another glyph-like structure rather than cleanly removing it.

## 1. High-Confidence Completion Gate

We changed the completion clean-delta gate from a tiered rule to a conservative high-confidence rule:

```text
R >= 0.50 -> enable completion clean-delta
R < 0.50  -> disable completion clean-delta and fall back to the default source command
```

New method name:

```text
support_v3_controller_rmsgap_completion_clean_delta_gated_highconf
```

Cross-seed run completed:

```text
6 tasks x 3 seeds = 18/18 complete
```

Artifacts:

```text
experiments/support_v3_2026-05-11/removal_completion_clean_delta_gated_highconf_seeds10_11_12_protocol.json
experiments/support_v3_2026-05-11/visual_gates/removal_completion_clean_delta_gated_highconf_seeds10_11_12_grid.png
experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_seed10_11_12.csv
experiments/support_v3_2026-05-11/prior_reliability/completion_reliability_vs_gain_highconf_scatter.png
```

Key readout:

```text
laptop_remove_sticker: R=0.751, gate=1.0, clean_delta benefit preserved
fridge removals: R~0.21-0.23, gate=0.0, falls back to default
whiteboard_remove_yellow_letter: R=0.206, gate=0.0, falls back to default
backpack_remove_toy_charm: R=0.163, gate=0.0, falls back to default
dog_remove_tennis_ball: R=0.092, gate=0.0, falls back to default
```

Important correction:

High-confidence gating suppresses completion-prior chasing, but it does not solve the whiteboard case. The default editor itself can still hallucinate glyph-like structure in the letter field.

## 2. Whiteboard Failure Interpretation

The whiteboard removal case should not be treated as a removal success.

Observed behavior:

```text
remove yellow letter I -> regenerate another letter/character-like structure
```

Interpretation:

```text
This is not only a completion-prior failure.
It is also a base target-formation / semantic-field hallucination failure.
The surrounding alphabet-magnet context pulls the model toward glyph-like completions.
```

Paper role:

```text
Use whiteboard removal as a limitation / stress case, not as a main benchmark success.
```

## 3. Whiteboard Replacement Probe

We tested whether the same whiteboard support can be redirected to a non-glyph target.

Probe variants:

```text
blank
blue letter T
red letter A
blue round magnet
red star sticker
```

Methods:

```text
support_v3_controller_rmsgap
support_v3_controller_rmsgap_replace_editor_v0
support_v3_controller_rmsgap_replace_editor_v1
```

Seeds:

```text
10, 11, 12
```

Completion:

```text
5 variants x 3 methods x 3 seeds = 45/45 complete
```

Artifact:

```text
experiments/support_v3_2026-05-11/visual_gates/whiteboard_replacement_probe_seeds10_11_12_grid.png
experiments/support_v3_2026-05-11/whiteboard_replacement_probe_protocol.json
```

Readout:

```text
blank: fails; glyph/letter-field residue remains
blue letter T: unstable; precise glyph control is weak
red letter A: unstable; sometimes letter-like but not controlled
blue round magnet: partial success; round blue object appears in several replace-editor outputs
red star sticker: strongest result; replace_editor_v0/v1 form a clear red star-like sticker across seeds
```

Conclusion:

```text
Whiteboard is not impossible to edit.
Removal and precise glyph replacement are unreliable, but strong non-glyph replacement targets with clear color/shape cues can work.
```

Paper role:

```text
Use red star sticker as a replacement stress probe.
Use blank/T/A as diagnostic evidence for glyph-field limitations.
```

## 4. Final Experiment Structure Decision

Current core-5:

```text
cat_crown
dog_sunglasses
mug_heart
tshirt_star
backpack_remove_toy_charm
```

Decision:

Do not expand the main benchmark with weak replacement/removal cases such as:

```text
dog_replace_tennis_ball_star
dog_remove_tennis_ball
whiteboard_remove_yellow_letter
fridge_remove_yellow_magnet
fridge_remove_peach_magnet
cat_replace_bell_heart_tag
backpack_replace_patch_blue
```

Recommended final structure:

```text
Main Core-6:
1. cat_crown
2. dog_sunglasses
3. mug_heart
4. tshirt_star
5. backpack_remove_toy_charm
6. recolor task to be run tomorrow

Extension probes:
1. laptop_remove_sticker with high-confidence completion prior
2. whiteboard_probe_red_star_sticker as non-glyph replacement

Limitations / diagnostics:
1. whiteboard_remove_yellow_letter
2. dog_remove_tennis_ball
3. dog_replace_tennis_ball_star
4. fridge removals
5. precise glyph replacement T/A
```

Rationale:

```text
Main benchmark should remain clean and defensible.
Completion-prior removal and replacement target formation should be written as operation-conditioned extensions/probes, not mixed silently into the main DeCE-RF controller.
```

## 5. Tomorrow Plan

Run one recolor / attribute-editing task to complete the main benchmark coverage.

Preferred candidate:

```text
red_chair_blue
```

Alternative:

```text
red_office_chair_to_blue_office_chair
```

Goal:

```text
Add localized attribute / recolor editing coverage so the main benchmark covers:

add/accessory
decal
removal
recolor
```

Recommended first step:

```text
Run seed-10 visual gate for the recolor candidate before expanding to seeds 10/11/12.
```

## 6. Experiment Design Update

After re-checking the current core-5, removal diagnostics, and replacement
probes, the final experiment design was updated across the paper planning
documents.

Updated files:

```text
paper/experiment_plan.md
paper/wacv_experiment_design.md
paper/figures.md
paper/tables.md
paper/results.md
paper/limitations.md
```

Final structure:

```text
Main Core-6:
1. cat_crown
2. dog_sunglasses
3. mug_heart
4. tshirt_star
5. backpack_remove_toy_charm
6. one recolor task after seed-10 gate

Preferred recolor:
red_chair_blue

Fallback recolor:
red_office_chair_to_blue_office_chair

Extension probes:
1. laptop_remove_sticker with high-confidence completion prior
2. whiteboard_probe_red_star_sticker with replacement target route

Limitations / diagnostics:
whiteboard_remove_yellow_letter
dog_remove_tennis_ball
dog_replace_tennis_ball_star
fridge_remove_yellow_magnet
fridge_remove_peach_magnet
precise glyph replacement T/A
```

Important decision:

```text
Do not promote dog_replace_tennis_ball_star into the core benchmark. It is a
partial replacement case: target pressure fires, but the object is not clean.
```

Writing rule:

```text
Do not silently mix operation-conditioned extension routes into the base
DeCE-RF main table. Label laptop as DeCE-RF + completion prior and whiteboard
red star as DeCE-RF + replacement route.
```
