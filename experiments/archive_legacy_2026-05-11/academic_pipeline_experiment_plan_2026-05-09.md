# Academic-Pipeline Experiment Plan

Date: 2026-05-09

Purpose: convert the current debugging state into a defensible paper experiment
plan. The central claim should be localized RF/ODE editing with automatic
support/reference construction, not pure deterministic recoloring.

## Current Diagnosis

The original main matrix is complete but mostly useful as failure analysis:

- `cat_crown` is the only consistently successful original main task.
- `backpack_blue`, `yellow_car_blue`, and `rabbit_sunglasses` expose useful
  limitations: color misses, broad drift, hybrid objects, and hard profile
  accessories.
- The newer `red_chair_blue` task is visually plausible after tuning, but it is
  a surface recolor case where deterministic YUV chroma replacement preserves
  texture and geometry better than the RF/ODE edit. It should not be a main
  success case.

## Recommended Main Qualitative Tasks

Use four tasks that cannot be reduced to simple chroma replacement:

| Role | Task | Why keep it |
| --- | --- | --- |
| Main accessory | `cat_crown` | Strongest success case; clean localized insertion. |
| Accessory + automatic support | `dog_sunglasses` | Shows SAM head anchor + automatic eye support + generated glasses reference. |
| Object decal | `mug_heart` | Clean local printed-symbol edit; easy to explain and audit. |
| Semantic removal | `backpack_remove_toy_charm` | Removes a localized dangling object while preserving the strap/zipper/fabric context; complements insertion and decal tasks. |

Current `backpack_remove_toy_charm` is stronger than adding another logo/decal
because it tests semantic deletion with an automatically derived support mask.
Current `tshirt_star` is usable as a smoke task but the star is blurred. Current
`tote_leaf` should not be a main task unless the leaf shape and placement are
fixed; the generated reference currently reads like a green diamond tag more
than a printed logo.

## Supplemental / Limitation Tasks

| Task | Recommended use |
| --- | --- |
| `red_chair_blue` | Limitation or appendix: RF/ODE edit can introduce structure drift on pure surface recoloring; deterministic color replacement is a strong non-generative baseline. |
| `backpack_replace_patch_blue` | Supplemental attribute-local replacement stress test: automatic semantic support changes the upper cartoon patch into a plain blue fabric patch while preserving the lower keychain/toy region. |
| `backpack_blue` | Failure analysis for object/texture recolor. |
| `yellow_car_blue` | Failure analysis for broad vehicle recolor and mask/support sensitivity. |
| `rabbit_sunglasses` | Failure analysis for side-profile accessories. |
| `tote_leaf` | Supplemental only after fixing leaf reference shape/box. |

## Experiments To Add

### E1. Pretty Main Smoke Matrix

Run seed 10 on the recommended main tasks with the compact method set:

```bash
TASKS="P1 P2 P3 P5" METHODS="M0 M1 M4" SEEDS="10" DEVICE=4 \
  bash scripts/run_pretty_matrix.sh
```

If `P5` remains too blurry, replace it with `P7` rather than using `P4`.
The current accepted main smoke set is `P1 P2 P3 P7`; use `P8` only as a
supplemental replacement probe.

### E2. Multi-Seed Confirmation

After visual audit of E1, run the accepted tasks over three seeds:

```bash
TASKS="P1 P2 P3 P7" METHODS="M0 M1 M4" SEEDS="10 11 12" DEVICE=4 \
  bash scripts/run_pretty_matrix.sh
```

This is the minimum matrix to support a qualitative and proxy-metric table.

### E3. Component Ablations

Use seed 10 only at first:

- full method
- no reconstruction correction
- no trajectory preservation
- no edit-reference guidance
- attention mask support instead of semantic/SAM support
- semantic/SAM support without generated reference

Critical task-specific ablations:

- `dog_sunglasses`: with vs. without automatic eye support; with vs. without
  glasses reference.
- `mug_heart` / decal task: with vs. without edit-reference guidance.

### E4. Surface Recolor Limitation

Keep `red_chair_blue` as a controlled limitation:

- RF/ODE edit output.
- Deterministic YUV-chroma reference.
- Short discussion: simple chroma replacement can outperform generative editing
  when the desired operation is pure recoloring and exact texture/geometry
  preservation is required.

Do not present deterministic YUV chroma as the RF/ODE method output.

### E5. Object Replacement / Attribute-Local Probe

Use `backpack_replace_patch_blue` as a focused replacement test:

```bash
TASKS="P8" METHODS="M0 M1 M4" SEEDS="10" DEVICE=7 \
  bash scripts/run_pretty_matrix.sh
```

The seed-10 review image is
`outputs/pretty_matrix/backpack_replace_patch_blue_seed10_review.png`, and the
three-seed M4 review is
`outputs/pretty_matrix/backpack_replace_patch_blue_full_seeds_review.png`.
M0/M1 drift into the lower keychain/toy region, while M4 confines the edit to
the upper semantic patch. Keep it as supplemental evidence for attribute-local
replacement rather than treating it as a full object-swap benchmark.

## Paper Framing

Defensible claim:

> RF/ODE editing with automatic localized support and reference construction
> improves the preservation/edit trade-off on localized semantic insertion and
> printed-symbol edits, and can localize harder object-replacement probes, while
> pure surface recoloring remains better handled by specialized deterministic
> color transforms or requires stronger texture constraints.

Avoid claiming:

- General-purpose color editing success.
- Strict geometry preservation for the chair task.
- Deterministic YUV recolor as model output.
