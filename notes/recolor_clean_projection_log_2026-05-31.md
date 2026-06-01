# Recolor Clean Projection Work Log - 2026-05-31

## Goal

Improve red-to-blue recolor while preserving source texture/shading and reducing residual red edges. Main test image:

- `data/pretty_free_candidates/pexels_red_armchair_room_6758347.jpg`

The user concern was that the chair texture should not be destroyed, and that the remaining red edge should be understood from the method's math rather than only tuned away.

## Current Code Changes

Modified files:

- `guidance_fields.py`
- `sd3_hrec.py`
- `run_edit_sd3.py`
- `tests/test_mask_and_box_helpers.py`

Added recolor texture/boundary loss:

- `masked_recolor_texture_boundary_loss`
- high-pass luma texture preservation
- object-side boundary chroma pressure

Added clean-space recolor projection:

- Decode current clean estimate.
- Build clean target from source luma texture plus target/reference chroma.
- Encode target back to latent.
- Convert clean delta to velocity and add it to edit guidance.

Added clean projection controls:

- `--edit-color-clean-projection-scale`
- `--edit-color-clean-projection-mode {soft,strict}`
- `--edit-color-clean-projection-texture-kernel-size`
- `--edit-color-clean-projection-alpha-power`
- `--edit-color-clean-projection-boundary-boost`
- `--edit-color-clean-projection-boundary-kernel-size`
- `--edit-color-clean-projection-composite-mode {blend,matte}`
- `--edit-color-clean-projection-background-kernel-size`

The newer `matte` composite mode changes the projection target from:

```text
target = (1 - alpha) * current + alpha * blue_foreground
```

to:

```text
target = (1 - alpha) * estimated_background + alpha * blue_foreground
```

This is closer to real foreground/background alpha compositing, but it still depends on the quality of alpha.

## Tests

Relevant tests passed after the changes:

```text
62 passed, 2 warnings, 1 subtests passed
```

Latest narrower smoke test:

```text
42 passed, 2 warnings
```

## Main Experiment Results

### Complex Red Armchair

Useful outputs:

- Baseline 896 texture run:
  - `outputs/pretty_matrix/red_chair_blue/support_v3_controller_rmsgap_recolor_texture_A_896/seed_10/result.png`
- Clean projection 0.18:
  - `outputs/pretty_matrix/red_chair_blue/support_v3_controller_rmsgap_recolor_cleanproj_soft_896_s018/seed_10/result.png`
- Alpha-edge 0.25:
  - `outputs/pretty_matrix/red_chair_blue/support_v3_controller_rmsgap_recolor_cleanproj_alphaedge_896_s025/seed_10/result.png`
- Matte projection 0.25:
  - `outputs/pretty_matrix/red_chair_blue/support_v3_controller_rmsgap_recolor_cleanproj_matte_896_s025/seed_10/result.png`
- Diagnostic compare images on local machine:
  - `/tmp/red_chair_alphaedge_compare_v2.png`
  - `/tmp/red_chair_matte_complex_compare.png`
  - `/tmp/red_chair_matte_complex_zoom.png`
  - `/tmp/red_edge_diagnostic.png`

Representative metrics:

| Version | Red edge | Outside blue | Inside blue | Texture corr |
|---|---:|---:|---:|---:|
| clean projection 0.18 | 6.6% | 2.9% | high | 0.451 |
| alpha-edge 0.25 | 3.5% | 6.8% | 99.6% | 0.464 |
| matte 0.25 | 3.5% | 8.3% | 99.7% | 0.465 |
| clean projection 0.35 | 3.6% | 9.6% | 99.6% | 0.452 |

Conclusion:

- Clean-space projection clearly improves texture compared with auxiliary loss.
- Alpha-edge 0.25 is currently the best tradeoff on the complex chair.
- Matte composite did not reduce red edge on the complex chair; it slightly increased outside blue.
- This suggests the bottleneck is alpha quality, not only the projection formula.

### Red Edge Diagnosis

Diagnostic result:

- Chair core region has almost no red residual.
- Most residual red near the object falls in low-alpha / boundary pixels.
- Output red near the object overlaps heavily with source red.

Interpretation:

- SAM/external mask finds the object region well enough for the body.
- The remaining issue is boundary mixed pixels, not general recolor failure.
- A binary or coarse soft segmentation mask is not a true foreground alpha matte.

Important distinction:

```text
SAM asks: is this pixel chair?
Matte asks: how much of this pixel is chair foreground?
```

The current alpha is derived from masks and reshaping, so it is not a true physical matte.

## SAM / Mask Clarification

For the complex red armchair:

- The pipeline reused pre-generated Grounded-SAM / semantic masks.
- `run_edit_sd3.py` did not call SAM live during each ODE run.
- Metadata points to:
  - `surface_refined_mask.png`
  - `surface_guidance_mask.png`
  - `operation_v3_grounding_mask.png`

So the correct description is:

```text
Grounded-SAM/semantic preprocessing generated external masks;
ODE runs consumed those masks.
```

For the white-background chair first trial:

- `--grounding-method grounded_sam` was only a label.
- No external SAM mask was passed.
- The automatic color mask missed the full chair.

## Simple White Chair Trial

Image:

- `data/pretty_free_candidates/pexels_red_chair_white_bg_4172380.jpg`

Outputs:

- Auto mask result:
  - `outputs/pretty_matrix/red_chair_white_bg_blue/cleanproj_alphaedge_896_s025/seed_10/result.png`
- Red pixel mask result:
  - `outputs/pretty_matrix/red_chair_white_bg_blue/cleanproj_alphaedge_masked_896_s025/seed_10/result.png`
- Dilated red mask result:
  - `outputs/pretty_matrix/red_chair_white_bg_blue/cleanproj_alphaedge_masked_dilate5_896_s025/seed_10/result.png`
- Matte composite result:
  - `outputs/pretty_matrix/red_chair_white_bg_blue/cleanproj_matte_masked_dilate5_896_s025/seed_10/result.png`

Metrics:

| Version | Red edge | Outside blue | Inside blue |
|---|---:|---:|---:|
| auto mask | 56.5% | 2.4% | 3.7% |
| red pixel mask | 13.6% | 2.3% | 97.4% |
| dilate5 mask | 10.9% | 3.9% | 98.0% |
| matte dilate5 | 10.5% | 1.6% | 98.0% |

Conclusion:

- The first white-chair run failed because no real external/SAM mask was passed.
- A simple red-pixel mask was enough to make the body recolor correctly.
- Matte composite reduced blue spill on white background, but only slightly reduced red edge.

## Current Best Setting

For the complex red armchair, the current best tradeoff is:

```text
--edit-color-clean-projection-scale 0.25
--edit-color-clean-projection-mode soft
--edit-color-clean-projection-texture-kernel-size 7
--edit-color-clean-projection-alpha-power 1.8
--edit-color-clean-projection-boundary-boost 2.0
--edit-color-clean-projection-boundary-kernel-size 9
--edit-color-clean-projection-composite-mode blend
```

`matte` mode is implemented and worth keeping, but on the complex chair it was not better than `blend`.

## Main Takeaway

The texture route is now in a good direction:

- Clean projection restores source-like fabric texture.
- It is better than just adding texture/boundary losses.

The unresolved edge issue is not simply "SAM bad" or "mask resolution too low"; it is more specifically:

```text
We need a true boundary alpha matte.
```

Until alpha is better, the system trades red edge against blue spill:

- More aggressive projection removes red but leaks blue.
- Conservative projection preserves background but leaves red.

## Suggested Next Steps

1. Keep `alpha-edge 0.25` as current best complex-chair result.
2. Do not continue increasing global projection scale.
3. Build a real narrow-band alpha estimator around the SAM boundary:
   - Use SAM mask as trimap.
   - Estimate foreground/background color models or use closed-form / random-walker matting.
   - Use the estimated alpha only in the boundary band.
4. Feed this matte into:
   - color mask
   - ref mask
   - clean projection alpha
5. Re-run complex chair and compare:
   - red edge
   - outside blue
   - texture corr
   - visual zoom around right/top chair boundary

