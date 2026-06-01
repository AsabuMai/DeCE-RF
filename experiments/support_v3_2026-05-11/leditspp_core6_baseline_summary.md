# LEDITS++ Core-6 Baseline Summary

Date: 2026-06-01

## Status

LEDITS++ has been run for the Core-6 task set with seeds 10, 11, and 12 using the diffusers-integrated `LEditsPPPipelineStableDiffusion` and `sd-legacy/stable-diffusion-v1-5`.

Outputs:

```text
outputs/external_baselines/<task>/ledits_pp/seed_<seed>/result.png
```

Metrics:

```text
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.csv
experiments/support_v3_2026-05-11/leditspp_core6_seed10_12_metrics.json
```

Visual grid:

```text
experiments/support_v3_2026-05-11/visual_gates/leditspp_core6_seed10_12_grid.png
```

## Aggregate Metrics

| metric | mean |
| --- | ---: |
| `clip_target_minus_source` | 0.0534 |
| `dino_source_similarity` | 0.4685 |
| `outside_mask_l1` | 0.2326 |
| `inside_mask_l1` | 0.2544 |
| `source_ssim_luma` | 0.4915 |
| `runtime_seconds` | 4.3157 |

## Per-Task Snapshot

| task | CLIP target-source delta | DINO/source sim | outside-mask L1 |
| --- | ---: | ---: | ---: |
| `backpack_remove_toy_charm` | 0.0082 | 0.6205 | 0.2545 |
| `cat_crown` | 0.1218 | 0.0724 | 0.2762 |
| `dog_sunglasses` | 0.0781 | 0.1447 | 0.2196 |
| `mug_heart` | 0.0151 | 0.7102 | 0.1710 |
| `red_chair_blue` | -0.0089 | 0.8120 | 0.2819 |
| `tshirt_star` | 0.1062 | 0.4511 | 0.1923 |

## Visual Readout

LEDITS++ is runnable and should be retained as an external diffusion-editing baseline, but the visual outputs show substantial nonlocal rewriting on several tasks. Cat/crown, dog/sunglasses, and backpack removal are especially nonlocal or identity-changing; mug/t-shirt show target formation with moderate reconstruction drift; red-chair recolor is not reliably blue and changes structure/color globally.

Reporting boundary: include these rows in the external baseline table with the backbone and protocol caveat. Do not merge them into the RF/flow-family baseline group or the internal DeCE-RF ablation table.
