# h-Edit Core-6 Smoke Summary

Date: 2026-06-01

## Status

h-Edit text-guided P2P can run on the server through an isolated compatibility environment:

```text
/workspace/baselines/envs/h-edit-py312
/workspace/baselines/src/h-edit/text-guided
```

The first Core-6 smoke used `mug_heart`, seed 10, implicit h-Edit-R + P2P, 20 diffusion steps, and a Core-6 YAML adapter.

Artifacts:

```text
outputs/external_baselines/mug_heart/h_edit/seed_10/result.png
experiments/support_v3_2026-05-11/hedit_mug_heart_seed10_smoke_metrics.csv
experiments/support_v3_2026-05-11/hedit_mug_heart_seed10_smoke_metrics.json
experiments/support_v3_2026-05-11/visual_gates/hedit_mug_heart_seed10_smoke.png
```

## Smoke Metrics

| metric | value |
| --- | ---: |
| `clip_target_minus_source` | -0.0266 |
| `dino_source_similarity` | 0.9778 |
| `outside_mask_l1` | 0.0362 |
| `inside_mask_l1` | 0.0130 |
| `source_ssim_luma` | 0.9132 |

## Readout

The runner is technically operational, but the default Core-6 mug-heart smoke does not form the red-heart target. It mostly reconstructs/preserves the source mug. Do not launch the full Core-6 h-Edit run until the Core-6 adapter or h-Edit prompt/P2P settings pass a target-formation smoke gate.

Recommended next h-Edit action: test a simpler P2P-compatible add/accessory case such as `dog_sunglasses` or tune the `blended_word`, `cfg_tar`, `cfg_src_edit`, and step count before expansion.
