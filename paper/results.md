# Current Results

Date: 2026-05-10

Internal method-matrix values below are computed from complete outputs in
`outputs/main_matrix`. External-baseline status and visual-score artifacts are
computed from `experiments/baseline_parity_manifest.csv`.

## Main Matrix

Coverage:

```text
T1-T4 x M0-M4 x seeds 10, 11, 12 = 60/60 complete
```

Metric artifact:

```text
experiments/main_metrics.csv
```

| Method | Rows | Outside-mask L1 | Luma SSIM | CLIP target-source | DINO source sim | Runtime s | Peak GB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base_only | 12 | 0.0864 | 0.8520 | -0.0373 | 0.5834 | 23.80 | 12.23 |
| direct_target | 12 | 0.1201 | 0.6385 | 0.0177 | 0.4793 | 25.17 | 12.23 |
| anchor_only | 12 | 0.1199 | 0.6371 | 0.0199 | 0.4656 | 27.29 | 12.23 |
| decoupled_rec | 12 | 0.1040 | 0.7109 | 0.0189 | 0.5498 | 27.53 | 12.23 |
| full | 12 | 0.0590 | 0.8745 | 0.0058 | 0.8738 | 54.98 | 13.73 |

Interpretation:

- Direct target and anchor-only variants improve CLIP target alignment more
  than base-only, but they drift heavily.
- The full method has the best outside-mask preservation and DINO source
  similarity, but weaker CLIP target-source gain.
- The defensible claim is therefore a preservation/edit-tradeoff claim, not
  a claim that the full method always edits more strongly.

## External Baselines

Artifacts:

```text
experiments/baseline_parity_manifest.csv
experiments/baseline_summary.csv
experiments/baseline_summary.md
experiments/baseline_visual_scores_seed10_12.csv
experiments/baseline_visual_score_template.csv
paper/stage2_5_integrity_precheck.md
```

Coverage:

```text
6 external baselines x 4 tasks x seeds 10,11,12 = 72 manifest rows
48 complete runnable rows
24 failed rows with recorded failure reasons
```

| Method | Runnable | Complete | Failed | Model family | Seed matched | Use in paper |
| --- | --- | ---: | ---: | --- | --- | --- |
| FlowEdit | yes | 12 | 0 | SD3 | yes | Qualitative baseline |
| SplitFlow | yes | 12 | 0 | SD3 + Mistral-7B | yes | Qualitative baseline |
| FireFlow | yes | 12 | 0 | FLUX-dev | yes | Qualitative baseline, not SD3-matched |
| RF-Solver-Edit | yes | 12 | 0 | FLUX-dev | no | Qualitative baseline, seed not exposed by upstream script |
| ReFlex | no | 0 | 12 | FLUX-dev | not run | Resource/code compatibility limitation |
| SteerFlow | no | 0 | 12 | unavailable | not run | No public runnable code found |

Internal visual-score averages, scale 1-5:

| Method | Rows | Edit success | Preservation | Locality | Artifact | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ours_full | 12 | 4.25 | 5.00 | 5.00 | 3.75 | 4.50 |
| FlowEdit | 12 | 3.75 | 1.75 | 1.75 | 2.75 | 2.25 |
| SplitFlow | 12 | 4.00 | 2.75 | 2.50 | 3.00 | 2.75 |
| FireFlow | 12 | 3.25 | 3.00 | 2.25 | 2.75 | 2.75 |
| RF-Solver-Edit | 12 | 2.75 | 3.00 | 2.25 | 2.75 | 2.75 |

Interpretation:

- The runnable external baselines often produce stronger target semantics than
  `ours_full`, especially on `dog_sunglasses`, where sunglasses are darker and
  more explicit.
- They frequently redraw the source identity, pose, local layout, or scene.
  This is most visible for `cat_crown`, `dog_sunglasses`, and
  `backpack_remove_toy_charm`.
- `backpack_remove_toy_charm` is the strongest qualitative evidence for our
  localized-preservation claim: `ours_full` removes the dangling toy while
  preserving the source patch, pink ring, zipper, and fabric; the baselines
  mostly preserve, redraw, or regenerate a dangling charm.
- `dog_sunglasses` should be described as a tradeoff case. External baselines
  produce more visible sunglasses, while `ours_full` preserves the original dog
  and background better.
- ReFlex and SteerFlow are not omitted. ReFlex failed under the available 24GB
  GPU and public-code constraints, and SteerFlow had no public runnable code as
  of 2026-05-10; both are recorded in the manifest with failure metadata.

Baseline figure candidates:

```text
outputs/paper_figures/baseline_backpack_remove_toy_charm_seed10.png
outputs/paper_figures/baseline_dog_sunglasses_seed10.png
```

## Failure Labels

Artifact:

```text
experiments/failure_flags.json
```

Coverage:

```text
60/60 rows have visual-review notes
57/60 rows have non-empty failure flags
3/60 rows are marked visually successful enough for the current table
```

Failure flag counts:

| Failure flag | Rows |
| --- | ---: |
| background_drift | 18 |
| color_miss | 18 |
| semantic_miss | 15 |
| hybrid_object | 3 |
| localization_error | 3 |
| success / blank | 3 |

## Ablations

Coverage:

```text
T1-T3 x seed 10 x {full plus six variants} = 21/21 complete
```

Metric artifact:

```text
experiments/ablation_metrics.csv
```

| Method | Rows | Outside-mask L1 | Luma SSIM | CLIP target-source | DINO source sim | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full | 3 | 0.0650 | 0.8716 | 0.0032 | 0.8359 | 54.28 |
| full_full_ref | 3 | 0.0614 | 0.8956 | 0.0027 | 0.8805 | 54.41 |
| full_no_rec | 3 | 0.0623 | 0.8932 | -0.0001 | 0.8588 | 50.69 |
| full_no_traj | 3 | 0.0712 | 0.8852 | 0.0055 | 0.8693 | 54.70 |
| full_attention_velocity | 3 | 0.0834 | 0.7877 | -0.0080 | 0.6332 | 52.07 |
| full_semantic_velocity | 3 | 0.0613 | 0.8948 | 0.0067 | 0.9013 | 51.65 |
| full_source_v_inject | 3 | 0.0579 | 0.9074 | -0.0078 | 0.9292 | 53.68 |

Interpretation:

- Attention support is visibly worse than semantic/SAM support on T1/T2.
- Source V injection improves preservation proxies on this seed-10 slice, but
  the CLIP target-source delta is negative and T3 remains weak, so it should
  remain an experimental ablation.
- Additional seeds are needed before treating ablation differences as stable.

## Main Qualitative Figures

```text
outputs/paper_figures/baseline_backpack_remove_toy_charm_seed10.png
outputs/paper_figures/baseline_dog_sunglasses_seed10.png
outputs/paper_figures/cat_crown_seed_10.png
outputs/paper_figures/cat_crown_seed_11.png
outputs/paper_figures/cat_crown_seed_12.png
outputs/paper_figures/backpack_blue_seed_10.png
outputs/paper_figures/backpack_blue_seed_11.png
outputs/paper_figures/backpack_blue_seed_12.png
outputs/paper_figures/yellow_car_blue_seed_10.png
outputs/paper_figures/yellow_car_blue_seed_11.png
outputs/paper_figures/yellow_car_blue_seed_12.png
outputs/paper_figures/rabbit_sunglasses_seed_10.png
outputs/paper_figures/rabbit_sunglasses_seed_11.png
outputs/paper_figures/rabbit_sunglasses_seed_12.png
```

## Ablation Figures

```text
outputs/paper_figures/cat_crown_ablation_seed_10.png
outputs/paper_figures/backpack_blue_ablation_seed_10.png
outputs/paper_figures/yellow_car_blue_ablation_seed_10.png
```
