# Baseline Summary

Date: 2026-05-10

## Run Status

| Method | Runnable | Complete | Failed | Model family | Seed matched | Notes |
| --- | --- | ---: | ---: | --- | --- | --- |
| `flowedit` | yes | 12 | 0 | SD3 | yes | Official FlowEdit SD3 runner; source resized/cropped to 512-compatible input. |
| `splitflow` | yes | 12 | 0 | SD3 + Mistral-7B prompt decomposition | yes | Official SplitFlow defaults; T_steps=50, n_max=33, tar guidance=13.5. |
| `fireflow` | yes | 12 | 0 | FLUX-dev | yes | Official FireFlow fast-editing recipe; qualitative FLUX baseline, not SD3-matched. |
| `rf_solver_edit` | yes | 12 | 0 | FLUX-dev | no | Official RF-Solver-Edit image script does not expose seed control. |
| `reflex` | no | 0 | 12 | FLUX-dev | not run | Failed under available 24GB GPU/code constraints; see failure metadata. |
| `steerflow` | no | 0 | 12 | unknown / unavailable | not run | No public runnable code or local repository found as of 2026-05-10. |

## Internal Visual Score Averages

Scores are a first-pass internal visual audit, not a user-study result. Scale: 1=failed, 3=usable/weak, 5=strong.

| Method | Rows | Edit success | Preservation | Locality | Artifact | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ours_full` | 12 | 4.25 | 5.00 | 5.00 | 3.75 | 4.50 |
| `flowedit` | 12 | 3.75 | 1.75 | 1.75 | 2.75 | 2.25 |
| `splitflow` | 12 | 4.00 | 2.75 | 2.50 | 3.00 | 2.75 |
| `fireflow` | 12 | 3.25 | 3.00 | 2.25 | 2.75 | 2.75 |
| `rf_solver_edit` | 12 | 2.75 | 3.00 | 2.25 | 2.75 | 2.75 |

## Reading

- `ours_full` is strongest on preservation and locality.
- SplitFlow, FlowEdit, FireFlow, and RF-Solver-Edit often improve target-object strength but redraw identity or layout.
- `backpack_remove_toy_charm` is the strongest evidence for localized preservation.
- `dog_sunglasses` is the clearest tradeoff case: external baselines make darker sunglasses, while `ours_full` preserves the dog better.
