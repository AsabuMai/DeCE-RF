# E2.3 Native FLUX Contextual Table

Backbone differs from the SD3 algorithmic comparison. These rows are native implementation context only.
All metrics use fixed Core-6 evaluation masks and a 512x512 eval/display copy; original native outputs are retained.

| Method | n | Backbone | Outside L1 down | Inside L1 | Source SSIM up | Excess outside L1 down | Edit audit up | Preserve audit up | Locality audit up | Artifact audit up | Overall audit up |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fireflow | 18 | FLUX.1-dev | 0.4036 | 0.1139 | -0.3035 | 0.3344 | 3.0000 | 2.8333 | 2.3333 | 4.0000 | 2.8333 |
| reflex | 18 | FLUX.1-dev | 0.2908 | 0.1899 | 0.1895 | 0.2215 | 3.8333 | 1.8333 | 2.3333 | 3.3333 | 2.5000 |
| rf_solver_edit | 18 | FLUX.1-dev | 0.3956 | 0.0928 | -0.2783 | 0.3263 | 2.8333 | 3.3333 | 3.0000 | 3.8333 | 3.1667 |

Interpretation: RF-Solver-Edit has the best visual overall score among the native FLUX rows because it preserves layout better on localized insertion/decal tasks. ReFlex has stronger target formation but substantially worse preservation/locality due to broad re-rendering. FireFlow is polished and preserves some structure, but misses several Core-6 goals, especially recolor, strip editing, and removal.

Paper wording: report this as E2.3 native FLUX implementation context, with backbone and normalization columns visible. Do not merge it into the E2.2 same-backbone SD3 algorithmic table.
