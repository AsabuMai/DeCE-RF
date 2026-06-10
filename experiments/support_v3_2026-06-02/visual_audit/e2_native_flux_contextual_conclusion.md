# E2.3 Native FLUX Contextual Visual Audit

Scope: FireFlow, RF-Solver-Edit, and ReFlex on strict Core-6, seeds 10/11/12.
Backbone: FLUX.1-dev. Outputs are evaluated as native implementation context, not as E2.2 same-backbone algorithmic evidence.

## Method Summary

| Method | n | Edit success | Preservation | Locality | Artifact | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fireflow | 18 | 3.00 | 2.83 | 2.33 | 4.00 | 2.83 |
| reflex | 18 | 3.83 | 1.83 | 2.33 | 3.33 | 2.50 |
| rf_solver_edit | 18 | 2.83 | 3.33 | 3.00 | 3.83 | 3.17 |

## Contextual Conclusion

The native FLUX rows are runnable and often generate visually polished images, but they do not uniformly solve the localized edit-preserve setting.
RF-Solver-Edit is the strongest contextual row in this audit because it preserves source layout best on localized insertion/decal tasks and passes the cat-crown, bowl-apple, and t-shirt-star probes reasonably well.
FireFlow behaves similarly on the successful insertion/decal cases, but it misses recolor, strip-edit, and removal goals while preserving much of the source.
ReFlex follows target semantics more aggressively on crown, apple, star, and chair-blue cases, but it frequently re-renders object identity, pose, crop, or the broader scene; this makes it a useful native-context baseline rather than a locality-preserving control.
All three native FLUX methods fail the exposed-object removal row visually because the toy charm remains or is re-rendered rather than removed.

Paper-safe wording: native FLUX editors can produce plausible target-looking images, but under fixed Core-6 evaluation masks their visual audit shows a recurring tradeoff between target formation and preservation/locality. These results should be reported as E2.3 implementation context with explicit backbone and normalization caveats.

Filled audit CSV: `/cluster/users/grad/2025/25t8103/project/experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_filled.csv`
Summary CSV: `/cluster/users/grad/2025/25t8103/project/experiments/support_v3_2026-06-02/visual_audit/e2_native_flux_visual_audit_summary.csv`
