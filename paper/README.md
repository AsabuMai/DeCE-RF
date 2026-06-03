# Paper Workspace

Current source of truth for the redesigned WACV experiment plan:

- paper/wacv_experiment_design.md
- paper/core6_phase1_images_prompts.md

Current paper-to-code alignment and cleanup boundary:

- docs/wacv_phase1_code_map.md
- scripts/run_wacv_phase1.sh

The current strict Core-6 design is category-based. After quick human audit, T1 is `cat_crown` rather than `dog_sunglasses`, and T3 is `tshirt_star` rather than `mug_heart`; the full 6-task x 4-method x 3-seed Phase 1 matrix is complete.

Archived legacy material from the previous Core-6/server-evidence pass lives in:

- paper/archive_old_core6_20260602/

Use archived old Core-6 results only as supplementary diagnostics. The active strict Phase 1 rows are documented in `docs/wacv_phase1_code_map.md` and `experiments/support_v3_2026-06-02/`.
