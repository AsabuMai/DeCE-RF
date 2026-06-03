# E2 Baseline Runnable Validation Audit

Date: 2026-06-03

Claim boundary: downloaded source, command-level smoke, and strict Core-6 output validation are separate states. A method enters the reduced RF comparison only after revised strict Core-6 generation is stable and is evaluated with the same human/metric gates as DeCE-RF.

Revised strict tasks: cat_crown, bowl_apple_inside, tshirt_star, red_chair_blue, pillow_vertical_fabric_strip, backpack_remove_toy_charm.

## Summary

- Downloaded repositories: 14/16.
- Command smoke `--help` passed: 1/16.
- Reduced RF comparison entries now: 3.
- E2-B preservation-aware RF candidates registered: 4 (`rf_solver_edit`, `ot_rf_otip`, `reflex`, `dvrf`).
- E2-B runnable status: `rf_solver_edit` and `reflex` are downloaded but blocked by gated FLUX.1-dev access; `ot_rf_otip` and `dvrf` are registered but still need repo verification, environment creation, smoke testing, and Core-6 adapters.
- Non-RF supplement candidates selected: 2 (`instruct_pix2pix`, `h_edit_r_p2p`). These are supplement-only positioning baselines and must not support RF-specific claims.
- Baseline audit rows not claimable now: 16.

## Audit Table

| baseline | family | priority | download_status | entrypoints_found | smoke_status | strict_complete | strict_failed | strict_pending | e2_bucket | strict_failure_example | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fireflow | rf_flow | high_rf | ok | src/edit.py;src/gradio_demo.py | help_timeout | 0 | 1 | 17 | baseline_audit | cat_crown/seed10: GatedRepoError: FLUX.1-dev checkpoint access blocked with 401 Unauthorized; strict target-mode generation cannot proceed without an authenticated token. | do_not_claim_beat |
| flowalign | rf_flow | high_rf | ok | run_edit.py;run_t2i.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| flowedit | rf_flow | high_rf | ok | run_script.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| masactrl | diffusion_attention_control | medium_non_rf | ok | run_synthesis_sdxl.py;app.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| rf_solver_edit | rf_solver | e2b_preservation_rf | ok | FLUX_Image_Edit/src/edit.py;FLUX_Image_Edit/src/gradio_demo.py;Hunyuanvideo_Video_Edit/edit_video.py | help_timeout | 0 | 1 | 17 | e2b_preservation_candidate | cat_crown/seed10: GatedRepoError: FLUX.1-dev checkpoint access blocked with 401 Unauthorized; strict target-mode generation cannot proceed without an authenticated token. | planned_e2b_do_not_claim_until_complete |
| reflex | rf_rectified_flow | e2b_preservation_rf | ok | img_edit.py | help_ok | 0 | 1 | 17 | e2b_preservation_candidate | cat_crown/seed10: GatedRepoError: FLUX.1-dev pipeline access blocked with 401 Unauthorized after fixing wrapper args and torchao compatibility; strict target-mode generation can... | planned_e2b_do_not_claim_until_complete |
| splitflow | rf_flow | high_rf | ok | run_script.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| zone | diffusion_inversion_attention | medium_non_rf | ok | inference.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| h_edit_r_p2p | diffusion_bridge_p2p | supplement_non_rf | ok | text-guided/main_demo.py | needs_env | 0 | 0 | 0 | non_rf_supplement_candidate | - | planned_supplement_do_not_use_for_rf_claim |
| instruct_pix2pix | diffusion_instruction_editing | supplement_non_rf | ok | edit_cli.py;main.py | needs_env | 0 | 0 | 0 | non_rf_supplement_candidate | - | planned_supplement_do_not_use_for_rf_claim |
| ledits_pp | diffusion_editing | medium_non_rf | ok | examples/LEdits.ipynb | not_run | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| pix2pix_zero | diffusion_inversion_direction | medium_non_rf | ok | app_gradio.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| prompt_to_prompt | diffusion_attention_control | medium_non_rf | ok | prompt-to-prompt_stable.ipynb | not_run | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| stable_flow | rf_flow | high_rf | ok | run_stable_flow.py | help_timeout | 0 | 0 | 18 | baseline_audit | - | do_not_claim_beat |
| ot_rf_otip | rf_preservation_aware_transport | e2b_preservation_rf | planned_not_downloaded | - | not_started | 0 | 0 | 18 | e2b_preservation_candidate | - | planned_e2b_do_not_claim_until_complete |
| dvrf | rf_preservation_aware_delta_velocity | e2b_preservation_rf | planned_not_downloaded | - | not_started | 0 | 0 | 18 | e2b_preservation_candidate | - | planned_e2b_do_not_claim_until_complete |

## E2-B Baseline Set

| baseline | paper-facing label | type | extra support? | current server state | next action |
| --- | --- | --- | --- | --- | --- |
| rf_solver_edit | RF-Solver-Edit / RF-Edit | RF solver / inversion-style image edit | no external DeCE mask | downloaded, env built, strict run blocked by FLUX.1-dev authentication | resolve gated checkpoint access or keep as blocked audit row |
| ot_rf_otip | OT-RF / OTIP-style | optimal-transport / trajectory-preserving RF inversion candidate | no external DeCE mask | registered only, repo and adapter pending | verify exact public repo, create env, smoke entrypoint, then add Core-6 adapter |
| reflex | ReFlex | RF/FLUX trajectory-attention edit candidate | no external DeCE mask | downloaded, env built, help smoke passes, strict run blocked by FLUX.1-dev authentication | resolve gated checkpoint access or keep as backup E2-B audit row |
| dvrf | DVRF / Delta Velocity RF | delta-velocity / path-aware RF candidate | no external DeCE mask | registered only, repo and adapter pending | download verified repo, create env, smoke entrypoint, then add Core-6 adapter |

## Non-RF Supplement Set

| baseline | paper-facing label | role | extra support? | current server state | next action |
| --- | --- | --- | --- | --- | --- |
| instruct_pix2pix | InstructPix2Pix | instruction-guided diffusion editor | no DeCE support mask | downloaded, adapter/smoke not yet validated for strict Core-6 | create supplement adapter and run 6 tasks x 2-3 seeds |
| h_edit_r_p2p | H-Edit / P2P-style | diffusion bridge / Prompt-to-Prompt-style comparator | no DeCE support mask | downloaded, adapter/smoke not yet validated for strict Core-6 | create supplement adapter and run 6 tasks x 2-3 seeds |

## Interpretation

- `e2b_preservation_candidate` means the method is part of the planned preservation-aware RF baseline pool, not a completed paper comparison row.
- `non_rf_supplement_candidate` means the method is selected for supplement positioning only; it must not be averaged into E2-A/E2-B or used for RF-specific claims.
- `baseline_audit` means the method remains useful for transparency, but it must not support a claim that DeCE-RF beats that baseline.
- `reduced_rf_comparison` means the method has completed the current strict Core-6 comparison cache and can be reported in the reduced E2-A table.
