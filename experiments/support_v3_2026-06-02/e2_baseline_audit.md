# E2 Baseline Runnable Validation Audit

Date: 2026-06-03

Claim boundary: downloaded source, command-level smoke, and strict Core-6 output validation are separate states. A method enters the SD3-matched reduced RF comparison only after strict Core-6 generation is stable and is evaluated with the same human/metric gates as DeCE-RF. Native-backbone RF / FLUX rows are contextual, not pure algorithmic controls.

Revised strict tasks: cat_crown, bowl_apple_inside, tshirt_star, red_chair_blue, pillow_vertical_fabric_strip, backpack_remove_toy_charm.

## Summary

- Downloaded repositories: 14/16.
- Command smoke `--help` passed: 1/16.
- SD3-matched reduced RF comparison entries now: 3.
- E2-B native-backbone contextual candidates registered: 6 (`fireflow`, `rf_solver_edit`, `reflex`, `stable_flow`, `ot_rf_otip`, `dvrf`).
- E2-B contextual runnable status: FLUX rows are blocked by gated FLUX.1-dev access or adapter gaps; `ot_rf_otip` and `dvrf` still need repo/backbone verification, environment creation, smoke testing, and Core-6 adapters.
- Non-RF supplement candidates selected: 2 (`instruct_pix2pix`, `h_edit_r_p2p`). These are supplement-only positioning baselines and must not support RF-specific claims.
- Baseline audit rows not claimable now: 16.

## Audit Table

| baseline | family | priority | download_status | entrypoints_found | smoke_status | strict_complete | strict_failed | strict_pending | e2_bucket | strict_failure_example | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fireflow | rf_flow | e2b_native_context_rf | ok | src/edit.py;src/gradio_demo.py | help_timeout | 0 | 1 | 17 | e2b_native_context_candidate | cat_crown/seed10: GatedRepoError: FLUX.1-dev checkpoint access blocked with 401 Unauthorized; strict target-mode generation cannot proceed without an authenticated token. | planned_e2b_context_do_not_use_for_algorithmic_claim |
| flowalign | rf_flow | high_rf | ok | run_edit.py;run_t2i.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| flowedit | rf_flow | high_rf | ok | run_script.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| masactrl | diffusion_attention_control | medium_non_rf | ok | run_synthesis_sdxl.py;app.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| rf_solver_edit | rf_solver | e2b_native_context_rf | ok | FLUX_Image_Edit/src/edit.py;FLUX_Image_Edit/src/gradio_demo.py;Hunyuanvideo_Video_Edit/edit_video.py | help_timeout | 0 | 1 | 17 | e2b_native_context_candidate | cat_crown/seed10: GatedRepoError: FLUX.1-dev checkpoint access blocked with 401 Unauthorized; strict target-mode generation cannot proceed without an authenticated token. | planned_e2b_context_do_not_use_for_algorithmic_claim |
| reflex | rf_rectified_flow | e2b_native_context_rf | ok | img_edit.py | help_ok | 0 | 1 | 17 | e2b_native_context_candidate | cat_crown/seed10: GatedRepoError: FLUX.1-dev pipeline access blocked with 401 Unauthorized after fixing wrapper args and torchao compatibility; strict target-mode generation can... | planned_e2b_context_do_not_use_for_algorithmic_claim |
| splitflow | rf_flow | high_rf | ok | run_script.py | help_timeout | 18 | 0 | 0 | reduced_rf_comparison | - | do_not_claim_beat |
| zone | diffusion_inversion_attention | medium_non_rf | ok | inference.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| h_edit_r_p2p | diffusion_bridge_p2p | supplement_non_rf | ok | text-guided/main_demo.py | needs_env | 0 | 0 | 0 | non_rf_supplement_candidate | - | planned_supplement_do_not_use_for_rf_claim |
| instruct_pix2pix | diffusion_instruction_editing | supplement_non_rf | ok | edit_cli.py;main.py | needs_env | 0 | 0 | 0 | non_rf_supplement_candidate | - | planned_supplement_do_not_use_for_rf_claim |
| ledits_pp | diffusion_editing | medium_non_rf | ok | examples/LEdits.ipynb | not_run | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| pix2pix_zero | diffusion_inversion_direction | medium_non_rf | ok | app_gradio.py | needs_env | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| prompt_to_prompt | diffusion_attention_control | medium_non_rf | ok | prompt-to-prompt_stable.ipynb | not_run | 0 | 0 | 0 | baseline_audit | - | do_not_claim_beat |
| stable_flow | rf_flow | e2b_native_context_rf | ok | run_stable_flow.py | help_timeout | 0 | 0 | 18 | e2b_native_context_candidate | - | planned_e2b_context_do_not_use_for_algorithmic_claim |
| ot_rf_otip | rf_preservation_aware_transport | e2b_native_context_rf | planned_not_downloaded | - | not_started | 0 | 0 | 18 | e2b_native_context_candidate | - | planned_e2b_context_do_not_use_for_algorithmic_claim |
| dvrf | rf_preservation_aware_delta_velocity | e2b_native_context_rf | planned_not_downloaded | - | not_started | 0 | 0 | 18 | e2b_native_context_candidate | - | planned_e2b_context_do_not_use_for_algorithmic_claim |

## E2-B Native-Backbone Contextual Set

| baseline | paper-facing label | native backbone | type | extra support? | current server state | next action |
| --- | --- | --- | --- | --- | --- | --- |
| fireflow | FireFlow | FLUX.1-dev | RF-flow editing | no external DeCE mask | downloaded, strict run blocked by FLUX.1-dev authentication | resolve access/adapter or keep as contextual blocked row |
| rf_solver_edit | RF-Solver-Edit / RF-Edit | FLUX.1-dev | RF solver / inversion-style image edit | no external DeCE mask | downloaded, env built, strict run blocked by FLUX.1-dev authentication | resolve access or keep as contextual blocked row |
| reflex | ReFlex | FLUX.1-dev | RF/FLUX trajectory-attention edit | no external DeCE mask | downloaded, env built, help smoke passes, strict run blocked by FLUX.1-dev authentication | resolve access or keep as contextual blocked row |
| stable_flow | stable-flow | FLUX.1-dev | RF-flow editing | no external DeCE mask | downloaded, adapter pending | resolve access/adapter or keep as contextual blocked row |
| ot_rf_otip | OT-RF / OTIP-style | TBD | optimal-transport / trajectory-preserving RF candidate | no external DeCE mask | registered only, repo/backbone/adapter pending | verify exact public repo and backbone, then smoke |
| dvrf | DVRF / Delta Velocity RF | TBD | delta-velocity / path-aware RF candidate | no external DeCE mask | registered only, repo/backbone/adapter pending | verify exact public repo and backbone, then smoke |

## Non-RF Supplement Set

| baseline | paper-facing label | role | extra support? | current server state | next action |
| --- | --- | --- | --- | --- | --- |
| instruct_pix2pix | InstructPix2Pix | instruction-guided diffusion editor | no DeCE support mask | downloaded, adapter/smoke not yet validated for strict Core-6 | create supplement adapter and run 6 tasks x 2-3 seeds |
| h_edit_r_p2p | H-Edit / P2P-style | diffusion bridge / Prompt-to-Prompt-style comparator | no DeCE support mask | downloaded, adapter/smoke not yet validated for strict Core-6 | create supplement adapter and run 6 tasks x 2-3 seeds |

## Interpretation

- `reduced_rf_comparison` means the method has completed the current SD3-matched strict Core-6 comparison cache and can be reported in Table 2a.
- `e2b_native_context_candidate` means the method is part of the native-backbone RF / FLUX contextual pool; it must not be used as pure algorithmic evidence against SD3-DeCE.
- `non_rf_supplement_candidate` means the method is selected for supplement positioning only; it must not be averaged into E2-A/E2-B or used for RF-specific claims.
- `baseline_audit` means the method remains useful for transparency, but it must not support a claim that DeCE-RF beats that baseline.
