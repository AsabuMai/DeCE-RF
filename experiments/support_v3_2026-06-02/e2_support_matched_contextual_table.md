# E2.4 Support-Matched Diagnostic

Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.
The post-hoc blend rows use the same fixed binary edit mask for localization only:
`output = M_edit * edited_output + (1 - M_edit) * source`.
These rows are diagnostic and must not be presented as main fair baselines.

| Method | n | Backbone | Outside L1 down | Inside L1 | Source SSIM up | Excess outside L1 down |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| direct_target_raw | 6 | SD3 | 0.0794 | 0.0984 | 0.7146 | 0.0246 |
| direct_target_mask_blend | 6 | SD3 | 0.0001 | 0.0924 | 0.9694 | -0.0547 |
| flowedit_mask_blend | 6 | SD3 | 0.0002 | 0.2162 | 0.9106 | -0.0546 |
| support_v3_controller_rmsgap | 6 | SD3 | 0.0374 | 0.0899 | 0.8908 | -0.0175 |

Interpretation: post-hoc localization sharply reduces outside-mask drift for direct-target and FlowEdit-style outputs, but it cannot recover missed edits, boundary coherence, or controller behavior. Full DeCE-RF is included as the target method; the fixed-weight DeCE displacement variant is reported separately as a component ablation rather than as an E2.4 support baseline.

Paper-safe wording: E2.4 separates localization from controller design. Binary localization or output blending can improve preservation metrics, but it is a diagnostic transformation rather than an editing algorithm under matched inference conditions.
