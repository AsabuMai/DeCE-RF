# E2.4 Support-Matched Diagnostic With Visual Audit

Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.
This is the recommended compact support-matched subset: attached accessory addition, surface decal, and exposed-object removal.
Post-hoc blend rows use the same fixed binary edit mask and are diagnostic only.

| Method | n | Outside L1 down | SSIM up | Edit | Preserve | Locality | Artifact | Overall | Failure mode |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| direct_target_raw | 6 | 0.0794 | 0.7146 | 1.67 | 1.67 | 1.33 | 3.33 | 1.33 | target_miss_global_identity_drift |
| direct_target_mask_blend | 6 | 0.0001 | 0.9694 | 1.33 | 4.00 | 3.00 | 2.33 | 2.00 | target_miss_blend_boundary |
| flowedit_mask_blend | 6 | 0.0002 | 0.9106 | 1.33 | 3.67 | 2.67 | 2.00 | 1.67 | target_miss_blend_boundary |
| support_v3_controller_rmsgap | 6 | 0.0374 | 0.8908 | 4.67 | 3.67 | 4.00 | 3.67 | 4.33 | none |

Conclusion: fixed binary output blending almost eliminates outside-mask metric error by construction, but visual audit shows it does not solve target correctness or boundary coherence. DeCE-RF is the only row that consistently performs the intended operation in this support-matched diagnostic. The fixed-weight DeCE displacement variant is reported separately as a component ablation rather than as an E2.4 support baseline.
