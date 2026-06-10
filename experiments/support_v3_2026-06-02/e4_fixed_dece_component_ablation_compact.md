# E4 Compact Component Ablation

Scope: cat_crown, tshirt_star, and backpack_remove_toy_charm; seeds 10/11.
`Fixed DeCE displacement` keeps operation-conditioned support and fixed clean-estimate edit-preserve displacement, but removes feedback-updated weights and projection/correction. It is a component ablation, not an external baseline or an E2.4 support-only row.

| Variant | n | Outside L1 down | SSIM up | Edit | Preserve | Locality | Artifact | Overall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct target guidance | 6 | 0.0794 | 0.7146 | 1.67 | 1.67 | 1.33 | 3.33 | 1.33 |
| Fixed DeCE displacement | 6 | 0.0378 | 0.8882 | 4.67 | 3.67 | 4.00 | 4.00 | 4.33 |
| DeCE-RF | 6 | 0.0374 | 0.8908 | 4.67 | 3.67 | 4.00 | 3.67 | 4.33 |

Interpretation: this compact ablation separates direct target guidance, fixed DeCE displacement, and the full DeCE-RF feedback controller. The fixed row should be used to discuss component structure and feedback, while E2.4 remains focused on whether binary localization alone explains the gain.
