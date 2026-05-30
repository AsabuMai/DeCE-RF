# Completion Prior Reliability Diagnosis, Seed 10

Purpose: diagnose whether a classical completion prior is reliable before running SD3 guidance. This is intended to explain when `completion_clean_delta` should help surface removals and when it should be gated off.

Signals:

- `R_boundary`: continuity between the completed inner mask boundary and the source outer boundary, including color, gradient, and inner-boundary smoothness.
- `R_agreement`: Telea vs Navier-Stokes agreement inside the mask.
- `R_host`: smoothness and low edge complexity of the host ring around the mask.
- `R = R_boundary * R_agreement * R_host`.

| Task | Group | R | R_boundary | R_agreement | R_host | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| laptop_remove_sticker | surface | 0.75 | 0.93 | 0.88 | 0.92 | High reliability; matches the stable clean_delta gain. |
| fridge_remove_yellow_magnet | surface | 0.21 | 0.77 | 0.87 | 0.32 | Medium-low reliability; cluttered host makes the prior fragile. |
| fridge_remove_peach_magnet | surface | 0.23 | 0.67 | 0.79 | 0.42 | Medium-low reliability; small target on busy host. |
| whiteboard_remove_yellow_letter | surface | 0.21 | 0.62 | 0.75 | 0.45 | Medium-low reliability; prior is plausible but nearby letters make host structure nontrivial. |
| backpack_remove_toy_charm | hard | 0.16 | 0.59 | 0.69 | 0.40 | Low reliability; Telea/NS agree on an implausible dangling-object completion. |
| dog_remove_tennis_ball | hard | 0.09 | 0.49 | 0.63 | 0.30 | Low reliability; occluded mouth/fur completion is not trustworthy. |

Takeaway: the reliability signals explain why the completion prior is useful for the planar laptop sticker case, partial/unstable for cluttered surface removals, and harmful or ineffective for backpack/dog hard removals. Backpack is especially useful as a failure case: classical methods can agree with each other while still producing a semantically wrong prior, so future gating should include a prior plausibility/contact-complexity check rather than relying on Telea/NS agreement alone.
