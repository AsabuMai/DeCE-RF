# Backpack Same-Support Inpainting Diagnostic

This is a removal-only diagnostic baseline for ackpack_remove_toy_charm, not a Core-6 main-table method. Telea/NS use the DeCE-RF support mask and classical OpenCV inpainting, so they receive extra same-support region information but no target-directed RF edit control.

| Method | n | CLIP target | CLIP delta | DINO/source | Outside L1 | Inside L1 | SSIM luma |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| same_support_inpaint_telea | 3 | 0.3528 | -0.0142 | 0.8962 | 0.0162 | 0.2267 | 0.8927 |
| same_support_inpaint_ns | 3 | 0.3528 | -0.0121 | 0.8975 | 0.0161 | 0.2318 | 0.8921 |
| support_v3_controller_rmsgap | 3 | 0.3459 | -0.0169 | 0.8842 | 0.0398 | 0.2200 | 0.7366 |

Readout: classical same-support inpainting has lower outside-mask drift and higher source similarity on this removal case, but visual inspection shows blocky or smeared fill artifacts around the strap/zipper area. DeCE-RF better preserves the semantic editing protocol and removes the dangling charm, but it locally smooths the occluded zipper/fabric. Report this as a diagnostic/limitation comparison rather than a headline baseline.
