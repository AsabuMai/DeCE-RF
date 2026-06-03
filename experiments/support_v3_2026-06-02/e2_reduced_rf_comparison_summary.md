# E2 Reduced RF Comparison Summary

Scope: revised strict Core-6 target-mode RF comparison, external FlowEdit/FlowAlign/SplitFlow vs DeCE-RF, seeds 10/11/12.

Claim boundary: this is a reduced target-mode comparison against runnable external RF baselines. Remaining RF baselines stay in adapter/generation validation audit and are not used for broad superiority claims.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | CLIP edit score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL | FlowEdit (external RF) | 18 | 0.1760 +- 0.0370 | 0.2581 +- 0.0831 | 0.4133 +- 0.1516 | 0.4092 +- 0.2116 | 0.0482 +- 0.0469 |
| ALL | FlowAlign (external RF) | 18 | 0.0769 +- 0.0214 | 0.1263 +- 0.0426 | 0.6406 +- 0.1484 | 0.6900 +- 0.1349 | 0.0401 +- 0.0361 |
| ALL | SplitFlow (external RF) | 18 | 0.0965 +- 0.0250 | 0.1402 +- 0.0636 | 0.5094 +- 0.1446 | 0.6159 +- 0.2123 | 0.0427 +- 0.0457 |
| ALL | DeCE-RF | 18 | 0.0406 +- 0.0192 | 0.1038 +- 0.0525 | 0.6735 +- 0.1260 | 0.8497 +- 0.1256 | 0.0395 +- 0.0419 |
| cat_crown | FlowEdit (external RF) | 3 | 0.2069 +- 0.0016 | 0.3781 +- 0.0383 | 0.3813 +- 0.0202 | 0.1999 +- 0.0154 | 0.1040 +- 0.0112 |
| cat_crown | FlowAlign (external RF) | 3 | 0.0832 +- 0.0043 | 0.1438 +- 0.0182 | 0.5877 +- 0.0075 | 0.6495 +- 0.0221 | 0.0993 +- 0.0061 |
| cat_crown | SplitFlow (external RF) | 3 | 0.1224 +- 0.0031 | 0.2459 +- 0.0064 | 0.4317 +- 0.0187 | 0.4240 +- 0.0482 | 0.1081 +- 0.0066 |
| cat_crown | DeCE-RF | 3 | 0.0572 +- 0.0001 | 0.0547 +- 0.0000 | 0.6503 +- 0.0002 | 0.9675 +- 0.0002 | 0.0988 +- 0.0019 |
| bowl_apple_inside | FlowEdit (external RF) | 3 | 0.2168 +- 0.0178 | 0.3075 +- 0.0162 | 0.2592 +- 0.0362 | 0.2649 +- 0.0583 | 0.0191 +- 0.0061 |
| bowl_apple_inside | FlowAlign (external RF) | 3 | 0.0785 +- 0.0086 | 0.1750 +- 0.0235 | 0.5082 +- 0.0073 | 0.6006 +- 0.0263 | 0.0062 +- 0.0104 |
| bowl_apple_inside | SplitFlow (external RF) | 3 | 0.1107 +- 0.0108 | 0.1542 +- 0.0156 | 0.3860 +- 0.0176 | 0.6316 +- 0.0452 | 0.0114 +- 0.0049 |
| bowl_apple_inside | DeCE-RF | 3 | 0.0533 +- 0.0000 | 0.0860 +- 0.0002 | 0.5367 +- 0.0001 | 0.7740 +- 0.0008 | 0.0202 +- 0.0002 |
| tshirt_star | FlowEdit (external RF) | 3 | 0.1243 +- 0.0128 | 0.1537 +- 0.0406 | 0.7113 +- 0.0280 | 0.4858 +- 0.0496 | 0.0943 +- 0.0113 |
| tshirt_star | FlowAlign (external RF) | 3 | 0.0524 +- 0.0092 | 0.0977 +- 0.0153 | 0.8562 +- 0.0069 | 0.5710 +- 0.0102 | 0.0738 +- 0.0055 |
| tshirt_star | SplitFlow (external RF) | 3 | 0.0515 +- 0.0014 | 0.0505 +- 0.0336 | 0.8085 +- 0.0083 | 0.6295 +- 0.1188 | 0.0890 +- 0.0069 |
| tshirt_star | DeCE-RF | 3 | 0.0176 +- 0.0000 | 0.0824 +- 0.0003 | 0.8780 +- 0.0000 | 0.6198 +- 0.0022 | 0.0798 +- 0.0001 |
| red_chair_blue | FlowEdit (external RF) | 3 | 0.1634 +- 0.0254 | 0.2279 +- 0.0107 | 0.3317 +- 0.0717 | 0.5451 +- 0.1470 | 0.0566 +- 0.0299 |
| red_chair_blue | FlowAlign (external RF) | 3 | 0.0908 +- 0.0051 | 0.1301 +- 0.0032 | 0.4733 +- 0.0087 | 0.7970 +- 0.0695 | 0.0200 +- 0.0064 |
| red_chair_blue | SplitFlow (external RF) | 3 | 0.0927 +- 0.0031 | 0.1497 +- 0.0076 | 0.4261 +- 0.0040 | 0.8469 +- 0.0144 | 0.0454 +- 0.0038 |
| red_chair_blue | DeCE-RF | 3 | 0.0653 +- 0.0001 | 0.2005 +- 0.0002 | 0.5220 +- 0.0005 | 0.8820 +- 0.0090 | 0.0055 +- 0.0011 |
| pillow_vertical_fabric_strip | FlowEdit (external RF) | 3 | 0.1888 +- 0.0359 | 0.2767 +- 0.0853 | 0.3767 +- 0.0629 | 0.2234 +- 0.0492 | 0.0404 +- 0.0097 |
| pillow_vertical_fabric_strip | FlowAlign (external RF) | 3 | 0.1057 +- 0.0132 | 0.1564 +- 0.0009 | 0.6079 +- 0.0071 | 0.6031 +- 0.0761 | 0.0312 +- 0.0095 |
| pillow_vertical_fabric_strip | SplitFlow (external RF) | 3 | 0.0863 +- 0.0076 | 0.0931 +- 0.0109 | 0.5017 +- 0.0156 | 0.3136 +- 0.0968 | 0.0214 +- 0.0089 |
| pillow_vertical_fabric_strip | DeCE-RF | 3 | 0.0207 +- 0.0000 | 0.0605 +- 0.0003 | 0.7173 +- 0.0001 | 0.9710 +- 0.0015 | 0.0494 +- 0.0007 |
| backpack_remove_toy_charm | FlowEdit (external RF) | 3 | 0.1556 +- 0.0122 | 0.2045 +- 0.0086 | 0.4195 +- 0.0361 | 0.7362 +- 0.0685 | -0.0253 +- 0.0045 |
| backpack_remove_toy_charm | FlowAlign (external RF) | 3 | 0.0509 +- 0.0021 | 0.0548 +- 0.0035 | 0.8102 +- 0.0073 | 0.9186 +- 0.0084 | 0.0102 +- 0.0014 |
| backpack_remove_toy_charm | SplitFlow (external RF) | 3 | 0.1152 +- 0.0092 | 0.1477 +- 0.0105 | 0.5022 +- 0.0118 | 0.8500 +- 0.0043 | -0.0190 +- 0.0009 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0295 +- 0.0000 | 0.1390 +- 0.0001 | 0.7366 +- 0.0004 | 0.8842 +- 0.0009 | -0.0169 +- 0.0005 |

Interpretation:

- FlowEdit, FlowAlign, and SplitFlow are runnable on the revised strict set and are included in the reduced target-mode RF comparison.
- External target-mode RF baselines often form the requested target object/attribute, but visual audit shows source identity, crop/layout, and background drift.
- DeCE-RF has much lower outside-mask change and higher source preservation metrics under the same fixed task masks.
- This table should be worded as a reduced target-mode RF comparison, not as evidence that DeCE-RF beats every RF baseline.
