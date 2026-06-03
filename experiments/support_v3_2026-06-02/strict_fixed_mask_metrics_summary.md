# Strict Core-6 Fixed-Mask Quantitative Audit

Fixed per-task evaluation masks are reused across all methods and seeds.

| Task | Method | n | Mask | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cat_crown | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0815 +- 0.0000 | 0.0916 +- 0.0000 | 0.5122 +- 0.0000 | 0.5048 +- 0.0000 | -0.0042 +- 0.0000 |
| cat_crown | Direct target guidance | 3 | fixed_eval_mask | 0.1246 +- 0.0000 | 0.1228 +- 0.0000 | 0.4358 +- 0.0000 | 0.4225 +- 0.0000 | 0.0007 +- 0.0000 |
| cat_crown | Generic support control | 3 | fixed_eval_mask | 0.0582 +- 0.0004 | 0.0574 +- 0.0007 | 0.6575 +- 0.0002 | 0.8458 +- 0.0074 | 0.0022 +- 0.0010 |
| cat_crown | DeCE-RF | 3 | fixed_eval_mask | 0.0572 +- 0.0001 | 0.0547 +- 0.0000 | 0.6503 +- 0.0002 | 0.9675 +- 0.0002 | 0.0988 +- 0.0019 |
| bowl_apple_inside | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0812 +- 0.0000 | 0.0426 +- 0.0000 | 0.4160 +- 0.0000 | 0.4853 +- 0.0000 | -0.0083 +- 0.0000 |
| bowl_apple_inside | Direct target guidance | 3 | fixed_eval_mask | 0.0925 +- 0.0000 | 0.0645 +- 0.0000 | 0.3450 +- 0.0000 | 0.7195 +- 0.0000 | 0.0054 +- 0.0000 |
| bowl_apple_inside | Generic support control | 3 | fixed_eval_mask | 0.0620 +- 0.0001 | 0.1435 +- 0.0043 | 0.5113 +- 0.0011 | 0.7290 +- 0.0190 | 0.0125 +- 0.0047 |
| bowl_apple_inside | DeCE-RF | 3 | fixed_eval_mask | 0.0533 +- 0.0000 | 0.0860 +- 0.0002 | 0.5367 +- 0.0001 | 0.7740 +- 0.0008 | 0.0202 +- 0.0002 |
| tshirt_star | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0218 +- 0.0000 | 0.0130 +- 0.0000 | 0.8889 +- 0.0000 | 0.9379 +- 0.0000 | -0.0070 +- 0.0000 |
| tshirt_star | Direct target guidance | 3 | fixed_eval_mask | 0.0410 +- 0.0000 | 0.0148 +- 0.0000 | 0.8446 +- 0.0000 | 0.8672 +- 0.0000 | -0.0006 +- 0.0000 |
| tshirt_star | Generic support control | 3 | fixed_eval_mask | 0.0286 +- 0.0000 | 0.0104 +- 0.0000 | 0.8648 +- 0.0001 | 0.9100 +- 0.0020 | -0.0092 +- 0.0004 |
| tshirt_star | DeCE-RF | 3 | fixed_eval_mask | 0.0176 +- 0.0000 | 0.0824 +- 0.0003 | 0.8780 +- 0.0000 | 0.6198 +- 0.0022 | 0.0798 +- 0.0001 |
| red_chair_blue | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.1247 +- 0.0000 | 0.0855 +- 0.0000 | 0.3246 +- 0.0000 | 0.5454 +- 0.0000 | -0.0244 +- 0.0000 |
| red_chair_blue | Direct target guidance | 3 | fixed_eval_mask | 0.1203 +- 0.0000 | 0.0918 +- 0.0000 | 0.3275 +- 0.0000 | 0.5603 +- 0.0000 | -0.0226 +- 0.0000 |
| red_chair_blue | Generic support control | 3 | fixed_eval_mask | 0.0652 +- 0.0000 | 0.1999 +- 0.0000 | 0.5210 +- 0.0002 | 0.8851 +- 0.0048 | -0.0016 +- 0.0026 |
| red_chair_blue | DeCE-RF | 3 | fixed_eval_mask | 0.0653 +- 0.0001 | 0.2005 +- 0.0002 | 0.5220 +- 0.0005 | 0.8820 +- 0.0090 | 0.0055 +- 0.0011 |
| pillow_vertical_fabric_strip | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0451 +- 0.0000 | 0.0533 +- 0.0000 | 0.5897 +- 0.0000 | 0.5551 +- 0.0000 | 0.0040 +- 0.0000 |
| pillow_vertical_fabric_strip | Direct target guidance | 3 | fixed_eval_mask | 0.0747 +- 0.0000 | 0.3339 +- 0.0000 | 0.4945 +- 0.0000 | 0.3051 +- 0.0000 | 0.1082 +- 0.0000 |
| pillow_vertical_fabric_strip | Generic support control | 3 | fixed_eval_mask | 0.0255 +- 0.0000 | 0.0851 +- 0.0002 | 0.6889 +- 0.0002 | 0.6822 +- 0.0057 | 0.0449 +- 0.0056 |
| pillow_vertical_fabric_strip | DeCE-RF | 3 | fixed_eval_mask | 0.0207 +- 0.0000 | 0.0605 +- 0.0003 | 0.7173 +- 0.0001 | 0.9710 +- 0.0015 | 0.0494 +- 0.0007 |
| backpack_remove_toy_charm | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0612 +- 0.0000 | 0.0892 +- 0.0000 | 0.6146 +- 0.0000 | 0.8077 +- 0.0000 | 0.0056 +- 0.0000 |
| backpack_remove_toy_charm | Direct target guidance | 3 | fixed_eval_mask | 0.0697 +- 0.0000 | 0.1442 +- 0.0000 | 0.5619 +- 0.0000 | 0.7595 +- 0.0000 | -0.0078 +- 0.0000 |
| backpack_remove_toy_charm | Generic support control | 3 | fixed_eval_mask | 0.0280 +- 0.0000 | 0.0800 +- 0.0000 | 0.7956 +- 0.0000 | 0.9471 +- 0.0005 | -0.0078 +- 0.0001 |
| backpack_remove_toy_charm | DeCE-RF | 3 | fixed_eval_mask | 0.0295 +- 0.0000 | 0.1390 +- 0.0001 | 0.7366 +- 0.0004 | 0.8842 +- 0.0009 | -0.0169 +- 0.0005 |
