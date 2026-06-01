# Core-6 Fixed-Mask Quantitative Audit

Fixed per-task evaluation masks are reused across all methods and seeds. Values are means over seeds 10, 11, and 12.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score | Inside blue |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| cat_crown | RF reconstruction / base reconstruction | 3 | 0.0831 | 0.0596 | 0.5122 | 0.5048 | -0.0042  0.0002 |
| cat_crown | Direct target guidance | 3 | 0.1251 | 0.1016 | 0.4358 | 0.4225 | 0.0007  0.0000 |
| cat_crown | Generic support control | 3 | 0.0595 | 0.0115 | 0.6577 | 0.8281 | -0.0020  0.1119 |
| cat_crown | DeCE-RF | 3 | 0.0542 | 0.1477 | 0.6504 | 0.9677 | 0.1018  0.0156 |
| dog_sunglasses | RF reconstruction / base reconstruction | 3 | 0.0615 | 0.0960 | 0.6137 | 0.9067 | -0.0251  0.0197 |
| dog_sunglasses | Direct target guidance | 3 | 0.1055 | 0.2187 | 0.5465 | 0.6598 | 0.0581  0.2859 |
| dog_sunglasses | Generic support control | 3 | 0.0422 | 0.1385 | 0.6338 | 0.9279 | 0.0966  0.1174 |
| dog_sunglasses | DeCE-RF | 3 | 0.0488 | 0.1192 | 0.6236 | 0.9592 | 0.0842  0.2181 |
| mug_heart | RF reconstruction / base reconstruction | 3 | 0.0142 | 0.0066 | 0.9590 | 0.8820 | -0.0287  0.0059 |
| mug_heart | Direct target guidance | 3 | 0.0349 | 0.0424 | 0.9138 | 0.8306 | -0.0276  0.0000 |
| mug_heart | Generic support control | 3 | 0.0082 | 0.0056 | 0.9654 | 0.9599 | -0.0274  0.0115 |
| mug_heart | DeCE-RF | 3 | 0.0076 | 0.0728 | 0.9574 | 0.8114 | 0.0425  0.0000 |
| tshirt_star | RF reconstruction / base reconstruction | 3 | 0.0220 | 0.0128 | 0.8889 | 0.9379 | -0.0070  0.5595 |
| tshirt_star | Direct target guidance | 3 | 0.0417 | 0.0152 | 0.8446 | 0.8672 | -0.0006  0.5808 |
| tshirt_star | Generic support control | 3 | 0.0292 | 0.0101 | 0.8648 | 0.9089 | -0.0094  0.3947 |
| tshirt_star | DeCE-RF | 3 | 0.0175 | 0.0657 | 0.8780 | 0.6217 | 0.0799  0.4941 |
| backpack_remove_toy_charm | RF reconstruction / base reconstruction | 3 | 0.0657 | 0.0892 | 0.6146 | 0.8077 | 0.0056  0.0000 |
| backpack_remove_toy_charm | Direct target guidance | 3 | 0.0755 | 0.2125 | 0.5619 | 0.7595 | -0.0078  0.0000 |
| backpack_remove_toy_charm | Generic support control | 3 | 0.0354 | 0.0913 | 0.7960 | 0.9477 | -0.0077  0.0000 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0397 | 0.2201 | 0.7369 | 0.8845 | -0.0164  0.0000 |
| red_chair_blue | RF reconstruction / base reconstruction | 3 | 0.1244 | 0.0874 | 0.3246 | 0.5454 | -0.0244  0.0000 |
| red_chair_blue | Direct target guidance | 3 | 0.1194 | 0.0959 | 0.3275 | 0.5603 | -0.0226  0.0000 |
| red_chair_blue | Generic support control | 3 | 0.0590 | 0.2238 | 0.5213 | 0.8872 | 0.0019  0.9036 |
| red_chair_blue | DeCE-RF | 3 | 0.0591 | 0.2251 | 0.5220 | 0.8857 | 0.0065  0.8981 |
