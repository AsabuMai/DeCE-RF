# Core-4 Fixed-Mask Quantitative Audit

Fixed per-task evaluation masks are reused across all methods and seeds.

| Task | Method | n | Mask | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cat_crown | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0812 +- 0.0000 | 0.1207 +- 0.0000 | 0.5122 +- 0.0000 | 0.5048 +- 0.0000 | -0.0042 +- 0.0000 |
| cat_crown | Direct target guidance | 3 | fixed_eval_mask | 0.1150 +- 0.0000 | 0.4203 +- 0.0000 | 0.4358 +- 0.0000 | 0.4225 +- 0.0000 | 0.0007 +- 0.0000 |
| cat_crown | Generic support control | 3 | fixed_eval_mask | 0.0554 +- 0.0002 | 0.1422 +- 0.0012 | 0.6577 +- 0.0008 | 0.8281 +- 0.0089 | -0.0020 +- 0.0050 |
| cat_crown | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0568 +- 0.0001 | 0.0884 +- 0.0000 | 0.6503 +- 0.0002 | 0.9667 +- 0.0002 | 0.0921 +- 0.0007 |
| cat_crown | DeCE-RF | 3 | fixed_eval_mask | 0.0559 +- 0.0001 | 0.0884 +- 0.0000 | 0.6504 +- 0.0001 | 0.9677 +- 0.0002 | 0.1018 +- 0.0014 |
| dog_sunglasses | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0662 +- 0.0000 | 0.0551 +- 0.0000 | 0.6137 +- 0.0000 | 0.9067 +- 0.0000 | -0.0251 +- 0.0000 |
| dog_sunglasses | Direct target guidance | 3 | fixed_eval_mask | 0.1031 +- 0.0000 | 0.4330 +- 0.0000 | 0.5465 +- 0.0000 | 0.6598 +- 0.0000 | 0.0581 +- 0.0000 |
| dog_sunglasses | Generic support control | 3 | fixed_eval_mask | 0.0536 +- 0.0000 | 0.0581 +- 0.0003 | 0.6338 +- 0.0001 | 0.9279 +- 0.0011 | 0.0966 +- 0.0007 |
| dog_sunglasses | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0538 +- 0.0000 | 0.1212 +- 0.0010 | 0.6227 +- 0.0002 | 0.9606 +- 0.0003 | 0.0896 +- 0.0005 |
| dog_sunglasses | DeCE-RF | 3 | fixed_eval_mask | 0.0534 +- 0.0001 | 0.1326 +- 0.0078 | 0.6236 +- 0.0003 | 0.9592 +- 0.0020 | 0.0842 +- 0.0111 |
| mug_heart | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0129 +- 0.0000 | 0.0359 +- 0.0000 | 0.9590 +- 0.0000 | 0.8820 +- 0.0000 | -0.0287 +- 0.0000 |
| mug_heart | Direct target guidance | 3 | fixed_eval_mask | 0.0246 +- 0.0000 | 0.2839 +- 0.0000 | 0.9138 +- 0.0000 | 0.8306 +- 0.0000 | -0.0276 +- 0.0000 |
| mug_heart | Generic support control | 3 | fixed_eval_mask | 0.0081 +- 0.0000 | 0.0080 +- 0.0000 | 0.9654 +- 0.0000 | 0.9599 +- 0.0003 | -0.0274 +- 0.0001 |
| mug_heart | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0106 +- 0.0000 | 0.0079 +- 0.0000 | 0.9573 +- 0.0000 | 0.8074 +- 0.0001 | 0.0420 +- 0.0002 |
| mug_heart | DeCE-RF | 3 | fixed_eval_mask | 0.0105 +- 0.0001 | 0.0079 +- 0.0000 | 0.9574 +- 0.0002 | 0.8114 +- 0.0006 | 0.0425 +- 0.0006 |
| backpack_remove_toy_charm | RF reconstruction / base reconstruction | 3 | fixed_eval_mask | 0.0657 +- 0.0000 | 0.0892 +- 0.0000 | 0.6146 +- 0.0000 | 0.8077 +- 0.0000 | 0.0056 +- 0.0000 |
| backpack_remove_toy_charm | Direct target guidance | 3 | fixed_eval_mask | 0.0755 +- 0.0000 | 0.2125 +- 0.0000 | 0.5619 +- 0.0000 | 0.7595 +- 0.0000 | -0.0078 +- 0.0000 |
| backpack_remove_toy_charm | Generic support control | 3 | fixed_eval_mask | 0.0354 +- 0.0000 | 0.0913 +- 0.0001 | 0.7960 +- 0.0000 | 0.9477 +- 0.0006 | -0.0077 +- 0.0003 |
| backpack_remove_toy_charm | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0403 +- 0.0001 | 0.2234 +- 0.0003 | 0.7392 +- 0.0005 | 0.8689 +- 0.0003 | -0.0180 +- 0.0004 |
| backpack_remove_toy_charm | DeCE-RF | 3 | fixed_eval_mask | 0.0397 +- 0.0000 | 0.2201 +- 0.0003 | 0.7369 +- 0.0004 | 0.8845 +- 0.0011 | -0.0164 +- 0.0004 |
