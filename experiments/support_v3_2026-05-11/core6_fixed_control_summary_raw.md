# Core-4 Fixed-Mask Quantitative Audit

Fixed per-task evaluation masks are reused across all methods and seeds.

| Task | Method | n | Mask | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| cat_crown | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0568 +- 0.0001 | 0.0884 +- 0.0000 | 0.6503 +- 0.0002 | 0.9667 +- 0.0002 | 0.0921 +- 0.0007 |
| cat_crown | DeCE-RF | 3 | fixed_eval_mask | 0.0559 +- 0.0001 | 0.0884 +- 0.0000 | 0.6504 +- 0.0001 | 0.9677 +- 0.0002 | 0.1018 +- 0.0014 |
| dog_sunglasses | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0538 +- 0.0000 | 0.1212 +- 0.0010 | 0.6227 +- 0.0002 | 0.9606 +- 0.0003 | 0.0896 +- 0.0005 |
| dog_sunglasses | DeCE-RF | 3 | fixed_eval_mask | 0.0534 +- 0.0001 | 0.1326 +- 0.0078 | 0.6236 +- 0.0003 | 0.9592 +- 0.0020 | 0.0842 +- 0.0111 |
| mug_heart | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0106 +- 0.0000 | 0.0079 +- 0.0000 | 0.9573 +- 0.0000 | 0.8074 +- 0.0001 | 0.0420 +- 0.0002 |
| mug_heart | DeCE-RF | 3 | fixed_eval_mask | 0.0105 +- 0.0001 | 0.0079 +- 0.0000 | 0.9574 +- 0.0002 | 0.8114 +- 0.0006 | 0.0425 +- 0.0006 |
| backpack_remove_toy_charm | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0403 +- 0.0001 | 0.2234 +- 0.0003 | 0.7392 +- 0.0005 | 0.8689 +- 0.0003 | -0.0180 +- 0.0004 |
| backpack_remove_toy_charm | DeCE-RF | 3 | fixed_eval_mask | 0.0397 +- 0.0000 | 0.2201 +- 0.0003 | 0.7369 +- 0.0004 | 0.8845 +- 0.0011 | -0.0164 +- 0.0004 |
| red_chair_blue | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0591 +- 0.0000 | 0.2241 +- 0.0006 | 0.5221 +- 0.0010 | 0.8896 +- 0.0007 | 0.0006 +- 0.0036 |
| red_chair_blue | DeCE-RF | 3 | fixed_eval_mask | 0.0591 +- 0.0001 | 0.2251 +- 0.0003 | 0.5220 +- 0.0004 | 0.8857 +- 0.0046 | 0.0065 +- 0.0024 |
| tshirt_star | Fixed DeCE displacement | 3 | fixed_eval_mask | 0.0175 +- 0.0000 | 0.0703 +- 0.0005 | 0.8773 +- 0.0001 | 0.6514 +- 0.0015 | 0.0772 +- 0.0002 |
| tshirt_star | DeCE-RF | 3 | fixed_eval_mask | 0.0175 +- 0.0000 | 0.0657 +- 0.0002 | 0.8780 +- 0.0000 | 0.6217 +- 0.0048 | 0.0799 +- 0.0006 |
