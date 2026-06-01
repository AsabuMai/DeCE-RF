# Core-6 Fixed-Control Ablation Summary

Fixed per-task evaluation masks are reused across all methods and seeds. Values are means over seeds 10, 11, and 12.

| Task | Method | n | Outside L1 | Inside L1 | Source SSIM | DINO/source | Edit score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cat_crown | Fixed DeCE displacement | 3 | 0.0540 | 0.1838 | 0.6503 | 0.9667 | 0.0921 |
| cat_crown | DeCE-RF | 3 | 0.0542 | 0.1477 | 0.6504 | 0.9677 | 0.1018 |
| dog_sunglasses | Fixed DeCE displacement | 3 | 0.0487 | 0.1179 | 0.6227 | 0.9606 | 0.0896 |
| dog_sunglasses | DeCE-RF | 3 | 0.0488 | 0.1192 | 0.6236 | 0.9592 | 0.0842 |
| mug_heart | Fixed DeCE displacement | 3 | 0.0077 | 0.0749 | 0.9573 | 0.8074 | 0.0420 |
| mug_heart | DeCE-RF | 3 | 0.0076 | 0.0728 | 0.9574 | 0.8114 | 0.0425 |
| tshirt_star | Fixed DeCE displacement | 3 | 0.0175 | 0.0703 | 0.8773 | 0.6514 | 0.0772 |
| tshirt_star | DeCE-RF | 3 | 0.0175 | 0.0657 | 0.8780 | 0.6217 | 0.0799 |
| backpack_remove_toy_charm | Fixed DeCE displacement | 3 | 0.0403 | 0.2234 | 0.7392 | 0.8689 | -0.0180 |
| backpack_remove_toy_charm | DeCE-RF | 3 | 0.0397 | 0.2201 | 0.7369 | 0.8845 | -0.0164 |
| red_chair_blue | Fixed DeCE displacement | 3 | 0.0591 | 0.2241 | 0.5221 | 0.8896 | 0.0006 |
| red_chair_blue | DeCE-RF | 3 | 0.0591 | 0.2251 | 0.5220 | 0.8857 | 0.0065 |
