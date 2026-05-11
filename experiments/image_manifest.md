# Paper Image Manifest

Date: 2026-05-07

This manifest separates publication candidates from debug-only images. Use
`paper_use=main` for paper tables/figures. Use `debug_only` for images copied
from other projects or images whose license/source is not clean enough for
publication.

## Usage Labels

```text
main                  usable for paper figures/tables after attribution
supplement_or_replace candidate only; prefer replacing before main-paper use
debug_only            internal debugging only
exclude               do not use
```

## Publication Candidates

| image_id | path | source | license | allowed_use | task_type | source_prompt | target_prompt | paper_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cat_grass_01 | `data/paper_images/cat_sitting_in_grass.jpg` | Wikimedia Commons, `File:Cat_sitting_in_grass.jpg` | CC BY-SA 3.0 | attribution + share-alike required | headwear insertion | A cat is sitting in grass. | A cat wearing a small golden crown is sitting in grass. | main | Good T3 candidate. Need attribution in paper/supplement. |
| cat_grass_02 | `data/paper_images/cat_on_grass.jpg` | Wikimedia Commons, `File:Cat_on_grass.jpg` | CC BY-SA 3.0 | attribution + share-alike required | headwear insertion / backup local edit | A cat is standing on grass. | A cat wearing a small golden crown is standing on grass. | main | Backup cat/crown candidate if cat_grass_01 composition is hard. |
| rabbit_side_01 | `data/paper_images/rabbit_side_view.jpg` | Wikimedia Commons, `File:Rabbit_side_view.JPG` | public domain | attribution not required but still cite source | side-profile accessory robustness | A rabbit is sitting outdoors in side profile. | A rabbit wearing small black sunglasses is sitting outdoors in side profile. | main | Good T4 robustness/failure case. |
| yellow_car_01 | `data/paper_images/yellow_car_side_unsplash.jpg` | Wikimedia Commons, `File:Yellow_car_side_(Unsplash).jpg` | CC0 | unrestricted; cite source anyway | color / attribute edit | A yellow car is parked on a street. | A blue car is parked on the same street. | main | Strong replacement for debug-only bus color edit. Vehicle surface is large and simple enough for first color task. |
| backpack_rocks_01 | `data/paper_images/herschel_backpack_by_rocks_unsplash.jpg` | Wikimedia Commons, `File:Herschel_backpack_by_rocks_(Unsplash).jpg` | CC0 | unrestricted; cite source anyway | color / attribute edit | A burgundy backpack is sitting on rocks outdoors. | A blue backpack is sitting on the same rocks outdoors. | main | Good non-vehicle color task; useful for testing general surface recolor beyond cars. |
| red_chair_01 | `data/paper_images/red_chair_cc0.jpg` | Wikimedia Commons, `File:Red_chair.jpg` | CC0 | unrestricted; cite source anyway | color / attribute edit | A red chair is indoors. | A blue chair is indoors. | main | Simple object-color task with clean license; useful as a sanity check. |
| dog_sitting_01 | `data/paper_images/dog_sitting_cc0.jpg` | Wikimedia Commons, `File:Dog-sitting.jpg` | CC0 | unrestricted; cite source anyway | local accessory / headwear insertion | A dog is sitting. | A dog wearing a small golden crown is sitting. | main | Low-resolution but legally clean local insertion candidate. Prefer replacing if quality is too low. |
| woman_portrait_01 | `data/paper_images/woman_portrait_50062201233.jpg` | Wikimedia Commons, `File:Woman_portrait_(50062201233).jpg` | CC BY 2.0 | attribution required; personality rights warning applies | local accessory insertion | A woman portrait without sunglasses. | A woman wearing black sunglasses. | supplement_or_replace | Do not use as main figure unless personality-rights risk is accepted or replaced by a safer image. |

## Existing Debug Images

| image_id | path | source | license | allowed_use | task_type | source_prompt | target_prompt | paper_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| panda_old_debug | `/home/Wu_25R8111/FlowEdit/Data/Images/panda.png` or existing panda assets | copied from existing editing projects / local assets | unclear in this project | debug continuity only | local accessory / object replacement | A panda is walking in a forest. | A panda wearing small black sunglasses is walking in a forest. | debug_only | Keep for continuity and ablation debugging, not main publication evidence. |
| bus_old_debug | `/home/Wu_25R8111/FlowEdit/Data/Images/bus.png` | copied from existing editing projects / local assets | unclear in this project | debug continuity only | color / attribute edit | A yellow vintage minibus is parked in a driveway. | A blue vintage minibus is parked in a driveway. | debug_only | Keep for color-edit debugging, not main publication evidence unless source/license is verified. |

## Immediate Gaps

- [x] Add a clean `main` image for local accessory insertion that does not carry
      personality-rights ambiguity. Prefer an animal or object-based accessory
      task if no safe human portrait is available.
- [x] Add a clean `main` image for clothing/bag color editing. This is needed
      because the current bus image is debug-only.
- [ ] Add exact attribution text for every `main` image before manuscript
      drafting.
