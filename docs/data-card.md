# Data card: CCT20 as used by Who Goes There

Written 2026-10-11 (item 3.04) from `python -m trailcam.data.metadata` and
`python -m trailcam.data.split --seed 42` run on the released annotation files; the day and night
numbers and the figures were added 2026-10-15 (item 8.07). Counts, shapes and class distributions
only: no image, sequence or file name appears here. Design and the reasons behind each rule:
`docs/plan.md` sections 2 and 3. Raw facts before filtering: `docs/data-inspection.md`.

## Source, licence and citation

Caltech Camera Traps, CCT20 benchmark subset, from LILA BC
(https://lila.science/datasets/caltech-camera-traps): 57,864 images from 20 camera locations in
the American Southwest, five COCO-style annotation files (train, cis_val, trans_val, cis_test,
trans_test) and the benchmark tarball `eccv_18_all_images_sm.tar.gz` (images resized to at most
1024 px per side). Originals are 2048 x 1494 or 2560 x 1920; every image has `location`, `seq_id`,
`frame_num`, `seq_num_frames`, `date_captured`, `width` and `height`.

Licence: Community Data License Agreement, Permissive, version 1.0
(https://cdla.dev/permissive-1-0/). The images and the metadata are not in this repository; the
only dataset images in it are the demo photos in `assets/` (see the last section). Wherever the
data or a photo from it is shown, cite:

> Sara Beery, Grant Van Horn, Pietro Perona. "Recognition in Terra Incognita." Proceedings of the
> European Conference on Computer Vision (ECCV), 2018. https://arxiv.org/abs/1807.04975

## The 12 classes

16 categories in the metadata, with non-contiguous ids; 12 are kept. The index order is the
official train count order and is defined once in `trailcam/data/classes.py`.

| index | class | CCT id |
| --- | --- | --- |
| 0 | opossum | 1 |
| 1 | rabbit | 10 |
| 2 | coyote | 9 |
| 3 | cat | 16 |
| 4 | squirrel | 5 |
| 5 | raccoon | 3 |
| 6 | dog | 8 |
| 7 | bobcat | 6 |
| 8 | car | 33 |
| 9 | bird | 11 |
| 10 | rodent | 99 |
| 11 | skunk | 7 |

## Dropped classes and rules

Applied per image to all five files, in this order (an image is counted under the first rule that
drops it):

1. **two_species**: the image carries two different species (62 of the 78 are bobcat + rabbit).
   Images with several annotations of one species (407 in official train) keep that label.
2. **empty** (CCT id 30): zero images in official train, so it cannot be learned from the files as
   given; filtered out of every split.
3. **rare**: badger (21), fox (51) and deer (34). Official train holds 9, 5 and 42 of them; deer
   has none in trans_test.
4. **unannotated**: an image with no annotation; none exist in the released files.

Images dropped per official file:

| rule | train | cis_val | trans_val | cis_test | trans_test | all |
| --- | --- | --- | --- | --- | --- | --- |
| two_species | 7 | 0 | 61 | 1 | 9 | 78 |
| empty | 0 | 995 | 61 | 1180 | 1778 | 4014 |
| rare | 50 | 4 | 0 | 168 | 15 | 237 |
| unannotated | 0 | 0 | 0 | 0 | 0 | 0 |
| kept | 13496 | 2485 | 1603 | 14478 | 21473 | 53535 |
| released | 13553 | 3484 | 1725 | 15827 | 23275 | 57864 |

Train's rare count is 50, not 9 + 5 + 42 = 56, because six badger images also carry a coyote and
fall under two_species first. trans_val loses 61 of its 70 rabbit images to two_species.

## Counts per split after filtering and re-splitting

The official train and cis_val files are pooled (17,037 images released, 15,981 kept) and re-split
by `seq_id` into train and val, about 85 to 15, stratified by the sequence's majority class and by
location (`trailcam/data/split.py`). cis_test, trans_val and trans_test pass through as filtered.
Seed 42:

| class | train | val | trans_val | cis_test | trans_test | all |
| --- | --- | --- | --- | --- | --- | --- |
| opossum | 2733 | 483 | 444 | 4524 | 5504 | 13688 |
| rabbit | 2494 | 442 | 9 | 1757 | 843 | 5545 |
| coyote | 1437 | 251 | 49 | 1326 | 2249 | 5312 |
| cat | 1257 | 222 | 72 | 1632 | 1418 | 4601 |
| squirrel | 1250 | 221 | 0 | 659 | 1050 | 3180 |
| raccoon | 1016 | 180 | 129 | 1047 | 5469 | 7841 |
| dog | 817 | 142 | 96 | 845 | 888 | 2788 |
| bobcat | 799 | 140 | 734 | 897 | 2329 | 4899 |
| car | 603 | 106 | 0 | 791 | 1113 | 2613 |
| bird | 509 | 89 | 7 | 573 | 224 | 1402 |
| rodent | 440 | 78 | 0 | 233 | 58 | 809 |
| skunk | 232 | 40 | 63 | 194 | 328 | 857 |
| **images** | 13587 | 2394 | 1603 | 14478 | 21473 | 53535 |
| sequences | 5334 | 940 | 538 | 5882 | 8258 | 20952 |
| locations | 10 | 10 | 1 | 10 | 9 | 20 |

- val holds 2,394 of the 15,981 pool images (0.150) and 940 of the 6,274 pool sequences (0.150);
  per pool location the val share of sequences is 0.147 to 0.154. The smallest val class is skunk
  with 40 images (the floor is 30).
- 3 pool sequences carry more than one label among their kept frames (0 majority ties); each image
  keeps its own label.
- train and val share all 10 cis locations; the 10 trans locations (1 in trans_val, 9 in
  trans_test) appear in no cis split.
- trans_val covers 9 of the 12 classes (no squirrel, car or rodent); it is a sanity check only and
  is reported with macro-F1 over the classes present.
- Imbalance: opossum is 20% of train; opossum and raccoon are 51% of trans_test after filtering.

## Bounding boxes

Boxes are stored as fractions of the original image size, clipped to the frame; they are
training-time augmentation (arm B) and attention scoring only, never a model input.

| split | images with at least one box | kept images | boxed fraction | boxes |
| --- | --- | --- | --- | --- |
| train | 11633 | 13587 | 0.856 | 12177 |
| val | 2072 | 2394 | 0.865 | 2133 |
| trans_val | 1472 | 1603 | 0.918 | 1551 |
| cis_test | 12551 | 14478 | 0.867 | 13116 |
| trans_test | 18010 | 21473 | 0.839 | 18752 |
| all | 45738 | 53535 | 0.854 | 47729 |

By official file the boxed fraction is 0.892 for train and 0.669 for cis_val (1,819 of its 3,582
annotations lack a box); the re-split mixes the two. 100 of the 47,729 kept boxes ran past the
frame and were clipped (33 train, 3 cis_val, 2 trans_val, 11 cis_test, 51 trans_test, by official
file).

## Day and night

The flag is computed from the pixels during the Kaggle resize (item 6.01,
`scripts/kaggle_prepare_cct20.py`: infrared frames are grayscale, saved single-band or with
near-identical channels) and written to the `day_night.json` sidecar of the item 6.03 cache. A
copy sits in the main checkout's `data/` (gitignored, never committed). Added 2026-10-15, item
8.07.

**Kept images per split** (the sidecar's flags, after the drop rules, split seed 42):

| split | night | images | night share |
| --- | --- | --- | --- |
| train | 8246 | 13587 | 0.607 |
| val | 1436 | 2394 | 0.600 |
| trans_val | 1014 | 1603 | 0.633 |
| cis_test | 9066 | 14478 | 0.626 |
| trans_test | 15384 | 21473 | 0.716 |
| all kept | 35146 | 53535 | 0.657 |

**All annotated images per official file** (the numbers the 6.03 cache build printed, over all
57,864 images, before the drop rules; not our split):

| official file | night | images | night share |
| --- | --- | --- | --- |
| train | 8110 | 13553 | 0.598 |
| cis_val | 2048 | 3484 | 0.588 |
| trans_val | 1050 | 1725 | 0.609 |
| cis_test | 9562 | 15827 | 0.604 |
| trans_test | 16006 | 23275 | 0.688 |
| all | 36776 | 57864 | 0.636 |

- About six in ten kept images are infrared night frames, and trans_test is the most night-heavy
  split (0.716 against 0.600 to 0.626 for the cis splits): the trans cameras are judged more on
  night frames than the model trains on (the day/night slices, item 8.04, measure the effect).
- The drop rules raise the night share (0.636 to 0.657): of the 4,329 dropped images only 1,630
  (0.377) are night frames.
- The gap is clean: over all annotated images 36,776 have a channel spread in [0, 0.5), none in
  [0.5, 2) and 4 in [2, 4), against the threshold 2.0. 551 source images were single-band (mode L).

Kept images per capture hour, all splits, with the night frames among them:

| hours | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| images | 197 | 3097 | 3329 | 3401 | 3288 | 2824 | 1978 | 2375 | 2631 | 2398 | 1958 | 1769 |
| night | 32 | 2970 | 3217 | 3281 | 3205 | 2729 | 1347 | 897 | 318 | 117 | 57 | 99 |

| hours | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| images | 4525 | 1124 | 1119 | 903 | 965 | 981 | 1162 | 1919 | 2773 | 3135 | 2664 | 3020 |
| night | 2990 | 52 | 84 | 78 | 60 | 239 | 596 | 1673 | 2624 | 3013 | 2550 | 2918 |

All annotated images per capture hour, before the drop rules (the 6.03 build's printout):

| hours | 00 | 01 | 02 | 03 | 04 | 05 | 06 | 07 | 08 | 09 | 10 | 11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| images | 203 | 3247 | 3485 | 3497 | 3420 | 2958 | 2162 | 2615 | 2810 | 2590 | 2168 | 2038 |
| night | 37 | 3087 | 3360 | 3370 | 3333 | 2850 | 1450 | 959 | 341 | 124 | 58 | 105 |

| hours | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| images | 5077 | 1492 | 1399 | 1103 | 1088 | 1086 | 1227 | 2030 | 2913 | 3272 | 2800 | 3184 |
| night | 3120 | 58 | 87 | 78 | 69 | 263 | 618 | 1758 | 2762 | 3144 | 2671 | 3074 |

**The hour-12 anomaly.** Hour 12 holds 4,525 kept images, 2,990 of them night frames, against
1,769 (99 night) at hour 11 and 1,124 (52 night) at hour 13; over all annotated images, 5,077 with
3,120 night. About 3,000 infrared frames carry a noon timestamp, so some camera clocks are wrong
and the clock cannot stand in for the light. The flag comes from the pixels.

## Figures

Five charts of this card's data, drawn from the metadata alone (no photo) by
`trailcam/data/datavis.py` (item 8.07), each with the citation above, in `docs/figures/`:

- `class_counts.png`: each class's share of each split, with its count.
- `camera_locations.png`: images per camera, cis vs trans, cameras ranked by image count (location
  ids are not shown).
- `day_night.png`: the night share of the kept images per split and the kept images per capture
  hour by day and night, with hour 12 marked (drawn with the sidecar).
- `box_fill.png`: the share of the image each box covers, log scale, cis vs trans, with arm B's 20%
  to 100% fill range shaded.
- `burst_length.png`: sequences and kept images by burst length, and the 222 bursts the official
  train and cis_val files share.

Regenerate: `uv run python -m trailcam.data.datavis --root <data dir> --seed 42 --out docs/figures`
with `--sidecar <data dir>/day_night.json` for the night shares; without it the day/night chart
falls back to the 6.03 build's numbers over all annotated images, and says so.

## Split seed and hash

| seed | split hash (SHA-256 over split, image id and label, sorted by image id) |
| --- | --- |
| 42 | `c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009` |

The split is not stored as a file: `uv run python -m trailcam.data.split --root <data dir> --seed 42`
rebuilds it and prints this hash. `tests/test_split.py` pins the counts and the hash above, and
every training run records the hash in `docs/ledger.md`.

## Where the paper and the released files disagree

1. **Image count.** The paper gives 57,868 images; the released files hold 57,864. The four
   splits the paper counts match the files exactly (train 13,553, cis_val 3,484, cis_test 15,827,
   trans_test 23,275), and with the 1,725 trans_val images they sum to 57,864, so the four-image
   discrepancy is in the paper's total, not in the files.
2. **Shared sequences.** The paper says no sequence is split across train and cis_val; the
   released files share 224 sequences between them (222 after filtering still have frames in both
   files). This breaks the one-burst-one-split invariant, which is why the two files are pooled and
   re-split by sequence. cis_test, trans_val and trans_test share no sequence with anything.

## Trans-test locations used for assets/

**Pending item 8.05.** The 6 to 12 demo photos in `assets/` (the Gradio examples and the only
images Grad-CAM overlays are made from) are chosen by the user in item 8.05, from trans_test
locations only, with at least two infrared night frames. Their locations are listed here then.
None are chosen yet; `assets/` does not exist.
