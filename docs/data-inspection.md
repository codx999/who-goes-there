# CCT20 annotation inspection

Generated 2026-10-10 from `eccv_18_annotations.tar.gz` (LILA, GCP mirror). Counts and shapes only.

## Categories

16 categories, identical across all five files.

| id | name |
| --- | --- |
| 1 | opossum |
| 3 | raccoon |
| 5 | squirrel |
| 6 | bobcat |
| 7 | skunk |
| 8 | dog |
| 9 | coyote |
| 10 | rabbit |
| 11 | bird |
| 16 | cat |
| 21 | badger |
| 30 | empty |
| 33 | car |
| 34 | deer |
| 51 | fox |
| 99 | rodent |

## Split sizes

| split | images | annotations | published | match | locations | sequences |
| --- | --- | --- | --- | --- | --- | --- |
| train | 13553 | 14071 | 13553 | yes | 10 | 5527 |
| cis_val | 3484 | 3582 | 3484 | yes | 10 | 1421 |
| trans_val | 1725 | 1865 | n/a | n/a | 1 | 589 |
| cis_test | 15827 | 16395 | 15827 | yes | 10 | 6424 |
| trans_test | 23275 | 24028 | 23275 | yes | 9 | 8982 |

## Overlaps between splits

| a | b | shared images | shared sequences | shared locations |
| --- | --- | --- | --- | --- |
| train | cis_val | 0 | 224 | 10 |
| train | trans_val | 0 | 0 | 0 |
| train | cis_test | 0 | 0 | 10 |
| train | trans_test | 0 | 0 | 0 |
| cis_val | trans_val | 0 | 0 | 0 |
| cis_val | cis_test | 0 | 0 | 10 |
| cis_val | trans_test | 0 | 0 | 0 |
| trans_val | cis_test | 0 | 0 | 0 |
| trans_val | trans_test | 0 | 0 | 0 |
| cis_test | trans_test | 0 | 0 | 0 |

Cis locations (train, cis_val, cis_test): 10. Trans locations (trans_val, trans_test): 10. Shared: 0. Total: 20.

train vs cis_val share locations: 10; train vs cis_test: 10.

## Labels per image

| split | images with 0 ann | with 1 ann | with >1 ann | >1 distinct species | empty-labelled | ann with bbox | ann without bbox |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0 | 13139 | 414 | 7 | 0 | 12617 | 1454 |
| cis_val | 0 | 3408 | 76 | 0 | 995 | 1763 | 1819 |
| trans_val | 0 | 1605 | 120 | 61 | 61 | 1673 | 192 |
| cis_test | 0 | 15469 | 358 | 1 | 1180 | 13264 | 3131 |
| trans_test | 0 | 22626 | 649 | 9 | 1778 | 18786 | 5242 |

Distinct-species combinations in multi-species images (all splits):

- bobcat + rabbit: 62
- badger + coyote: 6
- bobcat + rodent: 4
- bobcat + squirrel: 2
- coyote + rabbit: 2
- bird + rabbit: 1
- bobcat + opossum: 1

## Images per class per split

| class | train | cis_val | trans_val | cis_test | trans_test |
| --- | --- | --- | --- | --- | --- |
| opossum | 2726 | 490 | 444 | 4524 | 5505 |
| rabbit | 2443 | 493 | 70 | 1758 | 846 |
| coyote | 1409 | 285 | 51 | 1326 | 2249 |
| cat | 1287 | 192 | 72 | 1632 | 1418 |
| squirrel | 1250 | 222 | 0 | 659 | 1051 |
| raccoon | 1027 | 169 | 129 | 1047 | 5469 |
| dog | 820 | 139 | 96 | 845 | 888 |
| bobcat | 786 | 154 | 793 | 897 | 2338 |
| car | 668 | 41 | 0 | 791 | 1113 |
| bird | 484 | 114 | 7 | 574 | 224 |
| rodent | 375 | 143 | 0 | 233 | 62 |
| skunk | 229 | 43 | 63 | 194 | 328 |
| deer | 42 | 3 | 0 | 162 | 0 |
| badger | 9 | 1 | 0 | 4 | 14 |
| fox | 5 | 0 | 0 | 2 | 1 |
| empty | 0 | 995 | 61 | 1180 | 1778 |

Classes present: train 15, cis_val 15, trans_val 10, cis_test 16, trans_test 15.
Classes in trans_test but not train: ['empty'].
Classes in train but not trans_test: ['deer'].

## Image sizes and sequences

| split | distinct (w,h) | min w | max w | min h | max h | median seq len | max seq len |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 3 | 2048 | 2560 | 1494 | 1920 | 3 | 3 |
| cis_val | 3 | 2048 | 2560 | 1494 | 1920 | 3.0 | 3 |
| trans_val | 1 | 2048 | 2048 | 1494 | 1494 | 3 | 3 |
| cis_test | 3 | 2048 | 2560 | 1494 | 1920 | 3 | 3 |
| trans_test | 3 | 2048 | 2560 | 1494 | 1920 | 3 | 5 |

Note: width/height here are the original capture sizes; the benchmark tarball is resized to at most 1024 px per side.

## File name pattern

Extensions: {'.jpg': 57864}. Path depth (slashes): {0: 57864}. file_name stem equals image id: True.

## info block

Keys: ['contributor', 'date_created', 'description', 'version', 'year']

## Reachability

Two sampled image file names (one train, one trans_test) resolve with HTTP 200 on both the GCP mirror
(`storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/<file_name>`) and the Azure
mirror (`lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/<file_name>`).
Those are full-size originals. The 6.5 GB benchmark tarball (`eccv_18_all_images_sm.tar.gz`, resized to
1024 px) also answers HTTP 200; its internal folder layout is unknown until it is opened on Kaggle.

## Findings that change the plan

1. **Train has no empty images.** `empty` appears in cis_val (995), trans_val (61), cis_test (1180) and
   trans_test (1778) but never in train. The ECCV paper classified bounding-box crops, so empties were
   only a detection concern there. For whole-image classification we must decide: drop `empty` from the
   task (15 classes, filter empties out of val and test), or move empties from cis_val into train.
2. **The official train and cis_val splits share 224 sequences.** The published split leaks bursts between
   train and cis_val. Our "every burst in one split" invariant therefore cannot be satisfied by using the
   files as given; we need to re-split train + cis_val by `seq_id` ourselves. cis_test and both trans
   splits are clean (zero shared sequences with anything).
3. **Location split is clean.** 10 cis locations and 10 trans locations, no overlap; no image id appears
   in two files.
4. **Multi-species images are rare.** 78 images across all splits carry two different species (7 in
   train); the dominant pair is bobcat + rabbit. Rule: drop images with more than one distinct species.
   Images with several annotations of the same species (414 in train) keep that single label.
5. **Three classes are too rare to learn.** badger (9 train), fox (5 train), deer (42 train, 0 in
   trans_test). Decide: drop them, or merge into an `other` class. Also `rodent`, `squirrel` and `car`
   have zero images in trans_val, so trans_val covers only 10 classes.
6. **trans_val is tiny and single-location** (1725 images, 1 camera). Model selection should lean on
   the re-split cis val; trans_val is a sanity check only.
7. **Heavy imbalance.** opossum alone is 20% of train; opossum + raccoon are 47% of trans_test. Macro-F1
   or balanced accuracy must sit next to plain accuracy in the headline.
8. **Metadata fields are sufficient.** Every image has `location`, `seq_id`, `frame_num`,
   `seq_num_frames`, `date_captured`, `width`, `height`; file names are flat `<image_id>.jpg`.
   Sequences are almost all 3 frames. Originals are 2048x1494 or 2560x1920; the tarball is 1024 px max.
9. **Verified handoff facts.** Class list matches the 15 named plus `empty` (ids are not contiguous:
   1, 3, 5, 6, 7, 8, 9, 10, 11, 16, 21, 30, 33, 34, 51, 99, so the class-to-index map must be explicit).
   Published split counts match exactly. Format is COCO-style JSON with `images`, `annotations`,
   `categories`, `info`.
