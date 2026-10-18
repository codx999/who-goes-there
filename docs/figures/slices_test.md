# Slices: day and night, common and rare species

Approved by codx999, scored at 20261018T193117Z (UTC). Split seed 42, split hash c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009. ResNet-18 run seeds reported: 0, 1, 2.

- small_cnn seed 0: run small_cnn-seed0-20261015T223717Z, checkpoint sha256 18529bf80a299551e41970e380f36cb2d1d217ddb838322b56b596829c47573c
- resnet18 arm A seed 0: run resnet18_arm_a-seed0-20261017T002526Z, checkpoint sha256 2354a3a55c45360e299cc19cc317ca7912903138d7fcc2a839738f85ba420a76
- resnet18 arm B seed 0: run resnet18_arm_b-seed0-20261017T004731Z, checkpoint sha256 0963bb625f0e28a8564090afe44a8afc9ad5d5015f68eb41c06eb9471b82de31
- resnet18 arm A seed 1: run resnet18_arm_a-seed1-20261017T154310Z, checkpoint sha256 fddcd1a0a6a09820e370836fb8372d8ce5a2bea72ab2f6667f0fe34032b4a4de
- resnet18 arm B seed 1: run resnet18_arm_b-seed1-20261017T155451Z, checkpoint sha256 355444acc0f42905051b693623943a1a4aba979c750ba5f206eb16eff458aae6
- resnet18 arm A seed 2: run resnet18_arm_a-seed2-20261017T160630Z, checkpoint sha256 b68f7e1ea8c5466351e5ea526a0756e537297c0930089aeb1aad8b6004f51772
- resnet18 arm B seed 2: run resnet18_arm_b-seed2-20261017T162445Z, checkpoint sha256 a0a548893025c1a859045a79198a427fc0ef9dd8f554b1621d0565010a9124e9

The same predictions cut two ways; each split is one camera group (cis cameras were
seen in training, trans cameras never). Day and night come from the cache's day/night
flag. Rare species: fewer than 4% of the train images (seed-42 split,
fixed before any test scoring): bird, rodent, skunk; the other classes are common. Macro-F1
averages over the classes present in the slice. The two slices of a cut add up to the
split.

## small_cnn seed 0

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.5026 | 0.4436 | 0.5378 | 0.4504 | 0.3356 | 0.3671 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.3140 | 0.2961 | 0.3210 | 0.2041 | 0.1663 | 0.1472 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.5026 | 0.5079 | 0.4310 | 0.4504 | 0.4643 | 0.5976 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.3140 | 0.3226 | 0.0180 | 0.2041 | 0.2716 | 0.0655 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.5026 | 0.3140 | 0.1886 | 0.626 | 0.716 | 0.3117 | 0.1909 |
| cis_test | trans_test | macro-F1 | 0.4504 | 0.2041 | 0.2463 | 0.626 | 0.716 | 0.2075 | 0.2429 |

## resnet18 arm A seed 0

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6586 | 0.6101 | 0.6875 | 0.6277 | 0.5100 | 0.5466 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5131 | 0.5315 | 0.5058 | 0.4071 | 0.3838 | 0.3091 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6586 | 0.6676 | 0.5370 | 0.6277 | 0.6493 | 0.7214 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5131 | 0.5171 | 0.3754 | 0.4071 | 0.4926 | 0.3737 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6586 | 0.5131 | 0.1455 | 0.626 | 0.716 | 0.5154 | 0.1432 |
| cis_test | trans_test | macro-F1 | 0.6277 | 0.4071 | 0.2206 | 0.626 | 0.716 | 0.4166 | 0.2111 |

## resnet18 arm B seed 0

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6634 | 0.6486 | 0.6723 | 0.6269 | 0.5377 | 0.5398 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5435 | 0.5618 | 0.5362 | 0.4363 | 0.3925 | 0.3481 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6634 | 0.6764 | 0.4880 | 0.6269 | 0.6681 | 0.6523 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5435 | 0.5504 | 0.3066 | 0.4363 | 0.5270 | 0.3057 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6634 | 0.5435 | 0.1199 | 0.626 | 0.716 | 0.5458 | 0.1176 |
| cis_test | trans_test | macro-F1 | 0.6269 | 0.4363 | 0.1906 | 0.626 | 0.716 | 0.4434 | 0.1834 |

## resnet18 arm A seed 1

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6711 | 0.6201 | 0.7015 | 0.6071 | 0.5120 | 0.5179 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5216 | 0.5088 | 0.5267 | 0.4169 | 0.3585 | 0.3197 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6711 | 0.6876 | 0.4480 | 0.6071 | 0.6585 | 0.5920 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5216 | 0.5250 | 0.4033 | 0.4169 | 0.4957 | 0.4055 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6711 | 0.5216 | 0.1495 | 0.626 | 0.716 | 0.5200 | 0.1511 |
| cis_test | trans_test | macro-F1 | 0.6071 | 0.4169 | 0.1902 | 0.626 | 0.716 | 0.4232 | 0.1840 |

## resnet18 arm B seed 1

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6805 | 0.6537 | 0.6966 | 0.6352 | 0.5469 | 0.5456 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5326 | 0.5285 | 0.5343 | 0.4290 | 0.3923 | 0.3269 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6805 | 0.6922 | 0.5230 | 0.6352 | 0.6799 | 0.6864 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5326 | 0.5384 | 0.3361 | 0.4290 | 0.5113 | 0.3493 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6805 | 0.5326 | 0.1479 | 0.626 | 0.716 | 0.5321 | 0.1484 |
| cis_test | trans_test | macro-F1 | 0.6352 | 0.4290 | 0.2062 | 0.626 | 0.716 | 0.4354 | 0.1998 |

## resnet18 arm A seed 2

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6827 | 0.6317 | 0.7131 | 0.6299 | 0.5238 | 0.5676 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5065 | 0.5152 | 0.5031 | 0.4141 | 0.3613 | 0.3242 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6827 | 0.6980 | 0.4760 | 0.6299 | 0.6773 | 0.6655 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5065 | 0.5076 | 0.4705 | 0.4141 | 0.4963 | 0.4325 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6827 | 0.5065 | 0.1761 | 0.626 | 0.716 | 0.5076 | 0.1751 |
| cis_test | trans_test | macro-F1 | 0.6299 | 0.4141 | 0.2159 | 0.626 | 0.716 | 0.4199 | 0.2100 |

## resnet18 arm B seed 2

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.6986 | 0.6480 | 0.7288 | 0.6461 | 0.5305 | 0.5732 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.5399 | 0.5339 | 0.5423 | 0.4275 | 0.3840 | 0.3410 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.6986 | 0.7120 | 0.5180 | 0.6461 | 0.6852 | 0.7235 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.5399 | 0.5437 | 0.4098 | 0.4275 | 0.5190 | 0.3987 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.6986 | 0.5399 | 0.1586 | 0.626 | 0.716 | 0.5392 | 0.1594 |
| cis_test | trans_test | macro-F1 | 0.6461 | 0.4275 | 0.2187 | 0.626 | 0.716 | 0.4336 | 0.2125 |
