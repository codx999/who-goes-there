# Slices: day and night, common and rare species

Split seed 42, split hash c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009.

- resnet18 arm A: checkpoint sha256 2354a3a55c45360e299cc19cc317ca7912903138d7fcc2a839738f85ba420a76
- resnet18 arm B: checkpoint sha256 0963bb625f0e28a8564090afe44a8afc9ad5d5015f68eb41c06eb9471b82de31

The same predictions cut two ways; each split is one camera group (cis cameras were
seen in training, trans cameras never). Day and night come from the cache's day/night
flag. Rare species: fewer than 4% of the train images (seed-42 split,
fixed before any test scoring): bird, rodent, skunk; the other classes are common. Macro-F1
averages over the classes present in the slice. The two slices of a cut add up to the
split.

## resnet18 arm A

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | cis | 2394 | 958 | 1436 | 0.600 | 0.8647 | 0.8695 | 0.8614 | 0.8542 | 0.8713 | 0.8157 | 12 / 9 / 12 |
| trans_val | trans | 1603 | 589 | 1014 | 0.633 | 0.3718 | 0.3684 | 0.3738 | 0.2905 | 0.1576 | 0.2961 | 9 / 7 / 8 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | cis | 2394 | 2187 | 207 | 0.086 | 0.8647 | 0.8679 | 0.8309 | 0.8542 | 0.8721 | 0.9075 | 12 / 9 / 3 |
| trans_val | trans | 1603 | 1533 | 70 | 0.044 | 0.3718 | 0.3712 | 0.3857 | 0.2905 | 0.2989 | 0.3000 | 9 / 7 / 2 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | trans_val | accuracy | 0.8647 | 0.3718 | 0.4929 | 0.600 | 0.633 | 0.3716 | 0.4930 |
| val | trans_val | macro-F1 | 0.8542 | 0.2905 | 0.5638 | 0.600 | 0.633 | 0.2910 | 0.5632 |

## resnet18 arm B

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | cis | 2394 | 958 | 1436 | 0.600 | 0.8530 | 0.8424 | 0.8600 | 0.8427 | 0.8420 | 0.8198 | 12 / 9 / 12 |
| trans_val | trans | 1603 | 589 | 1014 | 0.633 | 0.3394 | 0.2869 | 0.3698 | 0.2594 | 0.1440 | 0.2601 | 9 / 7 / 8 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | cis | 2394 | 2187 | 207 | 0.086 | 0.8530 | 0.8596 | 0.7826 | 0.8427 | 0.8643 | 0.8806 | 12 / 9 / 3 |
| trans_val | trans | 1603 | 1533 | 70 | 0.044 | 0.3394 | 0.3405 | 0.3143 | 0.2594 | 0.2676 | 0.2588 | 9 / 7 / 2 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| val | trans_val | accuracy | 0.8530 | 0.3394 | 0.5136 | 0.600 | 0.633 | 0.3367 | 0.5163 |
| val | trans_val | macro-F1 | 0.8427 | 0.2594 | 0.5833 | 0.600 | 0.633 | 0.2602 | 0.5825 |
