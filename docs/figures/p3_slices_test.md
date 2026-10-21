# Slices: day and night, common and rare species

Phase 3's one test scoring (docs/phase3-plan.md section 7). Approved by codx999, scored at 20261020T213420Z (UTC). Split seed 42, split hash c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009. The finalist p3_convnext_cp, run seeds 0, 1, 2, recomputed from docs/ledger-phase3.md and resnet18_arm_a's rows of docs/ledger.md by the keep rule and the finalist rule:
- p3_convnext: kept, passed the screen (seed 0 trans_val macro-F1 +0.0767, val macro-F1 +0.0295); mean trans_val macro-F1 +0.0457 (more than +0.03: yes), rises on every seed: yes; mean val macro-F1 +0.0245 (no more than 0.03 down: yes)
- p3_convnext_cp: kept, passed the screen (seed 0 trans_val macro-F1 +0.1007, val macro-F1 +0.0031); mean trans_val macro-F1 +0.1028 (more than +0.03: yes), rises on every seed: yes; mean val macro-F1 +0.0063 (no more than 0.03 down: yes)
- The finalist: p3_convnext_cp (the higher mean trans_val macro-F1, by 0.0571, more than 0.03).

The test sets were used a second time here, for this one recipe, chosen on val and trans_val only, after phase 1's test scores (docs/test-ledger.md, 2026-10-18) had been seen. Report the two side by side, never one for the other.

- convnext_tiny arm C seed 0: run p3_convnext_cp-seed0-20261020T013607Z, checkpoint sha256 1f25b156f936025480671b29aac11227445c135b35d189dd378134f9046d6068
- convnext_tiny arm C seed 1: run p3_convnext_cp-seed1-20261020T021059Z, checkpoint sha256 58b6e589588362b50831754610db6e94c50cdccc63869d4f53870dfdd274791e
- convnext_tiny arm C seed 2: run p3_convnext_cp-seed2-20261020T021420Z, checkpoint sha256 67a6de97379f75e6c523a72ae358bf4ade2a76d53492cc7d55241bf3852e5fa6

The same predictions cut two ways; each split is one camera group (cis cameras were
seen in training, trans cameras never). Day and night come from the cache's day/night
flag. Rare species: fewer than 4% of the train images (seed-42 split,
fixed before any test scoring): bird, rodent, skunk; the other classes are common. Macro-F1
averages over the classes present in the slice. The two slices of a cut add up to the
split.

## convnext_tiny arm C seed 0

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.7579 | 0.7345 | 0.7719 | 0.7260 | 0.6274 | 0.6421 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.6662 | 0.6563 | 0.6701 | 0.5465 | 0.4749 | 0.5184 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.7579 | 0.7664 | 0.6430 | 0.7260 | 0.7498 | 0.7917 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.6662 | 0.6718 | 0.4738 | 0.5465 | 0.6408 | 0.4326 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.7579 | 0.6662 | 0.0917 | 0.626 | 0.716 | 0.6649 | 0.0930 |
| cis_test | trans_test | macro-F1 | 0.7260 | 0.5465 | 0.1794 | 0.626 | 0.716 | 0.5538 | 0.1722 |

## convnext_tiny arm C seed 1

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.7490 | 0.7378 | 0.7557 | 0.7171 | 0.6223 | 0.6171 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.6592 | 0.6369 | 0.6680 | 0.5319 | 0.4544 | 0.5070 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.7490 | 0.7579 | 0.6290 | 0.7171 | 0.7467 | 0.7819 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.6592 | 0.6645 | 0.4787 | 0.5319 | 0.6354 | 0.4308 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.7490 | 0.6592 | 0.0898 | 0.626 | 0.716 | 0.6564 | 0.0926 |
| cis_test | trans_test | macro-F1 | 0.7171 | 0.5319 | 0.1851 | 0.626 | 0.716 | 0.5377 | 0.1794 |

## convnext_tiny arm C seed 2

### Day and night

| split | camera group | images | day | night | night share | accuracy | accuracy, day | accuracy, night | macro-F1 | macro-F1, day | macro-F1, night | classes present (all / day / night) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 5412 | 9066 | 0.626 | 0.7627 | 0.7417 | 0.7752 | 0.7292 | 0.6246 | 0.6308 | 12 / 11 / 12 |
| trans_test | trans | 21473 | 6089 | 15384 | 0.716 | 0.6540 | 0.6329 | 0.6623 | 0.5315 | 0.4451 | 0.5126 | 12 / 12 / 11 |

### Common and rare species

| split | camera group | images | common | rare | rare share | accuracy | accuracy, common | accuracy, rare | macro-F1 | macro-F1, common | macro-F1, rare | classes present (all / common / rare) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | cis | 14478 | 13478 | 1000 | 0.069 | 0.7627 | 0.7734 | 0.6180 | 0.7292 | 0.7611 | 0.7800 | 12 / 9 / 3 |
| trans_test | trans | 21473 | 20863 | 610 | 0.028 | 0.6540 | 0.6595 | 0.4656 | 0.5315 | 0.6249 | 0.4210 | 12 / 9 / 3 |

### Cis and trans at the same day/night mix

Trans scores re-weighted to the cis night share p: each trans image counts p / q
at night and (1 - p) / (1 - q) by day, q the trans night share, and accuracy and
macro-F1 are computed from the weighted counts. The gap left at the same mix is
the part the day/night mix does not explain.

| cis split | trans split | metric | cis | trans | gap | cis night share | trans night share | trans at the cis mix | gap at the cis mix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cis_test | trans_test | accuracy | 0.7627 | 0.6540 | 0.1087 | 0.626 | 0.716 | 0.6513 | 0.1113 |
| cis_test | trans_test | macro-F1 | 0.7292 | 0.5315 | 0.1977 | 0.626 | 0.716 | 0.5362 | 0.1931 |
