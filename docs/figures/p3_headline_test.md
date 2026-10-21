# Phase 3's test scoring: cis_test and trans_test

Phase 3's one test scoring (docs/phase3-plan.md section 7). Approved by codx999, scored at 20261020T213420Z (UTC). Split seed 42, split hash c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009. The finalist p3_convnext_cp, run seeds 0, 1, 2, recomputed from docs/ledger-phase3.md and resnet18_arm_a's rows of docs/ledger.md by the keep rule and the finalist rule:
- p3_convnext: kept, passed the screen (seed 0 trans_val macro-F1 +0.0767, val macro-F1 +0.0295); mean trans_val macro-F1 +0.0457 (more than +0.03: yes), rises on every seed: yes; mean val macro-F1 +0.0245 (no more than 0.03 down: yes)
- p3_convnext_cp: kept, passed the screen (seed 0 trans_val macro-F1 +0.1007, val macro-F1 +0.0031); mean trans_val macro-F1 +0.1028 (more than +0.03: yes), rises on every seed: yes; mean val macro-F1 +0.0063 (no more than 0.03 down: yes)
- The finalist: p3_convnext_cp (the higher mean trans_val macro-F1, by 0.0571, more than 0.03).

The test sets were used a second time here, for this one recipe, chosen on val and trans_val only, after phase 1's test scores (docs/test-ledger.md, 2026-10-18) had been seen. Report the two side by side, never one for the other.

Accuracy and macro-F1 (over the classes present) on cis_test, cameras seen in training,
and trans_test, cameras never seen; the gap is cis minus trans. Rows in seed order,
the finalist's run seeds 0, 1 and 2, never by a score; the mean row averages its three
seeds.

| checkpoint | run id | cis_test images | trans_test images | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| convnext_tiny arm C seed 0 | p3_convnext_cp-seed0-20261020T013607Z | 14478 | 21473 | 0.7579 | 0.7260 | 0.6662 | 0.5465 | 0.0917 | 0.1794 |
| convnext_tiny arm C seed 1 | p3_convnext_cp-seed1-20261020T021059Z | 14478 | 21473 | 0.7490 | 0.7171 | 0.6592 | 0.5319 | 0.0898 | 0.1851 |
| convnext_tiny arm C seed 2 | p3_convnext_cp-seed2-20261020T021420Z | 14478 | 21473 | 0.7627 | 0.7292 | 0.6540 | 0.5315 | 0.1087 | 0.1977 |
| convnext_tiny arm C, mean of 3 seeds | 3 runs | 14478 | 21473 | 0.7565 | 0.7241 | 0.6598 | 0.5367 | 0.0967 | 0.1874 |

## Leakage alarms

None fired.

## The other outputs

- `<run id>/metrics.md` and its PNGs: per class, per camera and the confusion matrix of
  both splits; `<run id>/attention.md` and `attention_in_box.png`: attention-in-box and
  the animal-removed test; `<run id>/gradcam_overlay_<n>.png`: the `assets/` overlays.
- `slices.md`: day and night, common and rare species, cis and trans at one night share.
- `features_by_species.png`, `features_by_camera.png`: the ConvNeXt-Tiny feature maps.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.
