# Final scoring: cis_test and trans_test

Approved by codx999, scored at 20261018T193117Z (UTC). Split seed 42, split hash c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009. ResNet-18 run seeds reported: 0, 1, 2.

Accuracy and macro-F1 (over the classes present) on cis_test, cameras seen in training,
and trans_test, cameras never seen; the gap is cis minus trans. Rows in ledger order,
the order the runs were trained, never by a score; a mean row averages the seeds of one
model and arm.

| checkpoint | run id | cis_test images | trans_test images | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn seed 0 | small_cnn-seed0-20261015T223717Z | 14478 | 21473 | 0.5026 | 0.4504 | 0.3140 | 0.2041 | 0.1886 | 0.2463 |
| resnet18 arm A seed 0 | resnet18_arm_a-seed0-20261017T002526Z | 14478 | 21473 | 0.6586 | 0.6277 | 0.5131 | 0.4071 | 0.1455 | 0.2206 |
| resnet18 arm B seed 0 | resnet18_arm_b-seed0-20261017T004731Z | 14478 | 21473 | 0.6634 | 0.6269 | 0.5435 | 0.4363 | 0.1199 | 0.1906 |
| resnet18 arm A seed 1 | resnet18_arm_a-seed1-20261017T154310Z | 14478 | 21473 | 0.6711 | 0.6071 | 0.5216 | 0.4169 | 0.1495 | 0.1902 |
| resnet18 arm B seed 1 | resnet18_arm_b-seed1-20261017T155451Z | 14478 | 21473 | 0.6805 | 0.6352 | 0.5326 | 0.4290 | 0.1479 | 0.2062 |
| resnet18 arm A seed 2 | resnet18_arm_a-seed2-20261017T160630Z | 14478 | 21473 | 0.6827 | 0.6299 | 0.5065 | 0.4141 | 0.1761 | 0.2159 |
| resnet18 arm B seed 2 | resnet18_arm_b-seed2-20261017T162445Z | 14478 | 21473 | 0.6986 | 0.6461 | 0.5399 | 0.4275 | 0.1586 | 0.2187 |
| resnet18 arm A, mean of 3 seeds | 3 runs | 14478 | 21473 | 0.6708 | 0.6216 | 0.5137 | 0.4127 | 0.1571 | 0.2089 |
| resnet18 arm B, mean of 3 seeds | 3 runs | 14478 | 21473 | 0.6808 | 0.6361 | 0.5387 | 0.4309 | 0.1422 | 0.2051 |

## Leakage alarms

None fired.

## The other outputs

- `<run id>/metrics.md` and its PNGs: per class, per camera and the confusion matrix of
  both splits; `<run id>/attention.md` and `attention_in_box.png`: attention-in-box and
  the animal-removed test; `<run id>/gradcam_overlay_<n>.png`: the `assets/` overlays.
- `slices.md`: day and night, common and rare species, cis and trans at one night share.
- `features_by_species.png`, `features_by_camera.png`: the ResNet-18 feature maps.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.
