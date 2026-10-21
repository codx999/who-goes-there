# Test ledger, phase 3

One row per checkpoint of phase 3's finalist scored on cis_test and trans_test by phase 3's one
test scoring (`trailcam/evaluate/score_test.py --phase 3`, item 13.12; `docs/phase3-plan.md`
section 7), in the column order the script prints. The columns are those of `docs/test-ledger.md`,
whose notes on each column hold here too. This file is separate so that phase 1's run-once guard
on `docs/test-ledger.md` stays as it is, and it is committed with its header and no row so that
the scoring can refuse a missing file instead of creating one (a mistyped path never passes for
an empty test ledger).

The scoring runs once, for the finalist's three checkpoints (run seeds 0, 1 and 2) together, on
Kaggle, by the user, with the go-ahead (`APPROVED_BY` in `notebooks/kaggle_score_test.ipynb`,
with `PHASE = 3`). The finalist is not typed in anywhere: the script recomputes the plan's
screen, keep rule and finalist rule from `docs/ledger-phase3.md` and `resnet18_arm_a`'s rows of
`docs/ledger.md`, and refuses if nothing was kept. It prints each checkpoint's row as soon as it is
scored, and at the end appends all the rows to the copy in the Kaggle clone (`--test-ledger
<clone>/docs/test-ledger-phase3.md`). The user then pastes the printed rows once into this file in
the repo. A run that stops part way appends nothing and is run again from the start.

Disclosure (section 7): these scores are the test sets' second use, for one recipe chosen on val
and trans_val only, after phase 1's test scores (`docs/test-ledger.md`, 2026-10-18) had been seen.
They are reported beside phase 1's, never in their place, and they choose nothing: not the
finalist, chosen before, and not the Space's seed, chosen on val.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | split hash | checkpoint sha256 | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap | scored at | approved by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_convnext_cp-seed0-20261020T013607Z | convnext_tiny | C | p3_convnext_cp | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 1f25b156f936025480671b29aac11227445c135b35d189dd378134f9046d6068 | 0.7579 | 0.7260 | 0.6662 | 0.5465 | 0.0917 | 0.1794 | 20261020T213420Z | codx999 |
| p3_convnext_cp-seed1-20261020T021059Z | convnext_tiny | C | p3_convnext_cp | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 58b6e589588362b50831754610db6e94c50cdccc63869d4f53870dfdd274791e | 0.7490 | 0.7171 | 0.6592 | 0.5319 | 0.0898 | 0.1851 | 20261020T213420Z | codx999 |
| p3_convnext_cp-seed2-20261020T021420Z | convnext_tiny | C | p3_convnext_cp | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 67a6de97379f75e6c523a72ae358bf4ade2a76d53492cc7d55241bf3852e5fa6 | 0.7627 | 0.7292 | 0.6540 | 0.5315 | 0.1087 | 0.1977 | 20261020T213420Z | codx999 |
