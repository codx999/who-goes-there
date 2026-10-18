# Test ledger

One row per checkpoint scored on cis_test and trans_test by the final scoring
(`trailcam/evaluate/score_test.py`, item 9.01; plan.md section 7; R17, R40), in the column order
the script prints. The training rows stay in `docs/ledger.md`; this file is separate so that
table keeps its 16 columns (the user's decision in session 9.01).

The final scoring runs once, for every checkpoint together, on Kaggle, by the user (item 9.02).
The script prints each checkpoint's row as soon as it is scored, and at the end appends all the
rows to the test ledger it is given: the copy in the Kaggle clone (`--test-ledger
<clone>/docs/test-ledger.md`). The user then pastes the printed rows once into this file in the
repo. A run that stops part way appends nothing and is run again from the start.

- run id, model, arm, config, split hash: copied from the checkpoint's row in `docs/ledger.md`,
  found by the checkpoint's sha256.
- checkpoint sha256: of the scored `.pt` file, which never goes into git.
- cis_test and trans_test accuracy and macro-F1 (over the classes present); accuracy gap and
  macro-F1 gap: cis minus trans.
- scored at: the UTC time of the scoring, `YYYYMMDDTHHMMSSZ`. approved by: who gave the go-ahead
  (`--approved-by`, required).

Before it scores anything the script refuses, among others (the module docstring lists them
all): a run without `--approved-by` or without `--run-seeds`; a test ledger that already holds
any row (the scoring has run); a training ledger without a small CNN row, or without an arm A
and an arm B row for each run seed given, or with a row of another seed; two training rows of
one checkpoint or of one config and run seed; a checkpoint with no training row; and a training
row whose checkpoint is not given. When it appends, it refuses again if this file gained a row
during the run.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | split hash | checkpoint sha256 | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap | scored at | approved by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn-seed0-20261015T223717Z | small_cnn | A | small_cnn | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 18529bf80a299551e41970e380f36cb2d1d217ddb838322b56b596829c47573c | 0.5026 | 0.4504 | 0.3140 | 0.2041 | 0.1886 | 0.2463 | 20261018T193117Z | codx999 |
| resnet18_arm_a-seed0-20261017T002526Z | resnet18 | A | resnet18_arm_a | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 2354a3a55c45360e299cc19cc317ca7912903138d7fcc2a839738f85ba420a76 | 0.6586 | 0.6277 | 0.5131 | 0.4071 | 0.1455 | 0.2206 | 20261018T193117Z | codx999 |
| resnet18_arm_b-seed0-20261017T004731Z | resnet18 | B | resnet18_arm_b | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 0963bb625f0e28a8564090afe44a8afc9ad5d5015f68eb41c06eb9471b82de31 | 0.6634 | 0.6269 | 0.5435 | 0.4363 | 0.1199 | 0.1906 | 20261018T193117Z | codx999 |
| resnet18_arm_a-seed1-20261017T154310Z | resnet18 | A | resnet18_arm_a | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | fddcd1a0a6a09820e370836fb8372d8ce5a2bea72ab2f6667f0fe34032b4a4de | 0.6711 | 0.6071 | 0.5216 | 0.4169 | 0.1495 | 0.1902 | 20261018T193117Z | codx999 |
| resnet18_arm_b-seed1-20261017T155451Z | resnet18 | B | resnet18_arm_b | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 355444acc0f42905051b693623943a1a4aba979c750ba5f206eb16eff458aae6 | 0.6805 | 0.6352 | 0.5326 | 0.4290 | 0.1479 | 0.2062 | 20261018T193117Z | codx999 |
| resnet18_arm_a-seed2-20261017T160630Z | resnet18 | A | resnet18_arm_a | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | b68f7e1ea8c5466351e5ea526a0756e537297c0930089aeb1aad8b6004f51772 | 0.6827 | 0.6299 | 0.5065 | 0.4141 | 0.1761 | 0.2159 | 20261018T193117Z | codx999 |
| resnet18_arm_b-seed2-20261017T162445Z | resnet18 | B | resnet18_arm_b | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | a0a548893025c1a859045a79198a427fc0ef9dd8f554b1621d0565010a9124e9 | 0.6986 | 0.6461 | 0.5399 | 0.4275 | 0.1586 | 0.2187 | 20261018T193117Z | codx999 |
