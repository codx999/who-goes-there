# Ledger

One row per training run, appended by `trailcam/train/train.py` (plan.md section 5; R40), in the
column order the script prints. The dry run prints its row and never appends. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- run id: `<config>-seed<run seed>-<UTC start, YYYYMMDDTHHMMSSZ>`, one row per run id.
- config: the config's name in `config.yaml`. config hash: SHA-256 of the resolved config without
  `arm` and `run_seed` (`config_hash` in `trailcam/train/config.py`), so the arm A and arm B rows
  of one seed share it (the user's decision on review 5.02).
- split hash: `Split.hash` of `make_split` for the split seed (R14).
- epochs run includes the head-only epoch; best epoch is the epoch of the highest val macro-F1,
  whose weights are the checkpoint (R31).
- val accuracy and val macro-F1 are the best epoch's; macro-F1 averages over the classes present.
  trans_val accuracy and macro-F1 (over the classes present; the user's decision on review 5.03)
  are scored once on the restored checkpoint, reported and never selected on (R17).
- checkpoint sha256: of the saved `.pt` file, which never goes into git.

Check it after pasting a row: `uv run python -m trailcam.train.ledger --check docs/ledger.md`
(`trailcam/train/ledger.py`, item 7.01) exits 1 with every problem, or 0 with the val scores per
model, arm and seed. It refuses a missing or empty cell, a repeated run id, a malformed hash, seed,
score or run id, a `dryrun` row, one split seed with two split hashes (R14), arm A and arm B rows
of one model and run seed that differ in split hash or config hash (R25), rows of one model with
two config hashes, whatever their split seeds, and rows of one config hash that differ in model,
split seed or scheduler. So, by design, a logged run of a model with a changed recipe or split
seed fails the check (the user's decision on review 7.01).

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_cnn-seed0-20261015T223717Z | small_cnn | A | small_cnn | f40ed6e3960104f10a78e3d33a8f530cde82e23a555def76f694e19ab09a3e95 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 15 | 14 | cosine warmup 1 | 0.6625 | 0.6216 | 0.1965 | 0.0775 | 18529bf80a299551e41970e380f36cb2d1d217ddb838322b56b596829c47573c |
| resnet18_arm_a-seed0-20261017T002526Z | resnet18 | A | resnet18_arm_a | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8647 | 0.8542 | 0.3724 | 0.2915 | 2354a3a55c45360e299cc19cc317ca7912903138d7fcc2a839738f85ba420a76 |
| resnet18_arm_b-seed0-20261017T004731Z | resnet18 | B | resnet18_arm_b | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 10 | cosine warmup 1 | 0.8530 | 0.8427 | 0.3387 | 0.2588 | 0963bb625f0e28a8564090afe44a8afc9ad5d5015f68eb41c06eb9471b82de31 |
| resnet18_arm_a-seed1-20261017T154310Z | resnet18 | A | resnet18_arm_a | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 1 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 8 | cosine warmup 1 | 0.8659 | 0.8559 | 0.3525 | 0.3187 | fddcd1a0a6a09820e370836fb8372d8ce5a2bea72ab2f6667f0fe34032b4a4de |
| resnet18_arm_b-seed1-20261017T155451Z | resnet18 | B | resnet18_arm_b | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 1 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8509 | 0.8384 | 0.2926 | 0.3019 | 355444acc0f42905051b693623943a1a4aba979c750ba5f206eb16eff458aae6 |
| resnet18_arm_a-seed2-20261017T160630Z | resnet18 | A | resnet18_arm_a | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 2 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 8 | cosine warmup 1 | 0.8676 | 0.8569 | 0.4099 | 0.3200 | b68f7e1ea8c5466351e5ea526a0756e537297c0930089aeb1aad8b6004f51772 |
| resnet18_arm_b-seed2-20261017T162445Z | resnet18 | B | resnet18_arm_b | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 2 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 8 | cosine warmup 1 | 0.8571 | 0.8472 | 0.3699 | 0.3472 | a0a548893025c1a859045a79198a427fc0ef9dd8f554b1621d0565010a9124e9 |
