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
