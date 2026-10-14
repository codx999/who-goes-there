# Ledger

One row per training run, appended by `trailcam/train/train.py` (plan.md section 5; R40), in the
column order the script prints. The dry run prints its row and never appends. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- run id: `<config>-seed<run seed>-<UTC start, YYYYMMDDTHHMMSSZ>`, one row per run id.
- config: the config's name in `config.yaml`. config hash: SHA-256 of the resolved config without
  `arm` and `run_seed` (`config_hash` in `trailcam/train/config.py`), so the arm A and arm B rows
  of one seed share it (the user's decision on review 5.02; item 7.01 checks it).
- split hash: `Split.hash` of `make_split` for the split seed (R14).
- epochs run includes the head-only epoch; best epoch is the epoch of the highest val macro-F1,
  whose weights are the checkpoint (R31).
- val accuracy and val macro-F1 are the best epoch's; macro-F1 averages over the classes present.
  trans_val accuracy is scored once on the restored checkpoint, reported and never selected on
  (R17).
- checkpoint sha256: of the saved `.pt` file, which never goes into git.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
