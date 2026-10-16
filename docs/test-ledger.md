# Test ledger

One row per checkpoint scored on cis_test and trans_test by the final scoring
(`trailcam/evaluate/score_test.py`, item 9.01; plan.md section 7; R17, R40), in the column order
the script prints. The training rows stay in `docs/ledger.md`; this file is separate so that
table keeps its 16 columns (the user's decision in session 9.01). The final scoring is run once,
on Kaggle, by the user (item 9.02), who pastes the printed rows here.

- run id, model, arm, config, split hash: copied from the checkpoint's row in `docs/ledger.md`,
  found by the checkpoint's sha256.
- checkpoint sha256: of the scored `.pt` file, which never goes into git.
- cis_test and trans_test accuracy and macro-F1 (over the classes present); accuracy gap and
  macro-F1 gap: cis minus trans.
- scored at: the UTC time of the scoring, `YYYYMMDDTHHMMSSZ`. approved by: who gave the go-ahead
  (`--approved-by`, required).

The script reads this file before it scores anything and refuses a checkpoint that already has a
row here, a checkpoint with no row in `docs/ledger.md`, and a run without `--approved-by`.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | split hash | checkpoint sha256 | cis_test accuracy | cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap | macro-F1 gap | scored at | approved by |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
