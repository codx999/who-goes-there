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
