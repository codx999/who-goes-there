# Ledger, phase 3

One row per phase-3 training run (`docs/phase3-plan.md` section 8, item 13.11), appended by
`trailcam/train/train.py` with `--ledger <clone>/docs/ledger-phase3.md` (the training notebook's
`LEDGER = "docs/ledger-phase3.md"`), in the column order the script prints. The columns and the
header are those of `docs/ledger.md`, whose notes on each column hold here too. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- Only phase-3 configs, whose names start with `p3_`: `train.py` refuses a `p3_` config with any
  other ledger and any other config with this one, before training and at the append, and the
  check below refuses a row of another phase here and a `p3_` row in `docs/ledger.md` or
  `docs/ledger-phase2.md`. Phase-1 rows stay in `docs/ledger.md`, phase-2 rows in
  `docs/ledger-phase2.md`.
- `p3_convnext` and `p3_convnext_cp` are two recipes of one model, so the rows of one model may
  carry several config hashes; the rows of one config share one, and every row shares one split
  seed (phase 1's split, seed 42). `p3_baseline` is `resnet18_arm_a` exactly, so its rows carry
  arm A's config hash.
- Training still selects the checkpoint on val macro-F1 alone (R17); a candidate is kept or
  dropped on val and trans_val by the keep rule fixed before any run (`docs/phase3-plan.md`
  section 1), so trans_val here is optimistic (the plan's section 6). Test scores never go here:
  the finalist's, if any, go to `docs/test-ledger-phase3.md` (section 7).

Check it after pasting a row: `uv run python -m trailcam.train.ledger --check docs/ledger-phase3.md`
(`trailcam/train/ledger.py`) exits 1 with every problem, or 0 with the scores per config, model,
arm and seed.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p3_baseline-seed0-20261019T235600Z | resnet18 | A | p3_baseline | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8647 | 0.8542 | 0.3724 | 0.2915 | 01a7d632d487fe578bdaddb9b21bc70be30c9111a61b88143ed2362350902e6e |
| p3_convnext-seed0-20261020T001505Z | convnext_tiny | A | p3_convnext | 7240842c4d79b3d9401065374eb0aa35c4cdfc253f95917be54646fccc919e59 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 10 | cosine warmup 1 | 0.8964 | 0.8837 | 0.3830 | 0.3682 | 777d4e1c65a01b60caee719d6a4ac4b85440f81e2b474f7395465b4a6ca791a4 |
| p3_convnext_cp-seed0-20261020T013607Z | convnext_tiny | C | p3_convnext_cp | a8ed0edc19dfb978a3bc34b78f94c6ed5a4dd063ca0e3dad116989701c8a0a27 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8697 | 0.8573 | 0.4610 | 0.3922 | 1f25b156f936025480671b29aac11227445c135b35d189dd378134f9046d6068 |
| p3_convnext-seed1-20261020T015622Z | convnext_tiny | A | p3_convnext | 7240842c4d79b3d9401065374eb0aa35c4cdfc253f95917be54646fccc919e59 | 42 | 1 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8935 | 0.8801 | 0.3768 | 0.3514 | 3d95037676fd2c328d5044989338b91244c2c71c260a61ac4551e1f6bd25a41a |
| p3_convnext-seed2-20261020T015726Z | convnext_tiny | A | p3_convnext | 7240842c4d79b3d9401065374eb0aa35c4cdfc253f95917be54646fccc919e59 | 42 | 2 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8910 | 0.8767 | 0.3974 | 0.3477 | dee668bcf5961ca21b3c366365c860d32278df2f8beab4cffae7c45e8954de9c |
| p3_convnext_cp-seed1-20261020T021059Z | convnext_tiny | C | p3_convnext_cp | a8ed0edc19dfb978a3bc34b78f94c6ed5a4dd063ca0e3dad116989701c8a0a27 | 42 | 1 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 10 | cosine warmup 1 | 0.8784 | 0.8631 | 0.5159 | 0.4208 | 58b6e589588362b50831754610db6e94c50cdccc63869d4f53870dfdd274791e |
| p3_convnext_cp-seed2-20261020T021420Z | convnext_tiny | C | p3_convnext_cp | a8ed0edc19dfb978a3bc34b78f94c6ed5a4dd063ca0e3dad116989701c8a0a27 | 42 | 2 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 8 | cosine warmup 1 | 0.8768 | 0.8654 | 0.4822 | 0.4257 | 67a6de97379f75e6c523a72ae358bf4ade2a76d53492cc7d55241bf3852e5fa6 |
