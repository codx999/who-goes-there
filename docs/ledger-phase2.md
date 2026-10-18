# Ledger, phase 2

One row per phase-2 training run (`docs/phase2-plan.md`, item 13.02), appended by
`trailcam/train/train.py` with `--ledger <clone>/docs/ledger-phase2.md` (the training notebook's
`LEDGER = "docs/ledger-phase2.md"`), in the column order the script prints. The columns and the
header are those of `docs/ledger.md`, whose notes on each column hold here too. Kaggle runs are
started by the user, who pastes the printed row here; the row is the unit of review for run items.

- Only phase-2 configs, whose names start with `p2_`: `train.py` refuses a `p2_` config with any
  other ledger and a phase-1 config with this one, before training and at the append, and the
  check below refuses a row of either kind in the wrong ledger. Phase-1 rows stay in
  `docs/ledger.md`, which holds exactly the pre-registered runs the final scoring reads.
- Every phase-2 step is a new recipe of the same model, so the rows of one model may carry several
  config hashes; the rows of one config share one, and every row shares one split seed (phase 1's
  split, seed 42). `p2_baseline` is `resnet18_arm_a` exactly, so its rows carry arm A's config hash.
- Training still selects the checkpoint on val macro-F1 alone (R17); a step is kept or dropped on
  val and trans_val (`docs/phase2-plan.md` section 3), so trans_val here is slightly optimistic
  (the plan's honesty note). Test scores of phase 2 never go here.

Check it after pasting a row: `uv run python -m trailcam.train.ledger --check docs/ledger-phase2.md`
(`trailcam/train/ledger.py`) exits 1 with every problem, or 0 with the scores per config, model,
arm and seed.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.

| run id | model | arm | config | config hash | split seed | run seed | split hash | epochs run | best epoch | scheduler | val accuracy | val macro-F1 | trans_val accuracy | trans_val macro-F1 | checkpoint sha256 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p2_baseline-seed0-20261018T211046Z | resnet18 | A | p2_baseline | 483aa055ab221fe445add05623a061a980fdf728f63c819c78caccd5a94c6203 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8647 | 0.8542 | 0.3724 | 0.2915 | 9e49506caba07a75831098c87a3362ede03722124df60feb1c5b67247178d623 |
| p2_arm_c-seed0-20261018T214210Z | resnet18 | C | p2_arm_c | 5c7c51c2faea18b8f814eb58cb54ca34ade489302e2689781092ab633c313b51 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8379 | 0.8238 | 0.4435 | 0.3886 | 615ed4eee92902218dc7653e2e8a065361912dbbb87e248cb92111bc7fa6d20f |
| p2_res320-seed0-20261018T215623Z | resnet18 | A | p2_res320 | 5a20cf0771a6e3f81b2fbf4ee1a23bbb19738fcca761bf87ef07e7cf825febfd | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8847 | 0.8705 | 0.3618 | 0.3186 | 902f7b85ff2b0a56666ed1b0ba0eb9139a179dde707cf7089065848119dc134f |
| p2_res320-seed1-20261018T221422Z | resnet18 | A | p2_res320 | 5a20cf0771a6e3f81b2fbf4ee1a23bbb19738fcca761bf87ef07e7cf825febfd | 42 | 1 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 9 | cosine warmup 1 | 0.8789 | 0.8678 | 0.3568 | 0.3275 | 0f66984f3e2d56b5247c16fdf95ce233c377b807d5b92ea751714dd19da9f821 |
| p2_res320-seed2-20261018T222813Z | resnet18 | A | p2_res320 | 5a20cf0771a6e3f81b2fbf4ee1a23bbb19738fcca761bf87ef07e7cf825febfd | 42 | 2 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 10 | cosine warmup 1 | 0.8701 | 0.8586 | 0.4273 | 0.3316 | 7132b490eb75f721ce3ab575ed386f6be7c3451fc7f00e05a857facf69bad657 |
| p2_arm_c_p25-seed0-20261018T225723Z | resnet18 | C | p2_arm_c_p25 | 42d47bfda80f99a821b0257a1f1c74543d93587cf2f5966a735899df54410fb9 | 42 | 0 | c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009 | 10 | 8 | cosine warmup 1 | 0.8492 | 0.8391 | 0.3225 | 0.3408 | d4601e66a2e060b17e5b1733e979ed19f787c119bde01b3546b3af38aab10f6e |
