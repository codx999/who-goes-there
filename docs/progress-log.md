# Progress log

Dated "done / next" entries, newest last. One entry per session.

## 2026-10-10: session 0, bootstrap (Fable 5.1)

**Done.** Playbook steps 1 to 6 and 8: git and uv (Python 3.12), pyproject with ruff, strict
mypy and pytest, data guard extended with image and checkpoint suffixes and an `assets/`
allow-list (25 tests), git hooks installed and verified against a fake JPEG, Claude settings,
rules and the `/dryrun` and `/review-shapes` commands, `check.sh` green. CCT20 metadata
downloaded (3 MB) and inspected: `docs/data-inspection.md`. Class guideline and setup review saved
to `docs/course-guideline.md` and `docs/setup-review.md`. Public repo `codx999/who-goes-there`
created by the user (MIT, CCT20 citation in README); first CI run green (run 38096639477).

**Next.** Step 7 in a dedicated design session: `docs/plan.md` from HANDOFF.md, data-inspection.md,
course-guideline.md and setup-review.md; then CLAUDE.md, `docs/progress.json`. Then step 9
(Part B of `docs/review/playbook-reusable.md`) in its own session, and step 10.

Review: tier 0 for the docs; the guard script and tests are control-plane files and get their
first review with the step 9 machinery.

**Review of the bootstrap (same day).** `/code-review xhigh --max-findings all 9788ae4..HEAD`,
11 findings, none high. Fixed (8): notebook image outputs now refused; `assets/` images capped at
24 files and 12 MB over the index; JSON with more than 100 image or annotation records refused as
dataset metadata; video suffixes and the MP4 marker blocked; `.gitignore` regenerated from the
guard's lists with a drift test; `assets/` allow-list narrowed to jpg, jpeg, png, webp; `.env.local`
added to the Read deny list; `check.sh` fails outside CI when hooks are not enabled; unit tests for
`_parse_raw`, `pre_push_revs`, `read_blobs`. Each fix has a test that fails on the old guard (9 of
39 failed against the pre-fix copy). Skipped (3, low): mypy targets listed twice, version string in
three places, importlib loading in the test. Review: tier 1, ran /code-review xhigh. The fix commit
touches control-plane files, so it gets one fix-round review of its own range.

**Fix-round review (same day).** `/code-review xhigh --max-findings all 257bbfa..HEAD`, 9
findings, none high. Fixed (9): the notebook and metadata checks were rewritten as byte-level
scans on every text blob regardless of suffix (image MIME keys and data URIs followed by at least
1000 base64 characters; more than 100 dataset record keys), which also closes markdown
attachments, HTML outputs, top-level lists, nested dicts and renamed dumps; a second allow-listed
image folder `docs/figures/` (40 files, 20 MB) for generated plots, mirrored in `.gitignore` and
the drift test; image budgets enforced per commit in history mode; HEIC/AVIF recognised and
labelled as images; `check.sh` accepts an absolute hooks path; the set difference hoisted; the
clear-outputs workflow documented in the training rule. 49 tests; 12 of the new ones fail against
the previous guard. One judgment call: the payload threshold was added after the stricter scan
retroactively refused the previous commit's test file (a 400-character fake payload); the
threshold is a real improvement (docs and icons do not count) and avoided a force-push.
Review: tier 1, ran /code-review xhigh, two rounds; no further round unless requested.

## 2026-10-10: session 1, design document (Fable 5.1)

**Done.** Playbook step 7 in a dedicated session, explore then plan then implement: all seven
inputs read, the outline of `docs/plan.md` with its conflicts presented and approved in chat, then
four commits, `check.sh` green before each. `docs/plan.md` is now the source of truth: problem
statement (12 classes, 11 animals plus `car`), the explicit class-to-index map, the split design
(official location split kept; train and cis_val pooled and re-split by sequence because they share
224 sequences; cis and trans test scored once), the four leakage tests, small CNN plus fine-tuned
EfficientNet-B0 from timm, AdamW with cosine warmup and early stopping on val macro-F1, accuracy
and macro-F1 side by side, Gradio demo and Hugging Face Space with weights uploaded via
`huggingface_hub`, the "decisions and why" section, a dated schedule for sessions 2 to 12, open
questions, sources. `docs/HANDOFF.md` retired with a note pointing to the plan. `CLAUDE.md` (69
lines) from the playbook template. `docs/progress.json` with 36 items (after the review fixes) for sessions 2 to 12, 12 of
them owner=user (Kaggle runs, Hugging Face account and Space, the final scoring, example photos,
video, rehearsal); sessions 0 and 1 are logged here only. Worktree note: the app sets an absolute
`core.hooksPath` to the main checkout, so `check.sh` reported hooks disabled until
`git config --worktree core.hooksPath "$PWD/.githooks"`; recorded in CLAUDE.md.

Judgment calls, approved: `car` kept as the twelfth class; the playbook's "15 classes plus empty"
invariant rows are superseded by the plan and get corrected in session 2; fixture images are
generated at test time because the guard blocks them in git; the split is reproduced from seed
plus hash rather than committed.

**Next.** Session 2 (2026-10-11): Part B of `docs/review/playbook-reusable.md` writes
`invariants.md`, `risk_files.sh` with its test, both reviewer agents and `/review-session`
(items 2.01 to 2.03); then the user runs the first review (2.04) over the bootstrap and design
range. Open question 1 (the TA's reading of the torchvision dataset rule) is due 2026-10-13.

**Review (same day).** CLAUDE.md is a control-plane file written by Fable 5.1 and the review
machinery does not exist yet, so the user ran `/code-review xhigh --max-findings all main..HEAD`:
21 findings, none high, all fixed in one commit. Steps that could not pass after the merge (2.02
and 2.04 now name the explicit range 9788ae4..HEAD; 2.01 counts rows without assuming an id
prefix; 11.02 relies on the guard's Hugging Face token pattern instead of `git grep`); the dry run
now accepts `--dry-run=True` as `/dryrun` passes it and writes its own synthetic fixture
(`trailcam/data/synthetic.py`, shared with the tests, replacing a `tests/fixtures/` generator and
its needless 100-record cap); `docs/ledger.md` is created with its header in 5.03 before the first
Kaggle row is pasted; a packaging item 5.04 (build-system table, `uv build`, wheel install in a
fresh venv) precedes the Kaggle pip install; 5.01 adds the torch, torchvision and timm
dependencies with the mypy overrides; 6.01 uses `uvx --from nbconvert`; CLAUDE.md names the
notebook pip exception; Grad-CAM overlays are made only from the `assets/` photos so no dataset
image lands in `docs/figures/` (the asset selection moved from 10.02 to 8.03, before the val
evaluation 8.04; 8.02 writes fixture figures to a temporary `--out`); 11.01 pins grad-cam only if
open question 5 says so and adds `deploy_space.py` to the mypy targets; the TA question is due
2026-10-11 with Oxford-IIIT Pet named as the fallback; the sequence label for stratification is
the first kept frame; four numbers corrected (57,864 images, about 21 MB of weights, 9 of 12
classes in trans_val, 407 same-species multi-annotation images). Rejected: none.
Review: tier 0 for the docs plus CLAUDE.md, ran /code-review xhigh; one round.

## 2026-10-11: session 1 continued, design revision after the literature review (Fable 5.1)

**Done.** A parallel session wrote `docs/research-prompt.md` and `docs/research-report.md` (CCT20
literature, task options, a ranked list of research questions, an experimental design) and a draft
`scripts/kaggle_prepare_cct20.py`. Both documents are now committed under `docs/` (the report with
a note that plan.md wins where they differ). Decisions taken with the user and written into
`docs/plan.md` (now with a section 6, "Research question and experiment", and renumbered sections
after it): keep 12 classes, no `empty` (the report's 13-class option becomes an add-if-early run);
adopt the research question, box-guided crop augmentation at training time only, arm A vs arm B,
seed 0 first and seeds 1 and 2 only if arm B seed 0 is done by 10-16; measure accuracy, macro-F1,
attention-in-box (Grad-CAM mass inside the ground-truth box) and an animal-removed test (box
blanked, does the prediction survive); ResNet-18 from torchvision as the backbone with
EfficientNet-B0 as the Space fallback; 512 px longer-side cache as the prepare script already does;
stratify the re-split by sequence majority class and location with at least 30 val images per
class; Food-101 as the fallback dataset (the prompt's choice, replacing Oxford-IIIT Pet); keep
boxes, capture hour and a pixel-based day/night flag in the metadata (verified today: every image
has a full timestamp and about half are night shots; hour 12 is doubled and looks like a clock
default, so the pixel test is the label); descriptive slices (day/night, common/rare, per unseen
camera) and a two-colour feature map of backbone features; temperature scaling as add-if-early.
Rejected from the report: scoring the test set mid-way (plan keeps one scoring at the end) and a
committed split CSV (the guard refuses it; seed plus hash stays). CLAUDE.md gained two lines (the
report in the context list; boxes never a model input), 72 lines. `docs/progress.json` rewritten:
42 items, 13 owner=user; new items for the box-guided sampler (4.04), the prepare script with the
day/night test (6.01), arm A and arm B runs (6.05, 7.02, 7.03), attention and animal-removed
(8.03), slices and feature map (8.04), calibration if early (12.03). The draft Kaggle script
from the parallel session was deleted at the user's request; item 6.01 writes it fresh, type-checked and
with a test. The untracked copy of the research prompt in the main checkout was removed too (identical
to the committed one) so the merge is not blocked.

**Next.** Session 2 (2026-10-11, today): the review machinery (2.01 to 2.03), then the user's first
review (2.04) over 9788ae4..HEAD. Ask the TA about the torchvision dataset rule today (open
question 1).

Review: tier 0 for the docs plus CLAUDE.md (two added lines). The user decides whether the
CLAUDE.md change waits for the session 2 review or gets `/code-review xhigh --max-findings all
9c0c2e4..HEAD` now.

## 2026-10-11: session 2, the review machinery (Fable 5.1)

The user asked for Opus 5.5 at high; the session ran on Fable 5.1 and said so before writing
anything. Item 2.04 therefore replaces the Fable risk-reviewer with a second Opus run.

**Done.** Step 0, approved by the user: the two rules files brought in line with the final design.
`.claude/rules/training.md` lists the ledger-row fields of plan.md section 5 in order (run id,
model, arm, config, split seed, run seed, split hash, epochs run, best epoch, scheduler, val
accuracy, val macro-F1, trans_val accuracy, checkpoint hash) and states that arm A and arm B
configs differ only in the `arm` field; `.claude/rules/data.md` gains the rule that boxes are
training-time augmentation and attention scoring only, never a model input, with the sampler in
`trailcam/data/boxcrop.py` unreachable from the eval transform, and drops the `tests/fixtures/**`
path (the generator is `trailcam/data/synthetic.py`, covered by `trailcam/data/**`). Then items
2.01 to 2.03, one commit each, Part B of `docs/review/playbook-reusable.md` filled for this repo:
`docs/review/invariants.md` with 38 rows in one table (R01 to R07 what never enters git, R10 to R17
splits and test-set isolation, R20 to R27 transforms, boxes and the experiment, R30 to R38 model,
training, evaluation, demo and Space, R40 to R45 the ledger row and the control plane; the
playbook's "15 classes plus empty" starter row superseded by the 12-class map); `scripts/risk_files.sh`
(regex: `^trailcam/`, `^space/`, `\.ipynb$` plus the control plane; exit 2 on a bad range) with
`tests/test_risk_files.py` (12 tests: risky and safe samples, bad ranges, a move out of a risky
path, a non-ASCII path, the invariants table shape, and every "where" path matched by the regex);
`.claude/agents/code-reviewer.md` (Opus 5.5, xhigh, whole range), `.claude/agents/risk-reviewer.md`
(Fable 5.1, high, given files) and `.claude/commands/review-session.md` (tier 3 for 3.03 and 9.01,
run items through the ledger row R40). Acceptance steps run and passed: the grep count is 40, the
test file passes, `risk_files.sh 9788ae4..HEAD` prints CLAUDE.md, `scripts/guard_data.py` and
`scripts/risk_files.sh` and no `docs/` path, a bad range exits 2, `check.sh` green (61 tests).
Not changed: `.claude/settings.json` and CLAUDE.md (Part B steps 5 and 6 were done at bootstrap
and in session 1).

**Next.** Item 2.04 (user): the first review over the explicit range `9788ae4..HEAD`
(`/review-session` alone computes merge-base(main, HEAD)..HEAD and would cover only this session's
commits after the merge). Then session 3 (2026-10-12): the class map, the metadata loader, the
re-split by sequence with the leakage tests (tier 3, xhigh) and the data card.

Review: tier 2 (every changed file is control plane: `.claude/`, `scripts/`, `tests/`). Command
for the user: `/review-session 9788ae4..HEAD 2.01 2.02 2.03`, with a second Opus `code-reviewer`
run in place of the Fable `risk-reviewer` because Fable wrote the files; or
`/code-review xhigh --max-findings all 9788ae4..HEAD`. Not started by this session.

## 2026-10-11: item 2.04, the first review and its fixes (Opus 5.5)

**Done.** The user started `/review-session 9788ae4..HEAD 2.01 2.02 2.03` in this session after
fast-forwarding to the session 2 tip (e777e14). Tier 2: `risk_files.sh` printed 22 control-plane
files, no split or scoring code; 14 invariant rows selected (R01 to R04, R07, R14, R30, R37, R38,
R41 to R45), every row "spec changed" because the range creates the cited documents. The agent
types were not registered, so both reviewers ran as general-purpose agents on Opus at xhigh
following `.claude/agents/code-reviewer.md`: one on the whole range, one on the risk files in
place of the Fable risk-reviewer. 23 findings, 6 made by both; every one checked against the code,
reproduced with a scratchpad test or a temporary repository where possible, and every fix has a
regression test that fails on the old code. Eight fix commits: (1) the guard counts JSON-escaped
and single-quoted record keys and refuses any notebook output or attachment (high, both
reviewers); (2) more than 100 UUIDs in a blob is refused (every CCT image and sequence id is a
UUID, 57,864 of 57,864, counted) and trailing `.`, `~` or space no longer hides a suffix; (3) an
allowed image name must hold a JPEG, PNG or WebP, and WebP, TIFF and BMP signatures are refused
elsewhere; (4) `--pre-push` checks the image budgets over the whole pushed tree, with
`tests/test_guard_git.py` running the guard modes in a temporary repository; (5) `.gitignore`
ignores `.env.*` and `secrets.toml` anywhere, and the suffix set is pinned in both directions;
(6) `check.sh` fails when `.githooks/` is missing, and `tests/test_control_plane.py` pins hook
modes, CI, check.sh, pytest config, deny lines and reviewer tools; (7) `risk_files.sh` lists root
`config.yaml` and `configs/`, R25 names it; (8) `/review-session` makes the tier-3 tests tier 3
and falls back to the read-only `Plan` type, `/review-shapes` covers both models and the eval
transform, `/dryrun` says printed, rules `data.md` quotes the plan's eval transform. `check.sh`
green before each commit (97 tests).

Not fixed, the user's call: the Read deny rules are relative to the session's directory, so in a
worktree session the main checkout's `data/`, `models/`, `outputs/` and `.env*` are not denied
(high, both reviewers, reasoning only; needs absolute rules in the user's settings, which Claude
does not edit); the CLAUDE.md line "val and test get resize and normalise"; item 3.03's real-data
check may be skipped when `data/` is absent (items are not Claude's to edit); plan section 7 puts
a camera's background photo next to the per-camera chart, which R04 forbids unless it comes from
`assets/`; `.pdf` is not restricted by the guard.

Then, at the user's go-ahead: absolute Read deny rules for this project's `data/`, `models/`,
`outputs/`, `.env*` and any `kaggle.json` added to the user's `~/.claude/settings.json` (outside
the repo; the probe file could not be written from this session, so the user checks it); the
CLAUDE.md eval transform line (shorter side 256, centre crop 224, normalise; 73 lines); plan
section 7's per-camera chart made bars only; `.pdf` blocked by suffix and signature. Item 3.03's
steps edited by Claude at the user's explicit request in chat (an exception to "never edit
items"): the real-metadata coverage checks read `TRAILCAM_DATA_ROOT` and the item flips only when
`pytest -rs tests/test_split.py` reports 0 skipped; the skip is allowed only in CI.

Fix round, started by the user: `/review-session e777e14..HEAD`, tier 2, 12 risk files. Both
agent types still unregistered, so both ran as the built-in `Plan` type: code-reviewer on Opus at
xhigh over the range, risk-reviewer on Fable at high over the risk files (Opus wrote the fixes,
so Fable is the different model). Nothing above low; every first-round fix confirmed working
with a test that fails without it. Seven findings, all verified and fixed, plus two residual gaps
closed: pytest.toml and dot-prefixed override configs in the regex and the control-plane test,
no conftest.py (pytest 9.1.1's lookup order read from its source); the workflow folder, every
secrets form and the image budgets pinned; the Plan fallback's limits stated in its prompt; a
new `scripts/data_root.sh` so check.sh points `TRAILCAM_DATA_ROOT` at the main checkout's
`data/` outside CI; the dry run's never-append rule in plan section 5 and rules training.md;
citations in rules data.md and the UUID count in data-inspection.md; the whole pytest config and
every agent file pinned. `check.sh` green before each commit. No third round unless the user asks.

**Next.** Merged and pushed by Claude at the user's go-ahead (d80be8c). Item 2.04 (owner: user)
flipped to `passes: true` by Claude at the user's explicit request in chat. Open: the user checks
the Read deny rules with a probe file in `data/`. Then session 3 (2026-10-12). Then session 3 (2026-10-12): class map,
metadata loader, re-split with the leakage tests (tier 3).

Review: tier 2, ran /review-session 9788ae4..HEAD (code-reviewer Opus xhigh on the range, second Opus run on the risk files)

## 2026-10-11: session 3, item 3.01, the class map (Opus 5.5)

**Done.** `trailcam/data/classes.py`: the 12 classes of plan section 2 in index order 0 to 11
with their CCT ids, read-only name-to-index and CCT-id-to-index maps, `INDEX_TO_CCT_ID`,
`NUM_CLASSES`, the four dropped ids (30 empty, 21 badger, 51 fox, 34 deer) and lookups that
raise `DroppedClassError` for a dropped id or name and `KeyError` for an unknown one.
`tests/test_classes.py` reads the expected order and ids from the plan's class table and the
category table in `docs/data-inspection.md`, checks kept plus dropped equals the 16 categories,
and that the maps are read-only. R05's test cell names the file; the two-species filter stays
"none yet" until 3.02. `uv run pytest tests/test_classes.py` passes; item 3.01 flipped.

Unblocker, edited by the user: the project Read deny rules `./data/**`, `./models/**` and
`./outputs/**` were applied to folders of that name at any depth and refused `trailcam/data/`
(probe: `tests/data/` refused, `trailcam/model/` allowed). The user changed them to `/data/**`,
`/models/**`, `/outputs/**` (anchored at the project root) in this worktree's and in the main
checkout's `.claude/settings.json`; an app-made worktree also loads the main checkout's project
settings, so the worktree edit alone did not help. Probes after both edits: nested folders
readable; top-level `data/`, `models/`, `outputs/` in the worktree and the main checkout still
refused. `tests/test_control_plane.py` pins the new lines. The main checkout's copy is an
uncommitted edit: `git restore .claude/settings.json` there before merging this branch.

**Next.** Item 3.02, the metadata loader and the synthetic metadata writer.

## 2026-10-11: session 3, item 3.02, the metadata loader and the synthetic metadata (Opus 5.5)

**Done.** `trailcam/data/metadata.py`: finds the one `<split>_annotations.json` per official file
under `--root` (recursive), checks every category against the class map by id and name, and
builds one frozen `Record` per kept image (label index from `trailcam/data/classes.py`, location,
seq_id, frame_num, seq_num_frames, width, height, capture hour, every box as COCO x, y, w, h in
the original pixel size, and `night` from a `day_night.json` sidecar when present, else None).
Drop rules in order: two_species, empty, rare, unannotated, counted per file; repeated image ids,
orphan annotations, malformed boxes and unknown or renamed categories raise. The sidecar format
(image id to `{"night": bool}`) is defined here for item 6.01. `trailcam/data/synthetic.py`
(metadata part): 12 classes, cis locations 101 and 102, trans location 201, 3-frame sequences,
7 pool sequences per class and cis location, boxes on about 90% of animal annotations, planted
empty, rare, two-species and two-opossum frames and one burst split across train and cis_val; it
writes only the five files and the sidecar into the given directory.

`tests/test_metadata.py`: 23 tests, expectations read back from the written JSON; one runs on the
real metadata when `TRAILCAM_DATA_ROOT` is set (file totals, 78 two-species images, 4,014
empties, all as in `docs/data-inspection.md`). The CLI on the real `data/` printed counts only:
kept 53,535 (train 13,496, cis_val 2,485, trans_val 1,603, cis_test 14,478, trans_test 21,473);
dropped two_species 78, empty 4,014, rare 237, unannotated 0; boxed fraction 0.67 (cis_val) to
0.92 (trans_val); hour 12 holds 4,525 images against about 1,100 to 1,800 for its neighbours, as
plan section 2 says. Invariant test cells filled: R03, R05, R10, R15, R37.

Follow-up, at the user's go-ahead after reviewing the two open decisions: boxes are now stored as
fractions of the image's `width` and `height`, clipped to the frame in the loader
(`normalise_box`) and counted per file; on the real metadata 100 of 47,729 kept boxes are clipped
(train 33, cis_val 3, trans_val 2, cis_test 11, trans_test 51), and the CLI prints the count.
The sidecar format, `day_night.json` mapping image id to `{"night": bool, "hour": int}` with
extra keys allowed and its path passed with `--sidecar`, is written into plan section 2, along
with the box format; R24's scenario names the double-scaling failure. The synthetic sidecar
carries the hour too. `tests/test_metadata.py`: 26 tests (fractions against the written pixel
boxes, `normalise_box` cases, two planted overrunning boxes clipped and counted).

Review, started by the user: `/review-session 3.02`, tier 2, range 72613c9..bcc04ec, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`, `trailcam/data/synthetic.py`;
code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Nothing above low; every
scenario held (R01, R03, R05, R10, R15, R23, R37, R42 to R45; the split side of R10 and R15 and
the dry-run side of R37 skipped as known gaps). Spec changed: plan section 2 (rows R01, R05, R06,
R15, R21, R23, R32 re-read; still accurate). Three low findings, all verified and fixed in
fe6c301:
- both reviewers: `load_metadata` read a `day_night.json` next to the annotation files on its
  own, against plan section 2; the flag now comes only from `--sidecar`;
- code-reviewer: a sidecar keyed wrongly (by file name) left every flag unset and the summary
  said "no sidecar"; a sidecar missing any kept image id now raises;
- code-reviewer: the real-metadata test did not pin the box facts plan section 2 states; it now
  checks rare drops per file (50, 4, 0, 168, 15), 47,729 kept boxes, 100 clipped, and that boxes
  are in the original-size pixel space (some end past 0.9 of the frame, impossible if they were
  1024 px boxes over 2048 to 2560 px originals). That test passed on the old code, as a test-gap
  fix should; the four sidecar tests failed on it.
`check.sh` green (152 passed). The fix commit touches `trailcam/data/metadata.py` and modifies
one existing test, so one fix round on fe6c301 is due.

Fix round, started by the user: `/review-session bcc04ec..HEAD 3.02`, tier 2, risk files
`tests/test_metadata.py` and `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found in the range by either; every scenario held (R01,
R03, R05, R10, R15, R42 to R45). Spec changed: plan section 2's two sidecar rules, which the code
matches. One low note outside the range from code-reviewer (R03, reasoning only), fixed: two
real-metadata asserts called `len()` inside the assert, so a failure would have printed a real
record (with its image id and file name) or real box values; the counts are now taken first.
Checked by breaking both expectations in a scratch copy: the failures print counts only. No
third round.

**Next.** Item 3.03, the re-split and the leakage tests (tier 3, xhigh).

Review: tier 2, ran /review-session 3.02 (72613c9..bcc04ec) and the fix round /review-session
bcc04ec..HEAD 3.02 (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-11: session 3, item 3.03, the re-split and the leakage tests (Opus 5.5, xhigh)

**Done.** `trailcam/data/split.py`: `make_split(meta, seed)` pools the official train and cis_val
records, groups them by `seq_id` and re-splits whole sequences into train and val; trans_val,
cis_test and trans_test are the tuples `load_metadata` returned, passed through untouched. A
sequence's class is the majority label over its kept frames (ties to the lower class index; every
image keeps its own label); a pool sequence spanning two locations is refused. Strata are (class,
location): each class gets 15% of its sequences rounded half up (at least 1, at most n - 1 when
n >= 2), spread over its locations by largest remainder, so every stratum gets the floor or the
ceiling of 15%; within a stratum the val sequences are those with the smallest SHA-256 of (seed,
seq_id). No random generator is used, so the split does not depend on record order, dict or set
order, the global `random` state or the Python version: the hash computed on the Mac is the one a
Kaggle run will record. The split hash is a SHA-256 over (split, image id, label) for all five
splits, sorted by image id. Before returning, `check_split` refuses (with counts only) an image id
twice, a `seq_id` in two splits, a location in both a cis and a trans split, or a cis location
missing from train or val. Output splits are `train`, `val`, `trans_val`, `cis_test`,
`trans_test`; each record's `split` field names the split it is in (one docstring line in
`trailcam/data/metadata.py` says so). `Split.records` is read-only and `Split`'s repr shows the
seed, the hash and the sizes only. Nothing is written to a file.

`uv run python -m trailcam.data.split --root <main checkout data/> --seed 42` printed counts only.
Pool: 15,981 images in 6,274 sequences (222 of them with kept frames in both official files),
3 mixed-label sequences, 0 majority ties. Sizes: train 13,587, val 2,394 (0.150 of the pool's
images, 940 of 6,274 sequences, 0.147 to 0.154 per cis location), trans_val 1,603, cis_test
14,478, trans_test 21,473. Smallest val class: skunk with 40 images (rodent 78, bird 89); over 200
split seeds tried, the smallest val class never fell below 40. Locations: 10 in train and val,
1 in trans_val, 10 in cis_test, 9 in trans_test. Split hash for seed 42:
c78a9c1048f63aa52f34578ca17a6391315afdc2e478b4dbfd8a3aa954d1b009.

`tests/test_split.py`: 58 tests. Every leakage and coverage rule (plan section 3 tests 1 to 4)
runs on the synthetic metadata and on the real metadata; sizes, per-class counts, sequences and
the hash are pinned for seed 42 on both (test 5), so CI catches a change to the algorithm without
data. Also: same seed same split, different seeds differ, record order, the global `random` state
and `PYTHONHASHSEED` (in a subprocess) leave the hash unchanged, the hash covers assignment and
labels, the stratum sizes follow the majority class (a spy on `val_quotas`), `val_quotas` on 300
random size tables, six refusals of planted leaks, and a CLI run that prints no id and writes no
file. `TRAILCAM_DATA_ROOT=<abs path> uv run pytest -rs tests/test_split.py`: 58 passed, 0 skipped;
without it, 37 passed and 21 skipped (the CI case). Mutation check in a scratch copy: 15 planted
bugs (per-image draw, pooling every file, first-frame strata, hash over a set or over ids only,
subsampled trans_test, no runtime check, ties to the higher index, banker's rounding, per-stratum
rounding, the global `random`, set-order draws, unseeded tie-breaks, no location check, val taken
from the cis_val file) each fail at least one test both with and without data, and the leakage
tests catch the three leaking bugs on their own with the runtime check removed. R03 check of the
failure output with bugs planted, under the default, `--tb=long` and `-vv` styles: the first
version printed a real image id, because pytest prints a failing test's arguments (`Metadata` and
`Split` reprs) and the arguments of the frame that raised; fixed by a counts-only `Split` repr,
tests that take a `Case` whose repr names its source only, and `quietly`, which re-raises a failure
in `make_split` or the CLI as its type and message. After the fix: no id, `Record(` or
`Metadata(` in any failure output. Invariant test cells filled: R10, R11, R12, R13, R14, R15, R27.

Open: `Record` and `Metadata` keep the dataclass repr, which prints image ids. A later real-data
test that takes them as fixture arguments, or a notebook cell that echoes one, would show ids; an
id-free repr on both closes that for every caller. Added as item 3.05 with the user's OK (it
changes 3.02's module); not built in this session, so it stays out of the 3.03 review range.

Review, started by the user: `/code-review ultra` (cloud) and `/review-session 3.03`, tier 3,
range d889a44..e3a2cb5, risk files `tests/test_split.py`, `trailcam/data/metadata.py`,
`trailcam/data/split.py`; code-reviewer on Opus at xhigh, risk-reviewer on Fable at high. Nothing
above low; every scenario held (R01, R03, R05, R10 to R15, R27, R42 to R45; skipped as known gaps:
the loader side of R12, the ledger and data card of R14, the run seed of R27, the `Record` and
`Metadata` reprs of R03 (item 3.05), the notebook, Kaggle and assets parts of R01 and R43). Spec
changed: none. Two findings, both verified and fixed in 9d3221d:
- both local reviewers, low: the real-metadata tests skipped whenever `TRAILCAM_DATA_ROOT` was
  unset, though item 3.03 allows the skip only in CI. At the user's decision they now skip only
  when `CI` is set and fail otherwise, in `tests/test_split.py` and `tests/test_metadata.py`. A
  subprocess test runs them without the root as a local run (all 23 fail) and as CI (all 23
  skip); on the old code it failed (0 failed locally). R42's test cell names it.
- ultrareview, nit: `make_split` built each sequence's label `Counter` twice (inline and in
  `majority_label`); it now builds it once. Behaviour unchanged: the pinned hashes hold, and the
  first-frame, tie-rule and mixed-count mutations are still caught.
Notes, not fixed: the 30-val-image floor is tested for seed 42 only and not enforced at run time,
to be decided in 5.03 if the split seed can differ from 42 (code-reviewer); the pinned real values
have no independent source, so the data card (3.04) cross-checks them; the code-reviewer
reconciled the pool and trans_val totals with `docs/data-inspection.md` (risk-reviewer).
`check.sh` green (211 passed). The fix commit modifies existing test lines and touches
`trailcam/data/split.py`, so one fix round on it is due.

Fix round, started by the user: `/review-session e3a2cb5..HEAD 3.03`, tier 3, risk files
`tests/test_metadata.py`, `tests/test_split.py`, `trailcam/data/split.py`; code-reviewer on Opus
at xhigh, risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03,
R10 to R15, R27, R42 to R45; the same known gaps skipped). Spec changed: none. Both reproduced the
new subprocess test by hand (locally 3 failed and 20 errors, as CI 23 skipped). Notes, not fixed:
any non-empty `CI` counts as CI, so `CI=false` would skip too, the same rule `check.sh` uses (both
reviewers); item 3.05's step `uv run pytest tests/test_metadata.py` now needs
`TRAILCAM_DATA_ROOT` exported, or a run through `check.sh` (code-reviewer). No third round.

**Next.** Item 3.04, the data card (counts from the seed 42 run above), then item 3.05, the
id-free reprs (tier 2).

Review: tier 3, ran `/code-review ultra` and `/review-session 3.03` (d889a44..e3a2cb5) and the fix
round `/review-session e3a2cb5..HEAD 3.03` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-11: session 3, item 3.04, the data card (Opus 5.5)

**Done.** `docs/data-card.md`: licence (CDLA-Permissive 1.0) and the Beery et al. 2018 citation,
the 12 classes with CCT ids, the four drop rules with images dropped per official file, the
per-split per-class counts after the re-split (seed 42) with sequences and locations, the boxed
fraction per split, the capture-hour histogram, the seed and split hash, and the paper's two
contradictions with the released files (57,868 vs 57,864 images; 224 sequences shared by train
and cis_val). Counts come from `python -m trailcam.data.metadata` and `python -m
trailcam.data.split --seed 42` on the main checkout's `data/`; the boxed fraction per re-split
split was counted with the same modules (the metadata CLI prints it per official file only). A
scratch script parsed the card and compared it with both CLI outputs: the 15 rows of the split
table (12 classes, images, sequences, locations), the hash, the drop table and the hour table
match exactly. Hash `c78a9c10...b009`, as logged for 3.03 and pinned in `tests/test_split.py`.

Cross-check of the 3.03 pins against plan sections 2 and 3 and `docs/data-inspection.md`: the
pool is 13,553 + 3,484 = 17,037 released, 13,496 + 2,485 = 15,981 kept = train 13,587 + val
2,394; per file, released minus two_species, empty and rare gives the kept count in all five
files, and 57,864 - 78 - 4,014 - 237 = 53,535; empties 995 + 61 + 1,180 + 1,778 = 4,014 and
two-species 78 match the inspection; rare in train is 50, not 56, because the six badger + coyote
images are dropped as two_species first (coyote 1,409 to 1,403); trans_val's rabbit falls from 70
to 9 (the 59 bobcat + rabbit and 2 coyote + rabbit images); clipped boxes total 100 of 47,729, as
the plan says. No disagreement found.

Pending in the card, by design: the day/night share (the sidecar comes from 6.01, the user
records the share in 6.03) and the trans-test locations used for `assets/` (chosen by the user in
8.05; `assets/` does not exist yet). The item's steps (file exists, counts equal 3.03's for seed
42, `check.sh` green) all hold, so `passes` is flipped. `check.sh` green (211 passed).

**Next.** Item 3.05, the id-free reprs for `Record` and `Metadata` (tier 2).

## 2026-10-11: session 3, item 3.05, id-free reprs for Record and Metadata (Opus 5.5)

**Done.** `trailcam/data/metadata.py`: `Record` and `Metadata` are `repr=False` dataclasses with
their own `__repr__`, after the counts-only `Split` repr of 3.03. A record prints as
`Record(class=..., split=..., location=..., frame=n/m, boxes=k, night=...)` (a label outside the
class map prints as `label n` rather than raising); the metadata prints the records per split,
the drops per rule and the clipped boxes, each summed over the splits. No image id, file name or
seq id in either. Equality and hashing are unchanged (still every field, `eq` was not touched).
New test `test_reprs_show_no_image_id_seq_id_or_file_name` in `tests/test_metadata.py`: no
synthetic record's repr holds its image id, seq id or file name (nor `syn-` or `.jpg`), the exact
format of one record and of the metadata, two records differing only in ids print the same but
compare unequal, and a copy compares and hashes equal. R03's test cell in
`docs/review/invariants.md` names the test. `uv run pytest tests/test_metadata.py` 32 passed
(with `TRAILCAM_DATA_ROOT` exported); `check.sh` green (212 passed), so `passes` is flipped.

Not covered at first, noted: pytest's own explanation of a failing `assert record == other`
drills into the differing attributes and prints their values (checked on a scratch dataclass at
default verbosity), so an id field that differs is shown; this goes through pytest's dataclass
comparison, not the repr. Fixed after the review, below.

Review, started by the user: `/review-session 3.05`, tier 2, range 56bf5b0..6860117, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing above low; every scenario held (R01, R03, R05, R10, R15,
R42 to R45). Spec changed: none. Known gaps: none. One finding, verified and fixed in a99065c:
- both reviewers, low: a failed `==` between two `Record`s still printed the differing image id,
  file name and seq id, because pytest explains a dataclass with a generated `__eq__` field by
  field without calling the repr. `Record` now writes its `__eq__` out (every field in order, as
  before; the dataclass still adds the frozen `__hash__`), so pytest shows the id-free repr.
  Took the code-reviewer's fix over the risk-reviewer's docstring note, which would have left
  the gap open; a `conftest.py` hook was ruled out by R42 (both). New subprocess test
  `test_failed_record_equality_shows_no_id` runs a failing comparison at default and `-vv`
  verbosity; on the old code the ids appeared 2 and 24 times. R03's test cell names it.
Notes, not fixed: displays that go around the repr (`vars(r)`, `dataclasses.asdict(r)`,
`pd.DataFrame(records)`) still show ids, outside a repr change; how IPython and Rich show a
`Record` on Kaggle was reasoned, not run (code-reviewer). `check.sh` green (213 passed). The fix
commit touches `trailcam/data/metadata.py`, so one fix round on it is due.

Fix round, started by the user: `/review-session 6860117..HEAD 3.05`, tier 2, risk files
`tests/test_metadata.py`, `trailcam/data/metadata.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R05, R10,
R15, R42 to R45). Spec changed: none. Both traced pytest 9.1.1 and confirmed that the written-out
`__eq__` turns off its field-by-field listing; neither ran the new test against the old code
(read-only), which this session did before the fix (ids shown 2 and 24 times). No third round.

**Next.** Session 4, item 4.01, the Dataset and transforms.

Review: tier 2, ran `/review-session 3.05` (56bf5b0..6860117) and the fix round
`/review-session 6860117..HEAD 3.05` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-12: session 4, item 4.01, the Dataset and transforms (Opus 5.5)

**Done.** Dependencies first, brought forward from 5.01 by the user's decision at session start
(the Dataset needs them; asked rather than widened): `uv add torch torchvision pillow` (torch
2.14.1, torchvision 0.29.1, Pillow 12.3.0), with a CPU-only PyTorch index for every platform but
macOS in `[tool.uv.sources]`, so the lock gives Linux `torch 2.14.1+cpu` and holds no nvidia, cuda
or triton package; a mypy override (`ignore_missing_imports`) for torchvision, which ships no
`py.typed` (torch and Pillow ship types). timm and its override stay in 5.01. Plan section 10,
"Setup maintenance", records the move (spec changed).

`trailcam/data/dataset.py`: `CCT20Dataset`, the chapter 04 pattern (`classes`, `class_to_idx`,
`load_image`, `__len__`, `__getitem__`) over one split's records and an image folder, returning
`(float32 tensor (3, 224, 224), class index)`; a grayscale JPEG loads as three equal channels.
The training transform is called with the image and the record's boxes and hands the boxes to
its crop sampler; the eval transform is called with the image alone, and its `__call__` takes
nothing else. `TrainTransform` (arm A): the sampler's crop resized to 224, horizontal flip
p=0.5, colour jitter, random grayscale p=0.1, ImageNet normalisation; `ControlCropSampler`
draws torchvision's `RandomResizedCrop.get_params` at scale 0.65 to 1.0, aspect 3/4 to 4/3, and
ignores the boxes. `EvalTransform`: shorter side to 256, centre crop 224, normalise. The dataset
refuses the training transform for records outside train and any transform of another type
(R20). An unreadable image raises `ImageReadError` with the item's position, class and split,
never the file name (the image id), and the original exception suppressed (R03). All randomness
comes from torch's generator.

Chosen here, since the plan does not pin it: "mild" colour jitter is brightness, contrast and
saturation up to 20%, no hue. Left to their items: the box sampler and the `arm` switch (4.04,
which plugs in through `TrainTransform(sampler=...)`) and the fixture JPEG writer with drawn
boxes (4.03); the tests write their own noise JPEGs, grayscale for night records, for the
synthetic metadata's seed 42 split.

`tests/test_dataset.py`, 17 tests: shape, dtype and label (0 to 11, all 12 present, in record
order) for every train and val item; the eval transform equal over two calls under different
torch and Python seeds; six train outputs pairwise different; the train transform reproducible
under one torch seed whatever Python's `random` state; the eval transform's signature and a spy
in the dataset show it never receives a box; a spy sampler receives every record's boxes in
order; over 2,000 draws the control sampler's area fraction stays within 0.65 to 1.0 and spans
it; a black-third image shows the shorter-side resize and centre crop (a direct 224 resize
fails it); a grey image normalises to the ImageNet constants; the transform refusals; a missing
and a corrupt file raise without the file name; the dataset pickles. The pickling test found a
bug, fixed before the commit: `class_to_idx` held the class map's `mappingproxy`, which cannot
be pickled, so DataLoader workers started with spawn (macOS, the dry run) would have failed; it
is now a dict copy. Eight mutations of `dataset.py` (a direct 224 resize, a flip in eval,
torchvision's default scale from 0.08, boxes passed to eval, boxes dropped before the sampler,
no normalisation, Python's `random` in the train path, the path in the error) each fail their
test. The R20, R21 and R23 test cells name the file.

`uv run pytest tests/test_dataset.py` 17 passed; `check.sh` green (230 passed), so `passes` is
flipped.

Notes for later: the first CI run with torch is the user's push of this commit (the CPU wheels
download once, then `setup-uv` caches them). For 5.04, `uv add` wrote the lower bounds
`torch>=2.14.1` and `torchvision>=0.29.1`; if Kaggle's preinstalled torch is older, `pip install`
of the package would replace it with a PyPI CUDA build, so check the bounds against Kaggle's
versions there.

Review, started by the user: `/review-session 4.01`, tier 2, range c5b2a74..8ef7d6f, risk files
`pyproject.toml`, `tests/test_dataset.py`, `trailcam/data/dataset.py`, `uv.lock`; code-reviewer
on Opus at xhigh, risk-reviewer on Fable at high. Nothing above low; every scenario held (R01,
R03, R20, R21, R23, R24 hook only, R38, R42 to R45). Known gaps passed: the R24 sampler (4.04),
the R38 build-system table (5.04). Spec changed: plan section 10 only, cited by no row. One
finding, verified and fixed in 4bea66b:
- code-reviewer, low: the runtime minimums equalled the versions locked that day
  (`torch>=2.14.1`, `torchvision>=0.29.1`, `pillow>=12.3.0`), so pip on Kaggle, which ignores
  `[tool.uv.sources]`, would replace an older preinstalled CUDA torch with PyPI's build. Before
  lowering them, `tests/test_dataset.py` was run in a scratch venv outside the repo on torch
  2.3.0, torchvision 0.18.0 and Pillow 10.1.0 (the first torch built for NumPy 2, the first
  Pillow with Python 3.12 wheels; NumPy 2.5.4, deprecation warnings as errors): 17 passed. The
  minimums are now `torch>=2.3`, `torchvision>=0.18`, `pillow>=10.1`; the lock resolves the same
  versions. New `tests/test_dependencies.py` pins them and checks they sit below the locked
  versions; both tests fail on the old `pyproject.toml`. R38's test cell names it.
Noted, not a finding (code-reviewer): the dataset's refusal of the training transform checks
`split != "train"`, which cannot tell the re-split train from the official train file; 4.02 must
test that the train loader is built from `Split.records["train"]` (R12). `check.sh` green (232
passed). The fix commit touches `pyproject.toml`, so one fix round on it is due.

Fix round, started by the user: `/review-session 8ef7d6f..HEAD 4.01`, tier 2, risk files
`pyproject.toml`, `tests/test_dependencies.py`, `uv.lock`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R20,
R21, R23, R38, R42 to R45). Spec changed: none. The code-reviewer re-ran `tests/test_dataset.py`
and `tests/test_dependencies.py` on torch 2.3.0, torchvision 0.18.0 and Pillow 10.1.0 from the
uv cache (19 passed, no warnings), confirming the scratch-venv run above; the minimums were
checked on macOS arm64 only. Noted for 5.04 and 6.02 (code-reviewer): `requires-python =
">=3.12"` would block the pip install outright if Kaggle's image still runs Python 3.11, so check
Kaggle's Python as well as its torch when the wheel install is tested. No third round.

**Next.** Item 4.02, the DataLoaders (tier 2); its tests show the train loader is built from
`Split.records["train"]` (R12).

Review: tier 2, ran `/review-session 4.01` (c5b2a74..8ef7d6f) and the fix round
`/review-session 8ef7d6f..HEAD 4.01` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-13: session 4, item 4.02, the DataLoaders (Opus 5.5)

**Done.** `trailcam/data/loaders.py`: `make_loaders(split, image_root, batch_size, *,
num_workers, pin_memory, train_transform=None, generator=None)` returns a read-only mapping with
one loader per split in `SPLIT_NAMES` order. It takes the `Split` from `make_split` (a mapping of
records is refused with `TypeError`) and builds the train loader from `split.records["train"]`
itself, with the training transform (control arm by default; 4.04 passes the arm B transform), a
`RandomSampler` and `drop_last=True`, so batch norm never gets a batch of one; the reshuffle each
epoch still draws every image. val, trans_val, cis_test and trans_test get the eval transform, a
`SequentialSampler` and every image, the last short batch included. Before building anything it
refuses a train image id that also sits in another split, with split names and counts only (R12,
R16). `num_workers` and `pin_memory` are arguments, as the config file arrives in 5.02, which
wires them; `persistent_workers` is on when there are workers. `generator` seeds the shuffle and
the workers' base seed.

`tests/test_loaders.py`, 28 tests on the synthetic metadata's split for seed 42 with noise JPEGs
in a temporary directory: a batch is `(16, 3, 224, 224)` float32 with int64 labels for every
loader; the samplers (`RandomSampler` for train, `SequentialSampler` for the rest); R12, the train
loader holds `Split.records["train"]` record for record, equals the pool minus val, differs from
the official train file and shares no id with another split (a mutation that builds it from the
pool with the overlap check off fails the test); the eval loaders give labels in record order and
the same tensors on two passes; a train epoch is a permutation; a seeded generator reproduces the
first train batch; num_workers and pin_memory pass through; with two spawned workers (forced, so
Linux CI runs spawn too) the val batches equal the single-process ones, and the train dataset
survives a pickle round trip with the same augmentation for a seed; bad arguments and a tampered
split are refused. The spawn test takes about 15 s (torch imported in each worker), so the train
loader is not iterated with workers. `docs/review/invariants.md`: test cells filled for R12, R22
and, for the loader check, R16. `check.sh` green (260 passed).

Decision for the user to confirm: `drop_last=True` on the train loader is not in plan section 5.
With batch 64 it drops at most 63 images per epoch, a different set each epoch.

Review, started by the user: `/review-session 4.02`, tier 2, range c6c65f5..b11eff2, risk files
`tests/test_loaders.py`, `trailcam/data/loaders.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing above low; every scenario held (R01, R03, R12, R16 loader
part, R22, R42 to R45). Known gap passed: the R16 run alarms (5.03). Spec changed: none.
- code-reviewer, low, fixed in a76cb74: the overlap check caught a train id still present in
  another split, but not an image moved out of a test split into train and relabelled
  `split="train"`; the dataset check passes such a record and `split.hash` went stale unchecked.
  `make_loaders` now refuses a split whose recomputed `split_hash(split.records)` differs from
  `split.hash`, with the split sizes only. New
  `test_a_split_that_no_longer_matches_its_hash_is_refused` moves a cis_test image into train; it
  failed on the old code (DID NOT RAISE). The overlap test now recomputes the hash so it still
  reaches the overlap check. R12 and R16 test cells name the hash check.
- risk-reviewer, note folded into the same fix: the docstring said `generator` makes the
  augmentation reproducible; with `num_workers=0` the transform draws from torch's global
  generator, so `torch.manual_seed` is needed too. Docstring corrected.
- code-reviewer, low, resolved by the user's decision below: "num_workers and pin_memory from
  config" was handed to 5.02, but no item description named them.

Decisions, relayed from the coordinator session and confirmed by the user in this session
(c545b61):
- Item 5.02's scope now includes `num_workers` and `pin_memory` in every config (4 and true on
  Kaggle, 0 and false for the dry-run), with a step that checks them.
- `drop_last=True` on the train loader is the user's decision: a batch of one would fail batch
  norm in training mode, at most 63 images per epoch are skipped and a different set each epoch,
  and every step has the same batch size. Val and the test loaders keep every image.
`check.sh` green (261 passed). The fix touches `trailcam/data/loaders.py` and modifies a test
line, so one fix round on it was due.

Fix round, started by the user: `/review-session b11eff2..HEAD 4.02`, tier 2, risk files
`tests/test_loaders.py`, `trailcam/data/loaders.py`; code-reviewer on Opus at xhigh,
risk-reviewer on Fable at high. Nothing found by either; every scenario held (R01, R03, R12, R16
loader part, R22, R42 to R45). Spec changed: none. Noted, not findings (both reviewers): a
`Split` rebuilt with a recomputed hash that moves a test image into train passes both checks;
only `make_split` builds a `Split` today, and the pinned hash per seed (`tests/test_split.py`)
and the ledger's split hash (5.03) cover it. If a later item adds a helper that subsets a split
and recomputes the hash, `make_loaders` should also call `check_split` (code-reviewer). The
code-reviewer asked whether the 5.02 edit was approved; the user approved it in this session
before c545b61. No third round.

**Next.** Item 4.03, the synthetic fixture images (tier 2).

Review: tier 2, ran `/review-session 4.02` (c6c65f5..b11eff2) and the fix round
`/review-session b11eff2..HEAD 4.02` (code-reviewer Opus xhigh, risk-reviewer Fable high)

## 2026-10-13: session 4, item 4.03, the synthetic fixture images (Opus 5.5)

**Done.** `trailcam/data/synthetic.py`, the images part: `write_synthetic_fixture(out_dir, seed)`
writes the 3.02 metadata (five annotation files and the sidecar) into `out_dir` and one JPEG per
image of those five files into `out_dir/images/`, under the image's `file_name` at its `width` x
`height` (96 x 72), read back from the written files and the sidecar, not from the writer's state.
A JPEG is a dark per-location background (each camera sees one scene) blended with 25% noise,
and a bright rectangle in the annotation's class colour, outlined in white, at each box; a night
record (the sidecar's flag) is saved as a grayscale mode L JPEG, as 4.01's own writer did. Every
image of the files gets one, filtered ones included (684 JPEGs, 312 grayscale, 606 with a box,
610 rectangles, about 2.4 MB, under half a second). An image depends only on the seed and its
id; the same seed gives the same bytes. `out_dir` must be new or empty, so a mistyped path cannot
mix the fixture into real data. It returns a `Fixture` (the paths, the image root, the counts) for
5.03's dry run. `python -m trailcam.data.synthetic <dir>` or `--root <dir>` (both, since the
item names `--root` and its step a bare directory; giving neither or both is a usage error, as is
a non-empty directory) prints counts only: no ids, file names or paths.

New `tests/test_synthetic.py`, 13 tests: the fixture writes only into its directory (five files,
the sidecar, `images/` holding exactly the files' `file_name`s), refuses a non-empty directory and
accepts an empty one; every JPEG is a JPEG at its image's size, mode L exactly for night images,
and the counts match; every box's one-pixel edge averages at least 220 in luminance (measured
239 or more) and the ring just outside it at most 140 (measured 114 or less), an unboxed image
has no pixel at 160 or above; a day box's centre is its class colour within 12; same seed same
bytes, a different seed different images; the fixture loads through 3.02 (each record's night
flag matches its JPEG's mode), splits through 3.03 to the hash pinned in `tests/test_split.py`
(images leave the metadata unchanged) and every record of the five splits reads through 4.01's
`CCT20Dataset` as a (3, 224, 224) float32 tensor with its label; the CLI in both forms, its two
usage errors and `python -m` print counts only. Eleven mutations of the writer each fail a test:
exclusive corners (one pixel off), width and height taken as corners, x and y swapped, no
grayscale, grayscale saved as RGB, width and height swapped, the JPEGs flat in `out_dir`, no
non-empty refusal, colour by location, the path printed, only the first box of an image drawn.

`tests/test_dataset.py` now writes its images with `write_synthetic_fixture` instead of its own
noise writer; no assertion changed, its 17 tests pass, and the 4.01 mutations whose detection
depends on the images (a flip in eval, Python's `random` in the train path, a deterministic
training transform) still fail it. R03's and R37's test cells name `tests/test_synthetic.py`.

Steps: `uv run python -m trailcam.data.synthetic <scratch dir>` wrote the fixture and printed
counts only; `uv run python scripts/guard_data.py --all` passed; `python -m trailcam.data.metadata`
with `--root` and `--sidecar` on it loaded 661 kept images (302 night) and `python -m
trailcam.data.split --root <dir> --seed 42` split it (hash `7a13478e…`, the pinned synthetic
hash), both exit 0. `check.sh` green (245 passed), so `passes` is flipped.

Review, started by the user: `/review-session 4.03`, tier 2, range c6c65f5..0a46547, risk files
`tests/test_dataset.py`, `tests/test_synthetic.py`, `trailcam/data/synthetic.py`; code-reviewer
on Opus at xhigh, risk-reviewer on Fable at high. Nothing found by either; every scenario held
(R01, R03, R37 for the fixture, R42 to R45). Known gap passed: R37's dry run (5.03). Spec changed:
none. The code-reviewer's `check.sh` run was green (245 passed, none skipped).

Could not verify (code-reviewer): the new image code on the declared minimum Pillow 10.1. Checked
afterwards at the user's request, as in the 4.01 fix round: a scratch venv outside the repo,
offline from the uv cache, torch 2.3.0, torchvision 0.18.0, Pillow 10.1.0, NumPy 2.5.4,
deprecation warnings as errors. `tests/test_dataset.py` passes (17) and so do 12 of the 13 tests
in `tests/test_synthetic.py`, but one fails in every run (ten runs):
`test_a_rectangle_is_drawn_exactly_where_each_box_is` (an unboxed night image with a pixel at
166, above the 160 the test allows) or `test_same_seed_same_bytes` (two writes in one process
differ). The cause, isolated in scratch scripts, is Pillow 10.1's JPEG encoder on this Mac (macOS
arm64): over 300 drawn images the pixels before saving hash the same in every process, but the
saved mode L JPEGs differ between processes (2 of 6 runs), always in the last 8 x 8 block, by up
to 95 grey levels; the same grey images saved as three-channel JPEGs differ too (2 of 6 runs),
and 8 of 300 decode with unequal channels. On Pillow 12.3, the locked version, every hash is the
same in every run. So the writer is correct and Pillow 10.1 cannot write it reproducibly; writing
night frames as three-channel JPEGs would not avoid it. Only 10.1.0 and 12.3.0 are in the uv
cache, so the first fixed version is not known offline. `check.sh` and CI use the lock and are
unaffected; Kaggle never writes the fixture (the dry run is Mac only), and the Dataset there only
decodes, which passes on 10.1.

Not changed, pending the user's decision: the minimum `pillow>=10.1`. `tests/test_dependencies.py`
states it is verified; that holds for the Dataset and its tests, not for the fixture's tests. The
options are to keep it and document the limit, or to raise the floor to the first version that
writes reproducibly, found with network (touches `pyproject.toml`, `uv.lock` and
`tests/test_dependencies.py`, R38, and needs a fix round). For 6.01: the prepare script writes
the 512 px cache with Kaggle's Pillow, so check that version and that re-encoding an image there
is reproducible.

Noted, not fixed (risk-reviewer), by the user's decision: `mypy tests/test_synthetic.py` reports 7
errors (Pillow's union return types compared as ints at lines 135, 138 and 146; the loop variable
`split` reused for a `Split` at lines 196 to 207). Tests are outside `check.sh`'s mypy scope and
the code is correct at runtime; fixing them would touch a reviewed high-stakes file and need
another round. No fix commit, so no fix round.

**Next.** Item 4.04, the arm B box-crop sampler (tier 2); its tests can draw on the fixture's
rectangles to check a crop contains its box. Item 4.02 runs in a parallel session.

Review: tier 2, ran `/review-session 4.03` (c6c65f5..0a46547; code-reviewer Opus xhigh,
risk-reviewer Fable high)
