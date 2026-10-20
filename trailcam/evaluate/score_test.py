"""The final scoring: cis_test and trans_test, once, for every reported checkpoint (plan.md
section 7; R17, R26, R32, R40).

This is the one route to the test splits. `evaluate.py`, `attention.py` and `slices.py` refuse
them on the command line (`CLI_SPLITS`); this module calls their functions directly, after the
checks below, on the records of cis_test and trans_test and of no other split.

**Refusals (exit 1, before any checkpoint is loaded or any test image read).**

- No `--approved-by`, or an empty one, or one that would break the table (a `|`, a line break
  of any kind or another non-printable character): the user's go-ahead is recorded by name in
  every row.
- No `--run-seeds`: the run seeds the experiment reports, stated by the user. A run still
  training has no ledger row, so the ledger alone cannot show it is missing; the seeds given
  can (the user's decision on the ultra review of 9.01).
- A ledger that is not complete for those seeds (`check_complete`): no small CNN row, a ResNet-18
  arm A or arm B row missing for one of the seeds, or a ResNet-18 row of a seed not given (every
  seed is trained and reported for both arms, plan.md section 6). The message names each.
- Two ledger rows with one checkpoint sha256, or with one config and run seed (a rerun: the user
  picks one row and removes the other before scoring).
- A test ledger that already holds a row: the final scoring has run, and it runs once, for every
  checkpoint together (fix round of review 9.01). A run in the training ledger that was added
  after test scores were seen can therefore never be scored.
- A checkpoint with no row in `docs/ledger.md` (looked up by its sha256), and a ledger row whose
  checkpoint is not given: every run in the ledger is scored, in the one run.
- A config in `config.yaml` that no longer matches its row (config hash, model, arm, split seed),
  checkpoints of more than one split hash, a cache whose split hash differs from theirs, a ledger
  or test ledger that fails its check, a non-empty `--out`, or an overlay photo outside `assets/`.

Each checkpoint's config is the one its ledger row names, with the run seed of the row, so a
checkpoint cannot be scored under another recipe. A checkpoint whose file holds another model or
class map is refused too (`load_checkpoint`), also before any test image is read.

**One pass per checkpoint and split.** `predict_split` runs once on cis_test and once on
trans_test, and its predictions feed the headline, the metrics tables and figures
(`evaluate.write_report`) and the slice tables (`slices.write_report`): no second pass. Backbone
features are kept for the ResNet-18 and the ConvNeXt-Tiny (`FEATURE_MODELS`, 512 and 768 wide);
the small CNN's would be about 36k x 25,088 floats. `attention_pass` and `write_overlays` run
outside `torch.inference_mode()`, since Grad-CAM needs autograd. The overlays are drawn first,
from the `assets/` photos alone, so one that fails stops the run before any test split is scored.

**Output, under `--out`.**

- `headline.md`: accuracy and macro-F1 on cis_test and trans_test and the gap (cis minus trans)
  per checkpoint, and their means over the seeds of a model and arm; rows in ledger order (phase
  3: the finalist's seed order), never by a score. The leakage alarms of `train.py` are applied
  to the test scores.
- `slices.md`, and `features_by_species.png` and `features_by_camera.png` for the ResNet-18 (or,
  in phase 3, ConvNeXt-Tiny) checkpoints side by side.
- Per checkpoint, in a folder named by its run id: `metrics.md` with the per-class precision,
  recall and F1, per-camera scores and confusion matrix of both splits and three PNGs per split
  (R32); `attention.md` and `attention_in_box.png` (R26); and the Grad-CAM overlays of the
  `--overlay` photos, which must lie under `assets/`.

Counts, scores, camera location numbers, run ids and hashes only, never an image id or file name
(R03). Repo figures belong in `docs/figures/` (40 files, 20 MB), copied there by the user.

**The test ledger.** One row per checkpoint, the columns of `TEST_LEDGER_COLUMNS`: the run id,
model, arm and config of its training row, the split hash, the checkpoint sha256, the four test
scores, both gaps, the UTC time of the scoring and who approved it (the user's decision of
session 9.01: a file of its own, so `docs/ledger.md` and its 16 columns stay as they are). Like
`train.py`, each row and its alarms are printed before the rows are appended, so a failing append
loses nothing; the exit code is `ALARM_EXIT` when an alarm fired, else 0.

Each checkpoint's row and alarms are printed as soon as that checkpoint is scored (the user's
decision on review 9.01), so the log holds every row scored even if the run stops later. The rows
are appended together, last, after every checkpoint's tables and figures, the slice tables, the
feature maps and `headline.md` are written (the user's decision on the fix round). A run that
stops part way has appended nothing and is run again from the start with a new `--out`: the
checkpoints are pinned by sha256, so the scores are the same, and every output covers every
checkpoint.

On Kaggle (item 9.02), from a clone at the commit of the installed package:

    python -m trailcam.evaluate.score_test --approved-by "<name>" --run-seeds 0 1 2 \\
        --checkpoint <small_cnn run>.pt --checkpoint <arm A seed 0>.pt ... \\
        --data-root <cache> --config <clone>/config.yaml --ledger <clone>/docs/ledger.md \\
        --test-ledger <clone>/docs/test-ledger.md --out /kaggle/working/final \\
        --overlay <clone>/assets/<photo>.jpg=coyote --assets <clone>/assets

**The phase-1 mode** (`--phase 1`, the default) is the above, unchanged, and refuses by name a
phase-2 or phase-3 ledger (`ledger-phase2.md`, `ledger-phase3.md`), the phase-3 test ledger
(`test-ledger-phase3.md`) and `--baseline-ledger`, which is phase 3's.

**The phase-3 mode** (`--phase 3`, item 13.12; docs/phase3-plan.md sections 1, 2 and 7): phase
3's one further scoring, of the finalist's three checkpoints, into its own test ledger,
`docs/test-ledger-phase3.md`, so phase 1's run-once guard on `docs/test-ledger.md` stays as it is.
It takes `--ledger` named `ledger-phase3.md`, `--test-ledger` named `test-ledger-phase3.md` (which
must exist: a missing file is refused, never created, so a mistyped path cannot pass for an empty
test ledger) and `--baseline-ledger`, phase 1's `docs/ledger.md` (the default), whose
`resnet18_arm_a` rows are the keep rule's baseline. The finalist is never named on the command
line: `select_finalist` recomputes it from the two ledgers, before any checkpoint is hashed:

- every phase-3 row is of `p3_baseline`, `p3_convnext` or `p3_convnext_cp` and of run seed 0, 1
  or 2; arm A has one row per seed in the baseline ledger, and every row shares one split hash;
- the control, `p3_baseline`, has one row, of seed 0, with arm A's config hash, within phase 2's
  bands of arm A seed 0 (val macro-F1 0.003, trans_val macro-F1 0.03); if it is not bit for bit,
  a screen verdict that differs against the control and against arm A is refused (the user's);
- each candidate has a seed-0 row, and the screen is read on it: trans_val macro-F1 at least 0.03
  above arm A seed 0's, val macro-F1 at most 0.03 below. A candidate that fails it is not kept.
  One that passes must have seeds 1 and 2 too: if not, its confirm is not finished (a run may
  still be training) and the finalist cannot be chosen yet, so the scoring is refused;
- the keep rule, each delta paired by seed against arm A: the mean trans_val macro-F1 rises by
  more than 0.03, it rises on every seed, and the mean val macro-F1 drops by no more than 0.03;
- the finalist rule: the one kept candidate; of two, the higher mean trans_val macro-F1, unless
  the two means are within 0.03 (a gap of 0.03 or less), then the higher mean val macro-F1; equal
  there too is refused, for the user to decide. Nothing kept: refused, phase 3 has no scoring.

Every comparison is exact: the 4-decimal cells are read as `Decimal`, and a mean bar of 0.03 is
compared as a sum bar of 0.09 over the 3 seeds, so a gain of exactly 0.03 is not "more than".

Then it refuses `--run-seeds` other than exactly 0 1 2, and any `--checkpoint` that is not one of
the finalist's three: a file with no row in `ledger-phase3.md` (an unknown file, or one whose
bytes changed since training), a run of another config (the control, a candidate not kept, or one
kept but not the finalist), a file given twice, or a finalist seed with no file. Everything else
(the test ledger already holding a row, the configs, the cache, `--out`, the overlays, one pass
per checkpoint and split, the outputs, the rows appended last) is the phase-1 mode's. The
headline and every table say the test sets were used a second time (section 7, "Disclosure").

    python -m trailcam.evaluate.score_test --phase 3 --approved-by "<name>" --run-seeds 0 1 2 \\
        --checkpoint <finalist seed 0>.pt --checkpoint <seed 1>.pt --checkpoint <seed 2>.pt \\
        --data-root <cache> --config <clone>/config.yaml --ledger <clone>/docs/ledger-phase3.md \\
        --baseline-ledger <clone>/docs/ledger.md \\
        --test-ledger <clone>/docs/test-ledger-phase3.md --out /kaggle/working/score \\
        --overlay <clone>/assets/<photo>.jpg=coyote --assets <clone>/assets
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType
from typing import Final

import torch
from torch import nn

from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.evaluate import attention, evaluate, slices
from trailcam.evaluate.attention import SplitRun, attention_pass, load_checkpoint
from trailcam.evaluate.evaluate import SplitResult, parse_overlay, write_overlays
from trailcam.evaluate.gradcam import ASSETS_DIR, assets_image
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.evaluate.slices import SplitPredictions, predict_split, project, write_feature_maps
from trailcam.train.config import (
    CONFIG_PATH,
    Config,
    ConfigError,
    config_hash,
    load_config,
    with_overrides,
)
from trailcam.train.ledger import Row, check_ledger, split_cells
from trailcam.train.train import (
    ALARM_EXIT,
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    LEDGER_PATH,
    PHASE3_LEDGER_NAME,
    TRANS_GAP_ALARM,
    VAL_ACCURACY_ALARM,
    file_sha256,
    ledger_phase,
)

TEST_SPLITS: Final = ("cis_test", "trans_test")  # cis first: the gap is cis minus trans
TEST_LEDGER_PATH: Final = Path(__file__).resolve().parents[2] / "docs" / "test-ledger.md"
TEST_LEDGER_COLUMNS: Final = (
    "run id",
    "model",
    "arm",
    "config",
    "split hash",
    "checkpoint sha256",
    "cis_test accuracy",
    "cis_test macro-F1",
    "trans_test accuracy",
    "trans_test macro-F1",
    "accuracy gap",
    "macro-F1 gap",
    "scored at",
    "approved by",
)
HEADLINE_NAME: Final = "headline.md"
# The feature map's models, with the name the outputs give them; the small CNN's features are too
# large. ConvNeXt-Tiny's are 768 wide.
FEATURE_MODELS: Final = MappingProxyType(
    {"resnet18": "ResNet-18", "convnext_tiny": "ConvNeXt-Tiny"}
)
# How the headline's rows are ordered: phase 1's ledger order, phase 3's seed order.
LEDGER_ORDER: Final = (
    "Rows in ledger order,",
    "the order the runs were trained, never by a score; a mean row averages the seeds of one",
    "model and arm.",
)
SEED_ORDER: Final = (
    "Rows in seed order,",
    "the finalist's run seeds 0, 1 and 2, never by a score; the mean row averages its three",
    "seeds.",
)
REFUSED_EXIT: Final = 1
STAMP: Final = "%Y%m%dT%H%M%SZ"
CITATION: Final = "Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0."

# Phase 3 (docs/phase3-plan.md sections 1, 2 and 7; item 13.12).
PHASES: Final = (1, 3)  # --phase: phase 1's final scoring, or phase 3's one further scoring
PHASE3_TEST_LEDGER_NAME: Final = "test-ledger-phase3.md"
PHASE3_TEST_LEDGER_PATH: Final = TEST_LEDGER_PATH.with_name(PHASE3_TEST_LEDGER_NAME)
PHASE3_BASELINE: Final = "resnet18_arm_a"  # the keep rule's baseline, in docs/ledger.md (P3 §2)
PHASE3_CONTROL: Final = "p3_baseline"  # the baseline's seed 0 re-run, a check (P3 §2)
PHASE3_CANDIDATES: Final = ("p3_convnext", "p3_convnext_cp")  # (a) and (b) (P3 §3)
PHASE3_SEEDS: Final = (0, 1, 2)
VAL, TRANS_VAL = "val macro-F1", "trans_val macro-F1"  # the two scores the rules read
SCREEN_TRANS_GAIN: Final = Decimal("0.03")  # seed 0's trans_val: at least this above arm A's
SCREEN_VAL_DROP: Final = Decimal("0.03")  # seed 0's val: at most this below arm A's
KEEP_TRANS_GAIN: Final = Decimal("0.03")  # the mean trans_val rises by more than this
KEEP_VAL_DROP: Final = Decimal("0.03")  # the mean val drops by no more than this
FINALIST_BAND: Final = Decimal("0.03")  # two means within this: the higher mean val decides
CONTROL_VAL_BAND: Final = Decimal("0.003")  # p3_baseline seed 0 against arm A seed 0
CONTROL_TRANS_BAND: Final = Decimal("0.03")


class Refusal(Exception):
    """The scoring will not run; the message says why."""


@dataclass(frozen=True)
class Checkpoint:
    """A checkpoint given on the command line, its training ledger row and the config that row
    names, with the row's run seed."""

    path: Path
    sha256: str
    row: Row
    config: Config

    @property
    def run_id(self) -> str:
        return self.row["run id"]

    @property
    def label(self) -> str:
        """`small_cnn seed 0`, or the model, arm and seed, e.g. `resnet18 arm A seed 1`."""
        return f"{slices.checkpoint_label(self.config)} seed {self.config.run_seed}"


@dataclass(frozen=True, eq=False)
class Scored:
    """One checkpoint's single pass over each test split (in `TEST_SPLITS` order) and its
    attention-in-box and animal-removed runs."""

    checkpoint: Checkpoint
    runs: tuple[SplitPredictions, ...]
    attention: tuple[SplitRun, ...]


@dataclass(frozen=True)
class Headline:
    """Accuracy and macro-F1 on cis_test and trans_test of one checkpoint, or the mean over the
    `runs` seeds of a model and arm."""

    label: str
    run_id: str
    runs: int
    cis_images: int
    trans_images: int
    cis_accuracy: float
    cis_macro_f1: float
    trans_accuracy: float
    trans_macro_f1: float

    @property
    def accuracy_gap(self) -> float:
        return self.cis_accuracy - self.trans_accuracy

    @property
    def macro_f1_gap(self) -> float:
        return self.cis_macro_f1 - self.trans_macro_f1


def breaks_table(cell: str) -> bool:
    """A cell that would break a markdown table row: a `|`, or any character that is not
    printable, which covers every line break `str.splitlines` splits on (U+2028 among them)."""
    return "|" in cell or not cell.isprintable()


def check_approver(text: str | None) -> str:
    """The approver's name, stripped; refuses a missing or empty one, or one that would break the
    table (`breaks_table`)."""
    name = (text or "").strip()
    if not name:
        raise Refusal(
            "--approved-by is required: the final scoring runs only with the user's go-ahead"
        )
    if breaks_table(name):
        raise Refusal("--approved-by holds a |, a line break or another non-printable character")
    return name


def check_run_seeds(seeds: Sequence[int] | None) -> tuple[int, ...]:
    """The run seeds the experiment reports, sorted; refuses none given."""
    if not seeds:
        raise Refusal(
            "--run-seeds is required: the run seeds the experiment reports, so a run still "
            "training, which has no ledger row yet, is seen as missing"
        )
    return tuple(sorted(set(seeds)))


def scoring_header() -> str:
    """The two header lines of the test ledger table: the column names and the separator."""
    return "\n".join(
        [
            "| " + " | ".join(TEST_LEDGER_COLUMNS) + " |",
            "|" + " --- |" * len(TEST_LEDGER_COLUMNS),
        ]
    )


def read_test_ledger(path: Path) -> list[dict[str, str]]:
    """The rows of the test ledger at `path`. Refuses a missing file, a header that is not
    `scoring_header` exactly once, a line after it that is not a row, a row with an empty
    cell or the wrong number of cells, and a checkpoint sha256 in two rows."""
    if not path.is_file():
        raise Refusal(f"no test ledger at {path}; pass --test-ledger <clone>/docs/{path.name}")
    lines = path.read_text(encoding="utf-8").splitlines()
    header, separator = scoring_header().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1 or starts[0] + 1 >= len(lines) or lines[starts[0] + 1].strip() != separator:
        raise Refusal(f"{path.name} does not hold the header of TEST_LEDGER_COLUMNS exactly once")
    rows: list[dict[str, str]] = []
    for number, line in enumerate(lines[starts[0] + 2 :], start=starts[0] + 3):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1):
            raise Refusal(f"{path.name} line {number}: not a table row (the table ends the file)")
        cells = split_cells(stripped)
        if len(cells) != len(TEST_LEDGER_COLUMNS) or not all(cells):
            raise Refusal(f"{path.name} line {number}: not one non-empty cell per column")
        rows.append(dict(zip(TEST_LEDGER_COLUMNS, cells, strict=True)))
    for column in ("checkpoint sha256", "run id"):
        values = [row[column] for row in rows]
        if len(set(values)) != len(values):
            raise Refusal(f"{path.name}: two rows have one {column}")
    return rows


def read_ledger(path: Path) -> list[Row]:
    """The rows of the training ledger; refuses one that fails `trailcam.train.ledger`'s check."""
    rows, problems = check_ledger(path)
    if problems:
        raise Refusal(
            f"{path.name} fails its check ({len(problems)} problems, the first: {problems[0]}); "
            "run python -m trailcam.train.ledger --check"
        )
    return rows


def _config_of(row: Row, config_file: Path) -> Config:
    """The config the row names, with its run seed, checked against the row's cells."""
    try:
        name = row["config"]
        config = with_overrides(
            name, load_config(name, config_file), {"run_seed": int(row["run seed"])}
        )
    except ConfigError as error:
        raise Refusal(f"run {row['run id']}: {error}") from None
    if config.fixture:
        raise Refusal(f"run {row['run id']}: config {row['config']} is a fixture config")
    found = {
        "config hash": config_hash(config),
        "model": config.model,
        "arm": config.arm,
        "split seed": str(config.split_seed),
    }
    changed = [column for column, value in found.items() if value != row[column]]
    if changed:
        raise Refusal(
            f"run {row['run id']}: config {row['config']} in {config_file.name} no longer matches "
            f"its ledger row ({', '.join(changed)})"
        )
    return config


def check_complete(ledger: Sequence[Row], run_seeds: Sequence[int]) -> None:
    """Refuses a training ledger that does not hold every reported run (R17): a small CNN row, and
    a ResNet-18 arm A and arm B row for each of `run_seeds` and for no other seed (the user's
    decisions on review 9.01 and its ultra review). The message names what is missing."""
    missing = []
    if not any(row["model"] == "small_cnn" for row in ledger):
        missing.append("no small_cnn row")
    for arm in ("A", "B"):
        seeds = {
            int(row["run seed"])
            for row in ledger
            if (row["model"], row["arm"]) == ("resnet18", arm)
        }
        missing += [
            f"no resnet18 arm {arm} row for run seed {n}" for n in run_seeds if n not in seeds
        ]
        missing += [
            f"a resnet18 arm {arm} row of run seed {n}, which --run-seeds does not give"
            for n in sorted(seeds - set(run_seeds))
        ]
    if missing:
        raise Refusal(
            "the ledger does not hold exactly the reported runs, so a run may still be training: "
            + "; ".join(missing)
        )


def check_unique(ledger: Sequence[Row]) -> None:
    """Refuses two ledger rows with one checkpoint sha256, or with one config and run seed."""
    by_hash: dict[str, list[str]] = {}
    by_seed: dict[tuple[str, str], list[str]] = {}
    for row in ledger:
        by_hash.setdefault(row["checkpoint sha256"], []).append(row["run id"])
        by_seed.setdefault((row["config"], row["run seed"]), []).append(row["run id"])
    for digest, runs in by_hash.items():
        if len(runs) > 1:
            raise Refusal(f"ledger rows {', '.join(runs)} name one checkpoint (sha256 {digest})")
    for (config, seed), runs in by_seed.items():
        if len(runs) > 1:
            raise Refusal(
                f"ledger rows {', '.join(runs)} are runs of config {config} with run seed {seed}; "
                "the user must pick one and remove the other rows before the scoring"
            )


def resolve_checkpoints(
    paths: Sequence[Path],
    ledger: Sequence[Row],
    scored: Sequence[Mapping[str, str]],
    config_file: Path,
    run_seeds: Sequence[int],
) -> list[Checkpoint]:
    """The checkpoints given, each with its ledger row and config, in ledger order. Refuses a
    test ledger with any row, an incomplete ledger, two rows of one checkpoint or of one config
    and run seed, more than one split hash, an unknown checkpoint, one given twice, a ledger row
    whose checkpoint is not given, and two checkpoints with one label."""
    if not paths:
        raise Refusal("no --checkpoint given")
    if scored:
        runs = ", ".join(row["run id"] for row in scored)
        raise Refusal(
            f"the test ledger already holds {len(scored)} rows ({runs}): the final scoring has "
            "run, and every checkpoint is scored once, together"
        )
    check_unique(ledger)
    check_complete(ledger, run_seeds)
    split_hashes = {row["split hash"] for row in ledger}
    if len(split_hashes) != 1:
        raise Refusal(
            f"the ledger's runs were trained on {len(split_hashes)} split hashes, not one"
        )
    by_hash = {row["checkpoint sha256"]: row for row in ledger}
    given: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise Refusal(f"no checkpoint at {path}")
        digest = file_sha256(path)
        if digest in given:
            raise Refusal(
                f"{path.name} and {given[digest].name} are one checkpoint (sha256 {digest})"
            )
        if digest not in by_hash:
            raise Refusal(
                f"{path.name} (sha256 {digest}) has no row in the ledger: an unknown checkpoint"
            )
        given[digest] = path
    missing = [row["run id"] for row in ledger if row["checkpoint sha256"] not in given]
    if missing:
        raise Refusal(
            "every run in the ledger is scored in the one run; no checkpoint given for "
            + ", ".join(missing)
        )
    checkpoints = [
        Checkpoint(
            given[row["checkpoint sha256"]],
            row["checkpoint sha256"],
            row,
            _config_of(row, config_file),
        )
        for row in ledger
        if row["checkpoint sha256"] in given
    ]
    labels = [c.label for c in checkpoints]
    if len(set(labels)) != len(labels):
        raise Refusal(f"two checkpoints share a model, arm and seed: {labels}")
    return checkpoints


# Phase 3: the keep rule and the finalist rule, recomputed from the two ledgers (P3 §1, §2, §7).


@dataclass(frozen=True)
class Verdict:
    """One phase-3 candidate under the screen and the keep rule. `rows` are its rows of run
    seeds 0, 1 and 2 when it passed the screen, else its seed-0 row; the sums are over `rows`."""

    config: str
    rows: tuple[Row, ...]
    screened: bool
    kept: bool
    why: str
    trans_sum: Decimal
    val_sum: Decimal


@dataclass(frozen=True)
class Selection:
    """The verdicts of both candidates, in `PHASE3_CANDIDATES` order, and the finalist."""

    verdicts: tuple[Verdict, ...]
    finalist: Verdict
    rule: str  # how the finalist rule decided

    def lines(self) -> list[str]:
        """The verdicts and the finalist, for the log and the headline."""
        return [
            *(f"- {v.why}" for v in self.verdicts),
            f"- The finalist: {self.finalist.config} ({self.rule}).",
        ]


def _score(row: Row, column: str) -> Decimal:
    """A 4-decimal score cell, exactly (the ledger check has passed every cell's format)."""
    return Decimal(row[column])


def _by_seed(rows: Sequence[Row], config: str, ledger_name: str) -> dict[int, Row]:
    """The rows of `config`, by run seed; refuses two rows of one seed."""
    found: dict[int, Row] = {}
    for row in rows:
        if row["config"] != config:
            continue
        seed = int(row["run seed"])
        if seed in found:
            raise Refusal(
                f"{ledger_name} holds two rows of {config} with run seed {seed} "
                f"({found[seed]['run id']}, {row['run id']}): the user must pick one"
            )
        found[seed] = row
    return found


def _screen(candidate: Row, baseline: Row) -> tuple[bool, str]:
    """The screen on run seed 0 (P3 §1): trans_val macro-F1 at least `SCREEN_TRANS_GAIN` above
    the baseline's and val macro-F1 at most `SCREEN_VAL_DROP` below it."""
    trans = _score(candidate, TRANS_VAL) - _score(baseline, TRANS_VAL)
    val = _score(candidate, VAL) - _score(baseline, VAL)
    passed = trans >= SCREEN_TRANS_GAIN and val >= -SCREEN_VAL_DROP
    return passed, f"seed 0 trans_val macro-F1 {trans:+.4f}, val macro-F1 {val:+.4f}"


def _verdict(
    config: str, rows: Mapping[int, Row], baseline: Mapping[int, Row], control: Row
) -> Verdict:
    """The screen, then the keep rule, of one candidate against arm A, paired by run seed."""
    if 0 not in rows:
        raise Refusal(
            f"{PHASE3_LEDGER_NAME} has no {config} row of run seed 0: the screening batch is not "
            "complete (docs/phase3-plan.md section 4), so the finalist cannot be chosen"
        )
    passed, screen = _screen(rows[0], baseline[0])
    bit_for_bit = all(_score(control, c) == _score(baseline[0], c) for c in (VAL, TRANS_VAL))
    if not bit_for_bit and _screen(rows[0], control)[0] != passed:
        raise Refusal(
            f"{config}'s screen passes against one of {PHASE3_BASELINE} seed 0 and "
            f"{PHASE3_CONTROL} seed 0 and fails against the other: that verdict goes to the user "
            "(docs/phase3-plan.md section 2)"
        )
    if not passed:
        first = _score(rows[0], TRANS_VAL), _score(rows[0], VAL)
        return Verdict(
            config,
            (rows[0],),
            False,
            False,
            f"{config}: not kept, failed the screen ({screen})",
            *first,
        )
    missing = [str(n) for n in PHASE3_SEEDS if n not in rows]
    if missing:
        raise Refusal(
            f"{config} passed the screen ({screen}) but has no row of run seed "
            f"{', '.join(missing)}: its confirm is not finished, a run may still be training, so "
            "the keep rule and the finalist cannot be decided yet"
        )
    ordered = tuple(rows[n] for n in PHASE3_SEEDS)
    count = len(PHASE3_SEEDS)
    trans_sum = sum((_score(r, TRANS_VAL) for r in ordered), Decimal(0))
    val_sum = sum((_score(r, VAL) for r in ordered), Decimal(0))
    trans_gain = trans_sum - sum((_score(baseline[n], TRANS_VAL) for n in PHASE3_SEEDS), Decimal(0))
    val_change = val_sum - sum((_score(baseline[n], VAL) for n in PHASE3_SEEDS), Decimal(0))
    falling = [
        str(n) for n in PHASE3_SEEDS if _score(rows[n], TRANS_VAL) <= _score(baseline[n], TRANS_VAL)
    ]
    conditions = (
        trans_gain > KEEP_TRANS_GAIN * count,  # the mean bar as a sum bar: exact
        not falling,
        val_change >= -KEEP_VAL_DROP * count,
    )
    kept = all(conditions)
    why = (
        f"{config}: {'kept' if kept else 'not kept'}, passed the screen ({screen}); mean "
        f"trans_val macro-F1 {trans_gain / count:+.4f} (more than +{KEEP_TRANS_GAIN}: "
        f"{'yes' if conditions[0] else 'no'}), rises on every seed: "
        f"{'yes' if conditions[1] else 'no, not on seed ' + ', '.join(falling)}; mean val "
        f"macro-F1 {val_change / count:+.4f} (no more than {KEEP_VAL_DROP} down: "
        f"{'yes' if conditions[2] else 'no'})"
    )
    return Verdict(config, ordered, True, kept, why, trans_sum, val_sum)


def select_finalist(ledger: Sequence[Row], baseline: Sequence[Row]) -> Selection:
    """Phase 3's finalist, recomputed from `docs/ledger-phase3.md` (`ledger`) and arm A's rows
    of `docs/ledger.md` (`baseline`) by the plan's screen, keep rule and finalist rule (P3 §1),
    after the control check (P3 §2). Refuses a ledger the rules cannot be read on, and nothing
    kept (no test scoring, P3 §7). Reads scores only: no checkpoint is hashed or loaded."""
    check_unique(ledger)
    named = {PHASE3_CONTROL, *PHASE3_CANDIDATES}
    other = sorted({row["config"] for row in ledger} - named)
    if other:
        raise Refusal(
            f"{PHASE3_LEDGER_NAME} holds rows of {', '.join(other)}, which docs/phase3-plan.md "
            f"does not name: only {', '.join(sorted(named))}"
        )
    seeds = [row["run id"] for row in ledger if int(row["run seed"]) not in PHASE3_SEEDS]
    if seeds:
        raise Refusal(
            f"{PHASE3_LEDGER_NAME} rows of a run seed other than 0, 1 and 2: {', '.join(seeds)}"
        )
    base = _by_seed(baseline, PHASE3_BASELINE, "the baseline ledger")
    missing = [str(n) for n in PHASE3_SEEDS if n not in base]
    if missing or any((r["model"], r["arm"]) != ("resnet18", "A") for r in base.values()):
        raise Refusal(
            f"the baseline ledger does not hold one resnet18 arm A row of {PHASE3_BASELINE} "
            f"for each run seed 0, 1 and 2 (missing: {', '.join(missing) or 'none'}): the "
            "keep rule's baseline (docs/phase3-plan.md section 2)"
        )
    control = _by_seed(ledger, PHASE3_CONTROL, PHASE3_LEDGER_NAME)
    if set(control) != {0}:
        raise Refusal(
            f"{PHASE3_CONTROL} must have exactly one row, of run seed 0 (its seeds 1 and 2 are "
            f"phase 1's, docs/phase3-plan.md section 2), not of seeds {sorted(control)}"
        )
    check = control[0]
    if check["config hash"] != base[0]["config hash"]:
        raise Refusal(f"{PHASE3_CONTROL} does not have {PHASE3_BASELINE}'s config hash")
    split_hashes = {row["split hash"] for row in [*ledger, *base.values()]}
    if len(split_hashes) != 1:
        raise Refusal(f"the phase-3 runs and their baseline span {len(split_hashes)} split hashes")
    val_off = abs(_score(check, VAL) - _score(base[0], VAL))
    trans_off = abs(_score(check, TRANS_VAL) - _score(base[0], TRANS_VAL))
    if val_off > CONTROL_VAL_BAND or trans_off > CONTROL_TRANS_BAND:
        raise Refusal(
            f"{PHASE3_CONTROL} seed 0 is {val_off} from {PHASE3_BASELINE} seed 0 on val "
            f"macro-F1 (band {CONTROL_VAL_BAND}) and {trans_off} on trans_val macro-F1 (band "
            f"{CONTROL_TRANS_BAND}): outside a band, no screen is read until the difference is "
            "explained (docs/phase3-plan.md section 2)"
        )
    verdicts = tuple(
        _verdict(name, _by_seed(ledger, name, PHASE3_LEDGER_NAME), base, check)
        for name in PHASE3_CANDIDATES
    )
    kept = [v for v in verdicts if v.kept]
    if not kept:
        raise Refusal(
            "no phase-3 candidate was kept, so phase 3 has no test scoring (docs/phase3-plan.md "
            "section 7): " + "; ".join(v.why for v in verdicts)
        )
    if len(kept) == 1:
        return Selection(verdicts, kept[0], "the one candidate kept")
    first, second = kept
    band = FINALIST_BAND * len(PHASE3_SEEDS)  # the band of the means, as a band of the sums
    trans_gap, val_gap = first.trans_sum - second.trans_sum, first.val_sum - second.val_sum
    apart = f"{abs(trans_gap) / len(PHASE3_SEEDS):.4f}"
    if abs(trans_gap) > band:
        finalist = first if trans_gap > 0 else second
        return Selection(
            verdicts,
            finalist,
            f"the higher mean trans_val macro-F1, by {apart}, more than {FINALIST_BAND}",
        )
    if val_gap == 0:
        raise Refusal(
            f"both candidates were kept, their mean trans_val macro-F1 {apart} apart (within "
            f"{FINALIST_BAND}) and their mean val macro-F1 equal: the finalist rule does not "
            "decide, the user does"
        )
    finalist = first if val_gap > 0 else second
    return Selection(
        verdicts,
        finalist,
        f"the mean trans_val macro-F1 {apart} apart, within {FINALIST_BAND}, so the higher mean "
        f"val macro-F1, by {abs(val_gap) / len(PHASE3_SEEDS):.4f}",
    )


def _not_scored(config: str, selection: Selection) -> str:
    """Why a run of `config`, not the finalist's, is never scored."""
    verdict = next((v for v in selection.verdicts if v.config == config), None)
    if verdict is None:
        return "the control"
    return "kept, but not the finalist" if verdict.kept else "not kept under the keep rule"


def resolve_phase3_checkpoints(
    paths: Sequence[Path],
    ledger: Sequence[Row],
    baseline: Sequence[Row],
    scored: Sequence[Mapping[str, str]],
    config_file: Path,
    run_seeds: Sequence[int],
) -> tuple[Selection, list[Checkpoint]]:
    """The finalist's three checkpoints, each with its row and config, in seed order, and the
    selection. Refuses a test ledger with any row, run seeds other than 0, 1 and 2, every refusal
    of `select_finalist`, a file that is missing, given twice, has no row in the phase-3 ledger or
    is a run of another config, and a finalist seed with no file."""
    if not paths:
        raise Refusal("no --checkpoint given")
    if scored:
        runs = ", ".join(row["run id"] for row in scored)
        raise Refusal(
            f"the test ledger already holds {len(scored)} rows ({runs}): phase 3's test scoring "
            "has run, and the finalist's checkpoints are scored once, together"
        )
    if tuple(run_seeds) != PHASE3_SEEDS:
        raise Refusal(
            f"--run-seeds {' '.join(map(str, run_seeds))}: phase 3 scores run seeds 0 1 2, the "
            "keep rule's seeds (docs/phase3-plan.md section 1), and no others"
        )
    selection = select_finalist(ledger, baseline)
    finalist = selection.finalist
    by_hash = {row["checkpoint sha256"]: row for row in ledger}
    given: dict[str, Path] = {}
    for path in paths:
        if not path.is_file():
            raise Refusal(f"no checkpoint at {path}")
        digest = file_sha256(path)
        if digest in given:
            raise Refusal(
                f"{path.name} and {given[digest].name} are one checkpoint (sha256 {digest})"
            )
        row = by_hash.get(digest)
        if row is None:
            raise Refusal(
                f"{path.name} (sha256 {digest}) has no row in {PHASE3_LEDGER_NAME}: an unknown "
                "checkpoint, or a file whose bytes changed since it was trained"
            )
        if row["config"] != finalist.config:
            raise Refusal(
                f"{path.name} is run {row['run id']} of {row['config']}, "
                f"{_not_scored(row['config'], selection)}: phase 3 scores the finalist "
                f"{finalist.config}'s checkpoints only (docs/phase3-plan.md section 7)"
            )
        given[digest] = path
    missing = [r["run id"] for r in finalist.rows if r["checkpoint sha256"] not in given]
    if missing:
        raise Refusal(
            f"the finalist {finalist.config}'s three seeds are scored together, once; no "
            "checkpoint given for " + ", ".join(missing)
        )
    checkpoints = [
        Checkpoint(
            given[r["checkpoint sha256"]], r["checkpoint sha256"], r, _config_of(r, config_file)
        )
        for r in finalist.rows
    ]
    return selection, checkpoints


def finalist_files(paths: Sequence[Path], selection: Selection) -> list[Path]:
    """The files among `paths` (every `.pt` of the checkpoints' dataset, which may hold other
    runs too) whose sha256 is one of the finalist's rows, in seed order: what the notebook gives
    the scoring. Refuses a finalist row that no file matches (missing, or changed since training)
    or that two files match. The scoring checks the files given again."""
    wanted = {row["checkpoint sha256"]: row for row in selection.finalist.rows}
    found: dict[str, list[Path]] = {}
    for path in paths:
        digest = file_sha256(path)
        if digest in wanted:
            found.setdefault(digest, []).append(path)
    missing = [row["run id"] for digest, row in wanted.items() if digest not in found]
    if missing:
        raise Refusal(
            f"no file among the {len(paths)} .pt files has the sha256 of the finalist's run "
            f"{', '.join(missing)}: missing from the dataset, or changed since it was trained"
        )
    twice = [wanted[digest]["run id"] for digest, files in found.items() if len(files) > 1]
    if twice:
        raise Refusal(f"two files hold the checkpoint of run {', '.join(twice)}")
    return [found[digest][0] for digest in wanted]


def check_ledger_names(
    phase: int, ledger: Path, test_ledger: Path, baseline: Path | None
) -> Path | None:
    """Refuses another phase's ledgers, by file name; returns the baseline ledger (phase 3's
    default: `docs/ledger.md`), or None in phase 1. Phase 1 takes no `ledger-phase2.md` or
    `ledger-phase3.md`, no `test-ledger-phase3.md` and no `--baseline-ledger`; phase 3 takes
    `ledger-phase3.md`, `test-ledger-phase3.md` and a phase-1 baseline ledger only."""
    if phase == 1:
        if ledger_phase(ledger) != 1:
            raise Refusal(
                f"{ledger.name} is the phase-{ledger_phase(ledger)} ledger: the phase-1 scoring "
                "takes docs/ledger.md (phase 3's finalist: --phase 3)"
            )
        if test_ledger.name == PHASE3_TEST_LEDGER_NAME:
            raise Refusal(f"{test_ledger.name} is phase 3's test ledger (--phase 3)")
        if baseline is not None:
            raise Refusal("--baseline-ledger is phase 3's (--phase 3)")
        return None
    if ledger_phase(ledger) != 3:
        raise Refusal(
            f"{ledger.name} is not the phase-3 ledger: --phase 3 takes "
            f"--ledger <clone>/docs/{PHASE3_LEDGER_NAME}"
        )
    if test_ledger.name != PHASE3_TEST_LEDGER_NAME:
        raise Refusal(
            f"{test_ledger.name} is not phase 3's test ledger: --phase 3 takes "
            f"--test-ledger <clone>/docs/{PHASE3_TEST_LEDGER_NAME}"
        )
    chosen = LEDGER_PATH if baseline is None else baseline
    if ledger_phase(chosen) != 1:
        raise Refusal(
            f"--baseline-ledger {chosen.name} is a phase-{ledger_phase(chosen)} ledger: the keep "
            f"rule's baseline is {PHASE3_BASELINE} in docs/ledger.md"
        )
    return chosen


def check_out(out: Path) -> None:
    """`--out` must be new or empty, so one scoring's outputs never mix with another's."""
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise Refusal(f"{out} is not an empty folder")


def check_overlays(photos: Sequence[tuple[Path, int | None]], assets_root: Path) -> None:
    try:
        for path, _ in photos:
            assets_image(path, assets_root)
    except (ValueError, FileNotFoundError) as error:
        raise Refusal(str(error)) from None


def load_models(checkpoints: Sequence[Checkpoint]) -> list[nn.Module]:
    """Every checkpoint's model on the CPU; refuses a file of another model or class map."""
    models = []
    for checkpoint in checkpoints:
        try:
            models.append(load_checkpoint(checkpoint.path, checkpoint.config))
        except ValueError as error:
            raise Refusal(f"run {checkpoint.run_id}: {error}") from None
    return models


def load_test_records(
    data_root: Path, split_seed: int, split_hash: str
) -> dict[str, tuple[Record, ...]]:
    """The records of cis_test and trans_test, and of no other split; refuses a cache whose split
    hash differs from the checkpoints'."""
    meta = load_metadata(data_root / CACHE_ANNOTATIONS, data_root / DAY_NIGHT_SIDECAR)
    split = make_split(meta, split_seed)
    if split.hash != split_hash:
        raise Refusal(
            f"the cache's split hash {split.hash} for split seed {split_seed} is not the "
            f"checkpoints' {split_hash}"
        )
    return {name: split.records[name] for name in TEST_SPLITS}


def score_checkpoint(
    model: nn.Module,
    checkpoint: Checkpoint,
    records: Mapping[str, Sequence[Record]],
    image_root: Path,
    device: torch.device,
) -> Scored:
    """One `predict_split` and one `attention_pass` per test split, in `TEST_SPLITS` order. The
    attention pass runs outside inference mode: Grad-CAM needs autograd."""
    c = checkpoint.config
    model.to(device)
    runs = tuple(
        predict_split(
            model,
            records[name],
            name,
            image_root,
            device,
            batch_size=c.batch_size,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
            keep_features=c.model in FEATURE_MODELS,
        )
        for name in TEST_SPLITS
    )
    attention_runs = tuple(
        attention_pass(
            model,
            records[name],
            name,
            image_root,
            device,
            batch_size=c.batch_size,
            num_workers=c.num_workers,
            pin_memory=c.pin_memory,
        )
        for name in TEST_SPLITS
    )
    return Scored(checkpoint, runs, attention_runs)


def headline(scored: Scored) -> Headline:
    cis, trans = scored.runs
    return Headline(
        label=scored.checkpoint.label,
        run_id=scored.checkpoint.run_id,
        runs=1,
        cis_images=len(cis.labels),
        trans_images=len(trans.labels),
        cis_accuracy=accuracy(cis.predictions, cis.labels),
        cis_macro_f1=macro_f1(cis.predictions, cis.labels),
        trans_accuracy=accuracy(trans.predictions, trans.labels),
        trans_macro_f1=macro_f1(trans.predictions, trans.labels),
    )


def seed_means(scored: Sequence[Scored]) -> list[Headline]:
    """The mean scores over the seeds of each model and arm with more than one checkpoint."""
    groups: dict[str, list[Headline]] = {}
    for s in scored:
        groups.setdefault(slices.checkpoint_label(s.checkpoint.config), []).append(headline(s))
    means = []
    for label, rows in groups.items():
        if len(rows) < 2:
            continue
        means.append(
            Headline(
                label=f"{label}, mean of {len(rows)} seeds",
                run_id="",
                runs=len(rows),
                cis_images=rows[0].cis_images,
                trans_images=rows[0].trans_images,
                cis_accuracy=statistics.fmean(r.cis_accuracy for r in rows),
                cis_macro_f1=statistics.fmean(r.cis_macro_f1 for r in rows),
                trans_accuracy=statistics.fmean(r.trans_accuracy for r in rows),
                trans_macro_f1=statistics.fmean(r.trans_macro_f1 for r in rows),
            )
        )
    return means


def scoring_alarms(row: Headline) -> tuple[str, ...]:
    """The leakage alarms of `train.py` (R16) on the test scores: trans_test accuracy within
    `TRANS_GAP_ALARM` of cis_test accuracy (or above it), or cis_test accuracy above
    `VAL_ACCURACY_ALARM`."""
    alarms = []
    if row.trans_accuracy >= row.cis_accuracy - TRANS_GAP_ALARM:
        alarms.append(
            f"{row.label}: trans_test accuracy {row.trans_accuracy:.4f} is within "
            f"{TRANS_GAP_ALARM:.2f} of cis_test accuracy {row.cis_accuracy:.4f} (or above it)"
        )
    if row.cis_accuracy > VAL_ACCURACY_ALARM:
        alarms.append(
            f"{row.label}: cis_test accuracy {row.cis_accuracy:.4f} is above "
            f"{VAL_ACCURACY_ALARM:.2f}, beyond Beery et al.'s box crops"
        )
    return tuple(alarms)


def render_headline(
    rows: Sequence[Headline],
    alarms: Sequence[str],
    header: Sequence[str] = (),
    title: str = "Final scoring",
    order: Sequence[str] = LEDGER_ORDER,
    feature_models: Sequence[str] = ("resnet18",),
) -> str:
    """The headline table, the alarms and where the other outputs are. `order` says how the rows
    are ordered (`LEDGER_ORDER`, or phase 3's `SEED_ORDER`); `feature_models` are the models of
    the scored checkpoints whose feature maps were drawn, named by `FEATURE_MODELS`."""
    named = ", ".join(
        dict.fromkeys(FEATURE_MODELS[m] for m in feature_models if m in FEATURE_MODELS)
    )
    lines = [
        f"# {title}: cis_test and trans_test",
        "",
        *header,
        *([""] if header else []),
        "Accuracy and macro-F1 (over the classes present) on cis_test, cameras seen in training,",
        f"and trans_test, cameras never seen; the gap is cis minus trans. {order[0]}",
        *order[1:],
        "",
        "| checkpoint | run id | cis_test images | trans_test images | cis_test accuracy "
        "| cis_test macro-F1 | trans_test accuracy | trans_test macro-F1 | accuracy gap "
        "| macro-F1 gap |",
        "|" + " --- |" * 10,
        *(
            f"| {r.label} | {r.run_id or f'{r.runs} runs'} | {r.cis_images} | {r.trans_images} "
            f"| {r.cis_accuracy:.4f} | {r.cis_macro_f1:.4f} | {r.trans_accuracy:.4f} "
            f"| {r.trans_macro_f1:.4f} | {r.accuracy_gap:.4f} | {r.macro_f1_gap:.4f} |"
            for r in rows
        ),
        "",
        "## Leakage alarms",
        "",
        *([f"- {alarm}" for alarm in alarms] if alarms else ["None fired."]),
        "",
        "## The other outputs",
        "",
        "- `<run id>/metrics.md` and its PNGs: per class, per camera and the confusion matrix of",
        "  both splits; `<run id>/attention.md` and `attention_in_box.png`: attention-in-box and",
        "  the animal-removed test; `<run id>/gradcam_overlay_<n>.png`: the `assets/` overlays.",
        "- `slices.md`: day and night, common and rare species, cis and trans at one night share.",
        f"- `features_by_species.png`, `features_by_camera.png`: the {named} feature maps."
        if named
        else "- No feature maps: no checkpoint of a model in `FEATURE_MODELS`.",
        "",
        CITATION,
    ]
    return "\n".join(lines) + "\n"


def ledger_values(scored: Scored, stamp: str, approver: str) -> dict[str, str]:
    """One test ledger row, by `TEST_LEDGER_COLUMNS`."""
    row, h = scored.checkpoint.row, headline(scored)
    values = {
        "run id": row["run id"],
        "model": row["model"],
        "arm": row["arm"],
        "config": row["config"],
        "split hash": row["split hash"],
        "checkpoint sha256": scored.checkpoint.sha256,
        "cis_test accuracy": f"{h.cis_accuracy:.4f}",
        "cis_test macro-F1": f"{h.cis_macro_f1:.4f}",
        "trans_test accuracy": f"{h.trans_accuracy:.4f}",
        "trans_test macro-F1": f"{h.trans_macro_f1:.4f}",
        "accuracy gap": f"{h.accuracy_gap:.4f}",
        "macro-F1 gap": f"{h.macro_f1_gap:.4f}",
        "scored at": stamp,
        "approved by": approver,
    }
    assert tuple(values) == TEST_LEDGER_COLUMNS
    return values


def ledger_line(values: Mapping[str, str]) -> str:
    cells = [values[column] for column in TEST_LEDGER_COLUMNS]
    if any(breaks_table(cell) for cell in cells):
        raise ValueError("a test ledger cell holds a |, a line break or a non-printable character")
    return "| " + " | ".join(cells) + " |"


def append_rows(path: Path, lines: Sequence[str]) -> None:
    """Append `lines` to the table that ends the test ledger, rechecked first: it must still hold
    no row, since the run-once check in `prepare` came before the scoring (a second scoring, or
    rows pasted in while this one ran, are refused here)."""
    found = read_test_ledger(path)
    if found:
        raise Refusal(
            f"{path.name} gained {len(found)} rows during this run; nothing appended, the rows "
            "are printed above"
        )
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text + ("" if text.endswith("\n") else "\n") + "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )


def print_row(scored: Scored, stamp: str, approver: str) -> str:
    """Print one checkpoint's test ledger row and its alarms as soon as it is scored, and return
    the row; the rows are appended together at the end of the run (R40)."""
    line = ledger_line(ledger_values(scored, stamp, approver))
    print(f"test ledger row of {scored.checkpoint.label}:")
    print(scoring_header())
    print(line)
    sys.stdout.flush()
    for alarm in scoring_alarms(headline(scored)):
        print(f"LEAKAGE ALARM: {alarm}", file=sys.stderr)
    sys.stderr.flush()
    return line


def write_checkpoint_outputs(scored: Scored, out_dir: Path, header: Sequence[str]) -> list[Path]:
    """`metrics.md` with three PNGs per split and `attention.md` with its histogram of one
    checkpoint under `out_dir`, from its one pass."""
    results = [SplitResult(r.split, r.labels, r.predictions, r.locations) for r in scored.runs]
    table, figures = evaluate.write_report(results, out_dir, header)
    attention_table, histogram = attention.write_report(scored.attention, out_dir, header)
    return [table, *figures, attention_table, histogram]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.evaluate.score_test",
        description="The final scoring: every ledger checkpoint on cis_test and trans_test, once.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--approved-by", help="who gave the go-ahead for the final scoring (required)"
    )
    parser.add_argument(
        "--run-seeds",
        type=int,
        nargs="+",
        help="the run seeds the experiment reports, e.g. 0 1 2 (required)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="a checkpoint written by train.py with a row in the ledger; repeat for every run",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="the 512 px cache")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty output folder")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="the config.yaml path")
    parser.add_argument(
        "--phase",
        type=int,
        choices=PHASES,
        default=1,
        help="1: phase 1's final scoring (the default); 3: phase 3's finalist (item 13.12)",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=LEDGER_PATH,
        help="docs/ledger.md (phase 3: ledger-phase3.md)",
    )
    parser.add_argument(
        "--baseline-ledger",
        type=Path,
        default=None,
        help="phase 3 only: the keep rule's baseline ledger (default docs/ledger.md)",
    )
    parser.add_argument(
        "--test-ledger",
        type=Path,
        default=TEST_LEDGER_PATH,
        help="docs/test-ledger.md (phase 3: test-ledger-phase3.md)",
    )
    parser.add_argument(
        "--overlay",
        action="append",
        type=parse_overlay,
        default=[],
        metavar="PHOTO[=CLASS]",
        help="a photo under the assets folder for a Grad-CAM overlay; repeat for more",
    )
    parser.add_argument(
        "--assets", type=Path, default=ASSETS_DIR, help="the assets folder (the repo's)"
    )
    return parser


def prepare(
    args: argparse.Namespace,
) -> tuple[str, list[Checkpoint], list[nn.Module], dict[str, tuple[Record, ...]], Selection | None]:
    """Every check, in order, before any test image is read; then the test records. The
    selection is phase 3's (`--phase 3`), None in phase 1."""
    approver = check_approver(args.approved_by)
    run_seeds = check_run_seeds(args.run_seeds)
    baseline = check_ledger_names(args.phase, args.ledger, args.test_ledger, args.baseline_ledger)
    check_out(args.out)
    check_overlays(args.overlay, args.assets)
    ledger = read_ledger(args.ledger)
    scored = read_test_ledger(args.test_ledger)
    selection = None
    if baseline is not None:  # phase 3
        selection, checkpoints = resolve_phase3_checkpoints(
            args.checkpoint, ledger, read_ledger(baseline), scored, args.config, run_seeds
        )
    else:
        checkpoints = resolve_checkpoints(args.checkpoint, ledger, scored, args.config, run_seeds)
    models = load_models(checkpoints)
    first = checkpoints[0]
    records = load_test_records(args.data_root, first.config.split_seed, first.row["split hash"])
    return approver, checkpoints, models, records, selection


def phase3_header(selection: Selection, approver: str, stamp: str, first: Checkpoint) -> list[str]:
    """The header of every phase-3 output: the go-ahead, the split, the selection and the
    disclosure of the second use of the test sets (docs/phase3-plan.md section 7)."""
    return [
        f"Phase 3's one test scoring (docs/phase3-plan.md section 7). Approved by {approver}, "
        f"scored at {stamp} (UTC). Split seed {first.config.split_seed}, split hash "
        f"{first.row['split hash']}. The finalist {selection.finalist.config}, run seeds "
        f"{', '.join(map(str, PHASE3_SEEDS))}, recomputed from docs/ledger-phase3.md and "
        f"{PHASE3_BASELINE}'s rows of docs/ledger.md by the keep rule and the finalist rule:",
        *selection.lines(),
        "",
        "The test sets were used a second time here, for this one recipe, chosen on val and "
        "trans_val only, after phase 1's test scores (docs/test-ledger.md, 2026-10-18) had been "
        "seen. Report the two side by side, never one for the other.",
    ]


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        approver, checkpoints, models, records, selection = prepare(args)
    except Refusal as refusal:
        print(f"score_test refused: {refusal}", file=sys.stderr)
        return REFUSED_EXIT

    stamp = datetime.now(UTC).strftime(STAMP)
    first = checkpoints[0]
    if selection is None:
        title = "Final scoring"
        common = [
            f"Approved by {approver}, scored at {stamp} (UTC). Split seed "
            f"{first.config.split_seed}, split hash {first.row['split hash']}. ResNet-18 run "
            f"seeds reported: {', '.join(map(str, check_run_seeds(args.run_seeds)))}."
        ]
    else:
        title = "Phase 3's test scoring"
        common = phase3_header(selection, approver, stamp, first)
        print("\n".join(common), flush=True)
    out: Path = args.out
    for model, checkpoint in zip(models, checkpoints, strict=True):
        # The assets/ overlays need no test image, so they come first: a photo whose Grad-CAM
        # map is zero stops the run before any test split is scored. Outside inference mode.
        write_overlays(model, args.overlay, out / checkpoint.run_id, args.assets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored: list[Scored] = []
    ledger_lines: list[str] = []
    for model, checkpoint in zip(models, checkpoints, strict=True):
        result = score_checkpoint(model, checkpoint, records, args.data_root / CACHE_IMAGES, device)
        model.to("cpu")
        ledger_lines.append(print_row(result, stamp, approver))
        header = [
            f"{title} of {checkpoint.label}: run {checkpoint.run_id}, config "
            f"{checkpoint.row['config']}, checkpoint sha256 {checkpoint.sha256}.",
            *common,
        ]
        write_checkpoint_outputs(result, out / checkpoint.run_id, header)
        scored.append(result)

    rows = [headline(s) for s in scored]
    alarms = [alarm for row in rows for alarm in scoring_alarms(row)]
    table = out / HEADLINE_NAME
    names = [s.checkpoint.config.model for s in scored]
    order = LEDGER_ORDER if selection is None else SEED_ORDER
    text = render_headline([*rows, *seed_means(scored)], alarms, common, title, order, names)
    table.write_text(text, "utf-8")
    lines = [f"- {c.label}: run {c.run_id}, checkpoint sha256 {c.sha256}" for c in checkpoints]
    slices.write_report({s.checkpoint.label: s.runs for s in scored}, out, [*common, "", *lines])
    projections = {
        s.checkpoint.label: project(s.runs)
        for s in scored
        if s.checkpoint.config.model in FEATURE_MODELS
    }
    if projections:
        write_feature_maps(projections, out)
    print(table.read_text(encoding="utf-8"))
    sys.stdout.flush()
    try:  # last: a run that stops earlier appends nothing
        append_rows(args.test_ledger, ledger_lines)
    except Refusal as refusal:
        print(f"score_test refused to append: {refusal}", file=sys.stderr)
        return REFUSED_EXIT
    print(f"appended {len(ledger_lines)} rows to {args.test_ledger}; outputs under {out}")
    return ALARM_EXIT if alarms else 0


if __name__ == "__main__":
    raise SystemExit(main())
