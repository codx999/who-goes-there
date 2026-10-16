"""The final scoring: cis_test and trans_test, once, for every reported checkpoint (plan.md
section 7; R17, R26, R32, R40).

This is the one route to the test splits. `evaluate.py`, `attention.py` and `slices.py` refuse
them on the command line (`CLI_SPLITS`); this module calls their functions directly, after the
checks below, on the records of cis_test and trans_test and of no other split.

**Refusals (exit 1, before any checkpoint is loaded or any test image read).**

- No `--approved-by`, or an empty one: the user's go-ahead is recorded by name in every row.
- A ledger that is not complete (`check_complete`, the user's decision on review 9.01): no small
  CNN row, or ResNet-18 arm A and arm B rows whose run seeds differ (every seed is trained and
  reported for both arms, plan.md section 6). A run still training has no row yet, so this is
  how a scoring that would leave it out is refused; the message names what is missing.
- Two ledger rows with one checkpoint sha256, or with one config and run seed (a rerun: the user
  picks one row and removes the other before scoring).
- A checkpoint with no row in `docs/ledger.md` (looked up by its sha256), or one that already has
  a row in `docs/test-ledger.md`: each checkpoint is scored once.
- A ledger row whose checkpoint is neither given nor already scored: every run in the ledger is
  scored, in one run unless a run stopped part way (below).
- A config in `config.yaml` that no longer matches its row (config hash, model, arm, split seed),
  checkpoints of more than one split hash, a cache whose split hash differs from theirs, a ledger
  or test ledger that fails its check, a non-empty `--out`, or an overlay photo outside `assets/`.

Each checkpoint's config is the one its ledger row names, with the run seed of the row, so a
checkpoint cannot be scored under another recipe. A checkpoint whose file holds another model or
class map is refused too (`load_checkpoint`), also before any test image is read.

**One pass per checkpoint and split.** `predict_split` runs once on cis_test and once on
trans_test, and its predictions feed the headline, the metrics tables and figures
(`evaluate.write_report`) and the slice tables (`slices.write_report`): no second pass. Backbone
features are kept for the ResNet-18 only (`FEATURE_MODELS`); the small CNN's would be about
36k x 25,088 floats. `attention_pass` and `write_overlays` run outside `torch.inference_mode()`,
since Grad-CAM needs autograd. The overlays are drawn first, from the `assets/` photos alone, so
one that fails stops the run before any test split is scored.

**Output, under `--out`.**

- `headline.md`: accuracy and macro-F1 on cis_test and trans_test and the gap (cis minus trans)
  per checkpoint, and their means over the seeds of a model and arm; rows in ledger order, never
  by a score. The leakage alarms of `train.py` are applied to the test scores.
- `slices.md`, and `features_by_species.png` and `features_by_camera.png` for the ResNet-18
  checkpoints side by side.
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
`train.py`, each row and its alarms are printed before the row is appended, so a failing append
loses nothing; the exit code is `ALARM_EXIT` when an alarm fired, else 0.

Each checkpoint's row is printed and appended as soon as that checkpoint is scored, before its
tables and figures, the next checkpoint, the slice tables, the feature maps and `headline.md`
(the user's decision on review 9.01). A run that stops part way has therefore logged every
checkpoint it scored. To finish, run again with a new `--out` and the checkpoints that have no
test row yet: the ones already scored are refused if given, and count as present for the
completeness check. The slice tables, feature maps and headline of that second run hold its own
checkpoints only.

On Kaggle (item 9.02), from a clone at the commit of the installed package:

    python -m trailcam.evaluate.score_test --approved-by "<name>" \\
        --checkpoint <small_cnn run>.pt --checkpoint <arm A seed 0>.pt ... \\
        --data-root <cache> --config <clone>/config.yaml --ledger <clone>/docs/ledger.md \\
        --test-ledger <clone>/docs/test-ledger.md --out /kaggle/working/final \\
        --overlay <clone>/assets/<photo>.jpg=coyote --assets <clone>/assets
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
    TRANS_GAP_ALARM,
    VAL_ACCURACY_ALARM,
    file_sha256,
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
FEATURE_MODELS: Final = ("resnet18",)  # the feature map; the small CNN's features are too large
REFUSED_EXIT: Final = 1
STAMP: Final = "%Y%m%dT%H%M%SZ"
CITATION: Final = "Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0."


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


def check_approver(text: str | None) -> str:
    """The approver's name, stripped; refuses a missing or empty one, or one that would break the
    table (a | or a line break)."""
    name = (text or "").strip()
    if not name:
        raise Refusal(
            "--approved-by is required: the final scoring runs only with the user's go-ahead"
        )
    if "|" in name or "\n" in name or "\r" in name:
        raise Refusal("--approved-by holds a | or a line break")
    return name


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
        raise Refusal(f"no test ledger at {path}; pass --test-ledger <clone>/docs/test-ledger.md")
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
    hashes = [row["checkpoint sha256"] for row in rows]
    if len(set(hashes)) != len(hashes):
        raise Refusal(f"{path.name}: a checkpoint has more than one row")
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


def check_complete(ledger: Sequence[Row]) -> None:
    """Refuses a training ledger that does not hold every reported run (R17; the user's decision
    on review 9.01): a small CNN row, and ResNet-18 arm A and arm B rows with one non-empty set
    of run seeds. The message names what is missing."""
    missing = []
    if not any(row["model"] == "small_cnn" for row in ledger):
        missing.append("no small_cnn row")
    seeds = {
        arm: {
            int(row["run seed"])
            for row in ledger
            if (row["model"], row["arm"]) == ("resnet18", arm)
        }
        for arm in ("A", "B")
    }
    if not seeds["A"] | seeds["B"]:
        missing.append("no resnet18 row")
    for arm, other in (("A", "B"), ("B", "A")):
        for seed in sorted(seeds[other] - seeds[arm]):
            missing.append(f"no resnet18 arm {arm} row for run seed {seed} (arm {other} has one)")
    if missing:
        raise Refusal(
            "the ledger does not hold every reported run, so a run may still be training: "
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
) -> list[Checkpoint]:
    """The checkpoints given, each with its ledger row and config, in ledger order. Refuses an
    incomplete ledger, two rows of one checkpoint or of one config and run seed, more than one
    split hash, an unknown checkpoint, one given twice, one already scored, a ledger row whose
    checkpoint is neither given nor already scored, and two checkpoints with one label."""
    if not paths:
        raise Refusal("no --checkpoint given")
    check_unique(ledger)
    check_complete(ledger)
    split_hashes = {row["split hash"] for row in ledger}
    if len(split_hashes) != 1:
        raise Refusal(
            f"the ledger's runs were trained on {len(split_hashes)} split hashes, not one"
        )
    by_hash = {row["checkpoint sha256"]: row for row in ledger}
    done = {row["checkpoint sha256"]: row["run id"] for row in scored}
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
        if digest in done:
            raise Refusal(
                f"{path.name} (run {done[digest]}) was already scored on the test splits; "
                "each checkpoint is scored once"
            )
        given[digest] = path
    missing = [
        row["run id"]
        for row in ledger
        if row["checkpoint sha256"] not in given and row["checkpoint sha256"] not in done
    ]
    if missing:
        raise Refusal(
            "every run in the ledger is scored; no checkpoint given, and no test row, for "
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
    rows: Sequence[Headline], alarms: Sequence[str], header: Sequence[str] = ()
) -> str:
    """The headline table, the alarms and where the other outputs are."""
    lines = [
        "# Final scoring: cis_test and trans_test",
        "",
        *header,
        *([""] if header else []),
        "Accuracy and macro-F1 (over the classes present) on cis_test, cameras seen in training,",
        "and trans_test, cameras never seen; the gap is cis minus trans. Rows in ledger order,",
        "the order the runs were trained, never by a score; a mean row averages the seeds of one",
        "model and arm.",
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
        "- `features_by_species.png`, `features_by_camera.png`: the ResNet-18 feature maps.",
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
    if any("|" in cell or "\n" in cell for cell in cells):
        raise ValueError("a test ledger cell holds a | or a line break")
    return "| " + " | ".join(cells) + " |"


def append_rows(path: Path, lines: Sequence[str]) -> None:
    """Append `lines` to the table that ends the test ledger, rechecked first."""
    read_test_ledger(path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text + ("" if text.endswith("\n") else "\n") + "".join(f"{line}\n" for line in lines),
        encoding="utf-8",
    )


def log_row(scored: Scored, stamp: str, approver: str, test_ledger: Path) -> str:
    """Print one checkpoint's test ledger row and its alarms, then append the row: right after
    the checkpoint is scored, before anything else is written (R40)."""
    line = ledger_line(ledger_values(scored, stamp, approver))
    print(f"test ledger row of {scored.checkpoint.label}:")
    print(scoring_header())
    print(line)
    sys.stdout.flush()
    for alarm in scoring_alarms(headline(scored)):  # before the append, as in train.py
        print(f"LEAKAGE ALARM: {alarm}", file=sys.stderr)
    sys.stderr.flush()
    append_rows(test_ledger, [line])
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
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="a checkpoint written by train.py with a row in the ledger; repeat for every run",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="the 512 px cache")
    parser.add_argument("--out", type=Path, required=True, help="a new or empty output folder")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="the config.yaml path")
    parser.add_argument("--ledger", type=Path, default=LEDGER_PATH, help="docs/ledger.md")
    parser.add_argument(
        "--test-ledger", type=Path, default=TEST_LEDGER_PATH, help="docs/test-ledger.md"
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
) -> tuple[str, list[Checkpoint], list[nn.Module], dict[str, tuple[Record, ...]]]:
    """Every check, in order, before any test image is read; then the test records."""
    approver = check_approver(args.approved_by)
    check_out(args.out)
    check_overlays(args.overlay, args.assets)
    ledger = read_ledger(args.ledger)
    scored = read_test_ledger(args.test_ledger)
    checkpoints = resolve_checkpoints(args.checkpoint, ledger, scored, args.config)
    models = load_models(checkpoints)
    first = checkpoints[0]
    records = load_test_records(args.data_root, first.config.split_seed, first.row["split hash"])
    return approver, checkpoints, models, records


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        approver, checkpoints, models, records = prepare(args)
    except Refusal as refusal:
        print(f"score_test refused: {refusal}", file=sys.stderr)
        return REFUSED_EXIT

    stamp = datetime.now(UTC).strftime(STAMP)
    first = checkpoints[0]
    common = (
        f"Approved by {approver}, scored at {stamp} (UTC). Split seed {first.config.split_seed}, "
        f"split hash {first.row['split hash']}."
    )
    out: Path = args.out
    for model, checkpoint in zip(models, checkpoints, strict=True):
        # The assets/ overlays need no test image, so they come first: a photo whose Grad-CAM
        # map is zero stops the run before any test split is scored. Outside inference mode.
        write_overlays(model, args.overlay, out / checkpoint.run_id, args.assets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scored: list[Scored] = []
    for model, checkpoint in zip(models, checkpoints, strict=True):
        result = score_checkpoint(model, checkpoint, records, args.data_root / CACHE_IMAGES, device)
        model.to("cpu")
        log_row(result, stamp, approver, args.test_ledger)
        header = [
            f"Final scoring of {checkpoint.label}: run {checkpoint.run_id}, config "
            f"{checkpoint.row['config']}, checkpoint sha256 {checkpoint.sha256}.",
            common,
        ]
        write_checkpoint_outputs(result, out / checkpoint.run_id, header)
        scored.append(result)

    rows = [headline(s) for s in scored]
    alarms = [alarm for row in rows for alarm in scoring_alarms(row)]
    table = out / HEADLINE_NAME
    table.write_text(render_headline([*rows, *seed_means(scored)], alarms, [common]), "utf-8")
    lines = [f"- {c.label}: run {c.run_id}, checkpoint sha256 {c.sha256}" for c in checkpoints]
    slices.write_report({s.checkpoint.label: s.runs for s in scored}, out, [common, "", *lines])
    projections = {
        s.checkpoint.label: project(s.runs)
        for s in scored
        if s.checkpoint.config.model in FEATURE_MODELS
    }
    if projections:
        write_feature_maps(projections, out)
    print(table.read_text(encoding="utf-8"))
    print(f"appended {len(scored)} rows to {args.test_ledger}; outputs under {out}")
    return ALARM_EXIT if alarms else 0


if __name__ == "__main__":
    raise SystemExit(main())
