"""Check `docs/ledger.md` and summarise its val scores (plan.md sections 5 and 6; R14, R25, R40).

`python -m trailcam.train.ledger --check docs/ledger.md` reads the ledger, prints every problem
it finds to stderr and exits 1, or prints a summary of the val scores per model, arm and seed and
exits 0. A ledger with the header and no rows passes.

The columns are `LEDGER_COLUMNS` of `trailcam/train/train.py`, the header `ledger_header`; this
module keeps no copy of them. The header must appear once, and every line after it must be a row
of the table, since `append_row` adds rows at the end of the file. Each row is checked:

- it has one non-empty cell per column (a missing column shows as a short row);
- the run id is `<config>-seed<run seed>-<UTC start, YYYYMMDDTHHMMSSZ>` of its own config and run
  seed, and no other row has it (R40);
- the model and the arm are names `config.py` accepts, and the config is not `dryrun`: the dry
  run never appends (R37);
- the config hash, split hash and checkpoint sha256 are 64 lower-case hex digits, as `hexdigest`
  writes them;
- the seeds and epoch counts are whole numbers, the best epoch within the epochs run, the
  scheduler `<name> warmup <epochs>`, and the four scores four-decimal fractions in [0, 1].

Across rows: the rows of one config share model and arm; one split seed has one split hash and
one split hash one split seed (R14); the rows of one model and run seed, arm A and arm B alike,
share the split hash and the config hash (R25); every row of one model shares the config hash,
whatever its split seed, run seed and arm, so every seed pairing compares one recipe on the one
split seed of plan.md section 5 (section 6; review 7.01); and the rows of one config hash agree on
the cells it stands for: model, split seed and scheduler (review 7.01). The config hash leaves out
`arm` and `run_seed` (`config_hash` in `trailcam/train/config.py`, the user's decision on review
5.02), so a pair, or a seed, that differs in anything but the arm and the run seed fails here. By
design, so does any logged run of a model with a changed recipe or split seed (the user's decision
on review 7.01): such a run belongs in no ledger row next to the experiment's.

The summary (R17) lists val accuracy and val macro-F1 per model, arm, split seed and run seed,
sorted by those keys and never by a score; a group of rows shows their means, and a model and arm
with more than one row gets a line of the mean over all of them.
trans_val is shown beside them as reported, never ranked or picked by. Problems and the summary
hold column names, line numbers, run ids, seeds, hashes and scores only (R03).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from typing import Final

from trailcam.data.dataset import ARMS
from trailcam.model.factory import MODEL_NAMES
from trailcam.train.config import DRY_RUN, SCHEDULERS
from trailcam.train.train import LEDGER_COLUMNS, LEDGER_PATH, ledger_header

HASH_COLUMNS: Final = ("config hash", "split hash", "checkpoint sha256")
SCORE_COLUMNS: Final = (
    "val accuracy",
    "val macro-F1",
    "trans_val accuracy",
    "trans_val macro-F1",
)
SHARED_BY_SEED: Final = ("split hash", "config hash")  # arm A and B of one model and run seed
SHARED_BY_CONFIG_HASH: Final = ("model", "split seed", "scheduler")  # cells the hash stands for

_HEX64: Final = re.compile(r"[0-9a-f]{64}")
_WHOLE: Final = re.compile(r"0|[1-9][0-9]*")
_SCORE: Final = re.compile(r"[01]\.[0-9]{4}")
_CONFIG_NAME: Final = re.compile(r"[A-Za-z0-9_]+")
_SCHEDULER: Final = re.compile(rf"({'|'.join(map(re.escape, SCHEDULERS))}) warmup (0|[1-9][0-9]*)")
_STARTED: Final = "%Y%m%dT%H%M%SZ"


class LedgerError(ValueError):
    """The ledger has no table to read: missing, or its header is not `ledger_header`."""


@dataclass(frozen=True)
class Row:
    line: int  # from 1, in the ledger file
    values: Mapping[str, str]  # by `LEDGER_COLUMNS`; every cell present and non-empty once checked

    def __getitem__(self, column: str) -> str:
        return self.values[column]


def split_cells(line: str) -> list[str]:
    """The stripped cells of a markdown table line `| a | b |`."""
    inner = line.strip()
    return [cell.strip() for cell in inner[1:-1].split("|")]


def read_table(text: str) -> tuple[list[Row], list[str]]:
    """The rows under the ledger's header, and the problems of the lines that are not rows.

    A row with the wrong number of cells is a problem and is not returned; the cell checks are
    `check_rows`'. Refuses text without the header exactly once (`LedgerError`)."""
    lines = text.splitlines()
    header, separator = ledger_header().splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == header]
    if len(starts) != 1:
        raise LedgerError(f"the ledger holds the header of LEDGER_COLUMNS {len(starts)} times")
    start = starts[0]
    if start + 1 >= len(lines) or lines[start + 1].strip() != separator:
        raise LedgerError("the header of LEDGER_COLUMNS is not followed by its separator line")
    rows: list[Row] = []
    problems: list[str] = []
    for number, line in enumerate(lines[start + 2 :], start=start + 3):
        stripped = line.strip()
        if not (stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 1):
            problems.append(f"line {number}: not a table row (the table must end the ledger)")
            continue
        cells = split_cells(stripped)
        if len(cells) != len(LEDGER_COLUMNS):
            problems.append(
                f"line {number}: {len(cells)} cells, expected {len(LEDGER_COLUMNS)} "
                "(LEDGER_COLUMNS); a column is missing or a cell holds a |"
            )
            continue
        rows.append(Row(number, dict(zip(LEDGER_COLUMNS, cells, strict=True))))
    return rows, problems


def _cell_problems(row: Row) -> list[str]:
    """What is wrong with the cells of one row, column by column."""
    empty = [column for column in LEDGER_COLUMNS if not row[column]]
    if empty:
        return [f"empty column {column!r}" for column in empty]
    found: list[str] = []

    def expect(ok: bool, column: str, what: str) -> None:
        if not ok:
            found.append(f"{column} {row[column]!r} is not {what}")

    for column in HASH_COLUMNS:
        expect(bool(_HEX64.fullmatch(row[column])), column, "64 lower-case hex digits")
    for column in ("split seed", "run seed", "epochs run", "best epoch"):
        expect(bool(_WHOLE.fullmatch(row[column])), column, "a whole number")
    for column in SCORE_COLUMNS:
        ok = bool(_SCORE.fullmatch(row[column])) and float(row[column]) <= 1.0
        expect(ok, column, "a score in [0, 1] with four decimals")
    expect(row["model"] in MODEL_NAMES, "model", f"one of {MODEL_NAMES}")
    expect(row["arm"] in ARMS, "arm", f"one of {ARMS}")
    expect(bool(_CONFIG_NAME.fullmatch(row["config"])), "config", "a config name")
    if row["config"] == DRY_RUN:
        found.append(f"config {DRY_RUN!r}: the dry run never appends a row")
    expect(bool(_SCHEDULER.fullmatch(row["scheduler"])), "scheduler", "'<name> warmup <epochs>'")
    if _WHOLE.fullmatch(row["epochs run"]) and _WHOLE.fullmatch(row["best epoch"]):
        if not 1 <= int(row["best epoch"]) <= int(row["epochs run"]):
            found.append(
                f"best epoch {row['best epoch']} is not within the {row['epochs run']} epochs run"
            )
    prefix = f"{row['config']}-seed{row['run seed']}-"
    run_id = row["run id"]
    started = run_id.removeprefix(prefix)
    if started == run_id or not _is_timestamp(started):
        found.append(
            f"run id {run_id!r} is not '{prefix}<UTC start, YYYYMMDDTHHMMSSZ>' of its own "
            "config and run seed"
        )
    return found


def _is_timestamp(text: str) -> bool:
    try:
        return datetime.strptime(text, _STARTED).strftime(_STARTED) == text
    except ValueError:
        return False


def _disagreements(
    rows: Iterable[Row],
    key: Callable[[Row], tuple[str, ...]],
    columns: Sequence[str],
    what: Callable[[tuple[str, ...]], str],
) -> list[str]:
    """For each group of rows with one `key`, every column of `columns` it does not agree on."""
    groups: dict[tuple[str, ...], list[Row]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    found: list[str] = []
    for group_key, group in groups.items():
        for column in columns:
            if len({row[column] for row in group}) > 1:
                lines = ", ".join(str(row.line) for row in group)
                found.append(f"{what(group_key)} disagree on {column} (lines {lines})")
    return found


def check_rows(rows: Sequence[Row]) -> tuple[list[Row], list[str]]:
    """The rows whose cells are well formed, and every problem of the rows: cell by cell, then
    across the well-formed rows."""
    problems: list[str] = []
    good: list[Row] = []
    for row in rows:
        found = _cell_problems(row)
        problems += [f"line {row.line}: {problem}" for problem in found]
        if not found:
            good.append(row)

    by_id: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_id[row["run id"]].append(row.line)
    for run_id, lines in by_id.items():
        if run_id and len(lines) > 1:
            problems.append(f"run id {run_id!r} has {len(lines)} rows (lines {lines}), not one")

    problems += _disagreements(
        good, lambda r: (r["config"],), ("model", "arm"), lambda k: f"the rows of config {k[0]!r}"
    )
    problems += _disagreements(
        good, lambda r: (r["split seed"],), ("split hash",), lambda k: f"split seed {k[0]}'s rows"
    )
    problems += _disagreements(
        good, lambda r: (r["split hash"],), ("split seed",), lambda k: "the rows of one split hash"
    )
    problems += _disagreements(
        good,
        lambda r: (r["model"], r["run seed"]),
        SHARED_BY_SEED,
        lambda k: f"the rows of {k[0]} run seed {k[1]} (arm A and arm B alike)",
    )
    problems += _disagreements(
        good,
        lambda r: (r["model"],),
        ("config hash",),
        lambda k: f"the rows of {k[0]} (every split seed, run seed and arm)",
    )
    problems += _disagreements(
        good,
        lambda r: (r["config hash"],),
        SHARED_BY_CONFIG_HASH,
        lambda k: f"the rows of config hash {k[0]}",
    )
    return good, problems


def check_ledger(path: Path) -> tuple[list[Row], list[str]]:
    """The rows of the ledger at `path` and its problems; no problems means the ledger passes."""
    if not path.is_file():
        return [], [f"no ledger at {path}"]
    try:
        rows, problems = read_table(path.read_text(encoding="utf-8"))
    except LedgerError as error:
        return [], [str(error)]
    good, found = check_rows(rows)
    return good, problems + found


def summary(rows: Sequence[Row]) -> str:
    """A markdown table of the scores per model, arm, split seed and run seed, sorted by those
    keys (R17: never by a score); under the seeds of a model and arm with more than one row, the
    mean over all of them."""
    head = ("model", "arm", "split seed", "run seed", "rows", *SCORE_COLUMNS)
    lines = [f"ledger: {len(rows)} rows, every check passed"]
    if not rows:
        return "\n".join(lines)
    lines += ["", "| " + " | ".join(head) + " |", "|" + " --- |" * len(head)]

    def line(keys: Sequence[str], group: Sequence[Row]) -> str:
        scores = [f"{fmean(float(row[column]) for row in group):.4f}" for column in SCORE_COLUMNS]
        return "| " + " | ".join([*keys, str(len(group)), *scores]) + " |"

    by_arm: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for row in rows:
        by_arm[(row["model"], row["arm"])].append(row)
    for (model, arm), arm_rows in sorted(by_arm.items()):
        by_seed: dict[tuple[int, int], list[Row]] = defaultdict(list)
        for row in arm_rows:
            by_seed[(int(row["split seed"]), int(row["run seed"]))].append(row)
        for (split_seed, run_seed), group in sorted(by_seed.items()):
            lines.append(line((model, arm, str(split_seed), str(run_seed)), group))
        if len(arm_rows) > 1:
            lines.append(line((model, arm, "all", "all"), arm_rows))
    lines += [
        "",
        "Scores of a group of rows are means. trans_val is reported, never selected on (R17).",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m trailcam.train.ledger",
        description="Check a ledger's rows and print its val scores per model, arm and seed.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--check",
        type=Path,
        default=LEDGER_PATH,
        metavar="LEDGER",
        help="the ledger to check (default: docs/ledger.md of this checkout)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, problems = check_ledger(args.check)
    if problems:
        for problem in problems:
            print(f"ledger check: {problem}", file=sys.stderr)
        count = f"{len(problems)} problem{'s' if len(problems) > 1 else ''}"
        print(f"ledger check: {count} in {args.check}", file=sys.stderr)
        return 1
    print(summary(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
