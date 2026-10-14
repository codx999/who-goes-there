"""trailcam/train/ledger.py: the ledger checker and its val summary (plan.md sections 5 and 6;
R14 one split hash per split seed, R17 trans_val never ranked, R25 arm A and arm B rows of one seed
share split hash and config hash, R37 no dry-run row, R40 one well-formed row per run id).

Every ledger here is written into `tmp_path` from rows built with `train.ledger_row`, so the rows
have exactly the shape `train.py` appends; the hashes are SHA-256 digests of short labels.
"""

import hashlib
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from trailcam.train import ledger, train
from trailcam.train.ledger import check_ledger, main, summary
from trailcam.train.train import LEDGER_COLUMNS, LEDGER_PATH, ledger_header, ledger_row

REPO = Path(__file__).resolve().parents[1]
KEYS = {column.replace(" ", "_").replace("-", "_").lower(): column for column in LEDGER_COLUMNS}


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def values(**changes: str) -> dict[str, str]:
    """One well-formed row of `resnet18_arm_a`, run seed 0, with `changes` applied; a key is its
    column in lower case with `_` for spaces and dashes (`val_macro_f1`)."""
    row = {
        "run id": "resnet18_arm_a-seed0-20261015T101500Z",
        "model": "resnet18",
        "arm": "A",
        "config": "resnet18_arm_a",
        "config hash": sha("config"),
        "split seed": "42",
        "run seed": "0",
        "split hash": sha("split 42"),
        "epochs run": "8",
        "best epoch": "5",
        "scheduler": "cosine warmup 1",
        "val accuracy": "0.8125",
        "val macro-F1": "0.7400",
        "trans_val accuracy": "0.6010",
        "trans_val macro-F1": "0.5120",
        "checkpoint sha256": sha("checkpoint A0"),
    }
    for key, value in changes.items():
        row[KEYS[key]] = value
    return row


def arm_b(seed: int = 0, **changes: str) -> dict[str, str]:
    base = {
        "run_id": f"resnet18_arm_b-seed{seed}-20261016T090000Z",
        "arm": "B",
        "config": "resnet18_arm_b",
        "run_seed": str(seed),
        "checkpoint_sha256": sha(f"checkpoint B{seed}"),
    }
    return values(**{**base, **changes})


def arm_a(seed: int = 0, **changes: str) -> dict[str, str]:
    base = {
        "run_id": f"resnet18_arm_a-seed{seed}-20261015T101500Z",
        "run_seed": str(seed),
        "checkpoint_sha256": sha(f"checkpoint A{seed}"),
    }
    return values(**{**base, **changes})


def write(tmp_path: Path, *rows: Mapping[str, str], tail: str = "") -> Path:
    """A ledger with a preamble, the header and `rows`, as `append_row` leaves it."""
    lines = ["# Ledger", "", "Some prose.", "", ledger_header(), *map(ledger_row, rows)]
    path = tmp_path / "ledger.md"
    path.write_text("\n".join(lines) + "\n" + tail, encoding="utf-8")
    return path


def problems_of(path: Path) -> list[str]:
    return check_ledger(path)[1]


def test_the_columns_are_train_pys_own() -> None:
    assert ledger.LEDGER_COLUMNS is train.LEDGER_COLUMNS
    assert ledger.LEDGER_PATH is train.LEDGER_PATH
    assert len(LEDGER_COLUMNS) == 16


def test_the_real_ledger_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--check", str(LEDGER_PATH)]) == 0
    assert "every check passed" in capsys.readouterr().out


def test_the_command_line_exits_0_on_the_real_ledger_and_1_on_a_bad_row(tmp_path: Path) -> None:
    def run(path: Path) -> subprocess.CompletedProcess[str]:
        command = [sys.executable, "-m", "trailcam.train.ledger", "--check", str(path)]
        return subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)

    assert run(REPO / "docs" / "ledger.md").returncode == 0
    bad = write(tmp_path, arm_a(), arm_b(config_hash=sha("other")))
    result = run(bad)
    assert result.returncode == 1
    assert "config hash" in result.stderr


def test_a_well_formed_ab_pair_passes(tmp_path: Path) -> None:
    rows, problems = check_ledger(write(tmp_path, arm_a(), arm_b()))
    assert problems == []
    assert [row["arm"] for row in rows] == ["A", "B"]
    assert [row.line for row in rows] == [7, 8]


def test_a_missing_column_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    short = ledger_row(arm_b()).replace(f" {sha('checkpoint B0')} |", "")
    path = write(tmp_path, arm_a())
    path.write_text(path.read_text() + short + "\n")
    assert main(["--check", str(path)]) == 1
    err = capsys.readouterr().err
    assert "line 8: 15 cells, expected 16" in err


def test_an_empty_cell_fails(tmp_path: Path) -> None:
    row = ledger_row(arm_a()).replace("| A |", "|  |")
    path = write(tmp_path)
    path.write_text(path.read_text() + row + "\n")
    assert problems_of(path) == ["line 7: empty column 'arm'"]


@pytest.mark.parametrize("column", ["config_hash", "split_hash"])
def test_a_mismatched_ab_pair_fails(
    tmp_path: Path, column: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, arm_a(), arm_b(**{column: sha("other")}))
    assert main(["--check", str(path)]) == 1
    err = capsys.readouterr().err
    expected = f"the rows of resnet18 run seed 0 (arm A and arm B alike) disagree on {KEYS[column]}"
    assert f"ledger check: {expected} (lines 7, 8)" in err.splitlines()


def test_ab_pairs_are_matched_by_run_seed(tmp_path: Path) -> None:
    """Seed 1's pair may carry other hashes than seed 0's only where the run seed may change
    them: the config hash leaves out the run seed, so it is still shared; the checkpoint is not."""
    path = write(tmp_path, arm_a(0), arm_b(0), arm_a(1), arm_b(1))
    assert problems_of(path) == []
    path = write(tmp_path, arm_a(0), arm_b(0), arm_a(1), arm_b(1, config_hash=sha("other")))
    assert problems_of(path) == [
        "the rows of resnet18 run seed 1 (arm A and arm B alike) disagree on config hash "
        "(lines 9, 10)"
    ]


def test_one_split_seed_has_one_split_hash_and_back(tmp_path: Path) -> None:
    small = {
        "run_id": "small_cnn-seed0-20261015T080000Z",
        "model": "small_cnn",
        "config": "small_cnn",
        "config_hash": sha("small"),
        "checkpoint_sha256": sha("checkpoint small"),
    }
    path = write(tmp_path, arm_a(), values(**small, split_hash=sha("split other")))
    assert problems_of(path) == ["split seed 42's rows disagree on split hash (lines 7, 8)"]
    path = write(tmp_path, arm_a(), values(**small, split_seed="7"))
    assert problems_of(path) == ["the rows of one split hash disagree on split seed (lines 7, 8)"]


def test_a_repeated_run_id_fails(tmp_path: Path) -> None:
    path = write(tmp_path, arm_a(), arm_a())
    assert problems_of(path) == [
        "run id 'resnet18_arm_a-seed0-20261015T101500Z' has 2 rows (lines [7, 8]), not one"
    ]


@pytest.mark.parametrize(
    "column",
    ["config_hash", "split_hash", "checkpoint_sha256"],
)
@pytest.mark.parametrize(
    "bad",
    [sha("x").upper(), sha("x")[:63], sha("x") + "0", "g" + sha("x")[1:]],
    ids=["upper", "short", "long", "not-hex"],
)
def test_a_malformed_hash_fails(tmp_path: Path, column: str, bad: str) -> None:
    problems = problems_of(write(tmp_path, values(**{column: bad})))
    assert problems == [f"line 7: {KEYS[column]} {bad!r} is not 64 lower-case hex digits"]


def test_a_dry_run_row_fails(tmp_path: Path) -> None:
    row = values(run_id="dryrun-seed0-20261014T025936Z", arm="B", config="dryrun")
    assert problems_of(write(tmp_path, row)) == [
        "line 7: config 'dryrun': the dry run never appends a row"
    ]


@pytest.mark.parametrize(
    "run_id",
    [
        "resnet18_arm_b-seed0-20261015T101500Z",  # another config
        "resnet18_arm_a-seed1-20261015T101500Z",  # another run seed
        "resnet18_arm_a-seed0-20261015T101500",  # no Z
        "resnet18_arm_a-seed0-20261315T101500Z",  # month 13
        "resnet18_arm_a-seed0",
    ],
)
def test_a_run_id_not_of_its_own_config_and_seed_fails(tmp_path: Path, run_id: str) -> None:
    problems = problems_of(write(tmp_path, values(run_id=run_id)))
    assert len(problems) == 1
    assert problems[0].startswith(f"line 7: run id {run_id!r} is not 'resnet18_arm_a-seed0-")


@pytest.mark.parametrize(
    ("changes", "column"),
    [
        ({"model": "resnet50"}, "model"),
        ({"arm": "C"}, "arm"),
        ({"split_seed": "-1"}, "split seed"),
        ({"run_seed": "01", "run_id": "resnet18_arm_a-seed01-20261015T101500Z"}, "run seed"),
        ({"epochs_run": "eight"}, "epochs run"),
        ({"scheduler": "step warmup 1"}, "scheduler"),
        ({"scheduler": "cosine"}, "scheduler"),
        ({"val_accuracy": "81.25"}, "val accuracy"),
        ({"val_macro_f1": "1.0001"}, "val macro-F1"),
        ({"trans_val_accuracy": "0.60"}, "trans_val accuracy"),
        ({"trans_val_macro_f1": "nan"}, "trans_val macro-F1"),
    ],
)
def test_a_malformed_cell_fails(tmp_path: Path, changes: dict[str, str], column: str) -> None:
    problems = problems_of(write(tmp_path, values(**changes)))
    assert len(problems) == 1
    assert problems[0].startswith(f"line 7: {column} ")


@pytest.mark.parametrize(("epochs", "best"), [("8", "9"), ("8", "0")])
def test_the_best_epoch_lies_within_the_epochs_run(tmp_path: Path, epochs: str, best: str) -> None:
    problems = problems_of(write(tmp_path, values(epochs_run=epochs, best_epoch=best)))
    assert problems == [f"line 7: best epoch {best} is not within the {epochs} epochs run"]


def test_the_rows_of_one_config_share_model_and_arm(tmp_path: Path) -> None:
    other = values(
        run_id="resnet18_arm_a-seed0-20261016T101500Z",
        arm="B",
        checkpoint_sha256=sha("checkpoint other"),
    )
    assert problems_of(write(tmp_path, arm_a(), other)) == [
        "the rows of config 'resnet18_arm_a' disagree on arm (lines 7, 8)"
    ]


def test_a_missing_or_headerless_ledger_fails(tmp_path: Path) -> None:
    assert problems_of(tmp_path / "none.md") == [f"no ledger at {tmp_path / 'none.md'}"]
    path = tmp_path / "ledger.md"
    path.write_text("# Ledger\n")
    assert problems_of(path) == ["the ledger holds the header of LEDGER_COLUMNS 0 times"]
    path.write_text(ledger_header() + "\n" + ledger_header() + "\n")
    assert problems_of(path) == ["the ledger holds the header of LEDGER_COLUMNS 2 times"]
    path.write_text(ledger_header().splitlines()[0] + "\n")
    assert problems_of(path) == [
        "the header of LEDGER_COLUMNS is not followed by its separator line"
    ]
    reordered = ledger_header().replace("| model | arm |", "| arm | model |")
    path.write_text(reordered + "\n")
    assert problems_of(path) == ["the ledger holds the header of LEDGER_COLUMNS 0 times"]


def test_text_after_the_table_fails(tmp_path: Path) -> None:
    path = write(tmp_path, arm_a(), tail="\nA note.\n")
    assert problems_of(path) == [
        "line 8: not a table row (the table must end the ledger)",
        "line 9: not a table row (the table must end the ledger)",
    ]


def test_a_problem_shows_no_cell_but_the_offending_one(tmp_path: Path) -> None:
    """R03: problems name lines, columns and the bad cell, never a whole row."""
    path = write(tmp_path, arm_a(), arm_b(config_hash=sha("other"), arm="C"))
    for problem in problems_of(path):
        assert sha("checkpoint A0") not in problem
        assert "0.8125" not in problem


def test_the_summary_groups_by_model_arm_and_seed_never_by_score(tmp_path: Path) -> None:
    """Rows written with trans_val falling as the keys rise come out in key order: the summary
    ranks nothing by a score (R17)."""
    rows = [
        arm_b(1, val_macro_f1="0.7000", trans_val_macro_f1="0.1000"),
        arm_a(1, val_macro_f1="0.6000", trans_val_macro_f1="0.2000"),
        arm_b(0, val_macro_f1="0.8000", trans_val_macro_f1="0.3000"),
        arm_a(0, val_macro_f1="0.5000", trans_val_macro_f1="0.4000"),
    ]
    found, problems = check_ledger(write(tmp_path, *rows))
    assert problems == []
    text = summary(found)
    lines = text.splitlines()
    assert lines[0] == "ledger: 4 rows, every check passed"
    table = [line for line in lines if line.startswith("| resnet18")]
    assert [line.split(" | ")[1:4] for line in table] == [
        ["A", "42", "0"],
        ["A", "42", "1"],
        ["A", "all", "all"],
        ["B", "42", "0"],
        ["B", "42", "1"],
        ["B", "all", "all"],
    ]
    by_key = {tuple(line.split(" | ")[1:4]): line.split(" | ") for line in table}
    columns = ["model", "arm", "split seed", "run seed", "rows", *ledger.SCORE_COLUMNS]
    header = next(line for line in lines if line.startswith("| model"))
    assert header == "| " + " | ".join(columns) + " |"
    a_mean = by_key[("A", "all", "all")]
    assert a_mean[4] == "2"
    assert a_mean[6] == "0.5500"  # val macro-F1, mean of 0.5 and 0.6
    assert by_key[("B", "42", "1")][-1] == "0.1000 |"  # trans_val macro-F1 shown, as reported
    assert "best" not in text.lower()


def test_the_summary_of_an_empty_ledger_is_its_count(tmp_path: Path) -> None:
    found, problems = check_ledger(write(tmp_path))
    assert (found, problems) == ([], [])
    assert summary(found) == "ledger: 0 rows, every check passed"


def test_a_failing_check_prints_no_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write(tmp_path, arm_a(), arm_b(split_hash=sha("other")))
    assert main(["--check", str(path)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.splitlines()[-1] == f"ledger check: 2 problems in {path}"
