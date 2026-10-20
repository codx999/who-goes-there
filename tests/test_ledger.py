"""trailcam/train/ledger.py: the ledger checker and its val summary (plan.md sections 5 and 6;
R14 one split hash per split seed, R17 trans_val never ranked, R25 arm A and arm B rows of one seed
share split hash and config hash, R37 no dry-run row, R40 one well-formed row per run id; item
13.02: the phase-2 ledger, docs/ledger-phase2.md, takes `p2_` rows only and docs/ledger.md none;
item 13.11: the phase-3 ledger, docs/ledger-phase3.md, takes `p3_` rows only and neither other
ledger takes one).

Every ledger here is written into `tmp_path` from rows built with `train.ledger_row`, so the rows
have exactly the shape `train.py` appends; the hashes are SHA-256 digests of short labels.
"""

import hashlib
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from trailcam.train import ledger, train
from trailcam.train.ledger import check_ledger, main, summary
from trailcam.train.train import (
    LEDGER_COLUMNS,
    LEDGER_PATH,
    PHASE2_LEDGER_NAME,
    PHASE2_LEDGER_PATH,
    PHASE3_LEDGER_NAME,
    PHASE3_LEDGER_PATH,
    ledger_header,
    ledger_row,
)

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
    assert run(REPO / "docs" / "ledger-phase2.md").returncode == 0
    assert run(REPO / "docs" / "ledger-phase3.md").returncode == 0  # item 13.11
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
    """Seed 1's pair carries its own checkpoints but seed 0's config hash, which leaves out the
    run seed; an arm B seed 1 row with another config hash breaks its pair and the recipe."""
    path = write(tmp_path, arm_a(0), arm_b(0), arm_a(1), arm_b(1))
    assert problems_of(path) == []
    path = write(tmp_path, arm_a(0), arm_b(0), arm_a(1), arm_b(1, config_hash=sha("other")))
    assert problems_of(path) == [
        "the rows of resnet18 run seed 1 (arm A and arm B alike) disagree on config hash "
        "(lines 9, 10)",
        "the rows of resnet18 (every split seed, run seed and arm) disagree on config hash "
        "(lines 7, 8, 9, 10)",
    ]


def test_the_seeds_of_one_model_share_the_config_hash(tmp_path: Path) -> None:
    """Seed 1's pair agrees within itself but was trained to another recipe than seed 0's: the
    seed pairings of plan.md section 6 would compare two recipes (review 7.01)."""
    other = sha("other")
    path = write(
        tmp_path,
        arm_a(0),
        arm_b(0),
        arm_a(1, config_hash=other),
        arm_b(1, config_hash=other),
    )
    assert problems_of(path) == [
        "the rows of resnet18 (every split seed, run seed and arm) disagree on config hash "
        "(lines 7, 8, 9, 10)"
    ]


def test_a_seed_pair_on_another_split_seed_fails(tmp_path: Path) -> None:
    """Seed 1's pair trained on split seed 7 agrees within itself and with its split, but plan.md
    section 5 fixes one split seed: the "all" lines would average two splits (review 7.01, fix
    round)."""
    seven = {"split_seed": "7", "split_hash": sha("split 7"), "config_hash": sha("config 7")}
    path = write(tmp_path, arm_a(0), arm_b(0), arm_a(1, **seven), arm_b(1, **seven))
    assert problems_of(path) == [
        "the rows of resnet18 (every split seed, run seed and arm) disagree on config hash "
        "(lines 7, 8, 9, 10)"
    ]


def test_the_rows_of_one_config_hash_agree_on_the_cells_it_stands_for(tmp_path: Path) -> None:
    """An arm B row whose scheduler differs from its arm A partner's under one config hash: the
    hash says one recipe, the cells say two (review 7.01)."""
    path = write(tmp_path, arm_a(), arm_b(scheduler="cosine warmup 2"))
    assert problems_of(path) == [
        f"the rows of config hash {sha('config')} disagree on scheduler (lines 7, 8)"
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
        ({"model": "resnet101"}, "model"),
        ({"arm": "D"}, "arm"),  # arm C is phase 2's (item 13.04)
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


@pytest.mark.parametrize(
    ("changes", "column"),
    [
        ({"arm": "B"}, "arm"),
        ({"model": "small_cnn", "config_hash": sha("small")}, "model"),
    ],
)
def test_the_rows_of_one_config_share_model_and_arm(
    tmp_path: Path, changes: dict[str, str], column: str
) -> None:
    other = values(
        run_id="resnet18_arm_a-seed0-20261016T101500Z",
        checkpoint_sha256=sha("checkpoint other"),
        **changes,
    )
    assert problems_of(write(tmp_path, arm_a(), other)) == [
        f"the rows of config 'resnet18_arm_a' disagree on {column} (lines 7, 8)"
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


# Phase 2 (item 13.02): a ledger named ledger-phase2.md is checked as the phase-2 ledger.


def p2(config: str, seed: int = 0, **changes: str) -> dict[str, str]:
    """A well-formed phase-2 row of `config` (resnet18 arm A) and run seed `seed`, its config
    hash a digest of the config's name."""
    base = {
        "run_id": f"{config}-seed{seed}-20261019T090000Z",
        "config": config,
        "config_hash": sha(config),
        "run_seed": str(seed),
        "checkpoint_sha256": sha(f"checkpoint {config} {seed}"),
    }
    return values(**{**base, **changes})


def write_p2(tmp_path: Path, *rows: Mapping[str, str]) -> Path:
    return write(tmp_path, *rows).rename(tmp_path / PHASE2_LEDGER_NAME)


def test_the_real_phase2_ledger_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert PHASE2_LEDGER_PATH.name == PHASE2_LEDGER_NAME == "ledger-phase2.md"
    assert main(["--check", str(PHASE2_LEDGER_PATH)]) == 0
    assert capsys.readouterr().out.startswith("phase-2 ledger: ")


def test_a_p2_row_pasted_into_the_phase1_ledger_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The user pastes rows by hand: a `p2_` row added to a copy of docs/ledger.md fails its
    check, and with it check.sh's test of the real file and the final scoring's read."""
    path = tmp_path / "ledger.md"
    path.write_text(LEDGER_PATH.read_text() + ledger_row(p2("p2_head3")) + "\n")
    rows, problems = check_ledger(path)
    line = len(LEDGER_PATH.read_text().splitlines()) + 1
    assert problems == [
        f"line {line}: config 'p2_head3' is a phase-2 config: its rows go to "
        "docs/ledger-phase2.md, never to this ledger"
    ]
    assert main(["--check", str(path)]) == 1
    assert "phase-2 config" in capsys.readouterr().err


def test_a_phase1_row_in_the_phase2_ledger_fails(tmp_path: Path) -> None:
    assert problems_of(write_p2(tmp_path, p2("p2_baseline"), arm_a())) == [
        "line 8: config 'resnet18_arm_a' is not a phase-2 config (p2_...): its rows go to "
        "docs/ledger.md"
    ]


def test_the_steps_of_one_model_may_carry_several_config_hashes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Every phase-2 step is a new recipe of resnet18, so one model and run seed has several
    config hashes, which the phase-1 checks refuse. An arm pair may share its config's hash
    (here arm B against arm A) or carry its own: item 13.04's arm C hashes its `paste_p`."""
    rows = [
        p2("p2_baseline", config_hash=sha("arm A recipe")),
        p2("p2_baseline", 1, config_hash=sha("arm A recipe")),
        p2("p2_head3"),
        p2("p2_res320"),
        p2("p2_arm_b", arm="B", config_hash=sha("arm A recipe")),
        p2("p2_arm_c", arm="C"),
    ]
    path = write_p2(tmp_path, *rows)
    assert problems_of(path) == []
    assert main(["--check", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("phase-2 ledger: 6 rows, every check passed")
    unprefixed = [{**row, "config": row["config"][3:], "run id": row["run id"][3:]} for row in rows]
    phase1 = write(tmp_path, *unprefixed)
    assert any("disagree on config hash" in problem for problem in problems_of(phase1))


def test_the_rows_of_one_phase2_config_share_its_config_hash(tmp_path: Path) -> None:
    """A config edited between its seeds is two recipes under one name."""
    path = write_p2(tmp_path, p2("p2_head3"), p2("p2_head3", 1, config_hash=sha("edited")))
    assert problems_of(path) == [
        "the rows of config 'p2_head3' (every run seed) disagree on config hash (lines 7, 8)"
    ]


def test_the_phase2_rows_share_one_split_seed(tmp_path: Path) -> None:
    """Every comparison is paired by run seed on phase 1's split, so a row on another split
    seed fails, whatever its config."""
    other = p2("p2_res320", split_seed="7", split_hash=sha("split 7"))
    assert problems_of(write_p2(tmp_path, p2("p2_head3"), other)) == [
        "the rows of the phase-2 ledger disagree on split seed (lines 7, 8)"
    ]


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"config": "dryrun", "run_id": "dryrun-seed0-20261019T090000Z"}, "dry run never"),
        ({"run_id": "p2_head3-seed1-20261019T090000Z"}, "is not 'p2_head3-seed0-"),
        ({"val_macro_f1": "0.74"}, "four decimals"),
    ],
)
def test_the_phase2_ledger_keeps_the_cell_checks(
    tmp_path: Path, changes: dict[str, str], problem: str
) -> None:
    row = {**p2("p2_head3"), **{KEYS[key]: value for key, value in changes.items()}}
    found = problems_of(write_p2(tmp_path, row))
    assert len(found) >= 1 and problem in found[0]


def test_the_phase2_ledger_keeps_the_checks_of_one_config_and_one_hash(tmp_path: Path) -> None:
    """The rows of one config share model and arm, those of one config hash model, split seed
    and scheduler, and one split seed has one split hash."""
    assert problems_of(write_p2(tmp_path, p2("p2_r50"), p2("p2_r50", 1, model="resnet50"))) == [
        "the rows of config 'p2_r50' disagree on model (lines 7, 8)",
        f"the rows of config hash {sha('p2_r50')} disagree on model (lines 7, 8)",
    ]
    (tmp_path / PHASE2_LEDGER_NAME).unlink()
    other = p2("p2_head3", 1, scheduler="cosine warmup 2")
    assert problems_of(write_p2(tmp_path, p2("p2_head3"), other)) == [
        f"the rows of config hash {sha('p2_head3')} disagree on scheduler (lines 7, 8)"
    ]
    (tmp_path / PHASE2_LEDGER_NAME).unlink()
    moved = p2("p2_res320", split_hash=sha("another split"))
    assert problems_of(write_p2(tmp_path, p2("p2_head3"), moved)) == [
        "split seed 42's rows disagree on split hash (lines 7, 8)"
    ]


def test_the_phase2_summary_groups_by_config_never_by_score(tmp_path: Path) -> None:
    """Three configs of resnet18 arm A are three recipes: the summary never averages them
    together, and orders them by config and seeds, never by a score."""
    rows = [
        p2("p2_res320", val_macro_f1="0.9000"),
        p2("p2_baseline", 1, val_macro_f1="0.6000"),
        p2("p2_head3", val_macro_f1="0.7000"),
        p2("p2_baseline", val_macro_f1="0.5000"),
    ]
    found, problems = check_ledger(write_p2(tmp_path, *rows))
    assert problems == []
    lines = summary(found, phase=2).splitlines()
    assert lines[0] == "phase-2 ledger: 4 rows, every check passed"
    columns = ["config", "model", "arm", "split seed", "run seed", "rows", *ledger.SCORE_COLUMNS]
    assert "| " + " | ".join(columns) + " |" in lines
    table = [line.split(" | ") for line in lines if line.startswith("| p2_")]
    assert [cells[0][2:] for cells in table] == [
        "p2_baseline",
        "p2_baseline",
        "p2_baseline",
        "p2_head3",
        "p2_res320",
    ]
    assert [cells[4] for cells in table] == ["0", "1", "all", "0", "0"]
    assert table[2][7] == "0.5500"  # p2_baseline's mean val macro-F1, its own rows only
    assert "kept or dropped on val and trans_val" in lines[-1]
    assert "best" not in "\n".join(lines).lower()


# Phase 3 (item 13.11): a ledger named ledger-phase3.md is checked as the phase-3 ledger.


def p3(config: str, seed: int = 0, **changes: str) -> dict[str, str]:
    """A well-formed phase-3 row of `config` and run seed `seed`, its config hash a digest of the
    config's name; `p3_convnext*` rows are ConvNeXt-Tiny, arm C for `p3_convnext_cp`."""
    model = "convnext_tiny" if config.startswith("p3_convnext") else "resnet18"
    base = {
        "run_id": f"{config}-seed{seed}-20261023T090000Z",
        "model": model,
        "arm": "C" if config.endswith("_cp") else "A",
        "config": config,
        "config_hash": sha(config),
        "run_seed": str(seed),
        "checkpoint_sha256": sha(f"checkpoint {config} {seed}"),
    }
    return values(**{**base, **changes})


def write_p3(tmp_path: Path, *rows: Mapping[str, str]) -> Path:
    return write(tmp_path, *rows).rename(tmp_path / PHASE3_LEDGER_NAME)


def test_the_real_phase3_ledger_passes(capsys: pytest.CaptureFixture[str]) -> None:
    assert PHASE3_LEDGER_PATH.name == PHASE3_LEDGER_NAME == "ledger-phase3.md"
    assert main(["--check", str(PHASE3_LEDGER_PATH)]) == 0
    assert re.match(r"phase-3 ledger: \d+ rows, every check passed", capsys.readouterr().out)


@pytest.mark.parametrize("other", [LEDGER_PATH, PHASE2_LEDGER_PATH])
def test_a_p3_row_pasted_into_either_other_ledger_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], other: Path
) -> None:
    """A `p3_` row added to a copy of docs/ledger.md or docs/ledger-phase2.md fails its check,
    and with it check.sh's test of the real files."""
    path = tmp_path / other.name
    path.write_text(other.read_text() + ledger_row(p3("p3_convnext")) + "\n")
    line = len(other.read_text().splitlines()) + 1
    found = problems_of(path)
    expected = (
        f"line {line}: config 'p3_convnext' is a phase-3 config: its rows go to "
        "docs/ledger-phase3.md, never to this ledger"
        if other == LEDGER_PATH
        else f"line {line}: config 'p3_convnext' is not a phase-2 config (p2_...): its rows go "
        "to docs/ledger-phase3.md"
    )
    assert expected in found
    assert main(["--check", str(path)]) == 1
    assert "p3_convnext" in capsys.readouterr().err


def test_a_phase1_or_phase2_row_in_the_phase3_ledger_fails(tmp_path: Path) -> None:
    path = write_p3(tmp_path, p3("p3_baseline"), arm_a(), p2("p2_head3"))
    found = problems_of(path)
    assert found[:2] == [
        "line 8: config 'resnet18_arm_a' is not a phase-3 config (p3_...): its rows go to "
        "docs/ledger.md",
        "line 9: config 'p2_head3' is not a phase-3 config (p3_...): its rows go to "
        "docs/ledger-phase2.md",
    ]


def test_the_phase3_ledger_takes_two_convnext_recipes_and_the_control(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`p3_convnext` and `p3_convnext_cp` are two recipes of one model, so it carries two config
    hashes, which the phase-1 checks refuse; `p3_baseline` carries arm A's phase-1 hash. The rows
    of one config share its hash, and every row one split seed, as in phase 2."""
    rows = [
        p3("p3_baseline", config_hash=sha("arm A recipe")),
        p3("p3_convnext"),
        p3("p3_convnext_cp"),
        p3("p3_convnext", 1),
        p3("p3_convnext_cp", 1),
    ]
    path = write_p3(tmp_path, *rows)
    assert problems_of(path) == []
    assert main(["--check", str(path)]) == 0
    out = capsys.readouterr().out
    assert out.startswith("phase-3 ledger: 5 rows, every check passed")
    assert "| config | model | arm |" in out and "| p3_convnext_cp | convnext_tiny | C |" in out
    assert "a phase-3 candidate is kept or dropped on val and trans_val" in out
    (tmp_path / PHASE3_LEDGER_NAME).unlink()
    edited = write_p3(tmp_path, p3("p3_convnext"), p3("p3_convnext", 1, config_hash=sha("x")))
    assert problems_of(edited) == [
        "the rows of config 'p3_convnext' (every run seed) disagree on config hash (lines 7, 8)"
    ]
    (tmp_path / PHASE3_LEDGER_NAME).unlink()
    other = p3("p3_convnext_cp", split_seed="7", split_hash=sha("split 7"))
    assert problems_of(write_p3(tmp_path, p3("p3_convnext"), other)) == [
        "the rows of the phase-3 ledger disagree on split seed (lines 7, 8)"
    ]


def test_the_phase3_summary_groups_by_config_never_by_score(tmp_path: Path) -> None:
    rows = [
        p3("p3_convnext_cp", val_macro_f1="0.9000"),
        p3("p3_convnext", 1, val_macro_f1="0.6000"),
        p3("p3_baseline", val_macro_f1="0.7000"),
        p3("p3_convnext", val_macro_f1="0.5000"),
    ]
    found, problems = check_ledger(write_p3(tmp_path, *rows))
    assert problems == []
    lines = summary(found, phase=3).splitlines()
    assert lines[0] == "phase-3 ledger: 4 rows, every check passed"
    table = [line.split(" | ") for line in lines if line.startswith("| p3_")]
    assert [cells[0][2:] for cells in table] == [
        "p3_baseline",
        "p3_convnext",
        "p3_convnext",
        "p3_convnext",
        "p3_convnext_cp",
    ]
    assert table[3][7] == "0.5500"  # p3_convnext's mean val macro-F1, its own rows only
    assert "docs/phase3-plan.md section 1" in lines[-1]
    assert "best" not in "\n".join(lines).lower()
