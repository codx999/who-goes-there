"""trailcam/evaluate/score_test.py's phase-3 mode (item 13.12; docs/phase3-plan.md sections 1, 2
and 7; R17, R40), and the phase-1 mode's refusal of the phase-3 files.

Synthetic fixture only, as `tests/test_score_test.py`'s: a cache built from
`trailcam/data/synthetic.py` in a temporary directory, a phase-3 ledger (`ledger-phase3.md`) and a
baseline ledger (`ledger.md`, `resnet18_arm_a` seeds 0, 1 and 2) written there, never copied with
rows from `docs/` (the trap of items 6.04 and 9.03: a test that assumes a ledger's rows breaks when
the real file grows), and a header-only `test-ledger-phase3.md` built from `scoring_header`. The
finalist's three checkpoints are ResNet-18s with random weights: the fixture's `p3_` configs
name `resnet18`, since a ConvNeXt-Tiny pass over the fixture's test splits takes minutes on a CPU
(`score_test.py`'s ConvNeXt loading and scoring is `tests/test_phase3.py`'s). The other phase-3
rows name small stand-in files, which a refusal hashes and never loads. No real checkpoint and no
real test split is ever read; outputs go to pytest's temporary directories.

The real committed ledgers are read by two tests only: the finalist they give is recomputed
(`p3_convnext_cp`, its three rows), and nothing is scored.
"""

import ast
import contextlib
import hashlib
import io
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from tests.test_score_test import (
    APPROVER,
    _empty_test_ledger,
    _figures_snapshot,
    _forbid_reading,
    _ledger_values,
    _photo,
    _save_checkpoint,
    _write_ledger,
)
from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.evaluate import attention, evaluate, score_test, slices
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.evaluate.score_test import (
    PHASE3_CANDIDATES,
    PHASE3_SEEDS,
    PHASE3_TEST_LEDGER_PATH,
    REFUSED_EXIT,
    TEST_LEDGER_COLUMNS,
    TEST_LEDGER_PATH,
    TEST_SPLITS,
    Refusal,
    Selection,
    read_ledger,
    select_finalist,
)
from trailcam.train.config import CONFIG_PATH
from trailcam.train.ledger import Row, check_ledger
from trailcam.train.train import (
    ALARM_EXIT,
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    LEDGER_PATH,
    PHASE2_LEDGER_PATH,
    PHASE3_LEDGER_PATH,
    file_sha256,
    ledger_row,
)

FINALIST = "p3_convnext_cp"
ROOT = Path(__file__).resolve().parents[1]
# The repo files the phase-3 mode must never write (the run writes the clone's copy only).
REPO_DOCS = (
    PHASE3_TEST_LEDGER_PATH,
    TEST_LEDGER_PATH,
    LEDGER_PATH,
    PHASE2_LEDGER_PATH,
    PHASE3_LEDGER_PATH,
)
# (val macro-F1, trans_val macro-F1) by config and run seed: the real ledgers' values, so the
# default fixture keeps both candidates and picks p3_convnext_cp, as the real rows do.
REAL_LIKE: Mapping[tuple[str, int], tuple[str, str]] = {
    ("resnet18_arm_a", 0): ("0.8542", "0.2915"),
    ("resnet18_arm_a", 1): ("0.8559", "0.3187"),
    ("resnet18_arm_a", 2): ("0.8569", "0.3200"),
    ("p3_baseline", 0): ("0.8542", "0.2915"),
    ("p3_convnext", 0): ("0.8837", "0.3682"),
    ("p3_convnext", 1): ("0.8801", "0.3514"),
    ("p3_convnext", 2): ("0.8767", "0.3477"),
    ("p3_convnext_cp", 0): ("0.8573", "0.3922"),
    ("p3_convnext_cp", 1): ("0.8631", "0.4208"),
    ("p3_convnext_cp", 2): ("0.8654", "0.4257"),
}
Scores = Mapping[tuple[str, int], tuple[str, str]]


def _config_file(path: Path, **changes: Any) -> Path:
    """A copy of config.yaml whose arm A and `p3_` configs have no workers, small batches and no
    pretrained download, and whose two candidates are ResNet-18s (arm A and arm C); `changes`
    go into every one of them."""
    settings = yaml.safe_load(CONFIG_PATH.read_text())
    loader = {"batch_size": 16, "num_workers": 0, "pin_memory": False, "pretrained": False}
    configs = settings["configs"]
    for name in ("resnet18_arm_a", "p3_baseline", *PHASE3_CANDIDATES):
        configs[name] = {**configs[name], **loader, **changes}
    for name in PHASE3_CANDIDATES:
        configs[name]["model"] = "resnet18"
    path.write_text(yaml.safe_dump(settings))
    return path


@dataclass
class World:
    """A phase-3 scoring's inputs under one temporary directory: one checkpoint file for every
    phase-3 row (`files`, by config and run seed; real ResNet-18 weights for the finalist's,
    stand-ins for the rest), and the ledgers of `REAL_LIKE`."""

    root: Path
    cache: Path
    config_file: Path
    ledger: Path
    baseline: Path
    test_ledger: Path
    assets: Path
    photo: Path
    files: dict[tuple[str, int], Path]
    split_hash: str
    records: dict[str, tuple[Record, ...]] = field(default_factory=dict)

    def finalist(self, config: str = FINALIST) -> list[Path]:
        return [self.files[(config, seed)] for seed in PHASE3_SEEDS]

    def argv(
        self,
        out: Path,
        *,
        approver: str | None = APPROVER,
        checkpoints: Sequence[Path] | None = None,
        ledger: Path | None = None,
        baseline: Path | None = None,
        test_ledger: Path | None = None,
        config_file: Path | None = None,
        run_seeds: Sequence[int] | None = PHASE3_SEEDS,
        phase: int | None = 3,
    ) -> list[str]:
        given = self.finalist() if checkpoints is None else checkpoints
        argv = [arg for path in given for arg in ("--checkpoint", str(path))]
        argv += ["--data-root", str(self.cache), "--out", str(out)]
        argv += ["--config", str(config_file or self.config_file)]
        argv += ["--ledger", str(ledger or self.ledger)]
        argv += ["--test-ledger", str(test_ledger or self.test_ledger)]
        argv += ["--assets", str(self.assets), "--overlay", f"{self.photo}={CLASSES[0]}"]
        if phase is not None:
            argv += ["--phase", str(phase), "--baseline-ledger", str(baseline or self.baseline)]
        if approver is not None:
            argv += ["--approved-by", approver]
        if run_seeds is not None:
            argv += ["--run-seeds", *map(str, run_seeds)]
        return argv


def write_ledgers(
    world: World,
    folder: Path,
    scores: Scores = REAL_LIKE,
    *,
    drop: Sequence[tuple[str, int]] = (),
    extra: Sequence[dict[str, str]] = (),
) -> tuple[Path, Path]:
    """`ledger-phase3.md` and `ledger.md` in `folder`: a row per key of `scores` but `drop`
    (arm A's in the baseline, the rest in the phase-3 ledger, in `scores` order), each naming
    its file in `world.files`, then the `extra` rows; both pass `trailcam.train.ledger`'s
    check."""
    folder.mkdir(parents=True, exist_ok=True)
    phase3, base = [], []
    for (name, seed), (val, trans) in scores.items():
        if (name, seed) in drop:
            continue
        path = world.files.get((name, seed))
        digest = file_sha256(path) if path else score_test_digest(name, seed)
        row = _ledger_values(name, seed, digest, world.config_file, world.split_hash)
        row |= {"val macro-F1": val, "val accuracy": val}
        row |= {"trans_val macro-F1": trans, "trans_val accuracy": trans}
        (base if name == "resnet18_arm_a" else phase3).append(row)
    ledger = _write_ledger(folder / "ledger-phase3.md", [*phase3, *extra])
    baseline = _write_ledger(folder / "ledger.md", base)
    for path in (ledger, baseline):
        assert check_ledger(path)[1] == [], check_ledger(path)[1]
    return ledger, baseline


def score_test_digest(name: str, seed: int) -> str:
    """A checkpoint sha256 for a row with no file (arm A's, never given to the scoring)."""
    return hashlib.sha256(f"{name} {seed}".encode()).hexdigest()


def _build_world(root: Path) -> World:
    built = root / "built"
    fixture = synthetic.write_synthetic_fixture(built)
    cache = root / "cache"
    cache.mkdir()
    fixture.image_root.rename(cache / CACHE_IMAGES)
    Path(fixture.paths["sidecar"]).rename(cache / DAY_NIGHT_SIDECAR)
    built.rename(cache / CACHE_ANNOTATIONS)
    split = make_split(load_metadata(cache / CACHE_ANNOTATIONS, cache / DAY_NIGHT_SIDECAR), 42)

    files = {}
    for name, seed in REAL_LIKE:
        if name == "resnet18_arm_a":
            continue
        path = root / "checkpoints" / f"{name}-seed{seed}.pt"
        path.parent.mkdir(exist_ok=True)
        if name == FINALIST:
            _save_checkpoint(path, "resnet18", 20 + seed)
        else:
            path.write_bytes(f"a stand-in for {name} seed {seed}".encode())
        files[(name, seed)] = path
    assets = root / "assets"
    world = World(
        root=root,
        cache=cache,
        config_file=_config_file(root / "config.yaml"),
        ledger=root,  # set below
        baseline=root,
        test_ledger=_empty_test_ledger(root / "test-ledger-phase3.md"),
        assets=assets,
        photo=_photo(assets / "photo.jpg"),
        files=files,
        split_hash=split.hash,
        records={name: split.records[name] for name in split.records},
    )
    world.ledger, world.baseline = write_ledgers(world, root / "ledgers")
    return world


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return _build_world(tmp_path_factory.mktemp("score_test_phase3"))


@pytest.fixture
def fresh(world: World, tmp_path: Path) -> World:
    """`world` with an empty test ledger of its own, so a refusal test never depends on the full
    run's rows."""
    test_ledger = _empty_test_ledger(tmp_path / "test-ledger-phase3.md")
    return World(**{**world.__dict__, "test_ledger": test_ledger})


def _rows(world: World, folder: Path, scores: Scores = REAL_LIKE, **kw: Any) -> tuple[Any, Any]:
    """The rows of both ledgers of `write_ledgers`, as the scoring reads them."""
    ledger, baseline = write_ledgers(world, folder, scores, **kw)
    return read_ledger(ledger), read_ledger(baseline)


def _with(**changes: tuple[str, str]) -> dict[tuple[str, int], tuple[str, str]]:
    """`REAL_LIKE` with the scores of `changes`, keyed `<config>_<seed>`."""
    scores = dict(REAL_LIKE)
    for key, value in changes.items():
        name, seed = key.rsplit("_", 1)
        scores[(name, int(seed))] = value
    return scores


# The selection: the screen, the keep rule and the finalist rule, recomputed (P3 §1, §2).


def test_both_kept_and_the_higher_trans_val_mean_is_the_finalist(
    world: World, tmp_path: Path
) -> None:
    """The real ledgers' case: both candidates kept; p3_convnext_cp's mean trans_val macro-F1
    (0.4129) leads p3_convnext's (0.3558) by 0.0571, more than 0.03, so it is the finalist, and
    its rows are its three seeds in seed order."""
    ledger, baseline = _rows(world, tmp_path)
    selection = select_finalist(ledger, baseline)
    assert [(v.config, v.screened, v.kept) for v in selection.verdicts] == [
        ("p3_convnext", True, True),
        ("p3_convnext_cp", True, True),
    ]
    assert selection.finalist.config == FINALIST
    assert [int(r["run seed"]) for r in selection.finalist.rows] == [0, 1, 2]
    assert {r["config"] for r in selection.finalist.rows} == {FINALIST}
    assert selection.rule == "the higher mean trans_val macro-F1, by 0.0571, more than 0.03"
    text = "\n".join(selection.lines())
    assert "p3_convnext_cp: kept" in text and "mean trans_val macro-F1 +0.1028" in text
    assert "p3_convnext: kept" in text and "mean trans_val macro-F1 +0.0457" in text


def test_within_the_band_the_higher_val_mean_is_the_finalist(world: World, tmp_path: Path) -> None:
    """The two trans_val means exactly 0.03 apart are within the band (0.03 or less), so the
    higher mean val macro-F1 decides: p3_convnext's 0.8802 over p3_convnext_cp's 0.8619."""
    scores = _with(
        p3_convnext_cp_0=("0.8573", "0.3982"),
        p3_convnext_cp_1=("0.8631", "0.3814"),
        p3_convnext_cp_2=("0.8654", "0.3777"),
    )  # each seed exactly 0.0300 above p3_convnext's
    selection = select_finalist(*_rows(world, tmp_path, scores))
    assert [v.kept for v in selection.verdicts] == [True, True]
    assert selection.finalist.config == "p3_convnext"
    assert selection.rule.startswith("the mean trans_val macro-F1 0.0300 apart, within 0.03")
    assert selection.rule.endswith("the higher mean val macro-F1, by 0.0182")


def test_within_the_band_and_one_val_mean_the_user_decides(world: World, tmp_path: Path) -> None:
    scores = _with(
        p3_convnext_cp_0=("0.8837", "0.3700"),
        p3_convnext_cp_1=("0.8801", "0.3514"),
        p3_convnext_cp_2=("0.8767", "0.3477"),
    )
    with pytest.raises(Refusal, match="the finalist rule does not decide, the user does"):
        select_finalist(*_rows(world, tmp_path, scores))


@pytest.mark.parametrize(
    ("changes", "kept", "why"),
    [
        # mean trans_val gain exactly 0.03: not "more than" (exact, not float arithmetic)
        (
            {"0": ("0.8573", "0.3215"), "1": ("0.8631", "0.3487"), "2": ("0.8654", "0.3500")},
            False,
            "mean trans_val macro-F1 +0.0300 (more than +0.03: no)",
        ),
        (
            {"0": ("0.8573", "0.3215"), "1": ("0.8631", "0.3487"), "2": ("0.8654", "0.3501")},
            True,
            "mean trans_val macro-F1 +0.0300 (more than +0.03: yes)",
        ),
        # a mean gain of 0.0640 that does not rise on seed 1 (equal is no rise)
        (
            {"0": ("0.8573", "0.4100"), "1": ("0.8631", "0.3187"), "2": ("0.8654", "0.4000")},
            False,
            "rises on every seed: no, not on seed 1",
        ),
        # the mean val macro-F1 down exactly 0.03: kept; 0.0001 more: not kept
        (
            {"0": ("0.8242", "0.3922"), "1": ("0.8259", "0.4208"), "2": ("0.8269", "0.4257")},
            True,
            "mean val macro-F1 -0.0300 (no more than 0.03 down: yes)",
        ),
        (
            {"0": ("0.8242", "0.3922"), "1": ("0.8259", "0.4208"), "2": ("0.8268", "0.4257")},
            False,
            "(no more than 0.03 down: no)",
        ),
    ],
)
def test_the_keep_rule_s_three_conditions_at_their_bars(
    world: World,
    tmp_path: Path,
    changes: dict[str, tuple[str, str]],
    kept: bool,
    why: str,
) -> None:
    """Each condition at its bar, on p3_convnext_cp (which passes the screen every time); the
    other candidate stays kept, so a failed condition makes p3_convnext the finalist."""
    scores = _with(**{f"{FINALIST}_{seed}": value for seed, value in changes.items()})
    selection = select_finalist(*_rows(world, tmp_path, scores))
    (verdict,) = [v for v in selection.verdicts if v.config == FINALIST]
    assert verdict.screened and verdict.kept is kept
    assert why in verdict.why, verdict.why
    if not kept:
        assert selection.finalist.config == "p3_convnext"
        assert selection.rule == "the one candidate kept"


def test_the_exact_bar_is_exact() -> None:
    """Why the cells are read as `Decimal`: in floats the mean gain of the first case above is
    0.03 plus a rounding error, which would pass "more than 0.03"."""
    floats = (0.3215 + 0.3487 + 0.3500) / 3 - (0.2915 + 0.3187 + 0.3200) / 3
    assert floats > 0.03  # the float route would keep it
    exact = sum(map(Decimal, ("0.3215", "0.3487", "0.3500"))) - sum(
        map(Decimal, ("0.2915", "0.3187", "0.3200"))
    )
    assert exact == Decimal("0.0900")


@pytest.mark.parametrize(
    ("seed0", "screen"),
    [
        (("0.8573", "0.3214"), "seed 0 trans_val macro-F1 +0.0299, val macro-F1 +0.0031"),
        (("0.8241", "0.3922"), "seed 0 trans_val macro-F1 +0.1007, val macro-F1 -0.0301"),
    ],
)
def test_a_candidate_that_fails_the_screen_is_not_kept(
    world: World, tmp_path: Path, seed0: tuple[str, str], screen: str
) -> None:
    """The screen on seed 0 (trans_val at least +0.03, val at most -0.03): a candidate that
    fails it has its seed-0 row only, as the plan runs no confirm for it, and is not kept."""
    scores = _with(p3_convnext_cp_0=seed0)
    drop = [(FINALIST, 1), (FINALIST, 2)]
    selection = select_finalist(*_rows(world, tmp_path, scores, drop=drop))
    (verdict,) = [v for v in selection.verdicts if v.config == FINALIST]
    assert (verdict.screened, verdict.kept) == (False, False)
    assert verdict.why == f"{FINALIST}: not kept, failed the screen ({screen})"
    assert selection.finalist.config == "p3_convnext"


def test_the_screen_passes_at_its_bars(world: World, tmp_path: Path) -> None:
    """At least +0.03 and at most -0.03 on seed 0: both bars inclusive."""
    scores = _with(p3_convnext_cp_0=("0.8242", "0.3215"))
    selection = select_finalist(*_rows(world, tmp_path, scores))
    assert all(v.screened for v in selection.verdicts)


@pytest.mark.parametrize(
    ("drop", "why"),
    [
        ([(FINALIST, 2)], "has no row of run seed 2: its confirm is not finished"),
        (
            [(FINALIST, 1), (FINALIST, 2)],
            "has no row of run seed 1, 2: its confirm is not finished",
        ),
        ([("p3_convnext", 1)], "p3_convnext passed the screen (seed 0"),
        ([(FINALIST, 0)], "no p3_convnext_cp row of run seed 0: the screening batch is not"),
        ([("p3_baseline", 0)], "p3_baseline must have exactly one row, of run seed 0"),
        (
            [("resnet18_arm_a", 2)],
            "does not hold one resnet18 arm A row of resnet18_arm_a for each run seed 0, 1 and 2 "
            "(missing: 2)",
        ),
    ],
)
def test_fewer_than_three_seeds_or_a_missing_run_is_refused(
    world: World, tmp_path: Path, drop: list[tuple[str, int]], why: str
) -> None:
    """A candidate that passed the screen with fewer than 3 seeds (its confirm still training)
    cannot be judged, and judging the other one alone could pick the wrong finalist; a candidate
    never screened, the control or a baseline seed missing are refused too."""
    with pytest.raises(Refusal, match=re.escape(why)):
        select_finalist(*_rows(world, tmp_path, drop=drop))


def _extra(world: World, name: str, seed: int, digest: str, **cells: str) -> dict[str, str]:
    return _ledger_values(name, seed, digest, world.config_file, world.split_hash) | cells


def test_rows_the_plan_does_not_have_are_refused(world: World, tmp_path: Path) -> None:
    """A fourth seed, a second control seed, and a config the plan does not name."""
    fourth = _extra(world, FINALIST, 3, score_test_digest(FINALIST, 3))
    with pytest.raises(Refusal, match="run seed other than 0, 1 and 2: p3_convnext_cp-seed3-"):
        select_finalist(*_rows(world, tmp_path / "a", extra=[fourth]))
    control = _extra(world, "p3_baseline", 1, score_test_digest("p3_baseline", 1))
    with pytest.raises(Refusal, match=r"of run seed 0 .*, not of seeds \[0, 1\]"):
        select_finalist(*_rows(world, tmp_path / "b", extra=[control]))
    ledger, baseline = _rows(world, tmp_path / "c")
    other = Row(
        99,
        {**ledger[0].values, "config": "p3_other", "run id": "p3_other-seed0-x"}
        | {"checkpoint sha256": score_test_digest("p3_other", 0)},
    )
    with pytest.raises(Refusal, match="holds rows of p3_other, which docs/phase3-plan.md"):
        select_finalist([*ledger, other], baseline)


def test_the_control_is_checked_against_arm_a_seed_0(world: World, tmp_path: Path) -> None:
    """Inside phase 2's bands (val 0.003, trans_val 0.03) the screen is read; outside either,
    refused; inside but not bit for bit, a screen verdict that would differ against the control
    is refused (it goes to the user), and one that agrees passes."""
    inside = _with(p3_baseline_0=("0.8572", "0.3215"))
    assert select_finalist(*_rows(world, tmp_path / "a", inside)).finalist.config == FINALIST
    for n, control in enumerate([("0.8573", "0.2915"), ("0.8542", "0.3216")]):
        scores = _with(p3_baseline_0=control)
        with pytest.raises(Refusal, match="outside a band, no screen is read"):
            select_finalist(*_rows(world, tmp_path / f"b{n}", scores))
    # p3_convnext_cp's seed 0 trans_val: +0.0385 over arm A (passes), +0.0200 over this
    # control, 0.0185 from arm A's (fails)
    differs = _with(p3_baseline_0=("0.8542", "0.3100"), p3_convnext_cp_0=("0.8573", "0.3300"))
    with pytest.raises(Refusal, match="p3_convnext_cp's screen passes against one of"):
        select_finalist(*_rows(world, tmp_path / "c", differs))


def test_the_control_must_have_arm_a_s_config_hash(world: World, tmp_path: Path) -> None:
    ledger, baseline = _rows(world, tmp_path)
    control = next(r for r in ledger if r["config"] == "p3_baseline")
    changed = Row(control.line, {**control.values, "config hash": "0" * 64})
    with pytest.raises(Refusal, match="p3_baseline does not have resnet18_arm_a's config hash"):
        select_finalist([changed if r is control else r for r in ledger], baseline)


def test_nothing_kept_is_no_test_scoring(world: World, tmp_path: Path) -> None:
    """Both candidates fail: phase 3 has no test scoring, and the message gives both verdicts."""
    scores = _with(
        p3_convnext_0=("0.8837", "0.3100"),  # fails the screen
        p3_convnext_cp_2=("0.8654", "0.3100"),  # passes it, but does not rise on seed 2
    )
    drop = [("p3_convnext", 1), ("p3_convnext", 2)]
    with pytest.raises(Refusal, match="no phase-3 candidate was kept") as refused:
        select_finalist(*_rows(world, tmp_path, scores, drop=drop))
    assert "p3_convnext: not kept, failed the screen" in str(refused.value)
    assert "p3_convnext_cp: not kept" in str(refused.value)


# The real committed ledgers: the finalist they give, recomputed; nothing scored.


def test_on_the_real_ledgers_the_finalist_is_p3_convnext_cp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """docs/ledger-phase3.md against resnet18_arm_a's rows of docs/ledger.md: both candidates
    kept, and the finalist rule picks p3_convnext_cp's three rows, seeds 0, 1 and 2 (compared
    with the ledger's own rows of that config, not with run ids typed here). Their configs in
    config.yaml still hash to the rows and are ConvNeXt-Tiny. Nothing is hashed or loaded."""
    _forbid_reading(monkeypatch)
    monkeypatch.setattr(score_test, "file_sha256", _forbid_hashing)
    ledger, baseline = read_ledger(PHASE3_LEDGER_PATH), read_ledger(LEDGER_PATH)
    selection = select_finalist(ledger, baseline)
    assert [(v.config, v.kept) for v in selection.verdicts] == [
        ("p3_convnext", True),
        ("p3_convnext_cp", True),
    ]
    assert selection.finalist.config == FINALIST
    own = sorted((r for r in ledger if r["config"] == FINALIST), key=lambda r: int(r["run seed"]))
    assert list(selection.finalist.rows) == own
    assert [int(r["run seed"]) for r in own] == [0, 1, 2]
    assert selection.rule.startswith("the higher mean trans_val macro-F1")
    configs = [score_test._config_of(row, CONFIG_PATH) for row in selection.finalist.rows]
    assert {(c.model, c.arm) for c in configs} == {("convnext_tiny", "C")}


def _forbid_hashing(*_: Any) -> None:
    raise AssertionError("a checkpoint was hashed")


def test_the_repo_phase3_test_ledger_reads() -> None:
    """docs/test-ledger-phase3.md exists (the scoring refuses a missing one) with the header of
    `TEST_LEDGER_COLUMNS` once and only rows after it: none until the user's one run, then the
    finalist's three."""
    score_test.read_test_ledger(PHASE3_TEST_LEDGER_PATH)
    assert PHASE3_TEST_LEDGER_PATH.read_text().count(score_test.scoring_header()) == 1


def test_no_fixture_reads_the_repo_ledgers() -> None:
    """Only the two tests of the real files name them (the trap of 6.04 and 9.03), and the full
    run's check that it wrote none of them."""
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    names = ("PHASE3_TEST_LEDGER_PATH", "PHASE3_LEDGER_PATH", "LEDGER_PATH", "REPO_DOCS")
    naming = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
        if any(name in ast.unparse(node) for name in names)
        if node.name != "test_no_fixture_reads_the_repo_ledgers"
    }
    assert naming == {
        "test_on_the_real_ledgers_the_finalist_is_p3_convnext_cp",
        "test_the_repo_phase3_test_ledger_reads",
        "scoring",
        "test_the_run_writes_no_ledger_and_no_repo_file",
    }


# The full run: the finalist's three checkpoints, once each, into a fresh test ledger.


@dataclass
class Scoring:
    code: int
    stdout: str
    stderr: str
    out: Path
    before: dict[Path, bytes]  # the fixture's two ledgers
    repo_before: dict[Path, bytes]
    figures_before: list[Path]
    predict_calls: list[dict[str, Any]]
    attention_calls: list[str]


@pytest.fixture(scope="module")
def scoring(world: World) -> Iterator[Scoring]:
    predict_calls: list[dict[str, Any]] = []
    attention_calls: list[str] = []
    real_predict, real_attention = slices.predict_split, attention.attention_pass

    def predict_spy(
        model: torch.nn.Module, records: Sequence[Record], split: str, *a: Any, **k: Any
    ) -> Any:
        run = real_predict(model, records, split, *a, **k)
        predict_calls.append(
            {"model": id(model), "split": split, "splits": {r.split for r in records}, "run": run}
        )
        return run

    def attention_spy(
        model: torch.nn.Module, records: Sequence[Record], split: str, *a: Any, **k: Any
    ) -> Any:
        attention_calls.append(split)
        return real_attention(model, records, split, *a, **k)

    def second_pass(*_: Any, **__: Any) -> None:
        raise AssertionError("evaluate.predict was called: a second pass over a test split")

    out = world.root / "final"
    before = {path: path.read_bytes() for path in (world.ledger, world.baseline)}
    repo_before = {path: path.read_bytes() for path in REPO_DOCS}
    figures_before = _figures_snapshot()
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score_test, "predict_split", predict_spy)
        mp.setattr(score_test, "attention_pass", attention_spy)
        mp.setattr(evaluate, "predict", second_pass)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = score_test.main(world.argv(out))
    yield Scoring(
        code,
        stdout.getvalue(),
        stderr.getvalue(),
        out,
        before,
        repo_before,
        figures_before,
        predict_calls,
        attention_calls,
    )


def test_the_finalist_is_scored_once_per_checkpoint_and_test_split(scoring: Scoring) -> None:
    """Three checkpoints, each one pass over cis_test and one over trans_test (and one attention
    pass each), on those splits' records only; no other checkpoint is scored."""
    assert scoring.code in (0, ALARM_EXIT), scoring.stderr
    assert [c["split"] for c in scoring.predict_calls] == [*TEST_SPLITS] * 3
    assert [c["splits"] for c in scoring.predict_calls] == [{s} for s in TEST_SPLITS] * 3
    models = [c["model"] for c in scoring.predict_calls]
    assert len(set(models)) == 3 and models[0::2] == models[1::2]
    assert scoring.attention_calls == [*TEST_SPLITS] * 3


def test_one_row_per_finalist_checkpoint_with_the_approver(world: World, scoring: Scoring) -> None:
    """The fresh test ledger gains exactly the finalist's three rows, in seed order, with the
    training rows' run id, model, arm, config and split hash, each file's sha256, the scores of
    the one pass (by hand) and the approver."""
    rows = score_test.read_test_ledger(world.test_ledger)
    training = {r["run id"]: r for r in read_ledger(world.ledger)}
    assert [training[r["run id"]]["config"] for r in rows] == [FINALIST] * 3
    assert [int(training[r["run id"]]["run seed"]) for r in rows] == [0, 1, 2]
    assert [r["checkpoint sha256"] for r in rows] == [file_sha256(p) for p in world.finalist()]
    for row, (cis, trans) in zip(rows, _pairs(scoring), strict=True):
        source = training[row["run id"]]
        for column in ("model", "arm", "config", "split hash"):
            assert row[column] == source[column]
        assert row["approved by"] == APPROVER
        assert row["cis_test accuracy"] == f"{accuracy(cis.predictions, cis.labels):.4f}"
        assert row["cis_test macro-F1"] == f"{macro_f1(cis.predictions, cis.labels):.4f}"
        assert row["trans_test accuracy"] == f"{accuracy(trans.predictions, trans.labels):.4f}"
        assert row["trans_test macro-F1"] == f"{macro_f1(trans.predictions, trans.labels):.4f}"
    assert f"appended 3 rows to {world.test_ledger}" in scoring.stdout


def _pairs(scoring: Scoring) -> list[tuple[Any, Any]]:
    runs = [c["run"] for c in scoring.predict_calls]
    return list(zip(runs[0::2], runs[1::2], strict=True))


def test_the_outputs_say_which_recipe_why_and_that_the_test_sets_were_used_again(
    world: World, scoring: Scoring
) -> None:
    """The headline names phase 3, the verdicts and the finalist rule, and discloses the second
    use of the test sets (P3 §7); the selection is printed to the log before any scoring; the
    seed mean of the three seeds is a row; per-checkpoint outputs and the slices exist."""
    headline = (scoring.out / "headline.md").read_text()
    assert headline.startswith("# Phase 3's test scoring: cis_test and trans_test")
    for needle in (
        f"Approved by {APPROVER}",
        "The finalist p3_convnext_cp, run seeds 0, 1, 2",
        "- p3_convnext: kept",
        "- p3_convnext_cp: kept",
        "- The finalist: p3_convnext_cp (the higher mean trans_val macro-F1, by 0.0571",
        "The test sets were used a second time here, for this one recipe",
        "| resnet18 arm C, mean of 3 seeds | 3 runs |",
        "Beery et al. 2018",
    ):
        assert needle in headline, needle
    assert scoring.stdout.index("- The finalist: p3_convnext_cp") < scoring.stdout.index(
        "test ledger row of"
    )
    for row in score_test.read_test_ledger(world.test_ledger):
        metrics = (scoring.out / row["run id"] / "metrics.md").read_text()
        assert "Phase 3's test scoring of resnet18 arm C seed" in metrics
        assert "used a second time" in metrics
        assert (scoring.out / row["run id"] / "attention.md").is_file()
    assert "used a second time" in (scoring.out / "slices.md").read_text()
    assert (scoring.out / "features_by_camera.png").is_file()


def test_the_run_writes_no_ledger_and_no_repo_file(world: World, scoring: Scoring) -> None:
    """The training ledgers are read only; the repo's docs and figures are untouched."""
    assert {path: path.read_bytes() for path in scoring.before} == scoring.before
    assert {path: path.read_bytes() for path in REPO_DOCS} == scoring.repo_before
    assert _figures_snapshot() == scoring.figures_before


def test_a_second_run_is_refused(
    world: World,
    scoring: Scoring,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The test ledger now holds the three rows: the same command exits 1 before reading
    anything, and the rows stay as they are."""
    assert scoring.code in (0, ALARM_EXIT)
    _forbid_reading(monkeypatch)
    before = world.test_ledger.read_text()
    assert score_test.main(world.argv(world.root / "again")) == REFUSED_EXIT
    assert "phase 3's test scoring has run" in capsys.readouterr().err
    assert world.test_ledger.read_text() == before


# Refusals: exit 1 before any checkpoint is loaded or any test image read.


def _refused(world: World, argv: list[str], capsys: pytest.CaptureFixture[str], why: str) -> None:
    before = world.test_ledger.read_text()
    assert score_test.main(argv) == REFUSED_EXIT
    err = capsys.readouterr().err
    assert "score_test refused" in err and why in err, err
    assert world.test_ledger.read_text() == before


@pytest.mark.parametrize("approver", [None, "", "   ", "a | b", "Jane Doe"])
def test_without_an_approver_it_exits_1(
    fresh: World,
    tmp_path: Path,
    approver: str | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_reading(monkeypatch)
    out = tmp_path / "out"
    _refused(fresh, fresh.argv(out, approver=approver), capsys, "--approved-by")
    assert not out.exists()


def test_a_test_ledger_that_holds_a_row_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    row = dict.fromkeys(TEST_LEDGER_COLUMNS, "x")
    with fresh.test_ledger.open("a", encoding="utf-8") as ledger:
        ledger.write(score_test.ledger_line(row) + "\n")
    _forbid_reading(monkeypatch)
    _refused(fresh, fresh.argv(tmp_path / "out"), capsys, "already holds 1 rows")


@pytest.mark.parametrize(
    ("scores", "drop", "why"),
    [
        # p3_convnext_cp does not rise on seed 2; p3_convnext is kept and the finalist
        (_with(p3_convnext_cp_2=("0.8654", "0.3100")), (), "of p3_convnext_cp, not kept under"),
        # both kept, p3_convnext ahead by more than 0.03
        (
            _with(
                p3_convnext_0=("0.8837", "0.4682"),
                p3_convnext_1=("0.8801", "0.4514"),
                p3_convnext_2=("0.8767", "0.4477"),
            ),
            (),
            "of p3_convnext_cp, kept, but not the finalist",
        ),
        # nothing kept: no test scoring
        (
            _with(p3_convnext_0=("0.8837", "0.3100"), p3_convnext_cp_2=("0.8654", "0.3100")),
            (("p3_convnext", 1), ("p3_convnext", 2)),
            "no phase-3 candidate was kept, so phase 3 has no test scoring",
        ),
        # p3_convnext_cp's confirm not finished: fewer than 3 seeds
        (REAL_LIKE, ((FINALIST, 2),), "its confirm is not finished"),
    ],
)
def test_a_recipe_that_is_not_the_kept_finalist_exits_1(
    fresh: World,
    tmp_path: Path,
    scores: Scores,
    drop: Sequence[tuple[str, int]],
    why: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """p3_convnext_cp's checkpoints given against ledgers where it was not kept, was kept but is
    not the finalist, where nothing was kept, or where its confirm lacks a seed."""
    ledger, baseline = write_ledgers(fresh, tmp_path / "ledgers", scores, drop=drop)
    _forbid_reading(monkeypatch)
    argv = fresh.argv(tmp_path / "out", ledger=ledger, baseline=baseline)
    _refused(fresh, argv, capsys, why)


@pytest.mark.parametrize(
    ("extra", "why"),
    [
        (("p3_baseline", 0), "of p3_baseline, the control: phase 3 scores the finalist"),
        (("p3_convnext", 1), "of p3_convnext, kept, but not the finalist"),
    ],
)
def test_a_checkpoint_of_another_config_exits_1(
    fresh: World,
    tmp_path: Path,
    extra: tuple[str, int],
    why: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Beside the finalist's three, the control's or the other candidate's file (each with its
    own row): refused, so no other run is ever scored on a test split."""
    _forbid_reading(monkeypatch)
    paths = [*fresh.finalist(), fresh.files[extra]]
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, why)


@pytest.mark.parametrize(
    ("seeds", "why"),
    [
        ((0, 1), "--run-seeds 0 1: phase 3 scores run seeds 0 1 2"),
        ((0,), "--run-seeds 0: phase 3 scores run seeds 0 1 2"),
        ((0, 1, 2, 3), "--run-seeds 0 1 2 3: phase 3 scores run seeds 0 1 2"),
        (None, "--run-seeds is required"),
    ],
)
def test_a_seed_set_other_than_0_1_2_exits_1(
    fresh: World,
    tmp_path: Path,
    seeds: tuple[int, ...] | None,
    why: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_reading(monkeypatch)
    _refused(fresh, fresh.argv(tmp_path / "out", run_seeds=seeds), capsys, why)


def test_a_finalist_seed_without_its_checkpoint_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The three seeds are scored together: two of them alone are refused."""
    _forbid_reading(monkeypatch)
    paths = fresh.finalist()[:2]
    why = "three seeds are scored together, once; no checkpoint given for p3_convnext_cp-seed2-"
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, why)


@pytest.mark.parametrize("how", ["unknown", "changed", "twice", "missing"])
def test_a_checkpoint_with_no_row_or_a_changed_sha256_exits_1(
    fresh: World,
    tmp_path: Path,
    how: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A file no row names; a finalist file with one byte added (its sha256 changed since
    training); a file given twice; a path with no file."""
    seed0, seed1, seed2 = fresh.finalist()
    if how == "unknown":
        stranger = _save_checkpoint(tmp_path / "stranger.pt", "resnet18", 99)
        paths, why = [seed0, seed1, seed2, stranger], "has no row in ledger-phase3.md: an unknown"
    elif how == "changed":
        changed = tmp_path / seed0.name
        changed.write_bytes(seed0.read_bytes() + b"\0")
        paths, why = [changed, seed1, seed2], "or a file whose bytes changed since it was trained"
    elif how == "twice":
        paths, why = [seed0, seed1, seed2, seed1], "are one checkpoint (sha256"
    else:
        paths, why = [seed0, seed1, tmp_path / "gone.pt"], "no checkpoint at"
    _forbid_reading(monkeypatch)
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, why)


def test_a_phase1_or_phase2_ledger_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """By file name, before any ledger is read: the phase-3 mode takes `ledger-phase3.md`,
    `test-ledger-phase3.md` and a phase-1 baseline ledger only."""
    _forbid_reading(monkeypatch)
    monkeypatch.setattr(score_test, "read_ledger", _forbid_hashing)
    out = tmp_path / "out"
    for name in ("ledger.md", "ledger-phase2.md"):
        argv = fresh.argv(out, ledger=tmp_path / name)
        _refused(fresh, argv, capsys, f"{name} is not the phase-3 ledger: --phase 3 takes")
    test_ledger = _empty_test_ledger(tmp_path / "test-ledger.md")
    argv = fresh.argv(out, test_ledger=test_ledger)
    _refused(fresh, argv, capsys, "test-ledger.md is not phase 3's test ledger")
    for name, phase in (("ledger-phase3.md", 3), ("ledger-phase2.md", 2)):
        argv = fresh.argv(out, baseline=tmp_path / "base" / name)
        _refused(fresh, argv, capsys, f"--baseline-ledger {name} is a phase-{phase} ledger")
    assert not out.exists()


def test_a_missing_phase3_test_ledger_is_refused_not_created(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A mistyped path must not pass for an empty test ledger."""
    _forbid_reading(monkeypatch)
    missing = tmp_path / "elsewhere" / "test-ledger-phase3.md"
    assert score_test.main(fresh.argv(tmp_path / "out", test_ledger=missing)) == REFUSED_EXIT
    assert "no test ledger at" in capsys.readouterr().err
    assert not missing.exists()


def test_a_phase3_ledger_that_fails_its_check_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A phase-1 row pasted into the phase-3 ledger fails its check."""
    stray = _extra(fresh, "resnet18_arm_a", 0, score_test_digest("stray", 0))
    ledger, baseline = write_ledgers(fresh, tmp_path / "ledgers")
    ledger.write_text(ledger.read_text() + ledger_row(stray) + "\n")
    _forbid_reading(monkeypatch)
    argv = fresh.argv(tmp_path / "out", ledger=ledger, baseline=baseline)
    _refused(fresh, argv, capsys, "ledger-phase3.md fails its check")


def test_a_changed_config_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The finalist's config must still hash to its rows."""
    changed = _config_file(tmp_path / "config.yaml", weight_decay=0.5)
    _forbid_reading(monkeypatch)
    argv = fresh.argv(tmp_path / "out", config_file=changed)
    _refused(fresh, argv, capsys, "no longer matches its ledger row (config hash)")


# The phase-1 mode refuses the phase-3 files by name.


@pytest.mark.parametrize(
    ("change", "why"),
    [
        ({"ledger": "ledger-phase3.md"}, "ledger-phase3.md is the phase-3 ledger: the phase-1"),
        ({"ledger": "ledger-phase2.md"}, "ledger-phase2.md is the phase-2 ledger: the phase-1"),
        ({"test_ledger": "test-ledger-phase3.md"}, "is phase 3's test ledger (--phase 3)"),
        ({"baseline": "ledger.md"}, "--baseline-ledger is phase 3's (--phase 3)"),
    ],
)
def test_the_phase1_mode_refuses_the_phase3_files(
    fresh: World,
    tmp_path: Path,
    change: dict[str, str],
    why: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`--phase 1` (and no `--phase`, its default) with docs/ledger-phase3.md, docs/
    ledger-phase2.md or docs/test-ledger-phase3.md, or with --baseline-ledger: exit 1 before any
    ledger is read."""
    _forbid_reading(monkeypatch)
    monkeypatch.setattr(score_test, "read_ledger", _forbid_hashing)
    ledger = tmp_path / change.get("ledger", "ledger.md")
    test_ledger = _empty_test_ledger(tmp_path / change.get("test_ledger", "test-ledger.md"))
    for phase in ("--phase=1", None):
        argv = fresh.argv(
            tmp_path / "out", ledger=ledger, test_ledger=test_ledger, run_seeds=(0,), phase=None
        )
        argv += [phase] if phase else []
        if "baseline" in change:
            argv += ["--baseline-ledger", str(tmp_path / change["baseline"])]
        _refused(World(**{**fresh.__dict__, "test_ledger": test_ledger}), argv, capsys, why)


def test_the_parser_takes_phase_1_or_3() -> None:
    parser = score_test.build_parser()
    base = ["--data-root", "c", "--out", "o"]
    assert parser.parse_args(base).phase == 1
    assert parser.parse_args([*base, "--phase", "3"]).phase == 3
    assert parser.parse_args(base).baseline_ledger is None
    with pytest.raises(SystemExit):
        parser.parse_args([*base, "--phase", "2"])


def test_the_selection_lines_are_the_verdicts_and_the_finalist(
    world: World, tmp_path: Path
) -> None:
    selection: Selection = select_finalist(*_rows(world, tmp_path))
    lines = selection.lines()
    assert len(lines) == 3 and lines[-1].startswith("- The finalist: p3_convnext_cp (")


# The headline's labels and a run that stops part way, with the passes replaced (fast, no image).


def _fake_passes(
    monkeypatch: pytest.MonkeyPatch, fail_at: int | None = None, maps_fail: bool = False
) -> list[str]:
    """Replace loading, the overlays, the passes over the test splits and the figure writers with
    stand-ins: each checkpoint "predicts" four images, and the `fail_at`-th call of
    `score_checkpoint` (from 1) raises as a CUDA out-of-memory would; `maps_fail` makes the
    feature maps, written after every checkpoint is scored, raise. Returns the run ids scored."""
    scored: list[str] = []

    def score(model: Any, checkpoint: Any, *_: Any, **__: Any) -> score_test.Scored:
        if fail_at is not None and len(scored) + 1 == fail_at:
            raise RuntimeError("CUDA out of memory (a stand-in)")
        scored.append(checkpoint.run_id)
        runs = tuple(
            slices.SplitPredictions(split, (0, 0, 1, 1), (0, 1, 1, 1), (False,) * 4, (1,) * 4)
            for split in TEST_SPLITS
        )
        return score_test.Scored(checkpoint, runs, ())

    def outputs(result: Any, out_dir: Path, header: Any) -> list[Path]:
        out_dir.mkdir(parents=True, exist_ok=True)  # as the real writers make it
        return []

    def maps(*_: Any, **__: Any) -> None:
        if maps_fail:
            raise RuntimeError("CUDA out of memory in the feature maps (a stand-in)")

    monkeypatch.setattr(score_test, "load_models", lambda cs: [torch.nn.Identity() for _ in cs])
    monkeypatch.setattr(score_test, "write_overlays", lambda *_, **__: [])
    monkeypatch.setattr(score_test, "score_checkpoint", score)
    monkeypatch.setattr(score_test, "write_checkpoint_outputs", outputs)
    monkeypatch.setattr(score_test.slices, "write_report", lambda *_, **__: None)
    monkeypatch.setattr(score_test, "project", lambda runs: None)
    monkeypatch.setattr(score_test, "write_feature_maps", maps)
    return scored


def test_a_convnext_finalist_s_headline_names_its_model_and_the_seed_order(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Review 13.12, finding 2: with the candidates named `convnext_tiny` in the config (as the
    real ones are), the headline calls the feature maps ConvNeXt-Tiny's, from the scored
    checkpoints' configs, and its rows in seed order; phase 1's text stays "ResNet-18" and
    "ledger order". No ConvNeXt is built: the passes are stand-ins."""
    config_file = _config_file(tmp_path / "config.yaml")
    settings = yaml.safe_load(config_file.read_text())
    for name in PHASE3_CANDIDATES:
        settings["configs"][name]["model"] = "convnext_tiny"
    config_file.write_text(yaml.safe_dump(settings))
    convnext = World(**{**fresh.__dict__, "config_file": config_file})
    ledger, baseline = write_ledgers(convnext, tmp_path / "ledgers")
    assert {r["model"] for r in read_ledger(ledger) if r["config"] in PHASE3_CANDIDATES} == {
        "convnext_tiny"
    }
    _fake_passes(monkeypatch)
    out = tmp_path / "out"
    argv = convnext.argv(out, ledger=ledger, baseline=baseline, config_file=config_file)
    assert score_test.main(argv) in (0, ALARM_EXIT), capsys.readouterr().err
    headline = (out / "headline.md").read_text()
    assert "`features_by_camera.png`: the ConvNeXt-Tiny feature maps." in headline
    assert "ResNet-18" not in headline
    assert "Rows in seed order,\nthe finalist's run seeds 0, 1 and 2, never by a score" in headline
    assert "ledger order" not in headline
    assert "| convnext_tiny arm C, mean of 3 seeds | 3 runs |" in headline
    phase1 = score_test.render_headline([], [])
    assert "the ResNet-18 feature maps." in phase1 and "Rows in ledger order," in phase1


@pytest.mark.parametrize(("fail_at", "maps_fail", "printed"), [(2, False, 1), (None, True, 3)])
def test_a_run_that_stops_part_way_records_nothing_and_may_run_again(
    fresh: World,
    tmp_path: Path,
    fail_at: int | None,
    maps_fail: bool,
    printed: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Coordinator's check after review 13.12: an out-of-memory on the second checkpoint, or in
    the feature maps after all three are scored, stops the run with the rows scored printed and
    `test-ledger-phase3.md` still without a row (the rows are appended last, after every output),
    so the run-once guard lets a clean re-run with a new `--out` through, which appends the three
    rows once."""
    scored = _fake_passes(monkeypatch, fail_at, maps_fail)
    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        score_test.main(fresh.argv(tmp_path / "first"))
    out = capsys.readouterr().out
    assert out.count("test ledger row of") == printed == len(scored)
    assert score_test.read_test_ledger(fresh.test_ledger) == []
    assert not (tmp_path / "first" / "headline.md").exists() or maps_fail

    scored.clear()
    _fake_passes(monkeypatch)
    assert score_test.main(fresh.argv(tmp_path / "again")) in (0, ALARM_EXIT)
    rows = score_test.read_test_ledger(fresh.test_ledger)
    finalist = [r["run id"] for r in read_ledger(fresh.ledger) if r["config"] == FINALIST]
    assert [r["run id"] for r in rows] == finalist
    assert score_test.main(fresh.argv(tmp_path / "third")) == REFUSED_EXIT  # then, never again
