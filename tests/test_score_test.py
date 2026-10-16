"""trailcam/evaluate/score_test.py, the final scoring (item 9.01; R03, R17, R26, R32, R40).

Synthetic fixture only: a cache laid out as item 6.01 writes it, built from
`trailcam/data/synthetic.py` in a temporary directory, three checkpoints with random weights saved
as `train.py` saves them (the small CNN and ResNet-18 arm A and arm B, seed 0), a ledger with their
rows and an empty copy of `docs/test-ledger.md`. No real checkpoint and no real test split is ever
read here; outputs go to pytest's temporary directories, never to `docs/figures/`.

One full run is shared by the tests that read its outputs (`scoring`); the refusal tests replace
the passes over the images and the metadata loader with functions that fail, so a refusal is shown
to come before any test image is read.
"""

import contextlib
import dataclasses
import hashlib
import io
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.data.split import make_split
from trailcam.evaluate import attention, evaluate, score_test, slices
from trailcam.evaluate.metrics import accuracy, macro_f1
from trailcam.evaluate.score_test import (
    REFUSED_EXIT,
    TEST_LEDGER_COLUMNS,
    TEST_LEDGER_PATH,
    TEST_SPLITS,
    Checkpoint,
    Headline,
    Refusal,
    Scored,
)
from trailcam.evaluate.slices import SplitPredictions
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, config_hash, load_config
from trailcam.train.ledger import Row, check_ledger
from trailcam.train.train import (
    ALARM_EXIT,
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    LEDGER_COLUMNS,
    TRANS_GAP_ALARM,
    VAL_ACCURACY_ALARM,
    file_sha256,
    ledger_header,
    ledger_row,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
APPROVER = "Test Approver"
STARTED = "20261016T090000Z"
# (config name, model, arm, run seed, torch seed of the random weights)
RUNS = (
    ("small", "small_cnn", "A", 0, 1),
    ("arm_a", "resnet18", "A", 0, 2),
    ("arm_b", "resnet18", "B", 0, 3),
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _figures_snapshot() -> list[Path]:
    return sorted(FIGURES.rglob("*")) if FIGURES.exists() else []


def _config_file(path: Path, **changes: Any) -> Path:
    """A copy of config.yaml with the three test configs: no workers, small batches, no
    pretrained download; `changes` go into every one of them."""
    settings = yaml.safe_load(CONFIG_PATH.read_text())
    loader = {"batch_size": 16, "num_workers": 0, "pin_memory": False, **changes}
    settings["configs"] = {
        "small": {**settings["configs"]["small_cnn"], **loader},
        "arm_a": {**settings["configs"]["resnet18_arm_a"], "pretrained": False, **loader},
        "arm_b": {**settings["configs"]["resnet18_arm_b"], "pretrained": False, **loader},
    }
    path.write_text(yaml.safe_dump(settings))
    return path


def _ledger_values(
    name: str, seed: int, digest: str, config_file: Path, split_hash: str
) -> dict[str, str]:
    c = load_config(name, config_file)
    return {
        "run id": f"{name}-seed{seed}-{STARTED}",
        "model": c.model,
        "arm": c.arm,
        "config": name,
        "config hash": config_hash(c),
        "split seed": str(c.split_seed),
        "run seed": str(seed),
        "split hash": split_hash,
        "epochs run": "3",
        "best epoch": "2",
        "scheduler": f"{c.scheduler} warmup {c.warmup_epochs}",
        "val accuracy": "0.5000",
        "val macro-F1": "0.4000",
        "trans_val accuracy": "0.3000",
        "trans_val macro-F1": "0.2000",
        "checkpoint sha256": digest,
    }


def _write_ledger(path: Path, rows: Sequence[dict[str, str]]) -> Path:
    lines = ["# Ledger", "", ledger_header(), *map(ledger_row, rows)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _save_checkpoint(path: Path, model_name: str, torch_seed: int) -> Path:
    torch.manual_seed(torch_seed)
    model = build_model(model_name, pretrained=False)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": model_name, "classes": list(CLASSES), "state_dict": state}, path)
    return path


def _photo(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (320, 240), (90, 110, 70))
    image.paste((200, 180, 160), (100, 80, 160, 120))
    image.save(path, quality=95)
    return path


@dataclass
class World:
    """Everything a scoring run needs, under one temporary directory."""

    root: Path
    cache: Path
    config_file: Path
    ledger: Path
    test_ledger: Path
    assets: Path
    photo: Path
    checkpoints: dict[str, Path]  # by config name
    rows: dict[str, dict[str, str]]  # the ledger rows, by config name
    split_hash: str
    records: dict[str, tuple[Record, ...]] = field(default_factory=dict)

    def argv(
        self,
        out: Path,
        *,
        approver: str | None = APPROVER,
        checkpoints: Sequence[Path] | None = None,
        config_file: Path | None = None,
        ledger: Path | None = None,
    ) -> list[str]:
        given = self.checkpoints.values() if checkpoints is None else checkpoints
        argv = [arg for path in given for arg in ("--checkpoint", str(path))]
        argv += ["--data-root", str(self.cache), "--out", str(out)]
        argv += ["--config", str(config_file or self.config_file)]
        argv += ["--ledger", str(ledger or self.ledger), "--test-ledger", str(self.test_ledger)]
        argv += ["--assets", str(self.assets), "--overlay", f"{self.photo}={CLASSES[0]}"]
        if approver is not None:
            argv += ["--approved-by", approver]
        return argv


def _build_world(root: Path) -> World:
    built = root / "built"
    fixture = synthetic.write_synthetic_fixture(built)
    cache = root / "cache"
    cache.mkdir()
    fixture.image_root.rename(cache / CACHE_IMAGES)
    Path(fixture.paths["sidecar"]).rename(cache / DAY_NIGHT_SIDECAR)
    built.rename(cache / CACHE_ANNOTATIONS)
    meta = load_metadata(cache / CACHE_ANNOTATIONS, cache / DAY_NIGHT_SIDECAR)
    split = make_split(meta, 42)

    config_file = _config_file(root / "config.yaml")
    checkpoints, rows = {}, {}
    for name, model_name, _, seed, torch_seed in RUNS:
        path = _save_checkpoint(root / f"{name}-seed{seed}.pt", model_name, torch_seed)
        checkpoints[name] = path
        rows[name] = _ledger_values(name, seed, file_sha256(path), config_file, split.hash)
    ledger = _write_ledger(root / "ledger.md", list(rows.values()))
    assert check_ledger(ledger)[1] == []  # the rows pass trailcam.train.ledger's check
    test_ledger = root / "test-ledger.md"
    test_ledger.write_text(TEST_LEDGER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assets = root / "assets"
    return World(
        root=root,
        cache=cache,
        config_file=config_file,
        ledger=ledger,
        test_ledger=test_ledger,
        assets=assets,
        photo=_photo(assets / "photo.jpg"),
        checkpoints=checkpoints,
        rows=rows,
        split_hash=split.hash,
        records={name: split.records[name] for name in split.records},
    )


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    return _build_world(tmp_path_factory.mktemp("score_test"))


@dataclass
class Scoring:
    """One full run on `world`: exit code, output, and what the spies saw."""

    code: int
    stdout: str
    stderr: str
    out: Path
    ledger_before: str
    figures_before: list[Path]
    predict_calls: list[dict[str, Any]]
    attention_calls: list[dict[str, Any]]
    overlay_calls: list[dict[str, Any]]
    order: list[str]  # "overlay", "predict" and "attention", as the spies were called


@pytest.fixture(scope="module")
def scoring(world: World) -> Iterator[Scoring]:
    predict_calls: list[dict[str, Any]] = []
    attention_calls: list[dict[str, Any]] = []
    overlay_calls: list[dict[str, Any]] = []
    order: list[str] = []
    real_predict = slices.predict_split
    real_attention = attention.attention_pass
    real_overlays = evaluate.write_overlays

    def predict_spy(
        model: torch.nn.Module, records: Sequence[Record], split: str, *args: Any, **kwargs: Any
    ) -> SplitPredictions:
        order.append("predict")
        run = real_predict(model, records, split, *args, **kwargs)
        predict_calls.append(
            {
                "model": type(model).__name__,
                "split": split,
                "splits of records": {r.split for r in records},
                "records": len(records),
                "keep_features": kwargs.get("keep_features"),
                "run": run,
            }
        )
        return run

    def attention_spy(
        model: torch.nn.Module, records: Sequence[Record], split: str, *args: Any, **kwargs: Any
    ) -> attention.SplitRun:
        order.append("attention")
        attention_calls.append(
            {
                "split": split,
                "splits of records": {r.split for r in records},
                "inference mode": torch.is_inference_mode_enabled(),
                "grad": torch.is_grad_enabled(),
            }
        )
        return real_attention(model, records, split, *args, **kwargs)

    def overlay_spy(*args: Any, **kwargs: Any) -> list[Path]:
        order.append("overlay")
        overlay_calls.append(
            {"inference mode": torch.is_inference_mode_enabled(), "grad": torch.is_grad_enabled()}
        )
        return real_overlays(*args, **kwargs)

    def second_pass(*_: Any, **__: Any) -> None:
        raise AssertionError("evaluate.predict was called: a second pass over a test split")

    out = world.root / "final"
    ledger_before = world.ledger.read_text(encoding="utf-8")
    figures_before = _figures_snapshot()
    stdout, stderr = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(score_test, "predict_split", predict_spy)
        mp.setattr(score_test, "attention_pass", attention_spy)
        mp.setattr(score_test, "write_overlays", overlay_spy)
        mp.setattr(evaluate, "predict", second_pass)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = score_test.main(world.argv(out))
    yield Scoring(
        code,
        stdout.getvalue(),
        stderr.getvalue(),
        out,
        ledger_before,
        figures_before,
        predict_calls,
        attention_calls,
        overlay_calls,
        order,
    )


def _pairs(scoring: Scoring) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Per checkpoint, in `RUNS` order: (accuracy, macro-F1) on cis_test and on trans_test, by
    hand from the predictions of the one pass the spy saw."""
    scores = [
        (
            accuracy(c["run"].predictions, c["run"].labels),
            macro_f1(c["run"].predictions, c["run"].labels),
        )
        for c in scoring.predict_calls
    ]
    return list(zip(scores[0::2], scores[1::2], strict=True))


def _label(name: str) -> str:
    _, model, arm, seed, _ = next(run for run in RUNS if run[0] == name)
    return f"{model} seed {seed}" if model == "small_cnn" else f"{model} arm {arm} seed {seed}"


# The full run.


def test_the_run_finishes_and_reports_its_alarms(scoring: Scoring) -> None:
    """Exit 0, or `ALARM_EXIT` with the alarms printed (before the append, as `train.py` does)
    when trans_test accuracy is within two points of cis_test, which random weights can give."""
    expected = sum(
        (trans[0] >= cis[0] - TRANS_GAP_ALARM) + (cis[0] > VAL_ACCURACY_ALARM)
        for cis, trans in _pairs(scoring)
    )
    assert scoring.stderr.count("LEAKAGE ALARM") == expected
    assert scoring.code == (ALARM_EXIT if expected else 0), scoring.stderr
    headline = (scoring.out / "headline.md").read_text()
    assert ("None fired." in headline) is (expected == 0)


def test_headline_has_both_splits_both_metrics_and_the_gap(world: World, scoring: Scoring) -> None:
    """One row per checkpoint: cis_test and trans_test accuracy and macro-F1, by hand from the
    one pass's predictions, and the gap, cis minus trans, for both metrics (R32)."""
    text = (scoring.out / "headline.md").read_text()
    assert f"Approved by {APPROVER}" in text
    assert "Beery et al. 2018" in text
    for header in (
        "cis_test accuracy",
        "cis_test macro-F1",
        "trans_test accuracy",
        "trans_test macro-F1",
        "accuracy gap",
        "macro-F1 gap",
    ):
        assert f"| {header} |" in text
    for (name, row), (cis, trans) in zip(world.rows.items(), _pairs(scoring), strict=True):
        line = next(line for line in text.splitlines() if f"| {row['run id']} |" in line)
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        assert cells[0] == _label(name)
        images = [str(len(world.records["cis_test"])), str(len(world.records["trans_test"]))]
        assert cells[2:4] == images
        assert cells[4:] == [
            f"{cis[0]:.4f}",
            f"{cis[1]:.4f}",
            f"{trans[0]:.4f}",
            f"{trans[1]:.4f}",
            f"{cis[0] - trans[0]:.4f}",
            f"{cis[1] - trans[1]:.4f}",
        ]


def test_test_ledger_rows_carry_the_checkpoint_hashes_and_the_approver(
    world: World, scoring: Scoring
) -> None:
    """One row per checkpoint (R40): the training row's run id, model, arm, config and split
    hash, the file's sha256, the scores of the headline, the UTC time and who approved. The
    training ledger is not touched."""
    rows = score_test.read_test_ledger(world.test_ledger)
    assert len(rows) == len(RUNS)
    headline = (scoring.out / "headline.md").read_text()
    for row, (name, path) in zip(rows, world.checkpoints.items(), strict=True):
        trained = world.rows[name]
        assert row["checkpoint sha256"] == file_sha256(path) == trained["checkpoint sha256"]
        for column in ("run id", "model", "arm", "config", "split hash"):
            assert row[column] == trained[column]
        assert row["approved by"] == APPROVER
        assert re.fullmatch(r"\d{8}T\d{6}Z", row["scored at"])
        scores = [row[c] for c in TEST_LEDGER_COLUMNS[6:12]]
        assert f"| {' | '.join(scores)} |" in headline
    assert world.ledger.read_text(encoding="utf-8") == scoring.ledger_before
    assert "test ledger rows:" in scoring.stdout
    assert score_test.scoring_header() in scoring.stdout


def test_one_pass_per_checkpoint_and_test_split(scoring: Scoring) -> None:
    """`predict_split` runs once per checkpoint and split, on cis_test and trans_test records
    only, with features kept for the ResNet-18 alone; `evaluate.predict` never runs (its spy
    fails). Grad-CAM (the attention pass and the overlays) runs outside inference mode (R26)."""
    seen = [(c["model"], c["split"]) for c in scoring.predict_calls]
    assert seen == [(m, s) for m in ("SmallCNN", "ResNet", "ResNet") for s in TEST_SPLITS]
    for call in scoring.predict_calls:
        assert call["splits of records"] == {call["split"]}
        assert call["keep_features"] is (call["model"] == "ResNet")
        assert (call["run"].features is not None) is (call["model"] == "ResNet")
    assert [c["split"] for c in scoring.attention_calls] == list(TEST_SPLITS) * len(RUNS)
    for call in scoring.attention_calls:
        assert call["splits of records"] == {call["split"]}
        assert call["inference mode"] is False and call["grad"] is True
    assert len(scoring.overlay_calls) == len(RUNS)
    assert all(c["inference mode"] is False and c["grad"] for c in scoring.overlay_calls)


def test_the_overlays_come_before_any_test_pass(scoring: Scoring) -> None:
    """The overlays need no test image: all of them are drawn before the first test pass, so a
    failing one stops the run before anything is scored. Per checkpoint, predictions then the
    attention pass."""
    assert scoring.order == ["overlay"] * len(RUNS) + ["predict"] * 2 + ["attention"] * 2 + (
        ["predict"] * 2 + ["attention"] * 2
    ) * (len(RUNS) - 1)


def test_outputs_per_checkpoint_and_across_checkpoints(world: World, scoring: Scoring) -> None:
    """Per run id: the metrics table with per-class, per-camera and confusion matrix on both
    splits and three PNGs each (R32), the attention table and histogram (R26), the overlay; at
    the top, the slice tables with the cis_test / trans_test pair and the feature maps of the
    two ResNet-18 checkpoints. Nothing goes to docs/figures/."""
    for name, row in world.rows.items():
        folder = scoring.out / row["run id"]
        metrics = (folder / "metrics.md").read_text()
        for split, group in (("cis_test", "cis"), ("trans_test", "trans")):
            assert f"## {split} ({group} cameras)" in metrics
            for kind in ("confusion_matrix", "per_class", "per_camera"):
                assert (folder / f"{split}_{kind}.png").read_bytes().startswith(b"\x89PNG")
        assert metrics.count("### Per class") == 2 and metrics.count("### Confusion matrix") == 2
        assert f"checkpoint sha256 {row['checkpoint sha256']}" in metrics
        table = (folder / "attention.md").read_text()
        assert "| cis_test | cis | all |" in table and "| trans_test | trans | all |" in table
        assert (folder / "attention_in_box.png").is_file()
        assert (folder / "gradcam_overlay_01.png").is_file()
        assert len(list(folder.glob("*.png"))) == 6 + 1 + 1, name
    text = (scoring.out / "slices.md").read_text()
    for name in world.rows:
        assert f"## {_label(name)}" in text
    assert "| cis_test | trans_test | accuracy |" in text
    for figure in ("features_by_species.png", "features_by_camera.png"):
        assert (scoring.out / figure).is_file()
    assert _figures_snapshot() == scoring.figures_before


def test_outputs_hold_no_image_id_or_file_name(world: World, scoring: Scoring) -> None:
    """R03: counts, scores, locations, run ids and hashes only."""
    texts = [p.read_text() for p in scoring.out.rglob("*.md")]
    texts += [scoring.stdout, scoring.stderr, world.test_ledger.read_text()]
    for name in TEST_SPLITS:
        for record in world.records[name]:
            for text in texts:
                assert record.image_id not in text
                assert Path(record.file_name).name not in text


def test_a_scored_checkpoint_is_not_scored_again(
    world: World, scoring: Scoring, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every checkpoint now has a row in the test ledger: a second run exits 1 before reading
    anything, and the test ledger keeps its rows."""
    assert scoring.code in (0, ALARM_EXIT)
    _forbid_reading(monkeypatch)
    before = world.test_ledger.read_text()
    out = world.root / "again"
    assert score_test.main(world.argv(out)) == REFUSED_EXIT
    assert world.test_ledger.read_text() == before
    assert not out.exists()


# Refusals: exit 1 before any checkpoint is loaded or any test image read.


def _forbid_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("read past a refusal")

    for name in ("predict_split", "attention_pass", "load_metadata", "load_checkpoint"):
        monkeypatch.setattr(score_test, name, fail)


@pytest.fixture
def fresh(world: World, tmp_path: Path) -> World:
    """`world` with an empty test ledger of its own, so refusal tests do not depend on the full
    run's rows."""
    test_ledger = tmp_path / "test-ledger.md"
    test_ledger.write_text(TEST_LEDGER_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    return World(**{**world.__dict__, "test_ledger": test_ledger})


def _refused(world: World, argv: list[str], capsys: pytest.CaptureFixture[str], why: str) -> None:
    before = world.test_ledger.read_text()
    assert score_test.main(argv) == REFUSED_EXIT
    err = capsys.readouterr().err
    assert "score_test refused" in err and why in err, err
    assert world.test_ledger.read_text() == before


@pytest.mark.parametrize("approver", [None, "", "   ", "a | b"])
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


def test_without_an_approver_from_the_shell_it_exits_1(fresh: World, tmp_path: Path) -> None:
    """The module as Kaggle runs it, `python -m`: exit status 1, not argparse's 2."""
    import subprocess
    import sys

    argv = fresh.argv(tmp_path / "out", approver=None)
    result = subprocess.run(
        [sys.executable, "-m", "trailcam.evaluate.score_test", *argv],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == REFUSED_EXIT, result.stderr
    assert "--approved-by" in result.stderr
    assert not (tmp_path / "out").exists()


def test_an_unknown_checkpoint_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A checkpoint whose sha256 has no ledger row, beside the known ones."""
    stranger = _save_checkpoint(tmp_path / "stranger.pt", "small_cnn", 99)
    _forbid_reading(monkeypatch)
    paths = [*fresh.checkpoints.values(), stranger]
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, "unknown checkpoint")


def test_a_ledger_row_whose_checkpoint_is_missing_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """R17: the arms cannot be scored while the small CNN's checkpoint is left out (every run in
    the ledger is scored in the one run)."""
    _forbid_reading(monkeypatch)
    paths = [fresh.checkpoints["arm_a"], fresh.checkpoints["arm_b"]]
    _refused(
        fresh,
        fresh.argv(tmp_path / "out", checkpoints=paths),
        capsys,
        fresh.rows["small"]["run id"],
    )


def test_a_checkpoint_given_twice_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_reading(monkeypatch)
    paths = [*fresh.checkpoints.values(), fresh.checkpoints["small"]]
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, "one checkpoint")


def test_a_missing_checkpoint_file_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _forbid_reading(monkeypatch)
    paths = [*fresh.checkpoints.values(), tmp_path / "gone.pt"]
    _refused(fresh, fresh.argv(tmp_path / "out", checkpoints=paths), capsys, "no checkpoint at")


def test_a_changed_config_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The config a row names must still hash to the row's config hash."""
    changed = _config_file(tmp_path / "config.yaml", weight_decay=0.5)
    _forbid_reading(monkeypatch)
    argv = fresh.argv(tmp_path / "out", config_file=changed)
    _refused(fresh, argv, capsys, "no longer matches its ledger row (config hash)")


def test_a_ledger_that_fails_its_check_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = dict(fresh.rows["small"], **{"val accuracy": "1.5000"})
    ledger = _write_ledger(tmp_path / "ledger.md", [bad, fresh.rows["arm_a"], fresh.rows["arm_b"]])
    _forbid_reading(monkeypatch)
    _refused(fresh, fresh.argv(tmp_path / "out", ledger=ledger), capsys, "fails its check")


@pytest.mark.parametrize(
    "text",
    [
        "# Test ledger\n\nno table\n",
        "TABLE\n| a | b |\n",
        "TABLE\nprose after the table\n",
        "TABLE\nTABLE\n",
    ],
)
def test_a_malformed_test_ledger_exits_1(
    fresh: World,
    tmp_path: Path,
    text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fresh.test_ledger.write_text(text.replace("TABLE", score_test.scoring_header()))
    _forbid_reading(monkeypatch)
    assert score_test.main(fresh.argv(tmp_path / "out")) == REFUSED_EXIT
    assert "score_test refused" in capsys.readouterr().err


def test_a_non_empty_out_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "left over.md").write_text("an earlier run")
    _forbid_reading(monkeypatch)
    _refused(fresh, fresh.argv(out), capsys, "not an empty folder")


def test_an_overlay_outside_assets_exits_1(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = _photo(tmp_path / "elsewhere" / "photo.jpg")
    _forbid_reading(monkeypatch)
    argv = [*fresh.argv(tmp_path / "out"), "--overlay", str(outside)]
    assert score_test.main(argv) == REFUSED_EXIT
    assert "score_test refused" in capsys.readouterr().err


def test_a_cache_of_another_split_exits_1_before_any_prediction(
    fresh: World,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Rows whose split hash is not the cache's: the metadata is read, then the run stops before
    any image is predicted."""
    rows = [dict(row, **{"split hash": sha("another split")}) for row in fresh.rows.values()]
    ledger = _write_ledger(tmp_path / "ledger.md", rows)

    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("predicted past a refusal")

    monkeypatch.setattr(score_test, "predict_split", fail)
    monkeypatch.setattr(score_test, "attention_pass", fail)
    _refused(fresh, fresh.argv(tmp_path / "out", ledger=ledger), capsys, "split hash")


def test_a_file_of_another_model_is_refused(world: World) -> None:
    """The small CNN's row with the arm A file: `load_checkpoint` refuses it, as a refusal."""
    small = load_config("small", world.config_file)
    row = Row(1, world.rows["small"])
    wrong = Checkpoint(world.checkpoints["arm_a"], "0" * 64, row, small)
    with pytest.raises(Refusal, match="holds a resnet18"):
        score_test.load_models([wrong])


# The test splits have one route.


def test_the_other_command_lines_refuse_the_test_splits() -> None:
    """R17: evaluate.py, attention.py and slices.py share one `CLI_SPLITS`, val and trans_val;
    score_test.py has no --split and scores `TEST_SPLITS` alone. Their tests run each command
    line with cis_test and trans_test and see exit 2."""
    assert attention.CLI_SPLITS == ("val", "trans_val")
    assert evaluate.CLI_SPLITS is attention.CLI_SPLITS
    assert slices.CLI_SPLITS is attention.CLI_SPLITS
    assert TEST_SPLITS == ("cis_test", "trans_test")
    options = {a for action in score_test.build_parser()._actions for a in action.option_strings}
    assert "--split" not in options


def test_only_the_test_records_are_returned(world: World) -> None:
    records = score_test.load_test_records(world.cache, 42, world.split_hash)
    assert list(records) == list(TEST_SPLITS)
    for name, split_records in records.items():
        assert {r.split for r in split_records} == {name}
        assert len(split_records) == len(world.records[name])


# Pieces by hand.


def test_the_repo_test_ledger_reads() -> None:
    """docs/test-ledger.md holds the header of `TEST_LEDGER_COLUMNS` once and only rows after it."""
    score_test.read_test_ledger(TEST_LEDGER_PATH)
    assert TEST_LEDGER_PATH.read_text().count(score_test.scoring_header()) == 1


def test_a_row_with_another_run_seed_gets_that_seed(world: World) -> None:
    """A seed-1 run of arm A: its config is arm A's with run seed 1, and its config hash is arm
    A's (the hash leaves the run seed out)."""
    digest = sha("arm A seed 1")
    row = Row(
        1,
        dict(
            world.rows["arm_a"],
            **{"run id": f"arm_a-seed1-{STARTED}", "run seed": "1", "checkpoint sha256": digest},
        ),
    )
    config = score_test._config_of(row, world.config_file)
    assert config.run_seed == 1
    assert Checkpoint(Path("x.pt"), digest, row, config).label == "resnet18 arm A seed 1"


def _scored(
    name: str,
    seed: int,
    cis: tuple[tuple[int, ...], tuple[int, ...]],
    trans: tuple[tuple[int, ...], tuple[int, ...]],
    config_file: Path,
) -> Scored:
    config = dataclasses.replace(load_config(name, config_file), run_seed=seed)
    values = {column: "x" for column in LEDGER_COLUMNS} | {"run id": f"{name}-seed{seed}"}
    checkpoint = Checkpoint(Path("x.pt"), sha(f"{name}{seed}"), Row(1, values), config)
    runs = tuple(
        SplitPredictions(split, labels, predictions, (False,) * len(labels), (1,) * len(labels))
        for split, (labels, predictions) in zip(TEST_SPLITS, (cis, trans), strict=True)
    )
    return Scored(checkpoint, runs, ())


def test_the_headline_and_seed_means_by_hand(world: World) -> None:
    """Seed 0: cis 3 of 4 right, trans 1 of 4; seed 1: cis 4 of 4, trans 2 of 4. The mean row
    averages the two seeds, and its gap is the mean gap."""
    cis0 = ((0, 0, 1, 1), (0, 0, 1, 0))
    trans0 = ((0, 0, 1, 1), (1, 0, 0, 0))
    cis1 = ((0, 0, 1, 1), (0, 0, 1, 1))
    trans1 = ((0, 0, 1, 1), (0, 1, 1, 0))
    seeds = [
        _scored("arm_a", 0, cis0, trans0, world.config_file),
        _scored("arm_a", 1, cis1, trans1, world.config_file),
    ]
    first = score_test.headline(seeds[0])
    assert (first.cis_accuracy, first.trans_accuracy) == (0.75, 0.25)
    assert first.accuracy_gap == pytest.approx(0.5)
    # seed 0 cis: class 0 F1 = 2*2/(4+1+0) = 0.8, class 1 F1 = 2*1/(2+0+1) = 2/3
    assert first.cis_macro_f1 == pytest.approx((0.8 + 2 / 3) / 2)
    [mean] = score_test.seed_means(seeds)
    assert mean.label == "resnet18 arm A, mean of 2 seeds" and mean.runs == 2
    assert mean.cis_accuracy == pytest.approx(0.875)
    assert mean.trans_accuracy == pytest.approx(0.375)
    assert mean.accuracy_gap == pytest.approx(0.5)
    second = score_test.headline(seeds[1])
    assert mean.macro_f1_gap == pytest.approx((first.macro_f1_gap + second.macro_f1_gap) / 2)
    assert score_test.seed_means(seeds[:1]) == []
    text = score_test.render_headline([first, mean], [])
    assert "| resnet18 arm A, mean of 2 seeds | 2 runs |" in text
    assert "None fired." in text


@pytest.mark.parametrize(
    ("cis", "trans", "fired"),
    [(0.80, 0.60, 0), (0.80, 0.785, 1), (0.80, 0.85, 1), (0.96, 0.70, 1), (0.96, 0.95, 2)],
)
def test_the_leakage_alarms_on_the_test_scores(cis: float, trans: float, fired: int) -> None:
    row = Headline("m", "r", 1, 10, 10, cis, 0.5, trans, 0.5)
    alarms = score_test.scoring_alarms(row)
    assert len(alarms) == fired
    assert all(a.startswith("m: ") for a in alarms)
