"""Item 13.11: phase 3's ConvNeXt-Tiny end to end (docs/phase3-plan.md sections 5, 8 and 9; R17,
R26, R30, R40).

On a cache-shaped synthetic fixture, `train.py`'s command line runs `p3_convnext` into a copy of
docs/ledger-phase3.md (refused against docs/ledger.md first) and writes a ConvNeXt checkpoint
carrying its input size; then `evaluate.py` (with a Grad-CAM overlay), `attention.py` (Grad-CAM at
ConvNeXt's last stage) and `slices.py` (its 768 features, beside `p3_baseline`'s ResNet-18) run
their command lines on it, and `score_test.py` loads it from its ledger row and scores it.

Speed, said plainly: a real ConvNeXt epoch over the fixture's 440 training images takes minutes
on a CPU, so the training run's `fit` and trans_val scoring are replaced (random weights, as
item 13.02's arm C test does; the loop on ConvNeXt is tested in tests/test_train.py), and the
evaluation tools see each split trimmed to its first `TRIM` images (`make_split` wrapped, the
split hash kept). Everything else is the real code path. No network: no weights are downloaded.
"""

import shutil
import socket
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest
import torch
import yaml
from torch import nn

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES, NUM_CLASSES
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, load_metadata
from trailcam.data.split import Split, make_split
from trailcam.evaluate import attention, evaluate, score_test, slices
from trailcam.evaluate.attention import CHECKPOINT_KEYS, load_checkpoint
from trailcam.evaluate.score_test import Checkpoint, load_models, score_checkpoint
from trailcam.model.factory import build_model
from trailcam.train import train
from trailcam.train.config import CONFIG_PATH, config_hash, load_config
from trailcam.train.ledger import check_ledger
from trailcam.train.train import (
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    LEDGER_PATH,
    PHASE3_LEDGER_PATH,
    Scores,
    file_sha256,
    ledger_header,
)

TRIM = 6  # images per split the evaluation tools read here
CONFIGS = ("p3_baseline", "p3_convnext", "p3_convnext_cp")


def _refuse_connection(*args: object, **kwargs: object) -> None:
    raise AssertionError("a phase-3 test tried to open a network connection")


@pytest.fixture(scope="module", autouse=True)
def no_network() -> Iterator[None]:
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(socket.socket, "connect", _refuse_connection)
        yield


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic fixture laid out as the 512 px cache: annotations/, images/ and the
    sidecar."""
    tmp = tmp_path_factory.mktemp("phase3")
    root = tmp / "cache"
    written = synthetic.write_synthetic_fixture(tmp / "fixture")
    annotations = root / CACHE_ANNOTATIONS
    annotations.mkdir(parents=True)
    for path in (tmp / "fixture").glob("*_annotations.json"):
        shutil.copy(path, annotations / path.name)
    shutil.copytree(written.image_root, root / CACHE_IMAGES)
    shutil.copy(written.paths["sidecar"], root / DAY_NIGHT_SIDECAR)
    return root


@pytest.fixture(scope="module")
def config_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """config.yaml with the phase-3 configs small enough for the CPU: batch 4, no workers, no
    pinned memory. Nothing a checkpoint is checked against changes."""
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in CONFIGS:
        settings["configs"][name].update(batch_size=4, num_workers=0, pin_memory=False)
    path = tmp_path_factory.mktemp("config") / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


def _trimmed(split: Split) -> Split:
    return replace(split, records=MappingProxyType({k: v[:TRIM] for k, v in split.records.items()}))


@pytest.fixture
def trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluation tools' splits, each cut to its first `TRIM` images; the hash is kept."""
    for module in (evaluate, attention, slices):
        monkeypatch.setattr(
            module, "make_split", lambda meta, seed: _trimmed(make_split(meta, seed))
        )


@pytest.fixture(scope="module")
def run(cache: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """`train.py`'s command line for `p3_convnext`: first against a copy of docs/ledger.md, which
    it refuses before training, then against a copy of docs/ledger-phase3.md, into which it
    appends one row; random weights (no download), `fit` and the trans_val scoring replaced."""
    tmp = tmp_path_factory.mktemp("run")
    out = tmp / "out"
    argv = ["--config-name=p3_convnext", f"--data-root={cache}", f"--out-dir={out}"]
    phase1 = tmp / "ledger.md"
    phase1.write_text(LEDGER_PATH.read_text())
    with pytest.raises(SystemExit) as exit_info:
        train.main([*argv, f"--ledger={phase1}"])
    assert exit_info.value.code == 2 and not out.exists()
    assert phase1.read_text() == LEDGER_PATH.read_text()

    built: list[nn.Module] = []

    def build(name: str, pretrained: bool) -> nn.Module:
        built.append(build_model(name, pretrained=False))
        return built[-1]

    fake_fit = train.FitResult(
        (train.Epoch(1, True, 1.0, Scores(1.0, 0.5, 0.5)),), 1, (2, NUM_CLASSES)
    )
    ledger = tmp / "ledger-phase3.md"
    ledger.write_text(PHASE3_LEDGER_PATH.read_text())
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train, "build_model", build)
        patch.setattr(train, "fit", lambda *a, **k: fake_fit)
        patch.setattr(train, "evaluate", lambda *a, **k: Scores(1.0, 0.2, 0.1))
        assert train.main([*argv, f"--ledger={ledger}"]) == 0
    (checkpoint,) = out.glob("*.pt")
    return {"checkpoint": checkpoint, "ledger": ledger, "built": built}


def test_p3_convnext_trains_into_the_phase3_ledger_only(run: dict[str, Any]) -> None:
    """One row, which passes the phase-3 check, with `p3_convnext`'s model, arm and pinned
    config hash; the checkpoint holds a ConvNeXt-Tiny and its 224 px input size and loads under
    its config."""
    (model,) = run["built"]
    assert type(model).__name__ == "ConvNeXt"
    rows, problems = check_ledger(run["ledger"])
    assert problems == [] and len(rows) == 1
    row = rows[0]
    config = load_config("p3_convnext")
    assert (row["config"], row["model"], row["arm"]) == ("p3_convnext", "convnext_tiny", "A")
    assert row["config hash"] == config_hash(config)
    assert row["checkpoint sha256"] == file_sha256(run["checkpoint"])
    lines = run["ledger"].read_text().splitlines()
    assert lines[:-1] == PHASE3_LEDGER_PATH.read_text().splitlines()
    assert lines[-2] == ledger_header().splitlines()[1]
    saved = torch.load(run["checkpoint"], weights_only=True)
    assert set(saved) == CHECKPOINT_KEYS
    assert (saved["model"], saved["input_size"], saved["classes"]) == (
        "convnext_tiny",
        224,
        list(CLASSES),
    )
    loaded = load_checkpoint(run["checkpoint"], config, eval_size=config.input_size)
    assert type(loaded).__name__ == "ConvNeXt"
    for name in ("p3_baseline", "resnet18_arm_a"):  # a ResNet-18 config refuses it
        with pytest.raises(ValueError, match="holds a convnext_tiny"):
            load_checkpoint(run["checkpoint"], load_config(name))


def test_the_evaluation_tools_run_on_the_convnext_checkpoint(
    run: dict[str, Any],
    cache: Path,
    config_file: Path,
    tmp_path: Path,
    trimmed: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate.py (metrics, figures and a Grad-CAM overlay), attention.py (Grad-CAM at the last
    stage, `stages[-1]`) and slices.py (`p3_baseline` beside `p3_convnext`, each labelled by its
    config, the ConvNeXt's 768 features in the feature maps), through their command lines."""
    checkpoint = run["checkpoint"]
    targets: list[bool] = []
    real_target = attention.target_layer

    def spy(model: nn.Module) -> nn.Module:
        layer = real_target(model)
        stages = getattr(model, "stages", None)
        targets.append(isinstance(stages, nn.Sequential) and layer is stages[-1])
        return layer

    monkeypatch.setattr(attention, "target_layer", spy)
    data = ["--data-root", str(cache), "--split", "val", "--split", "trans_val"]

    assets = tmp_path / "assets"
    assets.mkdir()
    shutil.copy(sorted((cache / CACHE_IMAGES).rglob("*.jpg"))[0], assets / "photo.jpg")
    out = tmp_path / "evaluate"
    argv = ["--config", "p3_convnext", "--config-file", str(config_file)]
    argv += ["--checkpoint", str(checkpoint), *data, "--out", str(out)]
    argv += ["--overlay", str(assets / "photo.jpg"), "--assets", str(assets)]
    assert evaluate.main(argv) == 0
    table = (out / evaluate.METRICS_NAME).read_text(encoding="utf-8")
    assert "Model convnext_tiny, config p3_convnext, checkpoint sha256" in table
    assert "| val | cis |" in table and "| trans_val | trans |" in table
    assert len(list(out.glob("*.png"))) == 7  # three per split and the overlay

    out = tmp_path / "attention"
    argv = ["--config-name", "p3_convnext", "--config", str(config_file)]
    assert attention.main([*argv, "--checkpoint", str(checkpoint), *data, "--out", str(out)]) == 0
    assert targets and all(targets)
    text = (out / attention.TABLE_NAME).read_text(encoding="utf-8")
    assert "Model convnext_tiny, config p3_convnext" in text
    assert (out / attention.HISTOGRAM_NAME).exists()

    torch.manual_seed(1)
    baseline = tmp_path / "baseline.pt"
    torch.save(
        {
            "model": "resnet18",
            "classes": list(CLASSES),
            "state_dict": build_model("resnet18", pretrained=False).state_dict(),
            "input_size": 224,
        },
        baseline,
    )
    out = tmp_path / "slices"
    argv = ["--checkpoint", "p3_baseline", str(baseline), "--checkpoint", "p3_convnext"]
    argv += [str(checkpoint), "--config", str(config_file), "--data-root", str(cache)]
    assert slices.main([*argv, "--out", str(out)]) == 0
    table = (out / slices.TABLE_NAME).read_text(encoding="utf-8")
    assert "## p3_baseline" in table and "## p3_convnext" in table
    assert f"- p3_convnext: checkpoint sha256 {file_sha256(checkpoint)}" in table
    assert (out / slices.SPECIES_FIGURE).exists() and (out / slices.CAMERA_FIGURE).exists()


def test_the_convnext_feature_map_is_its_768_pooled_features(
    run: dict[str, Any], cache: Path
) -> None:
    """slices' features are the input of `head_module`, ConvNeXt's `head.fc`: 768 per image,
    after the pooling and the head's layer norm."""
    model = load_checkpoint(run["checkpoint"], load_config("p3_convnext"))
    meta = load_metadata(cache / CACHE_ANNOTATIONS, cache / DAY_NIGHT_SIDECAR)
    records = make_split(meta, 42).records["val"][:TRIM]
    result = slices.predict_split(
        model, records, "val", cache / CACHE_IMAGES, torch.device("cpu"), batch_size=3
    )
    assert result.features is not None and tuple(result.features.shape) == (TRIM, 768)
    assert len(result.predictions) == TRIM


@pytest.mark.parametrize(
    ("others", "message"),
    [(["p2_baseline"], "phase-2 and phase-3"), (["resnet18_arm_a"], "phase-1 and phase-3")],
)
def test_slices_never_mixes_phase3_with_another_phase(
    run: dict[str, Any],
    config_file: Path,
    cache: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    others: list[str],
    message: str,
) -> None:
    argv = ["--checkpoint", "p3_convnext", str(run["checkpoint"])]
    argv += ["--checkpoint", others[0], str(run["checkpoint"])]
    argv += ["--config", str(CONFIG_PATH), "--data-root", str(cache), "--out", str(tmp_path)]
    with pytest.raises(SystemExit) as exit_info:
        slices.main(argv)
    assert exit_info.value.code == 2
    assert f"{message} checkpoints in one run" in capsys.readouterr().err


def test_score_test_loads_and_scores_the_convnext_checkpoint_from_its_row(
    run: dict[str, Any], cache: Path
) -> None:
    """score_test.py's loading on a ConvNeXt (its phase-3 mode is item 13.12's): the row's config
    at the row's run seed matches the row, `load_models` builds and loads the model, and
    `score_checkpoint` makes one prediction pass and one attention pass per test split. The
    test splits here are the synthetic fixture's, cut to `TRIM` images."""
    rows, problems = check_ledger(run["ledger"])
    assert problems == []
    config = score_test._config_of(rows[0], CONFIG_PATH)
    checkpoint = Checkpoint(run["checkpoint"], rows[0]["checkpoint sha256"], rows[0], config)
    assert checkpoint.label == "convnext_tiny arm A seed 0"
    (model,) = load_models([checkpoint])
    assert type(model).__name__ == "ConvNeXt"
    meta = load_metadata(cache / CACHE_ANNOTATIONS, cache / DAY_NIGHT_SIDECAR)
    split = make_split(meta, config.split_seed)
    assert split.hash == rows[0]["split hash"]
    records = {name: split.records[name][:TRIM] for name in score_test.TEST_SPLITS}
    small = replace(
        checkpoint, config=replace(config, batch_size=3, num_workers=0, pin_memory=False)
    )
    scored = score_checkpoint(model, small, records, cache / CACHE_IMAGES, torch.device("cpu"))
    assert [r.split for r in scored.runs] == list(score_test.TEST_SPLITS)
    assert all(len(r.predictions) == TRIM for r in scored.runs)
    assert [r.split for r in scored.attention] == list(score_test.TEST_SPLITS)
    wrong = replace(checkpoint, config=load_config("p3_baseline"))
    with pytest.raises(score_test.Refusal, match="holds a convnext_tiny"):
        load_models([wrong])
