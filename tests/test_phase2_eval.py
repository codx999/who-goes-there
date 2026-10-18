"""Item 13.05: the evaluation tools at a phase-2 config's input size (docs/phase2-plan.md
sections 3 and 4; R17, R20, R26).

`evaluate.py`, `attention.py` and `slices.py` score a checkpoint at its config's `input_size`:
224 for every phase-1 config, 320 for `p2_res320`. Run end to end on a cache-shaped synthetic
fixture with random-weight checkpoints saved as `train.py` saves them: a 320 px checkpoint is
scored at 320 by all three (the metrics, the Grad-CAM overlay, attention-in-box and the slices),
a phase-1 checkpoint at 224 with the headers and text of before, and a checkpoint whose size is
not its config's is refused by each. The box geometry of attention-in-box follows the 320 px eval
transform, and the final scoring and the demo keep their 224 px defaults.
"""

import inspect
import re
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from torch import Tensor, nn

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.data.dataset import INPUT_SIZE, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, load_metadata
from trailcam.data.split import make_split
from trailcam.demo import predict as demo_predict
from trailcam.evaluate import attention, evaluate, gradcam, slices
from trailcam.evaluate.attention import (
    AttentionDataset,
    SplitRun,
    attention_pass,
    blank_region,
    box_in_frame,
    box_mask,
    eval_geometry,
    render_report,
)
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH, Config, load_config
from trailcam.train.train import CACHE_ANNOTATIONS, CACHE_IMAGES, file_sha256

SIZES = [(512, 384), (384, 512), (512, 512), (96, 72), (500, 281), (281, 500), (300, 299)]
CONFIGS = ("small_cnn", "resnet18_arm_a", "resnet18_arm_b", "p2_baseline", "p2_res320")
HEADER = r"Model resnet18, config {name}{size}, checkpoint sha256 [0-9a-f]{{64}}, split seed 42, "
HEADER += r"split hash [0-9a-f]{{64}}\."


@pytest.fixture(scope="module")
def cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The synthetic fixture laid out as the 512 px cache: annotations/, images/ and the
    sidecar."""
    tmp = tmp_path_factory.mktemp("phase2-eval")
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
    """config.yaml with the configs used here small enough for the CPU: batch 8, no workers, no
    pinned memory. Nothing a checkpoint is checked against changes."""
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in CONFIGS:
        settings["configs"][name].update(batch_size=8, num_workers=0, pin_memory=False)
    path = tmp_path_factory.mktemp("config") / "config.yaml"
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def assets(cache: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """An assets/ folder holding one photo of the fixture, for the Grad-CAM overlay."""
    folder = tmp_path_factory.mktemp("overlay") / "assets"
    folder.mkdir()
    first = sorted((cache / CACHE_IMAGES).iterdir())[0]
    shutil.copy(first, folder / "photo.jpg")
    return folder


def _checkpoint(path: Path, size: int | None, seed: int = 0) -> Path:
    """A ResNet-18 checkpoint as `train.py` saves it: with its input size since item 13.02, or
    without one (`size=None`), phase 1's format."""
    torch.manual_seed(seed)
    state = build_model("resnet18", pretrained=False).state_dict()
    saved: dict[str, object] = {"model": "resnet18", "classes": list(CLASSES), "state_dict": state}
    if size is not None:
        saved["input_size"] = size
    torch.save(saved, path)
    return path


@pytest.fixture
def seen(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[int, int]]]:
    """The height and width of every batch a model built by `load_checkpoint` is given, the
    Grad-CAM passes included: the one loader of the three tools builds its model through
    `attention.build_model`."""
    sides: list[tuple[int, int]] = []
    real = attention.build_model

    def record(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
        sides.append((int(inputs[0].shape[-2]), int(inputs[0].shape[-1])))

    def spy(name: str, **kwargs: bool) -> nn.Module:
        model = real(name, **kwargs)
        model.register_forward_pre_hook(record)
        return model

    monkeypatch.setattr(attention, "build_model", spy)
    yield sides


def _data(cache: Path, checkpoint: Path, out: Path) -> list[str]:
    """A real run's arguments: the checkpoint, the cache, val and trans_val, `--out`."""
    return [
        "--checkpoint",
        str(checkpoint),
        "--data-root",
        str(cache),
        "--split",
        "val",
        "--split",
        "trans_val",
        "--out",
        str(out),
    ]


def _evaluate(cache: Path, config_file: Path, name: str, checkpoint: Path, out: Path) -> int:
    argv = ["--config", name, "--config-file", str(config_file)]
    return evaluate.main(argv + _data(cache, checkpoint, out))


def _attention(cache: Path, config_file: Path, name: str, checkpoint: Path, out: Path) -> int:
    argv = ["--config-name", name, "--config", str(config_file)]
    return attention.main(argv + _data(cache, checkpoint, out))


def _slices(config_file: Path, cache: Path, pairs: list[tuple[str, Path]], out: Path) -> int:
    argv = [arg for name, path in pairs for arg in ("--checkpoint", name, str(path))]
    argv += ["--config", str(config_file), "--data-root", str(cache), "--out", str(out)]
    return slices.main(argv)


# End to end on the cache-shaped fixture


def test_a_320_checkpoint_is_evaluated_at_320_end_to_end(
    cache: Path,
    config_file: Path,
    assets: Path,
    tmp_path: Path,
    seen: list[tuple[int, int]],
) -> None:
    """`p2_res320`'s checkpoint, saved at 320 px: every batch of the metrics pass, the Grad-CAM
    overlay, both attention passes and the slices is 320 x 320, and each table says so."""
    res320 = _checkpoint(tmp_path / "res320.pt", 320)
    baseline = _checkpoint(tmp_path / "baseline.pt", 224, seed=1)

    out = tmp_path / "evaluate"
    argv = ["--config", "p2_res320", "--config-file", str(config_file)]
    argv += _data(cache, res320, out)
    argv += ["--overlay", str(assets / "photo.jpg"), "--assets", str(assets)]
    assert evaluate.main(argv) == 0
    assert seen and set(seen) == {(320, 320)}
    table = (out / evaluate.METRICS_NAME).read_text(encoding="utf-8")
    header = HEADER.format(name="p2_res320", size=", input size 320 px")
    assert re.search(f"^{header}$", table, flags=re.MULTILINE)
    assert f"checkpoint sha256 {file_sha256(res320)}" in table
    assert "| val | cis |" in table and "| trans_val | trans |" in table
    assert len(list(out.glob("*.png"))) == 7  # three per split and the overlay

    seen.clear()
    out = tmp_path / "attention"
    assert _attention(cache, config_file, "p2_res320", res320, out) == 0
    assert seen and set(seen) == {(320, 320)}  # the unblanked, Grad-CAM and blanked passes
    text = (out / attention.TABLE_NAME).read_text(encoding="utf-8")
    assert "normalised over the whole 320 x 320 input," in text
    assert re.search(f"^{header}$", text, flags=re.MULTILINE)
    assert (out / attention.HISTOGRAM_NAME).exists()

    seen.clear()
    out = tmp_path / "slices"
    assert _slices(config_file, cache, [("p2_baseline", baseline), ("p2_res320", res320)], out) == 0
    passes = len(seen) // 2
    assert seen[:passes] == [(224, 224)] * passes and seen[passes:] == [(320, 320)] * passes
    table = (out / slices.TABLE_NAME).read_text(encoding="utf-8")
    assert "## p2_baseline" in table and "## p2_res320" in table
    assert f"- p2_baseline: checkpoint sha256 {file_sha256(baseline)}" in table
    assert f"- p2_res320, input size 320 px: checkpoint sha256 {file_sha256(res320)}" in table
    assert (out / slices.SPECIES_FIGURE).exists() and (out / slices.CAMERA_FIGURE).exists()
    for written in tmp_path.rglob("*.md"):  # R17: val and trans_val only
        assert "cis_test" not in written.read_text() and "trans_test" not in written.read_text()


def test_a_phase1_checkpoint_is_scored_at_224_as_before(
    cache: Path, config_file: Path, tmp_path: Path, seen: list[tuple[int, int]]
) -> None:
    """Phase 1's format (no input size) under `resnet18_arm_a`: 224 x 224 batches everywhere,
    and the headers and text of the tables as they were before item 13.05."""
    arm_a = _checkpoint(tmp_path / "arm_a.pt", None)
    arm_b = _checkpoint(tmp_path / "arm_b.pt", None, seed=1)
    header = HEADER.format(name="resnet18_arm_a", size="")

    out = tmp_path / "evaluate"
    assert _evaluate(cache, config_file, "resnet18_arm_a", arm_a, out) == 0
    table = (out / evaluate.METRICS_NAME).read_text(encoding="utf-8")
    assert re.search(f"^{header}$", table, flags=re.MULTILINE) and "input size" not in table
    out = tmp_path / "attention"
    assert _attention(cache, config_file, "resnet18_arm_a", arm_a, out) == 0
    text = (out / attention.TABLE_NAME).read_text(encoding="utf-8")
    assert re.search(f"^{header}$", text, flags=re.MULTILINE) and "input size" not in text
    assert "normalised over the whole 224 x 224 input," in text
    out = tmp_path / "slices"
    pairs = [("resnet18_arm_a", arm_a), ("resnet18_arm_b", arm_b)]
    assert _slices(config_file, cache, pairs, out) == 0
    table = (out / slices.TABLE_NAME).read_text(encoding="utf-8")
    assert "## resnet18 arm A" in table and "## resnet18 arm B" in table
    assert f"- resnet18 arm A: checkpoint sha256 {file_sha256(arm_a)}" in table
    assert "input size" not in table
    assert seen and set(seen) == {(224, 224)}


def test_the_final_scoring_the_demo_and_the_space_keep_224() -> None:
    """Every size keyword item 13.05 added defaults to 224, so the callers that pass none (the
    final scoring, `score_test.py`; the demo and the Space through `predict.load_model` and
    `overlay_photo`) score and serve at 224 px as before, and `load_checkpoint` still refuses a
    320 px config for them."""
    for function, keyword in (
        (evaluate.predict, "size"),
        (evaluate.write_overlays, "size"),
        (attention.attention_pass, "size"),
        (attention.eval_geometry, "size"),
        (attention.box_in_frame, "size"),
        (attention.box_mask, "size"),
        (AttentionDataset, "size"),
        (slices.predict_split, "size"),
        (gradcam.overlay, "size"),
        (gradcam.overlay_photo, "size"),
        (gradcam.write_overlay, "size"),
        (attention.load_checkpoint, "eval_size"),
    ):
        assert inspect.signature(function).parameters[keyword].default == INPUT_SIZE == 224
    assert SplitRun("val", 0, 0, ()).input_size == 224
    source = inspect.getsource(demo_predict.load_model)
    assert "eval_size" not in source and "load_checkpoint(" in source


# Refused: a checkpoint whose size is not its config's, and a phase-1 and phase-2 mix


@pytest.mark.parametrize("tool", ["evaluate", "attention", "slices"])
@pytest.mark.parametrize(
    ("name", "size", "message"),
    [
        ("p2_baseline", 320, "trained at 320 px, but the config's input size is 224 px"),
        ("resnet18_arm_a", 320, "trained at 320 px, but the config's input size is 224 px"),
        ("p2_res320", None, "trained at 224 px, but the config's input size is 320 px"),
        ("p2_res320", 224, "trained at 224 px, but the config's input size is 320 px"),
    ],
    ids=["320-under-p2_baseline", "320-under-arm-a", "phase1-under-res320", "224-under-res320"],
)
def test_a_checkpoint_of_another_size_than_its_config_is_refused_by_each_tool(
    cache: Path,
    config_file: Path,
    tmp_path: Path,
    tool: str,
    name: str,
    size: int | None,
    message: str,
) -> None:
    """A mismatched pair is refused before any image is read and nothing is written: a model is
    never scored at a size it was not trained at."""
    checkpoint = _checkpoint(tmp_path / "run.pt", size)
    out = tmp_path / "out"
    runs: dict[str, Callable[[], int]] = {
        "evaluate": lambda: _evaluate(cache, config_file, name, checkpoint, out),
        "attention": lambda: _attention(cache, config_file, name, checkpoint, out),
        "slices": lambda: _slices(config_file, cache, [(name, checkpoint)], out),
    }
    with pytest.raises(ValueError, match=re.escape(message)):
        runs[tool]()
    assert not out.exists()


@pytest.mark.parametrize(
    ("name", "size", "message"),
    [
        ("p2_baseline", 320, "trained at 320 px, but the config's input size is 224 px"),
        ("resnet18_arm_a", 320, "trained at 320 px, but the config's input size is 224 px"),
        ("p2_res320", None, "trained at 224 px, but the config's input size is 320 px"),
        ("p2_res320", 224, "trained at 224 px, but the config's input size is 320 px"),
    ],
    ids=["320-under-p2_baseline", "320-under-arm-a", "phase1-under-res320", "224-under-res320"],
)
def test_slices_refuses_a_mismatched_second_checkpoint_before_scoring_the_first(
    cache: Path,
    config_file: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    size: int | None,
    message: str,
) -> None:
    """Review 13.05: a good first checkpoint of the pair (`p2_head3` at 224, or phase 1's arm B
    beside a phase-1 config) and a mismatched second. Every checkpoint is checked before any
    pass, so the first is never scored, no metadata is read and nothing is written."""
    phase1 = not name.startswith("p2_")
    if phase1:
        first = ("resnet18_arm_b", _checkpoint(tmp_path / "first.pt", None))
    else:
        first = ("p2_head3", _checkpoint(tmp_path / "first.pt", 224))
    second = (name, _checkpoint(tmp_path / "second.pt", size, seed=1))
    touched: list[str] = []

    def refuse(*args: object, **kwargs: object) -> None:
        touched.append("read")
        raise AssertionError("an image or the metadata was read before every checkpoint passed")

    monkeypatch.setattr(slices, "predict_split", refuse)
    monkeypatch.setattr(slices, "load_metadata", refuse)
    out = tmp_path / "out"
    with pytest.raises(ValueError, match=re.escape(message)):
        _slices(config_file, cache, [first, second], out)
    assert touched == [] and not out.exists()


def test_slices_refuses_a_phase1_and_a_phase2_checkpoint_together(
    config_file: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Phase 2 compares each variant with `p2_baseline`, never with a phase-1 run; and two
    checkpoints of one phase-2 config are one label twice."""
    a, b = _checkpoint(tmp_path / "a.pt", None), _checkpoint(tmp_path / "b.pt", 224)
    for pairs, message in (
        ([("resnet18_arm_a", a), ("p2_head3", b)], "phase-1 and phase-2 checkpoints in one run"),
        ([("p2_baseline", b), ("p2_baseline", b)], "per config"),
    ):
        with pytest.raises(SystemExit) as raised:
            _slices(config_file, tmp_path, pairs, tmp_path / "out")
        assert raised.value.code == 2
        assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_phase2_checkpoints_are_labelled_by_config_and_phase1_ones_by_model_and_arm() -> None:
    """`p2_baseline`, `p2_head3` and `p2_res320` are all ResNet-18 arm A, so a phase-2 label is
    the config name; a phase-1 label is the model and arm, as the final scoring's."""
    for name in ("p2_baseline", "p2_head3", "p2_res320", "p2_arm_c", "p2_r50"):
        config: Config = load_config(name)
        assert slices.model_label(name, config) == name
    for name, label in (
        ("small_cnn", "small_cnn"),
        ("resnet18_arm_a", "resnet18 arm A"),
        ("resnet18_arm_b", "resnet18 arm B"),
    ):
        config = load_config(name)
        assert slices.model_label(name, config) == slices.checkpoint_label(config) == label


# Attention-in-box at 320: the box in the 320 px eval frame


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_geometry_is_the_320_eval_transform_s(width: int, height: int) -> None:
    resize, crop = EvalTransform(320).pipeline.transforms[:2]
    image = Image.new("RGB", (width, height))
    resized = resize(image)
    new_width, new_height, left, top = eval_geometry(width, height, 320)
    assert resized.size == (new_width, new_height) and min(resized.size) == 366
    assert isinstance(crop, T.CenterCrop) and crop.size == (320, 320)
    marked = Image.new("L", resized.size)
    marked.putpixel((left, top), 255)
    assert np.asarray(crop(marked))[0, 0] == 255


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_a_mapped_box_lands_where_the_320_eval_transform_puts_its_pixels(
    width: int, height: int
) -> None:
    """As at 224 (`tests/test_attention.py`): a white box through the 320 px resize and crop
    lights the pixels of the box mapped at 320, give or take the resize's blur."""
    box = (0.35, 0.3, 0.3, 0.4)
    image = Image.new("L", (width, height))
    image.paste(255, blank_region(box, width, height))
    resize, crop = EvalTransform(320).pipeline.transforms[:2]
    seen = np.asarray(crop(resize(image))) > 127
    rows, columns = np.nonzero(seen)
    framed = box_in_frame(box, width, height, 320)
    assert framed is not None
    x0, y0, x1, y1 = framed
    tolerance = 1.5 + 366 / min(width, height)  # rounding the box outwards to whole pixels
    assert abs(columns.min() - x0) <= tolerance and abs(columns.max() + 1 - x1) <= tolerance
    assert abs(rows.min() - y0) <= tolerance and abs(rows.max() + 1 - y1) <= tolerance


def test_the_320_mask_covers_the_320_frame() -> None:
    """512 x 384 resizes to 488 x 366 and crops columns 84 to 404 (rows 23 to 343) at 320: a box
    over the whole image covers all 320 x 320 pixels, and one cut by the crop is clipped to it."""
    assert eval_geometry(512, 384, 320) == (488, 366, 84, 23)
    edge = box_mask([(0.0, 0.0, 1.0, 1.0)], 512, 384, 320)
    assert tuple(edge.shape) == (320, 320) and float(edge.sum()) == pytest.approx(320**2)
    clipped = box_in_frame((0.1, 0.5, 0.2, 0.2), 512, 384, 320)  # columns 48.8 to 146.4
    assert clipped == pytest.approx((0.0, 183 - 23, 146.4 - 84, 256.2 - 23))
    assert box_in_frame((0.0, 0.5, 0.1, 0.2), 512, 384, 320) is None  # columns 0 to 48.8


def test_the_attention_pass_at_320_scores_each_box_on_the_320_frame(cache: Path) -> None:
    """Images, blanked images and masks all at 320; each image's box share is its boxes' share
    of the 320 x 320 frame, computed here from the records."""
    meta = load_metadata(cache / CACHE_ANNOTATIONS, cache / DAY_NIGHT_SIDECAR)
    records = make_split(meta, 42).records["val"]
    boxed = [r for r in records if r.boxes]
    dataset = AttentionDataset(boxed, cache / CACHE_IMAGES, 320)
    image, blanked, mask, _ = dataset[0]
    assert tuple(image.shape) == tuple(blanked.shape) == (3, 320, 320)
    assert tuple(mask.shape) == (320, 320)

    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False)
    run = attention_pass(
        model, records, "val", cache / CACHE_IMAGES, torch.device("cpu"), batch_size=8, size=320
    )
    assert run.input_size == 320 and len(run.results) == len(boxed)
    for record, result in zip(boxed, run.results, strict=True):
        with Image.open(cache / CACHE_IMAGES / record.file_name) as opened:
            share = float(box_mask(record.boxes, *opened.size, 320).sum()) / 320**2
        assert result.box_share == pytest.approx(share)
        assert result.box_visible == (share > 0)


def test_a_report_of_splits_at_two_sizes_is_refused() -> None:
    with pytest.raises(ValueError, match="different input sizes"):
        render_report([SplitRun("val", 0, 0, (), 224), SplitRun("trans_val", 0, 0, (), 320)])


def test_the_overlay_at_320_is_the_320_eval_frame(
    assets: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grad-CAM reads the 320 px eval transform's output, and the overlay covers that frame."""
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    calls: list[Tensor] = []
    real = gradcam.gradcam

    def spy(
        model: nn.Module, images: Tensor, layer: nn.Module | None = None
    ) -> tuple[Tensor, Tensor]:
        calls.append(images.detach().clone())
        return real(model, images, layer)

    monkeypatch.setattr(gradcam, "gradcam", spy)
    photo = assets / "photo.jpg"
    result = gradcam.write_overlay(model, photo, tmp_path / "o.png", assets_root=assets, size=320)
    with Image.open(photo) as opened:
        batch = EvalTransform(320)(opened.convert("RGB")).unsqueeze(0)
    (given,) = calls
    torch.testing.assert_close(given, batch)
    assert result.image.size == (320, 320)
