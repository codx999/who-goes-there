"""trailcam/evaluate/figures.py, the Grad-CAM overlays of trailcam/evaluate/gradcam.py and the
evaluate.py entry point (item 8.02; R03, R04, R17, R26, R32, R35).

Figures are written into pytest's temporary directories, never into docs/figures/. Overlay photos
are drawn in the test and placed under a temporary `assets/` folder; the repo's `assets/` is not
created. Random-weight models only; no download.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
import yaml
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor, nn

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES, NUM_CLASSES, index_of_name
from trailcam.data.dataset import EVAL_RESIZE, INPUT_SIZE, EvalTransform
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, Record, load_metadata
from trailcam.evaluate import evaluate, gradcam, slices
from trailcam.evaluate.evaluate import SplitResult, parse_overlay, render_metrics
from trailcam.evaluate.figures import (
    camera_order,
    row_shares,
    write_confusion_matrix,
    write_per_camera,
    write_per_class,
)
from trailcam.evaluate.gradcam import (
    ASSETS_DIR,
    OVERLAY_ALPHA,
    OVERLAY_COLOUR,
    ZeroMapError,
    assets_image,
    blend,
    denormalise,
    write_overlay,
)
from trailcam.evaluate.metrics import (
    CameraScores,
    confusion_matrix,
    per_camera_scores,
    per_class_scores,
)
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH
from trailcam.train.train import CACHE_ANNOTATIONS, CACHE_IMAGES, file_sha256

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"
PNG = b"\x89PNG"


def _figures_snapshot() -> list[Path]:
    return sorted(FIGURES.rglob("*")) if FIGURES.exists() else []


def _photo(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    """A drawn photo: a grey-green field with a light rectangle, saved as a JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, (90, 110, 70))
    image.paste((200, 180, 160), (size[0] // 3, size[1] // 3, size[0] // 2, size[1] // 2))
    image.save(path, quality=95)
    return path


# figures.py


def test_row_shares_shade_each_row_by_its_own_total() -> None:
    matrix = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    matrix[0][0], matrix[0][3] = 3, 1  # class 0: 4 images, 3 right
    matrix[5][2] = 2  # class 5: 2 images, both called class 2
    shares = row_shares(matrix)
    assert shares[0, 0] == 0.75 and shares[0, 3] == 0.25
    assert shares[5, 2] == 1.0
    assert not shares[1].any()  # an absent class has an empty row, not a division by zero
    with pytest.raises(ValueError):
        row_shares([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        row_shares([[-1] * NUM_CLASSES] * NUM_CLASSES)


def test_camera_order_is_worst_accuracy_first_ties_by_location() -> None:
    scores = {
        40: CameraScores(10, 0.9, 0.8),
        7: CameraScores(10, 0.5, 0.4),
        33: CameraScores(10, 0.5, 0.6),
        2: CameraScores(10, 0.7, 0.7),
    }
    assert camera_order(scores) == [7, 33, 2, 40]


def test_the_three_writers_write_pngs(tmp_path: Path) -> None:
    labels = [0, 0, 1, 2, 2, 9]
    predictions = [0, 1, 1, 2, 5, 9]
    locations = [7, 7, 7, 3, 3, 3]
    paths = [
        write_confusion_matrix(confusion_matrix(predictions, labels), tmp_path / "a.png", "t"),
        write_per_class(per_class_scores(predictions, labels), tmp_path / "b.png", "t"),
        write_per_camera(
            per_camera_scores(predictions, labels, locations), 4 / 6, tmp_path / "c.png", "t"
        ),
    ]
    assert paths == [tmp_path / "a.png", tmp_path / "b.png", tmp_path / "c.png"]
    for path in paths:
        assert path.read_bytes().startswith(PNG)
    with pytest.raises(ValueError):
        write_per_class({}, tmp_path / "d.png", "t")
    with pytest.raises(ValueError):
        write_per_camera({}, 0.5, tmp_path / "e.png", "t")
    with pytest.raises(ValueError):
        write_per_camera({1: CameraScores(1, 1.0, 1.0)}, 1.5, tmp_path / "f.png", "t")


# gradcam.py: the overlays


def test_the_default_assets_folder_is_the_repos() -> None:
    assert ASSETS_DIR == ROOT / "assets"


def test_assets_image_accepts_a_photo_under_assets_only(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    inside = _photo(assets / "night" / "a.jpg")
    assert assets_image(inside, assets) == inside.resolve()
    outside = _photo(tmp_path / "data" / "b.jpg")
    refused = [
        outside,  # a dataset path
        assets / ".." / "data" / "b.jpg",  # leaves the folder through ..
        assets,  # the folder itself
    ]
    link = assets / "c.jpg"
    link.symlink_to(outside)  # leaves the folder through a link
    refused.append(link)
    for path in refused:
        with pytest.raises(ValueError) as raised:
            assets_image(path, assets)
        assert "b.jpg" not in str(raised.value) and str(tmp_path) not in str(raised.value)
    with pytest.raises(FileNotFoundError):
        assets_image(assets / "missing.jpg", assets)
    with pytest.raises(ValueError):  # a root not named assets, e.g. data/
        assets_image(outside, tmp_path / "data")


def test_denormalise_gives_back_the_eval_frame(tmp_path: Path) -> None:
    """The frame shown is the eval transform's input crop: Resize(256) and CenterCrop(224) of
    the photo, within one unit of rounding."""
    with Image.open(_photo(tmp_path / "x.jpg", (300, 200))) as opened:
        photo = opened.convert("RGB")
    crop = T.CenterCrop(INPUT_SIZE)(T.Resize(EVAL_RESIZE, antialias=True)(photo))
    frame = denormalise(EvalTransform()(photo))
    assert frame.size == (INPUT_SIZE, INPUT_SIZE)
    difference = np.abs(np.asarray(frame, dtype=int) - np.asarray(crop, dtype=int))
    assert difference.max() <= 1
    with pytest.raises(ValueError):
        denormalise(torch.zeros(1, 3, 4, 4))


def test_blend_by_hand() -> None:
    """Opacity is OVERLAY_ALPHA times the map over its maximum: the peak pixel moves
    OVERLAY_ALPHA of the way to the colour, a half-peak pixel half that, a zero pixel not at
    all."""
    frame = Image.new("RGB", (4, 2), (100, 100, 100))
    cam = torch.tensor([[2.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    out = np.asarray(blend(frame, cam), dtype=float)
    colour = np.asarray(OVERLAY_COLOUR, dtype=float)
    np.testing.assert_allclose(out[0, 0], np.round(100 + OVERLAY_ALPHA * (colour - 100)))
    np.testing.assert_allclose(out[0, 1], np.round(100 + OVERLAY_ALPHA / 2 * (colour - 100)))
    assert (out[0, 2:] == 100).all() and (out[1] == 100).all()
    flat = blend(frame, torch.zeros(2, 4))
    assert np.array_equal(np.asarray(flat), np.asarray(frame))
    with pytest.raises(ValueError):
        blend(frame, -cam)
    with pytest.raises(ValueError):
        blend(frame, torch.zeros(4, 2))


def test_the_overlay_draws_the_scored_gradcam_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R26: the overlay blends the map of `trailcam.evaluate.gradcam.gradcam`, the function
    attention-in-box scores, computed on the photo's eval-transform output, and reports that
    pass's top-1 class."""
    assets = tmp_path / "assets"
    photo = _photo(assets / "a.jpg")
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
    out = tmp_path / "out" / "overlay.png"
    result = write_overlay(model, photo, out, index_of_name("coyote"), assets)
    assert out.read_bytes().startswith(PNG)
    assert len(calls) == 1
    with Image.open(photo) as opened:
        batch = EvalTransform()(opened.convert("RGB")).unsqueeze(0)
    torch.testing.assert_close(calls[0], batch)
    maps, logits = real(model, batch)
    assert result.prediction == int(logits.argmax())
    assert result.confidence == pytest.approx(float(logits.softmax(dim=1).max()))
    frame = denormalise(batch[0])
    assert np.array_equal(np.asarray(result.image), np.asarray(blend(frame, maps[0])))


def test_the_overlay_png_holds_the_blended_image_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 8.02 (the user's decision): the PNG has one panel, the blended image; the
    unblended photo is neither drawn nor kept on the result, so no raw dataset image can reach
    docs/figures/ (.claude/rules/data.md)."""
    assets = tmp_path / "assets"
    photo = _photo(assets / "a.jpg")
    torch.manual_seed(0)
    model = build_model("resnet18", pretrained=False).eval()
    saved: list[Figure] = []
    real_savefig = Figure.savefig

    def keep(figure: Figure, *args: object, **kwargs: object) -> None:
        saved.append(figure)
        real_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", keep)
    out = tmp_path / "out" / "overlay.png"
    result = write_overlay(model, photo, out, assets_root=assets)
    assert out.read_bytes().startswith(PNG)
    assert len(saved) == 1 and len(saved[0].axes) == 1
    images = saved[0].axes[0].get_images()
    assert len(images) == 1
    assert np.array_equal(np.asarray(images[0].get_array()), np.asarray(result.image))
    with Image.open(photo) as opened:
        frame = denormalise(EvalTransform()(opened.convert("RGB")))
    assert not np.array_equal(np.asarray(result.image), np.asarray(frame))  # blended, not raw
    assert not hasattr(result, "frame")


def test_the_overlay_refuses_a_photo_outside_assets(tmp_path: Path) -> None:
    photo = _photo(tmp_path / "data" / "a.jpg")
    model = build_model("small_cnn")
    with pytest.raises(ValueError):
        write_overlay(model, photo, tmp_path / "o.png", assets_root=tmp_path / "assets")
    assert not (tmp_path / "o.png").exists()


def test_a_zero_map_is_refused_and_no_png_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 8.02's fix round: a map that is zero everywhere would leave the photo unblended,
    so the PNG would be the raw photo; it is refused, with a message that names no path."""
    assets = tmp_path / "assets"
    photo = _photo(assets / "a.jpg")
    model = build_model("small_cnn")
    real = gradcam.gradcam

    def zero(
        model: nn.Module, images: Tensor, layer: nn.Module | None = None
    ) -> tuple[Tensor, Tensor]:
        maps, logits = real(model, images, layer)
        return torch.zeros_like(maps), logits

    monkeypatch.setattr(gradcam, "gradcam", zero)
    out = tmp_path / "out" / "overlay.png"
    with pytest.raises(ZeroMapError) as raised:
        write_overlay(model, photo, out, assets_root=assets)
    assert "a.jpg" not in str(raised.value) and str(tmp_path) not in str(raised.value)
    assert not out.exists() and not out.parent.exists()


# evaluate.py


def test_render_metrics_by_hand() -> None:
    """Labels 0, 0, 1, 2, 2 at locations 7, 7, 7, 3, 3, predictions 0, 1, 1, 2, 5: accuracy
    3/5; F1 2/3 for each of classes 0, 1 and 2, so macro-F1 2/3 over the three present; nine
    classes absent. Location 3: accuracy 0.5, location 7: 2/3."""
    result = SplitResult("trans_val", (0, 0, 1, 2, 2), (0, 1, 1, 2, 5), (7, 7, 7, 3, 3))
    text = render_metrics([result], ["Model x."])
    assert "Model x." in text
    assert "| trans_val | trans | 5 | 3 | 0.6000 | 0.6667 |" in text
    absent = ", ".join(CLASSES[c] for c in range(3, NUM_CLASSES))
    assert f"left out of macro-F1: {absent}." in text
    assert "| opossum | 2 | 1 | 1.0000 | 0.5000 | 0.6667 |" in text
    assert "| raccoon |" not in text.split("### Confusion matrix")[0]  # absent: no per-class row
    assert "| 3 | 2 | 0.5000 | 0.6667 |" in text
    assert "| 7 | 3 | 0.6667 | 0.6667 |" in text
    assert "| coyote | 0 | 0 | 1 | 0 | 0 | 1 |" in text  # one right, one called raccoon


def test_split_results_are_checked() -> None:
    with pytest.raises(ValueError):
        SplitResult("train", (0,), (0,), (1,))
    with pytest.raises(ValueError):
        SplitResult("val", (), (), ())
    with pytest.raises(ValueError):
        SplitResult("val", (0, 1), (0,), (1, 1))


def test_predict_refuses_records_of_another_split(tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    fixture = synthetic.write_synthetic_fixture(root)
    meta = load_metadata(root, fixture.paths["sidecar"])
    model = build_model("small_cnn")
    with pytest.raises(ValueError):
        evaluate.predict(
            model,
            meta.records["trans_val"],
            "val",
            fixture.image_root,
            torch.device("cpu"),
            batch_size=4,
        )


def test_parse_overlay() -> None:
    assert parse_overlay("assets/a.jpg") == (Path("assets/a.jpg"), None)
    assert parse_overlay("assets/a.jpg=coyote") == (Path("assets/a.jpg"), index_of_name("coyote"))
    with pytest.raises(Exception, match="unknown class"):
        parse_overlay("assets/a.jpg=coyot")


def test_the_command_line_on_the_fixture_writes_a_table_and_three_pngs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The item's step: `--config dryrun --split val --out <tmp>`."""
    before = _figures_snapshot()
    out = tmp_path / "out"
    assert evaluate.main(["--config", "dryrun", "--split", "val", "--out", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == [
        "metrics.md",
        "val_confusion_matrix.png",
        "val_per_camera.png",
        "val_per_class.png",
    ]
    for png in out.glob("*.png"):
        assert png.read_bytes().startswith(PNG)
    table = (out / "metrics.md").read_text()
    assert "| val | cis | 72 |" in table
    for heading in ("### Per class", "### Per camera", "### Confusion matrix"):
        assert heading in table
    assert "cis_test" not in table and "trans_test" not in table and "trans_val" not in table
    assert _figures_snapshot() == before  # nothing written under docs/figures/

    # R03: no image id, sequence id or file name in the table or on stdout
    printed = capsys.readouterr().out
    written = synthetic.write_synthetic_fixture(tmp_path / "ids")
    meta = load_metadata(tmp_path / "ids", written.paths["sidecar"])
    names = {
        v for rs in meta.records.values() for r in rs for v in (r.image_id, r.file_name, r.seq_id)
    }
    assert names and not any(name in table or name in printed for name in names)


@pytest.mark.parametrize("split", ["cis_test", "trans_test", "train"])
def test_the_command_line_refuses_every_split_but_val_and_trans_val(
    split: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as raised:
        evaluate.main(["--config", "dryrun", "--split", split, "--out", str(tmp_path)])
    assert raised.value.code == 2
    assert not any(tmp_path.iterdir())


def test_the_command_line_reads_only_the_splits_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, set[str]]] = []

    def spy(
        model: object, records: tuple[Record, ...], split: str, *args: object, **kwargs: object
    ) -> SplitResult:
        seen.append((split, {r.split for r in records}))
        return SplitResult(split, (0,), (0,), (1,))

    monkeypatch.setattr(evaluate, "predict", spy)
    argv = ["--config", "dryrun", "--split", "trans_val", "--out", str(tmp_path / "a")]
    assert evaluate.main(argv) == 0
    assert seen == [("trans_val", {"trans_val"})]
    seen.clear()
    assert evaluate.main(["--config-name", "dryrun", "--out", str(tmp_path / "b")]) == 0
    assert seen == [("val", {"val"})]


def test_a_real_run_needs_a_checkpoint_and_the_cache(tmp_path: Path) -> None:
    for argv in (
        ["--config", "resnet18_arm_b"],
        ["--config", "resnet18_arm_b", "--checkpoint", "x.pt"],
        ["--config", "dryrun", "--checkpoint", "x.pt"],
        ["--config", "dryrun", "--data-root", str(tmp_path)],
        ["--config", "no_such_config"],
    ):
        with pytest.raises(SystemExit) as raised:
            evaluate.main([*argv, "--out", str(tmp_path / "out")])
        assert raised.value.code == 2
    assert not (tmp_path / "out").exists()


def test_the_command_line_writes_overlays_from_assets_only(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    first = _photo(assets / "a.jpg")
    second = _photo(assets / "b.jpg", (240, 320))
    outside = _photo(tmp_path / "data" / "c.jpg")
    base = ["--config", "dryrun", "--assets", str(assets)]

    with pytest.raises(SystemExit) as raised:  # refused before anything is evaluated
        evaluate.main(
            [
                *base,
                "--overlay",
                str(first),
                "--overlay",
                str(outside),
                "--out",
                str(tmp_path / "refused"),
            ]
        )
    assert raised.value.code == 2
    assert not (tmp_path / "refused").exists()

    out = tmp_path / "out"
    argv = [*base, "--overlay", f"{first}=coyote", "--overlay", str(second), "--out", str(out)]
    assert evaluate.main(argv) == 0
    overlays = sorted(p.name for p in out.glob("gradcam_*"))
    assert overlays == ["gradcam_overlay_01.png", "gradcam_overlay_02.png"]
    for name in overlays:
        assert (out / name).read_bytes().startswith(PNG)


def test_a_real_run_on_a_cache_shaped_fixture(tmp_path: Path) -> None:
    """A checkpoint saved as train.py saves it, a cache laid out as item 6.01 writes it, and a
    non-fixture config (a copy of config.yaml with one small_cnn config, no workers): the run
    scores val and trans_val and the header names the checkpoint's sha256."""
    built = tmp_path / "built"
    fixture = synthetic.write_synthetic_fixture(built)
    cache = tmp_path / "cache"
    cache.mkdir()
    fixture.image_root.rename(cache / CACHE_IMAGES)
    Path(fixture.paths["sidecar"]).rename(cache / DAY_NIGHT_SIDECAR)
    built.rename(cache / CACHE_ANNOTATIONS)

    settings = yaml.safe_load(CONFIG_PATH.read_text())
    settings["configs"] = {
        "check": {
            **settings["configs"]["small_cnn"],
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
        }
    }
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(settings))

    torch.manual_seed(0)
    model = build_model("small_cnn")
    checkpoint = tmp_path / "run.pt"
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model": "small_cnn", "classes": list(CLASSES), "state_dict": state}, checkpoint)

    out = tmp_path / "out"
    argv = [
        "--config",
        "check",
        "--config-file",
        str(config_file),
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
    assert evaluate.main(argv) == 0
    table = (out / "metrics.md").read_text()
    assert f"checkpoint sha256 {file_sha256(checkpoint)}" in table
    assert "| val | cis |" in table and "| trans_val | trans |" in table
    assert len(list(out.glob("*.png"))) == 6

    only_val = tmp_path / "only_val"  # R17: the splits given and no other
    assert evaluate.main([*argv[:-6], "--split", "val", "--out", str(only_val)]) == 0
    assert "trans_val" not in (only_val / "metrics.md").read_text()
    assert len(list(only_val.glob("*.png"))) == 3

    torch.save({"model": "resnet18", "classes": list(CLASSES), "state_dict": state}, checkpoint)
    with pytest.raises(ValueError):  # a checkpoint of another model than the config's
        evaluate.main([*argv[:-1], str(tmp_path / "other")])


def test_slices_predictions_score_here_without_a_second_pass() -> None:
    """The hook in evaluate.py's docstring: 8.04's `SplitPredictions` carries what `SplitResult`
    needs, so one pass serves the slice tables and the metrics table."""
    run = slices.SplitPredictions(
        split="val",
        labels=(0, 0, 1),
        predictions=(0, 1, 1),
        night=(True, False, False),
        locations=(7, 7, 3),
    )
    result = SplitResult(run.split, run.labels, run.predictions, run.locations)
    assert "| val | cis | 3 | 2 | 0.6667 | 0.6667 |" in render_metrics([result])
