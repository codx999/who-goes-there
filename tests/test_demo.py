"""The demo's prediction and its Gradio app (item 10.01; R04, R33, R35).

The model is the small CNN with random weights, saved as `train.py` saves a checkpoint, and the
photo is a synthetic-fixture JPEG: no trained run and no dataset image is needed.
"""

import inspect
import os
import subprocess
import sys
import time
from pathlib import Path

import gradio as gr
import gradio.analytics as gradio_analytics
import pytest
import torch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor, nn

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.data.dataset import INPUT_SIZE, EvalTransform
from trailcam.demo import app, predict
from trailcam.demo.app import build_app, build_parser, classify, example_photos, main
from trailcam.demo.predict import TOP_K, Prediction, load_model
from trailcam.evaluate import gradcam
from trailcam.evaluate.gradcam import ASSETS_DIR, ZeroMapError, overlay_photo
from trailcam.model.factory import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def photo(tmp_path_factory: pytest.TempPathFactory) -> Image.Image:
    root = synthetic.write_synthetic_fixture(tmp_path_factory.mktemp("fixture")).image_root
    with Image.open(sorted(root.rglob("*.jpg"))[0]) as image:
        return image.convert("RGB")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("run") / "small_cnn.pt"
    state = build_model("small_cnn").state_dict()
    torch.save({"model": "small_cnn", "classes": list(CLASSES), "state_dict": state}, path)
    return path


@pytest.fixture(scope="module")
def model(checkpoint: Path) -> nn.Module:
    return load_model(checkpoint, "small_cnn")


def _count_forwards(model: nn.Module) -> list[int]:
    calls: list[int] = []
    model.register_forward_hook(lambda *_: calls.append(1))
    return calls


def test_predict_on_a_fixture_image_returns_three_pairs_and_seconds(
    model: nn.Module, photo: Image.Image
) -> None:
    result = predict.predict(model, photo)
    assert isinstance(result, Prediction)
    assert len(result.top) == TOP_K == 3
    labels = [label for label, _ in result.top]
    probabilities = [probability for _, probability in result.top]
    assert len(set(labels)) == 3 and set(labels) <= set(CLASSES)
    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-6
    assert isinstance(result.seconds, float) and result.seconds > 0


def test_a_request_costs_a_warm_up_and_one_timed_pass(checkpoint: Path, photo: Image.Image) -> None:
    """The prediction is decoded from the timed pass's logits: two forwards, not three."""
    model = load_model(checkpoint, "small_cnn")
    calls = _count_forwards(model)
    result = predict.predict(model, photo)
    assert len(calls) == 2
    with torch.no_grad():
        expected = model(EvalTransform()(photo).unsqueeze(0))[0].softmax(dim=0)
    values, indices = expected.topk(3)
    assert [label for label, _ in result.top] == [CLASSES[int(i)] for i in indices]
    assert [p for _, p in result.top] == pytest.approx(values.tolist())


def test_the_transform_runs_before_the_clock_starts(
    checkpoint: Path, photo: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R33: the eval transform (and the decoding before it) are outside the timed region."""
    events: list[str] = []
    readings = iter([1.0, 1.125])

    class _Logged(EvalTransform):
        def __call__(self, image: Image.Image) -> Tensor:
            events.append("transform")
            return super().__call__(image)

    def clock() -> float:
        events.append("clock")
        return next(readings)

    model = load_model(checkpoint, "small_cnn")
    model.register_forward_hook(lambda *_: events.append("forward"))
    monkeypatch.setattr(predict, "EvalTransform", _Logged)
    monkeypatch.setattr(time, "perf_counter", clock)
    result = predict.predict(model, photo)
    assert events == ["transform", "forward", "clock", "forward", "clock"]
    assert result.seconds == pytest.approx(0.125)


def test_load_model_refuses_a_checkpoint_of_another_model(checkpoint: Path) -> None:
    with pytest.raises(ValueError, match="small_cnn"):
        load_model(checkpoint, "resnet18_arm_b")


def test_the_in_memory_overlay_writes_nothing(
    model: nn.Module, photo: Image.Image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An upload is not under assets/: its overlay is returned in memory, never saved."""

    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("the demo's overlay must not write a file")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(Image.Image, "save", refuse)
    monkeypatch.setattr(Figure, "savefig", refuse)
    result = overlay_photo(model, photo.convert("L"))  # a night photo is grayscale
    assert result.image.size == (INPUT_SIZE, INPUT_SIZE)
    assert 0 <= result.prediction < len(CLASSES) and 0.0 < result.confidence <= 1.0
    assert os.listdir(tmp_path) == []


def test_the_path_overlay_keeps_its_assets_containment(model: nn.Module, tmp_path: Path) -> None:
    photo = tmp_path / "photo.jpg"
    Image.new("RGB", (64, 48)).save(photo)
    with pytest.raises(ValueError, match="assets"):
        gradcam.overlay(model, photo)


def test_example_photos_come_from_assets_only(tmp_path: Path) -> None:
    """R35: a missing or empty folder gives no examples; photos under it are listed, other files
    skipped; a link out of the folder is refused."""
    assert example_photos(tmp_path / "assets") == []
    assets = tmp_path / "assets"
    assets.mkdir()
    assert example_photos(assets) == []
    for name in ("b.PNG", "a.jpg"):
        Image.new("RGB", (8, 8)).save(assets / name, format="PNG" if name.endswith("PNG") else None)
    (assets / "notes.txt").write_text("not a photo")
    assert example_photos(assets) == [(assets / "a.jpg").resolve(), (assets / "b.PNG").resolve()]
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (8, 8)).save(outside)
    (assets / "link.jpg").symlink_to(outside)
    with pytest.raises(ValueError, match="assets"):
        example_photos(assets)


def test_the_default_examples_are_the_repos_assets() -> None:
    """R35: both defaults name the repo's assets/, which may still be empty (item 8.05)."""
    default = inspect.signature(example_photos).parameters["assets_root"].default
    assert default == ASSETS_DIR == ROOT / "assets"
    assert build_parser().parse_args([]).assets == ROOT / "assets"
    examples = example_photos()
    assert all(path.is_relative_to((ROOT / "assets").resolve()) for path in examples)


def test_the_model_receives_the_eval_transform_of_the_photo_alone(
    checkpoint: Path, photo: Image.Image
) -> None:
    """R20, R21, R23: every forward of a request, the prediction's and Grad-CAM's, takes one
    argument, the eval transform's output for the photo (shorter side 256, centre crop 224,
    ImageNet normalisation); no box or other input."""
    model = load_model(checkpoint, "small_cnn")
    inputs: list[tuple[Tensor, ...]] = []
    model.register_forward_hook(lambda _, args, __: inputs.append(args))
    classify(model, photo, show_gradcam=True)
    expected = EvalTransform()(photo).unsqueeze(0)
    assert len(inputs) == 3  # warm-up, timed pass, Grad-CAM
    for args in inputs:
        assert len(args) == 1
        assert torch.equal(args[0].detach(), expected)
    assert list(inspect.signature(classify).parameters) == ["model", "photo", "show_gradcam"]
    assert list(inspect.signature(predict.predict).parameters) == ["model", "photo"]


def test_the_ui_calls_the_probabilities_confidence_never_accuracy() -> None:
    """R34: the shown probabilities are the model's own estimate."""
    source = Path(app.__file__).read_text().lower()
    assert "accuracy" not in source
    assert "model confidence" in source


def test_the_app_builds_with_and_without_examples(model: nn.Module, tmp_path: Path) -> None:
    assert isinstance(build_app(model, []), gr.Blocks)
    assets = tmp_path / "assets"
    assets.mkdir()
    Image.new("RGB", (8, 8)).save(assets / "a.jpg")
    assert isinstance(build_app(model, example_photos(assets)), gr.Blocks)


def test_the_app_is_built_with_analytics_off(
    model: nn.Module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No build calls api.gradio.app: analytics off, so the version check never starts."""
    calls: list[str] = []
    monkeypatch.setattr(gradio_analytics, "version_check", lambda: calls.append("version"))
    built = build_app(model, [])
    assert built.analytics_enabled is False
    assert calls == []


def test_classify_returns_the_top_three_the_seconds_and_the_overlay_on_request(
    model: nn.Module, photo: Image.Image
) -> None:
    scores, seconds, heatmap = classify(model, photo, show_gradcam=False)
    assert len(scores) == 3 and set(scores) <= set(CLASSES)
    assert seconds.endswith(" s (forward pass only)")
    assert heatmap is None
    _, _, heatmap = classify(model, photo, show_gradcam=True)
    assert heatmap is not None and heatmap.size == (INPUT_SIZE, INPUT_SIZE)


def test_classify_refuses_no_photo_and_survives_a_zero_map(
    model: nn.Module, photo: Image.Image, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(gr.Error):
        classify(model, None, show_gradcam=False)

    def zero(*_: object) -> None:
        raise ZeroMapError("zero")

    monkeypatch.setattr(app, "overlay_photo", zero)
    with pytest.warns(UserWarning, match="zero everywhere"):
        scores, _, heatmap = classify(model, photo, show_gradcam=True)
    assert len(scores) == 3 and heatmap is None


def test_main_check_builds_without_launching(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("--check must not launch")

    monkeypatch.setattr(gr.Blocks, "launch", refuse)
    assert main(["--check", "--assets", str(tmp_path / "assets")]) == 0
    assert main(["--check", "--checkpoint", str(checkpoint), "--config-name", "small_cnn"]) == 0
    assert "not launched" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([])  # a launch needs a checkpoint


def test_share_launches_with_a_public_link_and_check_never_launches(
    checkpoint: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--share` passes share=True to launch (mocked: no network); the default is local only;
    `--check --share` builds and opens no tunnel."""
    launches: list[dict[str, object]] = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda _, **kwargs: launches.append(kwargs))
    run = ["--checkpoint", str(checkpoint), "--config-name", "small_cnn"]
    assert main([*run, "--share"]) == 0
    assert main(run) == 0
    assert launches == [{"share": True}, {"share": False}]
    assert main(["--check", "--share"]) == 0
    assert len(launches) == 2


def test_the_share_certificate_folder_is_ignored() -> None:
    """`--share` writes Gradio's .gradio/certificate.pem into the working directory."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", ".gradio/certificate.pem"], cwd=ROOT, check=False
    )
    assert ignored.returncode == 0


def test_the_module_check_exits_zero() -> None:
    """The item's step: `python -m trailcam.demo.app --check` builds the interface and exits 0."""
    done = subprocess.run(
        [sys.executable, "-m", "trailcam.demo.app", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert "interface built, not launched" in done.stdout
