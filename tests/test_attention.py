"""trailcam/evaluate/attention.py: attention-in-box and the animal-removed test (plan.md
sections 6 and 7; R23 boxes never reach the model, R26 the two measures, R17 test isolation,
R03 counts and scores only).

The R26 failure modes each have a test: a map normalised after cropping to the box (an even map
must score the box's share of the frame, not 1), a box blanked with zeros (the fill must be the
image mean), and an attention pass on a test split from the command line (refused). The box
geometry is checked against `EvalTransform` itself. The end-to-end tests run on the synthetic
fixture with random weights; no download.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.data.dataset import INPUT_SIZE, EvalTransform
from trailcam.data.metadata import Record, load_metadata
from trailcam.data.split import Split, make_split
from trailcam.evaluate import attention
from trailcam.evaluate.attention import (
    AttentionDataset,
    ImageResult,
    SplitRun,
    attention_pass,
    blank_boxes,
    blank_region,
    box_in_frame,
    box_mask,
    eval_geometry,
    fraction_inside,
    image_mean,
    summarise,
)
from trailcam.model.factory import build_model
from trailcam.model.small_cnn import SmallCNN
from trailcam.train.config import load_config

SEED = 0
SIZES = [(512, 384), (384, 512), (512, 512), (96, 72), (500, 281), (281, 500), (300, 299)]


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Split, Path]:
    tmp = tmp_path_factory.mktemp("attention")
    written = synthetic.write_synthetic_fixture(tmp)
    split = make_split(load_metadata(tmp, written.paths["sidecar"]), SEED)
    return split, written.image_root


# Attention-in-box: the score


def test_a_map_inside_the_box_scores_one_and_outside_zero() -> None:
    mask = box_mask([(0.3, 0.3, 0.4, 0.4)], 512, 512)
    inside = (mask == 1).float()
    assert fraction_inside(inside * 7.0, mask) == pytest.approx(1.0)
    assert fraction_inside((mask == 0).float() * 0.3, mask) == pytest.approx(0.0)


def test_the_map_is_normalised_over_the_whole_frame_not_the_box() -> None:
    """R26: an even map scores the box's share of the frame; half the mass in the box is 0.5.
    A map normalised after cropping to the box would score 1 in both cases."""
    mask = box_mask([(0.25, 0.25, 0.5, 0.5)], 512, 512)
    assert fraction_inside(torch.ones(INPUT_SIZE, INPUT_SIZE), mask) == pytest.approx(
        float(mask.sum()) / INPUT_SIZE**2
    )
    half = (mask == 1).float() / float((mask == 1).sum()) + (mask == 0).float() / float(
        (mask == 0).sum()
    )
    assert fraction_inside(half, mask) == pytest.approx(0.5)


def test_a_zero_map_has_no_score_and_a_negative_map_is_refused() -> None:
    mask = box_mask([(0.25, 0.25, 0.5, 0.5)], 512, 512)
    assert np.isnan(fraction_inside(torch.zeros(INPUT_SIZE, INPUT_SIZE), mask))
    with pytest.raises(ValueError):
        fraction_inside(-torch.ones(INPUT_SIZE, INPUT_SIZE), mask)
    with pytest.raises(ValueError):
        fraction_inside(torch.ones(10, 10), mask)


# Attention-in-box: the box in the eval frame


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_the_geometry_is_the_eval_transform_s(width: int, height: int) -> None:
    resize, crop = EvalTransform().pipeline.transforms[:2]
    image = Image.new("RGB", (width, height))
    resized = resize(image)
    new_width, new_height, left, top = eval_geometry(width, height)
    assert resized.size == (new_width, new_height)
    assert isinstance(crop, T.CenterCrop)
    marked = Image.new("L", resized.size)
    marked.putpixel((left, top), 255)
    assert np.asarray(crop(marked))[0, 0] == 255


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_a_mapped_box_lands_where_the_eval_transform_puts_its_pixels(
    width: int, height: int
) -> None:
    """A white box on a black image, through resize and centre crop: the bright pixels sit
    inside the mapped box, give or take the resize's one-pixel blur."""
    box = (0.35, 0.3, 0.3, 0.4)
    image = Image.new("L", (width, height))
    image.paste(255, blank_region(box, width, height))
    resize, crop = EvalTransform().pipeline.transforms[:2]
    seen = np.asarray(crop(resize(image))) > 127
    rows, columns = np.nonzero(seen)
    framed = box_in_frame(box, width, height)
    assert framed is not None
    x0, y0, x1, y1 = framed
    tolerance = 1.5 + 256 / min(width, height)  # rounding the box outwards to whole pixels
    assert abs(columns.min() - x0) <= tolerance and abs(columns.max() + 1 - x1) <= tolerance
    assert abs(rows.min() - y0) <= tolerance and abs(rows.max() + 1 - y1) <= tolerance


def test_a_box_cut_by_the_centre_crop_is_clipped_and_one_outside_it_is_dropped() -> None:
    """512 x 384 resizes to 341 x 256 and crops columns 58 to 282 (rows 16 to 240)."""
    assert eval_geometry(512, 384) == (341, 256, 58, 16)
    clipped = box_in_frame((0.1, 0.5, 0.2, 0.2), 512, 384)  # columns 34.1 to 102.3
    assert clipped == pytest.approx((0.0, 128 - 16, 102.3 - 58, 179.2 - 16))
    assert box_in_frame((0.0, 0.5, 0.1, 0.2), 512, 384) is None  # columns 0 to 34.1
    assert float(box_mask([(0.0, 0.5, 0.1, 0.2)], 512, 384).sum()) == 0.0
    edge = box_mask([(0.0, 0.0, 1.0, 1.0)], 512, 384)
    assert float(edge.sum()) == pytest.approx(INPUT_SIZE**2)


def test_several_boxes_count_as_their_union_once() -> None:
    """On a 256 x 256 image the frame is the centre crop, offset 16: two 0.25-wide squares
    overlapping by half cover 1.5 x 64 by 64 pixels, not 2 x."""
    a = (0.25, 0.25, 0.25, 0.25)
    b = (0.375, 0.25, 0.25, 0.25)
    mask = box_mask([a, b], 256, 256)
    assert float(mask.sum()) == pytest.approx(64 * 96)
    assert float(mask.max()) == pytest.approx(1.0)


def test_partly_covered_pixels_count_by_their_share() -> None:
    mask = box_mask([(0.5, 0.5, 0.01, 0.01)], 224, 224)  # resize to 256: a 2.56 px square
    assert float(mask.sum()) == pytest.approx(2.56**2)


# The animal-removed test: blanking


def _noise(width: int, height: int, mode: str = "RGB") -> Image.Image:
    rng = np.random.default_rng(0)
    channels = 3 if mode == "RGB" else 1
    pixels = rng.integers(0, 256, size=(height, width, channels), dtype=np.uint8)
    return Image.fromarray(pixels.squeeze(-1) if channels == 1 else pixels, mode)


def test_blanking_changes_pixels_only_inside_the_box_and_fills_the_image_mean() -> None:
    image = _noise(120, 90)
    box = (0.2, 0.3, 0.25, 0.4)
    blanked = np.asarray(blank_boxes(image, [box]))
    before = np.asarray(image)
    left, top, right, bottom = blank_region(box, 120, 90)
    assert (left, top, right, bottom) == (24, 27, 54, 63)
    outside = np.ones(before.shape[:2], dtype=bool)
    outside[top:bottom, left:right] = False
    assert np.array_equal(blanked[outside], before[outside])
    mean = before.reshape(-1, 3).mean(axis=0).round().astype(np.uint8)
    assert np.array_equal(np.asarray(image_mean(image)), mean)
    assert (blanked[~outside] == mean).all()
    assert mean.min() > 0  # the R26 failure fills zeros; noise has a mean near 128
    assert np.array_equal(np.asarray(image), before)  # the input is not modified


def test_the_blanked_region_is_the_box_rounded_outwards() -> None:
    """On 100 x 50, (0.105, 0.21, 0.3, 0.3) spans columns 10.5 to 40.5 and rows 10.5 to 25.5:
    every pixel the box touches is blanked, so no animal pixel survives at an edge."""
    assert blank_region((0.105, 0.21, 0.3, 0.3), 100, 50) == (10, 10, 41, 26)
    assert blank_region((0.9, 0.9, 0.2, 0.2), 100, 50) == (90, 45, 100, 50)


def test_blanking_a_grayscale_image_and_several_boxes() -> None:
    image = _noise(80, 60, "L")
    boxes = [(0.1, 0.1, 0.2, 0.2), (0.6, 0.5, 0.3, 0.3)]
    blanked = np.asarray(blank_boxes(image, boxes))
    gray = np.asarray(image.convert("RGB"))
    changed = (blanked != gray).any(axis=2)
    allowed = np.zeros(changed.shape, dtype=bool)
    for box in boxes:
        left, top, right, bottom = blank_region(box, 80, 60)
        allowed[top:bottom, left:right] = True
        assert (blanked[top:bottom, left:right] == image_mean(image)).all()
    assert not (changed & ~allowed).any()


def test_the_model_sees_both_images_through_the_eval_transform(
    fixture: tuple[Split, Path],
) -> None:
    split, root = fixture
    boxed = [r for r in split.records["val"] if r.boxes]
    dataset = AttentionDataset(boxed, root)
    transform = EvalTransform()
    for index in range(3):
        original, blanked, mask, label = dataset[index]
        image = dataset.images.load_image(index)
        torch.testing.assert_close(original, transform(image), rtol=0, atol=0)
        expected = transform(blank_boxes(image, boxed[index].boxes))
        torch.testing.assert_close(blanked, expected, rtol=0, atol=0)
        torch.testing.assert_close(mask, box_mask(boxed[index].boxes, *image.size).float())
        assert label == boxed[index].label
        assert not torch.equal(original, blanked)
    with pytest.raises(ValueError):
        AttentionDataset(split.records["val"], root)  # some records have no box


# The pass over a split


class _Spy(SmallCNN):
    """The small CNN, recording every input it is given (R23)."""

    def __init__(self) -> None:
        super().__init__(num_classes=12)
        self.inputs: list[tuple[object, ...]] = []

    def forward(self, *args: Tensor) -> Tensor:
        self.inputs.append(args)
        return super().forward(*args)


def test_the_pass_feeds_the_model_image_tensors_only(fixture: tuple[Split, Path]) -> None:
    split, root = fixture
    records = split.records["trans_val"]
    torch.manual_seed(0)
    model = _Spy()
    run = attention_pass(model, records, "trans_val", root, torch.device("cpu"), batch_size=8)
    assert run.group == "trans"
    assert run.images == len(records)
    assert run.unboxed == sum(1 for r in records if not r.boxes) > 0
    assert len(run.results) == run.images - run.unboxed
    assert [r.label for r in run.results] == [r.label for r in records if r.boxes]
    assert model.inputs and all(len(args) == 1 for args in model.inputs)
    for (tensor,) in model.inputs:
        assert isinstance(tensor, Tensor) and tensor.dtype == torch.float32
        assert tensor.shape[1:] == (3, INPUT_SIZE, INPUT_SIZE)
    # one Grad-CAM pass and one blanked pass per batch
    assert sum(len(args[0]) for args in model.inputs) == 2 * len(run.results)
    for result in run.results:
        assert 0.0 < result.confidence <= 1.0 and 0.0 < result.blanked_confidence <= 1.0
        assert (result.box_share > 0) == result.box_visible
        if result.box_visible and not np.isnan(result.attention):
            assert 0.0 <= result.attention <= 1.0


def test_the_pass_refuses_records_of_another_split(fixture: tuple[Split, Path]) -> None:
    split, root = fixture
    model = build_model("small_cnn")
    with pytest.raises(ValueError, match="not from val"):
        attention_pass(
            model, split.records["trans_val"], "val", root, torch.device("cpu"), batch_size=4
        )
    with pytest.raises(ValueError):
        attention_pass(
            model, split.records["train"], "train", root, torch.device("cpu"), batch_size=4
        )


# The summary


def _result(
    label: int,
    prediction: int,
    attention_score: float,
    blanked: int,
    share: float = 0.1,
    visible: bool = True,
) -> ImageResult:
    return ImageResult(label, prediction, 0.8, visible, attention_score, share, blanked, 0.4)


def test_the_summary_by_hand() -> None:
    """Five boxed images, one with its box outside the crop (not scored). Of the four scored,
    labels 0 1 2 3, predictions 0 1 5 3: three correct. Attention 0.9, 0.7, 0.2 and a zero map.
    Blanked predictions 0 4 5 7: blanked accuracy 1/4, prediction kept 2/4.
    Correct: attention 0.9 and 0.7 (the zero map has none): mean 0.8, median 0.8; blanked
    accuracy 1/3, kept 1/3. Wrong: attention 0.2, blanked accuracy 0, kept 1. Box shares 0.1,
    0.2 and 0.3 for the images with a score (the zero map's 0.5 is left out with its score):
    0.2 over all, 0.15 correct, 0.3 wrong."""
    nan = float("nan")
    run = SplitRun(
        "val",
        images=7,
        unboxed=2,
        results=(
            _result(0, 0, 0.9, 0, share=0.1),
            _result(1, 1, 0.7, 4, share=0.2),
            _result(2, 5, 0.2, 5, share=0.3),
            _result(3, 3, nan, 7, share=0.5),
            _result(4, 4, nan, 4, share=0.0, visible=False),
        ),
    )
    assert (run.outside_crop, len(run.scored), run.zero_maps) == (1, 4, 1)
    everything, correct, wrong = summarise(run)
    assert [r.subset for r in (everything, correct, wrong)] == ["all", "correct", "wrong"]
    assert (everything.n, everything.attention_n) == (4, 3)
    assert everything.attention_mean == pytest.approx(0.6)
    assert everything.attention_median == pytest.approx(0.7)
    assert everything.accuracy == pytest.approx(0.75)
    assert everything.blanked_accuracy == pytest.approx(0.25)
    assert everything.kept == pytest.approx(0.5)
    assert everything.confidence == pytest.approx(0.8)
    assert everything.blanked_confidence == pytest.approx(0.4)
    assert (everything.box_share, correct.box_share, wrong.box_share) == (
        pytest.approx(0.2),
        pytest.approx(0.15),
        pytest.approx(0.3),
    )
    assert (correct.n, correct.attention_n) == (3, 2)
    assert correct.attention_mean == pytest.approx(0.8)
    assert correct.blanked_accuracy == pytest.approx(1 / 3)
    assert correct.kept == pytest.approx(1 / 3)
    assert (wrong.n, wrong.attention_median, wrong.blanked_accuracy, wrong.kept) == (
        1,
        pytest.approx(0.2),
        0.0,
        1.0,
    )
    assert everything.group == "cis"


def test_an_empty_subset_is_reported_as_not_available(tmp_path: Path) -> None:
    run = SplitRun("trans_val", images=1, unboxed=0, results=(_result(0, 0, 0.5, 0),))
    wrong = summarise(run)[2]
    assert wrong.n == 0 and wrong.attention_mean is None and wrong.accuracy is None
    table, histogram = attention.write_report([run], tmp_path)
    assert "n/a" in table.read_text()
    assert histogram.read_bytes().startswith(b"\x89PNG")


# The command line


def test_the_command_line_on_the_fixture_writes_a_table_and_a_histogram(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    figures = Path(__file__).resolve().parents[1] / "docs" / "figures"
    before = sorted(figures.rglob("*")) if figures.exists() else []
    out = tmp_path / "out"
    code = attention.main(
        ["--dry-run=True", "--split", "val", "--split", "trans_val", "--out", str(out)]
    )
    assert code == 0
    assert sorted(p.name for p in out.iterdir()) == ["attention.md", "attention_in_box.png"]
    assert (out / "attention_in_box.png").read_bytes().startswith(b"\x89PNG")
    table = (out / "attention.md").read_text()
    for heading in ("## Images", "## Attention-in-box", "## Animal-removed test"):
        assert heading in table
    for split, group in (("val", "cis"), ("trans_val", "trans")):
        assert f"| {split} | {group} | all |" in table
        assert f"| {split} | {group} | wrong |" in table
    assert "cis_test" not in table and "trans_test" not in table
    after = sorted(figures.rglob("*")) if figures.exists() else []
    assert after == before  # nothing written under docs/figures/

    # R03: no image id, sequence id or file name in the table or on stdout
    printed = capsys.readouterr().out
    meta = load_metadata(*_fixture_paths(tmp_path / "ids"))
    names = {
        v for rs in meta.records.values() for r in rs for v in (r.image_id, r.file_name, r.seq_id)
    }
    assert names and not any(name in table or name in printed for name in names)


def _fixture_paths(directory: Path) -> tuple[Path, Path]:
    written = synthetic.write_synthetic_fixture(directory)
    return directory, written.paths["sidecar"]


@pytest.mark.parametrize("split", ["cis_test", "trans_test", "train"])
def test_the_command_line_refuses_every_split_but_val_and_trans_val(
    split: str, tmp_path: Path
) -> None:
    with pytest.raises(SystemExit) as raised:
        attention.main(["--dry-run=True", "--split", split, "--out", str(tmp_path)])
    assert raised.value.code == 2
    assert not any(tmp_path.iterdir())


def test_the_command_line_reads_only_the_splits_it_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, set[str]]] = []

    def spy(
        model: object, records: tuple[Record, ...], split: str, *args: object, **kwargs: object
    ) -> SplitRun:
        seen.append((split, {r.split for r in records}))
        return SplitRun(split, len(records), len(records), ())

    monkeypatch.setattr(attention, "attention_pass", spy)
    assert attention.main(["--dry-run=True", "--split", "trans_val", "--out", str(tmp_path)]) == 0
    assert seen == [("trans_val", {"trans_val"})]
    seen.clear()
    assert attention.main(["--dry-run=True", "--out", str(tmp_path / "default")]) == 0
    assert seen == [("val", {"val"})]


def test_a_real_run_needs_a_checkpoint_and_the_cache(tmp_path: Path) -> None:
    for argv in (
        ["--config-name", "resnet18_arm_b", "--out", str(tmp_path)],
        ["--config-name", "resnet18_arm_b", "--checkpoint", "x.pt", "--out", str(tmp_path)],
        ["--dry-run=True", "--checkpoint", "x.pt", "--out", str(tmp_path)],
        [
            "--config-name",
            "dryrun",
            "--checkpoint",
            "x.pt",
            "--data-root",
            ".",
            "--out",
            str(tmp_path),
        ],
    ):
        with pytest.raises(SystemExit) as raised:
            attention.main(argv)
        assert raised.value.code == 2


def test_a_checkpoint_of_another_model_or_class_map_is_refused(tmp_path: Path) -> None:
    """The format `train.py` saves: the model name, the class list and the state dict."""
    config = load_config("small_cnn")
    model = build_model("small_cnn")
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    good = tmp_path / "good.pt"
    torch.save({"model": "small_cnn", "classes": list(CLASSES), "state_dict": state}, good)
    loaded = attention.load_checkpoint(good, config)
    for key, value in loaded.state_dict().items():
        torch.testing.assert_close(value, state[key])
    with pytest.raises(ValueError, match="holds a small_cnn"):
        attention.load_checkpoint(good, load_config("resnet18_arm_a"))
    other = tmp_path / "other.pt"
    classes = list(reversed(CLASSES))
    torch.save({"model": "small_cnn", "classes": classes, "state_dict": state}, other)
    with pytest.raises(ValueError, match="class map"):
        attention.load_checkpoint(other, config)
