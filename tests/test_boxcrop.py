"""trailcam/data/boxcrop.py: the arm B crop sampler and the `arm` switch of the training transform
(plan.md section 6; R24 the box-guided crop always holds the whole box at a fill of 20% to 100%
and is reachable from the arm B training transform only, R23 boxes never reach the eval
transform).

The sampler is driven on blank images of the CCT20 sizes (2048 x 1494 original, 512 x 373 cache)
and of the synthetic metadata (96 x 72). The dataset tests use the synthetic metadata's split for
seed 42 with noise JPEGs written into a temporary directory (never committed).
"""

import ast
import inspect
import math
import pickle
import random
import subprocess
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.boxcrop import BOX_P, FILL, BoxCropSampler, to_pixels
from trailcam.data.dataset import (
    CROP_RATIO,
    CCT20Dataset,
    ControlCropSampler,
    Crop,
    EvalTransform,
    TrainTransform,
)
from trailcam.data.metadata import Box, Record, load_metadata
from trailcam.data.split import Split, make_split

SEED = 42
SAMPLES = 1000
REPO = Path(__file__).resolve().parents[1]
CENTRE: Box = (0.4, 0.4, 0.15, 0.2)  # 76.8 x 74.6 px at 512 x 373, aspect ratio 1.03


class SpyBoxSampler(BoxCropSampler):
    """Records every call of the box path: the boxes it got and the crop it returned."""

    def __init__(self, p: float) -> None:
        super().__init__(p)
        self.calls: list[tuple[Box, ...]] = []
        self.crops: list[Crop | None] = []

    def box_crop(self, width: int, height: int, boxes: Sequence[Box]) -> Crop | None:
        crop = super().box_crop(width, height, boxes)
        self.calls.append(tuple(boxes))
        self.crops.append(crop)
        return crop


def fill_of(box: Box, size: tuple[int, int], crop: Crop) -> float:
    """The box's area over the crop's area, both in pixels."""
    width, height = size
    _, _, h, w = crop
    return (box[2] * width) * (box[3] * height) / (w * h)


def holds(box: Box, size: tuple[int, int], crop: Crop) -> bool:
    """The crop lies inside the image and holds the whole box, edges included."""
    width, height = size
    top, left, h, w = crop
    x, y, bw, bh = box
    inside = 0 <= top and 0 <= left and 1 <= h and 1 <= w
    inside = inside and top + h <= height and left + w <= width
    across = left <= x * width and (x + bw) * width <= left + w
    down = top <= y * height and (y + bh) * height <= top + h
    return inside and across and down


def arm_a_shaped(crop: Crop) -> bool:
    """Within a pixel of a shape in arm A's aspect-ratio range: the sampler rounds its shape
    down to whole pixels, as torchvision rounds arm A's."""
    _, _, h, w = crop
    return CROP_RATIO[0] <= (w + 1) / h and w / (h + 1) <= CROP_RATIO[1]


def arm_a_crop_exists(box: Box, size: tuple[int, int]) -> bool:
    """Whether some whole-pixel crop in arm A's aspect-ratio range holds the box inside the image
    at a fill of 0.2 or more, by trying every crop width and height."""
    width, height = size
    x, y, bw, bh = box
    need_w = math.ceil((x + bw) * width) - math.floor(x * width)
    need_h = math.ceil((y + bh) * height) - math.floor(y * height)
    w = torch.arange(need_w, width + 1, dtype=torch.float64)[:, None]
    h = torch.arange(need_h, height + 1, dtype=torch.float64)[None, :]
    shaped = (w / h >= CROP_RATIO[0]) & (w / h <= CROP_RATIO[1])
    filled = (bw * width) * (bh * height) / (w * h) >= FILL[0]
    return bool((shaped & filled).any())


def sample(
    sampler: BoxCropSampler, size: tuple[int, int], boxes: Sequence[Box], seed: int
) -> list[Crop]:
    image = Image.new("RGB", size)
    torch.manual_seed(seed)
    return [sampler(image, boxes) for _ in range(SAMPLES)]


# Image size, box, and whether a crop in arm A's aspect-ratio range can hold the box at a fill of
# 0.2: not for a box as wide as a 512 x 373 frame (any crop holding it is at least 512 / 373 =
# 1.37 times as wide as tall) nor for one 10 px wide and 373 px tall (a crop at 3/4 holding it is
# over five times its area).
CASES = {
    "centre, cache size": ((512, 373), CENTRE, True),
    "top-left corner": ((512, 373), (0.0, 0.0, 0.1, 0.1), True),
    "bottom-right corner": ((512, 373), (0.9, 0.9, 0.1, 0.1), True),
    "touching the right edge, 2:1": ((512, 373), (0.7, 0.3, 0.3, 0.2), True),
    "touching the bottom edge": ((1024, 746), (0.2, 0.75, 0.3, 0.25), True),
    "the whole image": ((512, 373), (0.0, 0.0, 1.0, 1.0), False),
    "81% of the image": ((512, 373), (0.05, 0.05, 0.9, 0.9), True),
    "wide strip, 27:1": ((512, 373), (0.0, 0.45, 1.0, 0.05), False),
    "tall strip, 1:36": ((512, 373), (0.5, 0.0, 0.02, 1.0), False),
    "small, off the pixel grid": ((512, 373), (0.3331, 0.6667, 0.0123, 0.0171), True),
    "original size, 1:2.7": ((2048, 1494), (0.61, 0.12, 0.08, 0.3), True),
    "portrait image": ((373, 512), (0.2, 0.5, 0.3, 0.1), True),
    "synthetic size": ((96, 72), (12 / 96, 10 / 72, 30 / 96, 20 / 72), True),
}


@pytest.mark.parametrize(("size", "box", "shaped"), CASES.values(), ids=CASES.keys())
def test_box_guided_crops_hold_the_whole_box_at_a_fill_of_20_to_100_percent(
    size: tuple[int, int], box: Box, shaped: bool
) -> None:
    """R24, over 1,000 samples per case through the sampler with the box path always on: every
    crop is the box path's, lies inside the image, holds the whole box (also at the image edge,
    after the jitter) and the box's area is 20% to 100% of the crop's (float rounding allowed).
    Where a crop in arm A's aspect-ratio range can hold the box, every crop is in that range."""
    assert arm_a_crop_exists(box, size) == shaped
    spy = SpyBoxSampler(p=1.0)
    crops = sample(spy, size, (box,), seed=0)
    assert len(spy.calls) == SAMPLES and spy.crops == crops
    for crop in crops:
        assert holds(box, size, crop)
        assert FILL[0] * (1 - 1e-12) <= fill_of(box, size, crop) <= FILL[1] * (1 + 1e-12)
        assert arm_a_shaped(crop) or not shaped


def test_crops_leave_arm_a_shapes_only_when_none_holds_the_box() -> None:
    """Over 300 random boxes of 1% to 100% of each side (20 crops each): every crop holds its box
    at a fill of 20% to 100%, and stays in arm A's aspect-ratio range whenever some crop in it
    holds the box inside the image at a fill of 0.2 or more, so the resize to 224 x 224 never
    stretches more than arm A's. An elongated box tops out below a fill of 1.0: a 2:1 box at 2/3."""
    rng = random.Random(0)
    size = (512, 373)
    image = Image.new("RGB", size)
    sampler = BoxCropSampler(p=1.0)
    torch.manual_seed(12)
    shaped_boxes = 0
    for _ in range(300):
        bw, bh = (math.exp(rng.uniform(math.log(0.01), 0.0)) for _ in range(2))
        box: Box = (rng.uniform(0.0, 1.0 - bw), rng.uniform(0.0, 1.0 - bh), bw, bh)
        shaped = arm_a_crop_exists(box, size)
        shaped_boxes += shaped
        for _ in range(20):
            crop = sampler(image, (box,))
            assert holds(box, size, crop)
            assert FILL[0] * (1 - 1e-12) <= fill_of(box, size, crop) <= FILL[1] * (1 + 1e-12)
            assert arm_a_shaped(crop) or not shaped
    assert 100 < shaped_boxes < 300
    size = (512, 384)
    two_to_one: Box = (0.25, 0.25, 0.375, 0.25)  # 192 x 96 px on whole pixels
    fills = [fill_of(two_to_one, size, c) for c in sample(sampler, size, (two_to_one,), seed=13)]
    assert 0.65 < max(fills) <= 2 / 3 + 0.01  # 192 x 144 at the tightest, rounded down


def test_the_fill_is_on_the_area_and_drawn_uniformly_from_20_to_100_percent() -> None:
    """The lower end 0.2 is reached on the area: a fill on the side (box width over crop width)
    would put the area fill down to 0.04, and a square root of it never near 0.2. Uniform in the
    fill: the mean is near 0.6."""
    size = (512, 373)
    fills = [fill_of(CENTRE, size, c) for c in sample(BoxCropSampler(p=1.0), size, (CENTRE,), 1)]
    assert FILL[0] * (1 - 1e-12) <= min(fills) < 0.22
    assert 0.95 < max(fills) <= 1.0
    assert 0.56 < sum(fills) / len(fills) < 0.64


def test_the_position_is_jittered() -> None:
    """The box sits anywhere in the crop, not always in its centre."""
    size = (512, 373)
    crops = sample(BoxCropSampler(p=1.0), size, (CENTRE,), seed=2)
    centre_x = (CENTRE[0] + CENTRE[2] / 2) * size[0]
    where = [(centre_x - left) / w for _, left, _, w in crops]
    assert len({(top, left) for top, left, _, _ in crops}) > 200
    assert min(where) < 0.35 and max(where) > 0.65


def test_the_aspect_ratio_spans_arm_a_s_range_when_the_box_allows() -> None:
    """For a box of aspect ratio 1.03, crops span arm A's 3/4 to 4/3 (whole pixels allowed)."""
    size = (512, 373)
    ratios = [w / h for _, _, h, w in sample(BoxCropSampler(p=1.0), size, (CENTRE,), seed=3)]
    assert CROP_RATIO[0] - 0.03 <= min(ratios) < 0.8
    assert 1.28 < max(ratios) <= CROP_RATIO[1] + 0.03


def test_a_box_larger_than_any_allowed_crop() -> None:
    """A box covering the whole 512 x 373 image gets the whole image every time: no crop in arm
    A's aspect-ratio range holds it. A box covering 81% of it cannot fill less than its share of
    the largest crop in that range, 497 x 373 (as arm A, which never takes the whole frame
    either): its fill starts at 0.83 and its loosest crop is about that size."""
    size = (512, 373)
    whole: Box = (0.0, 0.0, 1.0, 1.0)
    assert set(sample(BoxCropSampler(p=1.0), size, (whole,), seed=4)) == {(0, 0, 373, 512)}
    big: Box = (0.05, 0.05, 0.9, 0.9)
    crops = sample(BoxCropSampler(p=1.0), size, (big,), seed=5)
    fills = [fill_of(big, size, c) for c in crops]
    largest = 373 * 4 / 3 * 373
    assert (0.9 * 512) * (0.9 * 373) / largest * (1 - 1e-12) <= min(fills) < 0.85
    assert max(w for _, _, _, w in crops) <= 498
    assert max(w * h for _, _, h, w in crops) > 0.99 * largest


def test_boxes_are_fractions_converted_to_pixels_once() -> None:
    """R24: the box is a fraction of the image (plan section 2); x and width scale with the
    width, y and height with the height, once. A box in pixels is refused, not clipped."""
    assert to_pixels((0.1, 0.6, 0.2, 0.3), 1000, 500) == pytest.approx((100, 300, 300, 450))
    small = to_pixels(CENTRE, 512, 373)
    assert to_pixels(CENTRE, 2048, 1492) == pytest.approx(tuple(4 * v for v in small))
    image = Image.new("RGB", (512, 373))
    for pixels in [(100.0, 50.0, 40.0, 30.0), (0.5, 0.5, 40.0, 30.0)]:
        with pytest.raises(ValueError, match=r"^a box must be fractions of the image"):
            BoxCropSampler(p=1.0)(image, (pixels,))


def test_with_several_boxes_one_is_drawn_uniformly() -> None:
    """Two animals at opposite corners: each crop holds one of them whole at a fill of 20% to
    100%, never both (their union would be most of the frame), each in about half the crops."""
    size = (512, 373)
    first: Box = (0.05, 0.05, 0.1, 0.1)
    second: Box = (0.85, 0.85, 0.1, 0.1)
    held = {first: 0, second: 0}
    for crop in sample(BoxCropSampler(p=1.0), size, (first, second), seed=6):
        inside = [box for box in (first, second) if holds(box, size, crop)]
        assert len(inside) == 1
        assert FILL[0] * (1 - 1e-12) <= fill_of(inside[0], size, crop) <= 1.0
        held[inside[0]] += 1
    assert 400 < held[first] < 600 and held[first] + held[second] == SAMPLES


def test_a_box_too_small_for_any_crop_is_skipped() -> None:
    """A box under a pixel cannot fill 20% of any whole-pixel crop around it: next to a usable
    box it is never drawn, and alone it leaves the image to the control sampler, crop for crop."""
    size = (512, 373)
    tiny: Box = (0.5003, 0.5003, 0.0005, 0.0005)
    for crop in sample(BoxCropSampler(p=1.0), size, (tiny, CENTRE), seed=7):
        assert holds(CENTRE, size, crop)
    spy = SpyBoxSampler(p=1.0)
    image = Image.new("RGB", size)
    for seed in range(50):
        torch.manual_seed(seed)
        crop = spy(image, (tiny,))
        torch.manual_seed(seed)
        assert crop == ControlCropSampler()(image, ())
    assert len(spy.calls) == 50 and set(spy.crops) == {None}


def test_half_of_the_boxed_images_take_the_box_path() -> None:
    spy = SpyBoxSampler(p=BOX_P)
    sample(spy, (512, 373), (CENTRE,), seed=8)
    assert BOX_P == 0.5
    assert 430 < len(spy.calls) < 570


def test_images_without_boxes_never_call_the_box_path() -> None:
    """Even with the box path always on, an image without boxes never reaches it and gets the
    control sampler's crop for the same seed."""
    spy = SpyBoxSampler(p=1.0)
    image = Image.new("RGB", (512, 373))
    sample(spy, (512, 373), (), seed=9)
    for seed in range(50):
        torch.manual_seed(seed)
        crop = spy(image, ())
        torch.manual_seed(seed)
        assert crop == ControlCropSampler()(image, ())
    assert spy.calls == []


def write_jpegs(records: Sequence[Record], root: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    for r in records:
        noise = rng.randbytes(r.width * r.height * 3)
        Image.frombytes("RGB", (r.width, r.height), noise).save(root / r.file_name, quality=90)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Split, Path]:
    tmp = tmp_path_factory.mktemp("boxcrop")
    paths = synthetic.write_synthetic_metadata(tmp / "meta")
    split = make_split(load_metadata(tmp / "meta", paths["sidecar"]), SEED)
    write_jpegs([r for name in ("train", "val") for r in split.records[name]], tmp / "images")
    return split, tmp / "images"


def test_the_dataset_sends_only_boxed_images_to_the_box_path(fixture: tuple[Split, Path]) -> None:
    split, root = fixture
    records = split.records["train"]
    boxed = [r.boxes for r in records if r.boxes]
    assert 0 < len(boxed) < len(records)
    spy = SpyBoxSampler(p=1.0)
    dataset = CCT20Dataset(records, root, TrainTransform(sampler=spy))
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert tuple(image.shape) == (3, 224, 224) and image.dtype == torch.float32
    assert spy.calls == boxed
    assert None not in spy.crops


def test_arm_a_and_arm_b_are_identical_when_the_box_path_is_disabled(
    fixture: tuple[Split, Path],
) -> None:
    """With the box path off, arm B's transform gives arm A's tensor for every train image and
    seed, boxed or not: the sampler draws nothing more than the control sampler. With it on,
    the boxed images change."""
    split, root = fixture
    records = split.records["train"]
    arm_a = CCT20Dataset(records, root, TrainTransform("A"))
    disabled = CCT20Dataset(records, root, TrainTransform(sampler=BoxCropSampler(p=0.0)))
    arm_b = CCT20Dataset(records, root, TrainTransform("B"))
    changed = 0
    for index, record in enumerate(records):
        torch.manual_seed(index)
        expected = arm_a[index][0]
        torch.manual_seed(index)
        assert torch.equal(disabled[index][0], expected)
        torch.manual_seed(index)
        same = torch.equal(arm_b[index][0], expected)
        assert same or record.boxes  # without boxes, arm B gives arm A's tensor too
        changed += not same
    assert changed > 0


def test_the_arms_differ_only_in_the_crop_sampler() -> None:
    """The `arm` switch picks the sampler and nothing else (R25 for the transform)."""
    arm_a, arm_b = TrainTransform("A"), TrainTransform("B")
    assert TrainTransform().arm == "A" and (arm_a.arm, arm_b.arm) == ("A", "B")
    assert set(vars(arm_a)) == set(vars(arm_b)) == {"arm", "sampler", "after_crop"}
    assert repr(arm_a.after_crop) == repr(arm_b.after_crop)
    assert type(arm_a.sampler) is ControlCropSampler
    assert type(arm_b.sampler) is BoxCropSampler
    assert arm_b.sampler.p == BOX_P and type(arm_b.sampler.control) is ControlCropSampler


def test_the_eval_transform_has_no_code_path_that_reads_boxes() -> None:
    """R23, R24: the eval transform takes the image alone, names no box or sampler anywhere in
    its code, and its pipeline is exactly resize, centre crop, tensor and normalise."""
    assert list(inspect.signature(EvalTransform.__call__).parameters) == ["self", "image"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(EvalTransform)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert "pipeline" in names and "image" in names
    assert not [n for n in names if "box" in n.lower() or "sampler" in n.lower()]
    pipeline = EvalTransform().pipeline
    assert isinstance(pipeline, T.Compose)
    steps = [type(t) for t in pipeline.transforms]
    assert steps == [T.Resize, T.CenterCrop, T.ToTensor, T.Normalize]


def test_loading_the_eval_transform_never_loads_the_box_sampler() -> None:
    """In a fresh interpreter, the dataset and loader modules and an eval pass leave the box
    sampler module unloaded; building arm B's transform loads it."""
    code = textwrap.dedent(
        """
        import sys
        from PIL import Image
        import trailcam.data.loaders
        from trailcam.data.dataset import EvalTransform, TrainTransform
        EvalTransform()(Image.new("RGB", (300, 200)))
        TrainTransform("A")
        print("trailcam.data.boxcrop" in sys.modules)
        TrainTransform("B")
        print("trailcam.data.boxcrop" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "True"]


def test_arm_b_draws_from_torch_only() -> None:
    """The same torch seed gives the same crops whatever Python's `random` state is."""
    image = Image.new("RGB", (512, 373))
    sampler = BoxCropSampler()
    runs = []
    for python_seed in (1, 2):
        torch.manual_seed(7)
        random.seed(python_seed)
        runs.append([sampler(image, (CENTRE,)) for _ in range(20)])
    assert runs[0] == runs[1]


def test_arm_b_transform_survives_pickling(fixture: tuple[Split, Path]) -> None:
    """DataLoader workers started with spawn (macOS) receive the transform pickled."""
    split, root = fixture
    records = split.records["train"]
    index = next(i for i, r in enumerate(records) if r.boxes)
    dataset = CCT20Dataset(records, root, TrainTransform("B"))
    copy = pickle.loads(pickle.dumps(dataset))
    assert isinstance(copy.transform, TrainTransform)
    assert isinstance(copy.transform.sampler, BoxCropSampler)
    torch.manual_seed(11)
    expected = dataset[index][0]
    torch.manual_seed(11)
    assert torch.equal(copy[index][0], expected)


def test_bad_arguments_are_refused(fixture: tuple[Split, Path]) -> None:
    for p in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match=r"^p must be between 0 and 1"):
            BoxCropSampler(p=p)
    for arm in ("C", "b", ""):
        with pytest.raises(ValueError, match=r"^arm must be one of"):
            TrainTransform(arm)
    with pytest.raises(ValueError, match=r"^pass an arm or a sampler, not both"):
        TrainTransform("B", sampler=ControlCropSampler())
    image = Image.new("RGB", (512, 373))
    for box in [(0.5, 0.5, 0.0, 0.1), (0.95, 0.1, 0.1, 0.1), (-0.01, 0.1, 0.1, 0.1)]:
        with pytest.raises(ValueError, match=r"^a box must be fractions of the image"):
            BoxCropSampler(p=1.0)(image, (box,))
    split, root = fixture
    with pytest.raises(ValueError, match=r"^the training transform is for train records only"):
        CCT20Dataset(split.records["val"], root, TrainTransform("B"))
