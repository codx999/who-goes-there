"""trailcam/data/dataset.py: the Dataset over a split and its two transforms (plan.md sections 5
and 6; R20 augmentation on the training transform only, R21 ImageNet normalisation, R23 boxes
never a model input and never seen by the eval transform).

The images are the synthetic fixture's JPEGs (trailcam/data/synthetic.py, item 4.03), written
into a temporary directory (never committed) and split with seed 42: one per image at its size, a
noisy background with a drawn rectangle per box, and a grayscale (mode L) JPEG for a night record.
"""

import inspect
import pickle
import random
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch
import torchvision.transforms as T
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.classes import NUM_CLASSES
from trailcam.data.dataset import (
    CROP_SCALE,
    EVAL_RESIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_SIZE,
    CCT20Dataset,
    ControlCropSampler,
    Crop,
    EvalTransform,
    ImageReadError,
    TrainTransform,
)
from trailcam.data.metadata import Box, load_metadata
from trailcam.data.split import Split, make_split

SEED = 42


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Split, Path]:
    tmp = tmp_path_factory.mktemp("dataset")
    written = synthetic.write_synthetic_fixture(tmp)
    split = make_split(load_metadata(tmp, written.paths["sidecar"]), SEED)
    return split, written.image_root


@pytest.fixture(scope="module")
def train_set(fixture: tuple[Split, Path]) -> CCT20Dataset:
    split, root = fixture
    return CCT20Dataset(split.records["train"], root, TrainTransform())


@pytest.fixture(scope="module")
def val_set(fixture: tuple[Split, Path]) -> CCT20Dataset:
    split, root = fixture
    return CCT20Dataset(split.records["val"], root, EvalTransform())


def test_fixture_has_boxed_unboxed_and_night_images(fixture: tuple[Split, Path]) -> None:
    train = fixture[0].records["train"]
    boxed = sum(1 for r in train if r.boxes)
    night = sum(1 for r in train if r.night)
    assert 0 < boxed < len(train)
    assert 0 < night < len(train)


@pytest.mark.parametrize("name", ["train_set", "val_set"])
def test_items_are_float32_224_tensors_and_class_indices(
    name: str, request: pytest.FixtureRequest
) -> None:
    dataset: CCT20Dataset = request.getfixturevalue(name)
    torch.manual_seed(0)
    labels = []
    for index in range(len(dataset)):
        item = dataset[index]
        assert len(item) == 2  # the tensor and the class index, never a box (R23)
        tensor, label = item
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == (3, INPUT_SIZE, INPUT_SIZE)
        assert tensor.dtype == torch.float32
        assert type(label) is int
        labels.append(label)
    assert all(0 <= label < NUM_CLASSES for label in labels)
    assert set(labels) == set(range(NUM_CLASSES))
    assert labels == [r.label for r in dataset.records]


def test_dataset_follows_chapter_04(val_set: CCT20Dataset) -> None:
    assert len(val_set) == len(val_set.records)
    assert len(val_set.classes) == NUM_CLASSES
    assert [val_set.class_to_idx[c] for c in val_set.classes] == list(range(NUM_CLASSES))
    assert val_set.load_image(0).mode == "RGB"


def test_grayscale_jpeg_loads_as_three_equal_channels(val_set: CCT20Dataset) -> None:
    night = [i for i, r in enumerate(val_set.records) if r.night]
    assert night
    image = val_set.load_image(night[0])
    assert image.mode == "RGB"
    red, green, blue = image.split()
    assert red.tobytes() == green.tobytes() == blue.tobytes()


def test_eval_transform_is_deterministic(val_set: CCT20Dataset) -> None:
    image = val_set.load_image(0)
    transform = EvalTransform()
    torch.manual_seed(0)
    first = transform(image)
    torch.manual_seed(1)
    random.seed(1)
    second = transform(image)
    assert torch.equal(first, second)
    assert torch.equal(val_set[0][0], val_set[0][0])


def test_train_transform_is_not_deterministic(train_set: CCT20Dataset) -> None:
    image = train_set.load_image(0)
    boxes = train_set.records[0].boxes
    transform = TrainTransform()
    torch.manual_seed(0)
    outputs = [transform(image, boxes) for _ in range(6)]
    assert all(not torch.equal(a, b) for i, a in enumerate(outputs) for b in outputs[i + 1 :])
    assert not torch.equal(train_set[0][0], train_set[0][0])


def test_train_transform_draws_from_torch_only(train_set: CCT20Dataset) -> None:
    """The same torch seed gives the same tensor whatever Python's `random` state is, so the
    run seed and the DataLoader's per-worker seeds govern the augmentation."""
    image = train_set.load_image(0)
    boxes = train_set.records[0].boxes
    transform = TrainTransform()
    torch.manual_seed(7)
    random.seed(1)
    first = transform(image, boxes)
    torch.manual_seed(7)
    random.seed(2)
    second = transform(image, boxes)
    assert torch.equal(first, second)


def test_eval_transform_takes_the_image_only() -> None:
    params = list(inspect.signature(EvalTransform.__call__).parameters)
    assert params == ["self", "image"]


class SpyEval(EvalTransform):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.calls.append((args, kwargs))
        (image,) = args
        assert isinstance(image, Image.Image)
        return super().__call__(image)


def test_eval_transform_never_receives_boxes(fixture: tuple[Split, Path]) -> None:
    split, root = fixture
    records = split.records["val"]
    assert any(r.boxes for r in records)
    spy = SpyEval()
    dataset = CCT20Dataset(records, root, spy)
    for index in range(len(dataset)):
        dataset[index]
    assert len(spy.calls) == len(records)
    for args, kwargs in spy.calls:
        assert len(args) == 1 and isinstance(args[0], Image.Image)
        assert kwargs == {}


class SpySampler(ControlCropSampler):
    def __init__(self) -> None:
        self.boxes: list[tuple[Box, ...]] = []

    def __call__(self, image: Image.Image, boxes: Sequence[Box]) -> Crop:
        self.boxes.append(tuple(boxes))
        return super().__call__(image, boxes)


def test_training_hands_each_records_boxes_to_the_crop_sampler(
    fixture: tuple[Split, Path],
) -> None:
    split, root = fixture
    records = split.records["train"]
    spy = SpySampler()
    dataset = CCT20Dataset(records, root, TrainTransform(sampler=spy))
    for index in range(len(dataset)):
        dataset[index]
    assert spy.boxes == [r.boxes for r in records]


def test_control_sampler_draws_crops_at_scale_065_to_1() -> None:
    """Area fraction within the plan's 0.65 to 1.0 (rounding to whole pixels allowed), aspect
    ratio within torchvision's 3/4 to 4/3, the crop inside the image, the boxes ignored."""
    width, height = 512, 373  # a 512 px cache image of a 2048 x 1494 CCT20 frame
    image = Image.new("RGB", (width, height))
    sampler = ControlCropSampler()
    box: Box = (0.1, 0.2, 0.3, 0.4)
    fractions = []
    torch.manual_seed(0)
    for _ in range(2000):
        top, left, h, w = sampler(image, ())
        assert 0 <= top and top + h <= height and 0 <= left and left + w <= width
        assert 3 / 4 - 0.02 <= w / h <= 4 / 3 + 0.02
        fractions.append(w * h / (width * height))
    assert CROP_SCALE[0] - 0.005 <= min(fractions) < 0.67
    assert 0.97 < max(fractions) <= CROP_SCALE[1]
    torch.manual_seed(3)
    without = sampler(image, ())
    torch.manual_seed(3)
    assert sampler(image, (box,)) == without


def test_eval_transform_resizes_the_shorter_side_then_centre_crops() -> None:
    """A 384 x 192 image, black left third: resized to 512 x 256 and centre-cropped at columns 144
    to 368, the edge lands at column 26.7 of the crop. A direct 224 x 224 resize would put it at
    74.7, a 256 x 256 resize and crop at 69.3."""
    image = Image.new("RGB", (384, 192), (255, 255, 255))
    image.paste((0, 0, 0), (0, 0, 128, 192))
    out = EvalTransform()(image)
    assert tuple(out.shape) == (3, INPUT_SIZE, INPUT_SIZE)
    assert EVAL_RESIZE == 256
    assert bool((out[:, :, 20] < 0).all())  # black, normalised
    assert bool((out[:, :, 33] > 0).all())  # white, normalised


def test_normalisation_uses_imagenet_mean_and_std() -> None:
    assert IMAGENET_MEAN == (0.485, 0.456, 0.406)
    assert IMAGENET_STD == (0.229, 0.224, 0.225)
    out = EvalTransform()(Image.new("RGB", (300, 200), (128, 128, 128)))
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        expected = torch.full((INPUT_SIZE, INPUT_SIZE), (128 / 255 - mean) / std)
        assert torch.allclose(out[channel], expected, atol=1e-5)


def test_training_transform_is_refused_outside_train(fixture: tuple[Split, Path]) -> None:
    split, root = fixture
    with pytest.raises(ValueError, match=r"^the training transform is for train records only"):
        CCT20Dataset(split.records["val"], root, TrainTransform())
    with pytest.raises(ValueError):
        CCT20Dataset(split.records["train"] + split.records["val"][:1], root, TrainTransform())
    compose = T.Compose([T.Resize(EVAL_RESIZE), T.CenterCrop(INPUT_SIZE), T.ToTensor()])
    with pytest.raises(TypeError):
        CCT20Dataset(split.records["val"], root, compose)


def test_unreadable_image_error_names_no_file(fixture: tuple[Split, Path], tmp_path: Path) -> None:
    """A missing or corrupt file raises ImageReadError with the item's position, class and split;
    neither the message nor a chained exception carries the file name (R03)."""
    split, _ = fixture
    records = split.records["val"][:2]
    (tmp_path / records[1].file_name).write_bytes(b"not a jpeg")
    dataset = CCT20Dataset(records, tmp_path, EvalTransform())
    for index in range(2):
        with pytest.raises(ImageReadError) as caught:
            dataset[index]
        message = str(caught.value)
        assert message.startswith(f"item {index} of 2 (")
        for r in records:
            assert r.file_name not in message and r.image_id not in message
        assert "syn-" not in message
        assert caught.value.__cause__ is None and caught.value.__suppress_context__


def test_dataset_survives_pickling(val_set: CCT20Dataset) -> None:
    """DataLoader workers started with spawn (macOS) receive the dataset pickled."""
    copy = pickle.loads(pickle.dumps(val_set))
    assert torch.equal(copy[0][0], val_set[0][0])
    assert copy[0][1] == val_set[0][1]
