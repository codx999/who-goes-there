"""trailcam/data/loaders.py: the DataLoaders for the five splits (plan.md sections 3 and 5; R12 the
train loader is built from the re-split train and sees no id of another split, R22 shuffle on
train only and the other loaders in record order).

The images are synthetic JPEGs written into a temporary directory (never committed), one per
record of every split of the synthetic metadata's split for seed 42, as in `tests/test_dataset.py`;
item 4.03 writes the fixture images for the dry run.
"""

import pickle
import random
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from trailcam.data import synthetic
from trailcam.data.classes import NUM_CLASSES
from trailcam.data.dataset import INPUT_SIZE, CCT20Dataset, EvalTransform, TrainTransform
from trailcam.data.loaders import EVAL_SPLITS, Loader, make_loaders
from trailcam.data.metadata import Metadata, Record, load_metadata
from trailcam.data.split import POOLED_FILES, SPLIT_NAMES, Split, make_split, split_hash

SEED = 42
BATCH = 16


def write_jpegs(records: Sequence[Record], root: Path, seed: int = 0) -> None:
    rng = random.Random(seed)
    root.mkdir(parents=True, exist_ok=True)
    for r in records:
        mode, bands = ("L", 1) if r.night else ("RGB", 3)
        noise = rng.randbytes(r.width * r.height * bands)
        Image.frombytes(mode, (r.width, r.height), noise).save(root / r.file_name, quality=90)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> tuple[Split, Path, Metadata]:
    """The split, the image folder and the loaded metadata."""
    tmp = tmp_path_factory.mktemp("loaders")
    paths = synthetic.write_synthetic_metadata(tmp / "meta")
    meta = load_metadata(tmp / "meta", paths["sidecar"])
    split = make_split(meta, SEED)
    write_jpegs([r for name in SPLIT_NAMES for r in split.records[name]], tmp / "images")
    return split, tmp / "images", meta


@pytest.fixture(scope="module")
def loaders(fixture: tuple[Split, Path, Metadata]) -> Mapping[str, Loader]:
    split, root, _ = fixture
    return make_loaders(split, root, BATCH, num_workers=0, pin_memory=False)


def dataset_of(loader: Loader) -> CCT20Dataset:
    assert isinstance(loader.dataset, CCT20Dataset)
    return loader.dataset


def test_one_loader_per_split_in_order(loaders: Mapping[str, Loader]) -> None:
    assert tuple(loaders) == SPLIT_NAMES
    assert EVAL_SPLITS == ("val", "trans_val", "cis_test", "trans_test")
    assert isinstance(loaders, MappingProxyType)


@pytest.mark.parametrize("name", SPLIT_NAMES)
def test_a_batch_is_n_by_3_by_224_by_224(name: str, loaders: Mapping[str, Loader]) -> None:
    torch.manual_seed(0)
    images, labels = next(iter(loaders[name]))
    assert tuple(images.shape) == (BATCH, 3, INPUT_SIZE, INPUT_SIZE)
    assert images.dtype == torch.float32
    assert tuple(labels.shape) == (BATCH,)
    assert labels.dtype == torch.int64
    assert bool(((labels >= 0) & (labels < NUM_CLASSES)).all())


def test_train_shuffles_and_every_other_loader_is_sequential(
    loaders: Mapping[str, Loader],
) -> None:
    train = loaders["train"]
    assert type(train.sampler) is RandomSampler
    assert train.drop_last
    for name in EVAL_SPLITS:
        assert type(loaders[name].sampler) is SequentialSampler, name
        assert not loaders[name].drop_last, name


def test_train_loader_is_built_from_the_re_split_train(
    fixture: tuple[Split, Path, Metadata], loaders: Mapping[str, Loader]
) -> None:
    """R12: the train loader holds `split.records["train"]` itself: not the pooled official
    train and cis_val, not the official train file, and no image id of another split. The
    dataset's own check (`split == "train"`) cannot tell the re-split train from the official
    train file, whose records carry the same split name."""
    split, _, meta = fixture
    train = dataset_of(loaders["train"])
    assert isinstance(train.transform, TrainTransform)
    assert train.records == split.records["train"]
    assert all(a is b for a, b in zip(train.records, split.records["train"], strict=True))

    ids = [r.image_id for r in train.records]
    assert len(set(ids)) == len(ids)
    for name in EVAL_SPLITS:
        assert not set(ids) & {r.image_id for r in split.records[name]}, name

    official_train = {r.image_id for r in meta.records["train"]}
    pooled = {r.image_id for name in POOLED_FILES for r in meta.records[name]}
    val = {r.image_id for r in split.records["val"]}
    assert all(r.split == "train" for r in meta.records["train"])  # what the dataset check sees
    assert set(ids) != official_train  # the re-split moved whole bursts between the files
    assert set(ids) == pooled - val
    assert len(ids) < len(pooled)


def test_eval_loaders_hold_their_split_with_the_eval_transform(
    fixture: tuple[Split, Path, Metadata], loaders: Mapping[str, Loader]
) -> None:
    split, _, _ = fixture
    for name in EVAL_SPLITS:
        dataset = dataset_of(loaders[name])
        assert type(dataset.transform) is EvalTransform, name
        assert dataset.records == split.records[name], name


@pytest.mark.parametrize("name", EVAL_SPLITS)
def test_eval_loaders_keep_record_order_and_every_image(
    name: str, fixture: tuple[Split, Path, Metadata], loaders: Mapping[str, Loader]
) -> None:
    """R22: predictions line up with the records, so labels come out in record order, the last
    short batch included, and two passes give the same tensors."""
    split, _, _ = fixture
    records = split.records[name]
    assert len(records) % BATCH  # the last batch is short, and kept
    first = list(loaders[name])
    second = list(loaders[name])
    labels = torch.cat([batch[1] for batch in first]).tolist()
    assert labels == [r.label for r in records]
    for (a, _), (b, _) in zip(first, second, strict=True):
        assert torch.equal(a, b)


def test_train_epoch_is_a_permutation_of_the_train_split(loaders: Mapping[str, Loader]) -> None:
    """Every train image is drawn once per epoch in a new order; only the last incomplete batch
    is dropped."""
    train = loaders["train"]
    n = len(dataset_of(train))
    torch.manual_seed(0)
    first = list(train.sampler)
    second = list(train.sampler)
    assert sorted(first) == sorted(second) == list(range(n))
    assert first != second
    assert first != sorted(first)
    assert len(train) == n // BATCH


def test_generator_makes_the_train_order_and_augmentation_reproducible(
    fixture: tuple[Split, Path, Metadata],
) -> None:
    split, root, _ = fixture

    def first_batch(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator().manual_seed(seed)
        loader = make_loaders(
            split, root, BATCH, num_workers=0, pin_memory=False, generator=generator
        )["train"]
        torch.manual_seed(seed)
        images, labels = next(iter(loader))
        return images, labels

    a, b, c = first_batch(1), first_batch(1), first_batch(2)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert not torch.equal(a[0], c[0])


def test_train_transform_argument_reaches_the_train_loader_only(
    fixture: tuple[Split, Path, Metadata],
) -> None:
    split, root, _ = fixture
    transform = TrainTransform()
    built = make_loaders(
        split, root, BATCH, num_workers=0, pin_memory=False, train_transform=transform
    )
    assert dataset_of(built["train"]).transform is transform
    for name in EVAL_SPLITS:
        assert type(dataset_of(built[name]).transform) is EvalTransform


def test_eval_transform_argument_reaches_the_eval_loaders_only(
    fixture: tuple[Split, Path, Metadata],
) -> None:
    """Item 13.02: a phase-2 input size reaches the four eval splits through one eval transform,
    and a training transform of another size is refused, so train and eval never differ in it."""
    split, root, _ = fixture
    train_transform, eval_transform = TrainTransform(size=320), EvalTransform(320)
    built = make_loaders(
        split,
        root,
        BATCH,
        num_workers=0,
        pin_memory=False,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )
    assert dataset_of(built["train"]).transform is train_transform
    for name in EVAL_SPLITS:
        assert dataset_of(built[name]).transform is eval_transform
    images, _ = next(iter(built["val"]))
    assert tuple(images.shape[1:]) == (3, 320, 320)
    default = make_loaders(split, root, BATCH, num_workers=0, pin_memory=False)
    assert all(dataset_of(default[n]).transform.size == 224 for n in SPLIT_NAMES)
    with pytest.raises(ValueError, match="outputs 320 px, the eval transform 224 px"):
        make_loaders(
            split, root, BATCH, num_workers=0, pin_memory=False, train_transform=train_transform
        )


@pytest.mark.parametrize("pin_memory", [False, True])
def test_num_workers_and_pin_memory_are_passed_through(
    pin_memory: bool, fixture: tuple[Split, Path, Metadata]
) -> None:
    split, root, _ = fixture
    built = make_loaders(split, root, BATCH, num_workers=3, pin_memory=pin_memory)
    for name in SPLIT_NAMES:
        assert built[name].num_workers == 3
        assert built[name].pin_memory is pin_memory
        assert built[name].persistent_workers
        assert built[name].batch_size == BATCH


def test_spawned_workers_give_the_same_batches(
    fixture: tuple[Split, Path, Metadata], loaders: Mapping[str, Loader]
) -> None:
    """Workers started with spawn, the macOS default, forced here so Linux CI runs it too: the
    dataset reaches the two workers pickled and the val batches, split between them, come back
    in record order and equal to the single-process ones. Each spawned worker imports torch
    (seconds), so only the val loader is iterated; the train dataset's pickling is checked
    directly below."""
    split, root, _ = fixture
    val = make_loaders(split, root, BATCH, num_workers=2, pin_memory=False)["val"]
    val.multiprocessing_context = "spawn"
    batches = list(val)
    assert len(batches) == len(loaders["val"]) > 2
    for (a, la), (b, lb) in zip(batches, loaders["val"], strict=True):
        assert torch.equal(a, b) and torch.equal(la, lb)


def test_train_dataset_survives_pickling(loaders: Mapping[str, Loader]) -> None:
    """What a spawned train worker receives: the copy draws the same augmentation for a seed."""
    train = dataset_of(loaders["train"])
    copy = pickle.loads(pickle.dumps(train))
    assert copy.records == train.records
    torch.manual_seed(5)
    a, label_a = copy[3]
    torch.manual_seed(5)
    b, label_b = train[3]
    assert torch.equal(a, b) and label_a == label_b


def test_records_instead_of_a_split_are_refused(fixture: tuple[Split, Path, Metadata]) -> None:
    split, root, _ = fixture
    with pytest.raises(TypeError, match="Split"):
        make_loaders(split.records, root, BATCH, num_workers=0, pin_memory=False)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("batch_size", "num_workers", "message"),
    [(0, 0, "batch_size"), (BATCH, -1, "num_workers"), (10_000, 0, "fewer than one batch")],
)
def test_bad_arguments_are_refused(
    batch_size: int, num_workers: int, message: str, fixture: tuple[Split, Path, Metadata]
) -> None:
    split, root, _ = fixture
    with pytest.raises(ValueError, match=message):
        make_loaders(split, root, batch_size, num_workers=num_workers, pin_memory=False)


@pytest.mark.parametrize("name", EVAL_SPLITS)
def test_a_train_id_in_another_split_is_refused_with_counts_only(
    name: str, fixture: tuple[Split, Path, Metadata]
) -> None:
    """R12, R16: a Split assembled by hand that puts a train image in another split, with its
    hash recomputed so the hash check passes, is refused before any loader exists; the message
    names the split and a count, never the id."""
    split, root, _ = fixture
    leaked = replace(split.records["train"][0], split=name)
    records = dict(split.records)
    records[name] = records[name] + (leaked,)
    tampered = replace(split, records=MappingProxyType(records), hash=split_hash(records))
    with pytest.raises(
        ValueError, match=rf"^train image ids also appear in other splits: {name}=1$"
    ):
        make_loaders(tampered, root, BATCH, num_workers=0, pin_memory=False)


def test_a_split_that_no_longer_matches_its_hash_is_refused(
    fixture: tuple[Split, Path, Metadata],
) -> None:
    """R12, R16: a filter after the split that moves a cis_test image into train, relabelled so
    the dataset's `split == "train"` check passes and no id is in two splits, is refused because
    the records no longer match `split.hash`. The message carries counts only."""
    split, root, _ = fixture
    moved = split.records["cis_test"][0]
    records = dict(split.records)
    records["cis_test"] = records["cis_test"][1:]
    records["train"] = records["train"] + (replace(moved, split="train"),)
    tampered = replace(split, records=MappingProxyType(records))
    with pytest.raises(
        ValueError, match=r"^the split's records no longer match its hash"
    ) as caught:
        make_loaders(tampered, root, BATCH, num_workers=0, pin_memory=False)
    message = str(caught.value)
    assert moved.image_id not in message and moved.file_name not in message
    assert f"train={len(records['train'])}" in message
