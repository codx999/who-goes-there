"""trailcam/data/copypaste.py: arm C's background pool and paste step (item 13.04;
docs/phase2-plan.md section 6, step 1; R23 boxes never reach the eval transform, R25 the arms
differ only in the sampler, R28 the pool's rules and arm C's isolation from evaluation).

The pool rules run on the synthetic metadata's split for seed 42 with empty frames planted by hand,
each breaking one rule and passing the others, and on the real metadata in `TRAILCAM_DATA_ROOT`
(check.sh sets it locally; CI has no data and skips; anywhere else a missing root fails), counts
and the hash only. The paste runs on blank and solid-colour images in memory and on the synthetic
fixture's JPEGs, written into a temporary directory (never committed).
"""

import ast
import inspect
import math
import os
import pickle
import random
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.copypaste import (
    EVAL_SPLITS,
    PASTE_P,
    POOL_FILES,
    BackgroundPool,
    CopyPaste,
    PoolError,
    background_pool,
    check_pool,
    paste_box,
    pool_hash,
)
from trailcam.data.dataset import (
    CCT20Dataset,
    ControlCropSampler,
    EvalTransform,
    TrainTransform,
)
from trailcam.data.metadata import SPLITS, Box, EmptyFrame, Metadata, load_metadata
from trailcam.data.split import SPLIT_NAMES, Split, make_split
from trailcam.train import train
from trailcam.train.config import load_config

SEED = 42
SAMPLES = 1000
REPO = Path(__file__).resolve().parents[1]
REAL_ROOT = os.environ.get("TRAILCAM_DATA_ROOT", "")

# The real pool for split seed 42 (item 13.04): official cis_val's 995 empty frames, every one at a
# train location and in no evaluation sequence; the night flags are the cache sidecar's.
REAL_POOL = {"frames": 995, "day": 531, "night": 464, "locations": 10}
REAL_POOL_HASH = "87c5d379bfad28f7c5ede2d5ab8093dc7299fd6143782d2616db223e86c1d8bb"


def real_root() -> Path:
    """The real metadata root. CI has no data, so the real-data tests skip there; anywhere else a
    missing root fails (review 3.03)."""
    if REAL_ROOT:
        return Path(REAL_ROOT)
    if os.environ.get("CI"):
        pytest.skip("TRAILCAM_DATA_ROOT is not set (CI has no data)")
    pytest.fail("TRAILCAM_DATA_ROOT is not set; bash scripts/check.sh sets it", pytrace=False)


# --- the synthetic fixture with planted empty frames ---


def frame(
    name: str, split: str, location: int, seq_id: str, night: bool | None = False
) -> EmptyFrame:
    return EmptyFrame(f"plant-{name}", f"plant-{name}.jpg", split, location, seq_id, night)


def plant(meta: Metadata, split: Split) -> tuple[Metadata, set[str], dict[str, str]]:
    """`meta` with planted empty frames added to its files: the metadata, the image ids the pool
    must hold, and each excluded frame's id with the one rule that excludes it."""
    train_101 = next(r for r in split.records["train"] if r.location == 101)
    train_102 = next(r for r in split.records["train"] if r.location == 102)
    val = split.records["val"][0]
    cis_test = split.records["cis_test"][0]
    trans_val = split.records["trans_val"][0]
    trans_test = split.records["trans_test"][0]
    kept = [
        frame("keep-cis-val", "cis_val", 101, "plant-seq-a", night=False),
        frame("keep-train-file", "train", 102, "plant-seq-b", night=True),
        frame("keep-train-burst", "cis_val", 102, train_102.seq_id, night=False),
        frame("keep-train-burst-101", "cis_val", 101, train_101.seq_id, night=True),
    ]
    excluded = {
        # The sequence rule alone: a pooled file, a train location, a night flag.
        "val-burst": (frame("val-burst", "cis_val", val.location, val.seq_id), "sequence"),
        "cis-test-burst": (
            frame("cis-test-burst", "cis_val", cis_test.location, cis_test.seq_id),
            "sequence",
        ),
        "trans-val-burst": (frame("trans-val-burst", "cis_val", 101, trans_val.seq_id), "sequence"),
        "trans-test-burst": (
            frame("trans-test-burst", "train", 102, trans_test.seq_id),
            "sequence",
        ),
        # The location rule alone: a pooled file, a sequence of no split.
        "trans-location": (frame("trans-location", "cis_val", 201, "plant-seq-c"), "location"),
        "unseen-location": (frame("unseen-location", "cis_val", 999, "plant-seq-d"), "location"),
        # The file rule alone: a train location, a sequence of no split.
        "cis-test-file": (frame("cis-test-file", "cis_test", 101, "plant-seq-e"), "file"),
        "trans-val-file": (frame("trans-val-file", "trans_val", 102, "plant-seq-f"), "file"),
        "trans-test-file": (frame("trans-test-file", "trans_test", 101, "plant-seq-g"), "file"),
    }
    added = [*kept, *(f for f, _ in excluded.values())]
    empties = {s: (*meta.empties[s], *(f for f in added if f.split == s)) for s in SPLITS}
    planted = replace(meta, empties=MappingProxyType(empties))
    expected = {f.image_id for f in meta.empties["cis_val"]} | {f.image_id for f in kept}
    return planted, expected, {f"plant-{k}": rule for k, (_, rule) in excluded.items()}


def write_jpegs(frames: Sequence[EmptyFrame], root: Path) -> None:
    """Noise JPEGs at the synthetic size under each frame's `file_name`."""
    rng = random.Random(0)
    root.mkdir(parents=True, exist_ok=True)
    for f in frames:
        noise = rng.randbytes(synthetic.WIDTH * synthetic.HEIGHT * 3)
        image = Image.frombytes("RGB", (synthetic.WIDTH, synthetic.HEIGHT), noise)
        image.save(root / f.file_name, quality=90)


class Fixture:
    """The synthetic metadata (with the sidecar), its split for SEED, the planted metadata and
    the JPEG folder holding every image of the five files and every planted frame."""

    def __init__(self, tmp: Path) -> None:
        written = synthetic.write_synthetic_fixture(tmp / "fixture")
        self.root = written.image_root
        self.meta = load_metadata(tmp / "fixture", written.paths["sidecar"])
        self.split = make_split(self.meta, SEED)
        self.planted, self.expected, self.excluded = plant(self.meta, self.split)
        planted_frames = [f for s in SPLITS for f in self.planted.empties[s]]
        write_jpegs([f for f in planted_frames if f.image_id.startswith("plant-")], self.root)


@pytest.fixture(scope="module")
def fx(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return Fixture(tmp_path_factory.mktemp("copypaste"))


# --- the pool rules (R28) ---


def test_the_pool_keeps_exactly_the_frames_its_rules_allow(fx: Fixture) -> None:
    """The synthetic metadata's cis_val empties and the planted frames that break no rule: from a
    pooled file, at a train location, in no evaluation sequence (a train burst's frame stays)."""
    pool = background_pool(fx.planted, fx.split)
    assert {f.image_id for f in pool.frames} == fx.expected
    assert len(pool.frames) == len(fx.expected) == 3 + 4
    assert [f.image_id for f in pool.frames] == sorted(fx.expected)
    assert not fx.expected & set(fx.excluded)


def test_each_planted_frame_breaks_one_rule_and_passes_the_others(fx: Fixture) -> None:
    """So each exclusion above is that rule's work: drop the rule and its frame would enter."""
    train_locations = {r.location for r in fx.split.records["train"]}
    eval_sequences = {r.seq_id for n in EVAL_SPLITS for r in fx.split.records[n]}
    frames = {f.image_id: f for s in SPLITS for f in fx.planted.empties[s]}
    for image_id, rule in fx.excluded.items():
        f = frames[image_id]
        broken = {
            "file": f.split not in POOL_FILES,
            "location": f.location not in train_locations,
            "sequence": f.seq_id in eval_sequences,
        }
        assert [name for name, yes in broken.items() if yes] == [rule], image_id
        assert f.night is not None
    assert Counter(fx.excluded.values()) == {"sequence": 4, "location": 2, "file": 3}


def test_an_eval_sequences_empty_frame_is_excluded_though_otherwise_eligible(fx: Fixture) -> None:
    """Val shares its cameras with train: an empty frame of a val burst is from cis_val, at a train
    location, with a night flag, and still never a background; nor one of a test burst."""
    val_seq = {r.seq_id for r in fx.split.records["val"]}
    burst = next(f for f in fx.planted.empties["cis_val"] if f.image_id == "plant-val-burst")
    assert burst.seq_id in val_seq
    assert burst.location in {r.location for r in fx.split.records["train"]}
    pool = background_pool(fx.planted, fx.split)
    ids = {f.image_id for f in pool.frames}
    assert not ids & {"plant-val-burst", "plant-cis-test-burst", "plant-trans-val-burst"}
    assert "plant-trans-test-burst" not in ids


def test_every_background_is_at_a_train_location_and_in_no_evaluation_split(fx: Fixture) -> None:
    """The plan's tests: every background's location is a train location; no background's image
    id or sequence is in val, cis_test, trans_val or trans_test; none is from an evaluation file
    (cis_test, trans_val, trans_test), whose empties are evaluation frames even at a training
    camera."""
    pool = background_pool(fx.planted, fx.split)
    records = fx.split.records
    train_locations = {r.location for r in records["train"]}
    eval_locations = {r.location for n in EVAL_SPLITS for r in records[n]} - train_locations
    for f in pool.frames:
        assert f.location in train_locations and f.location not in eval_locations
        assert f.split in POOL_FILES == ("train", "cis_val")
        for name in EVAL_SPLITS:
            assert f.seq_id not in {r.seq_id for r in records[name]}, name
        for name in SPLIT_NAMES:
            assert f.image_id not in {r.image_id for r in records[name]}, name
    assert EVAL_SPLITS == ("val", "trans_val", "cis_test", "trans_test")


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (lambda fx: frame("x", "cis_test", 101, "plant-seq-x"), "from a file other than"),
        (lambda fx: frame("x", "trans_test", 101, "plant-seq-x"), "from a file other than"),
        (lambda fx: frame("x", "cis_val", 201, "plant-seq-x"), "location of an evaluation split"),
        (lambda fx: frame("x", "cis_val", 999, "plant-seq-x"), "not at a train location"),
        (
            lambda fx: frame("x", "cis_val", 101, fx.split.records["val"][0].seq_id),
            "1 frames in a sequence with an image in val",
        ),
        (
            lambda fx: frame("x", "cis_val", 101, fx.split.records["trans_test"][0].seq_id),
            "1 frames in a sequence with an image in val",
        ),
        (
            lambda fx: replace(
                fx.meta.empties["cis_val"][0], image_id=fx.split.records["train"][0].image_id
            ),
            "1 frames with the image id of a split's image",
        ),
        (lambda fx: fx.meta.empties["cis_val"][0], "1 frames repeated"),
        (lambda fx: frame("x", "cis_val", 101, "plant-seq-x", night=None), "without a day/night"),
    ],
)
def test_check_pool_refuses_each_broken_rule(fx: Fixture, bad: object, message: str) -> None:
    """The independent re-check: one bad frame beside good ones is refused, with counts only."""
    good = list(fx.meta.empties["cis_val"])
    check_pool(good, fx.split)
    extra = bad(fx)  # type: ignore[operator]
    with pytest.raises(PoolError, match=message) as error:
        check_pool([*good, extra], fx.split)
    text = str(error.value)
    assert "plant-" not in text and "syn-" not in text and ".jpg" not in text


def test_the_pool_is_built_from_make_splits_output_only(fx: Fixture) -> None:
    """A split edited after `make_split` (a val image moved into train) no longer matches its hash
    and is refused before any frame is chosen."""
    records = dict(fx.split.records)
    moved = replace(records["val"][0], split="train")
    records["train"] = (*records["train"], moved)
    records["val"] = records["val"][1:]
    edited = replace(fx.split, records=MappingProxyType(records))
    with pytest.raises(PoolError, match="no longer match its hash"):
        background_pool(fx.planted, edited)


def test_frames_without_a_night_flag_are_refused(fx: Fixture, tmp_path: Path) -> None:
    """Metadata loaded without the sidecar has no night flags, so day/night matching is
    impossible: the pool refuses it instead of matching on None."""
    paths = synthetic.write_synthetic_metadata(tmp_path / "meta")
    unflagged = load_metadata(tmp_path / "meta")
    assert paths["sidecar"].is_file()
    with pytest.raises(PoolError, match=r"3 frames without a day/night flag"):
        background_pool(unflagged, make_split(unflagged, SEED))


def test_the_pool_is_a_function_of_the_metadata_and_the_split_seed(fx: Fixture) -> None:
    """Same metadata and seed, same pool and hash, whatever the order of the empty frames; the
    hash covers the image ids, locations and night flags. Another split seed moves the val burst
    the planted frame belongs to, and the pool follows."""
    pool = background_pool(fx.planted, fx.split)
    shuffled = {s: tuple(reversed(fx.planted.empties[s])) for s in SPLITS}
    again = background_pool(replace(fx.planted, empties=MappingProxyType(shuffled)), fx.split)
    assert again == pool and again.hash == pool.hash == pool_hash(pool.frames)
    assert pool_hash(reversed(pool.frames)) == pool.hash
    flipped = replace(pool.frames[0], night=not pool.frames[0].night)
    assert pool_hash((flipped, *pool.frames[1:])) != pool.hash
    hashes = {pool.hash}
    for seed in range(5):
        other = make_split(fx.planted, seed)
        hashes.add(background_pool(fx.planted, other).hash)
    assert len(hashes) > 1


def test_the_pool_shows_counts_and_its_hash_only(fx: Fixture) -> None:
    pool = background_pool(fx.planted, fx.split)
    day = sum(1 for f in pool.frames if not f.night)
    assert (pool.day, pool.night) == (day, len(pool.frames) - day)
    assert repr(pool) == f"BackgroundPool(day={pool.day}, night={pool.night}, hash={pool.hash})"
    assert pool.summary() == (
        f"background pool (arm C, training only): {len(pool.frames)} empty frames, "
        f"day {pool.day}, night {pool.night}, hash {pool.hash}"
    )
    for text in (repr(pool), pool.summary(), str(pool)):
        assert "plant-" not in text and "syn-" not in text and ".jpg" not in text
    assert len(pool.hash) == 64 and int(pool.hash, 16) >= 0


def test_the_real_pool_follows_its_rules_and_is_pinned() -> None:
    """On the real metadata and the cache's sidecar (counts only): official cis_val's 995 empty
    frames, all at train locations and in no evaluation sequence; none of the 1,180 cis_test,
    61 trans_val or 1,778 trans_test empties. Every boxed train image has backgrounds of its
    day/night flag at other locations, so arm C never falls back for want of one."""
    root = real_root()
    meta = load_metadata(root, root / "day_night.json")
    split = make_split(meta, SEED)
    pool = background_pool(meta, split)
    records = split.records
    train_locations = {r.location for r in records["train"]}
    eval_sequences = {r.seq_id for n in EVAL_SPLITS for r in records[n]}
    split_ids = {r.image_id for n in SPLIT_NAMES for r in records[n]}
    counts = {
        "frames": len(pool.frames),
        "day": pool.day,
        "night": pool.night,
        "locations": len({f.location for f in pool.frames}),
    }
    assert counts == REAL_POOL
    assert pool.hash == REAL_POOL_HASH
    assert sum(1 for f in pool.frames if f.split != "cis_val") == 0
    assert sum(1 for f in pool.frames if f.location not in train_locations) == 0
    assert sum(1 for f in pool.frames if f.seq_id in eval_sequences) == 0
    assert sum(1 for f in pool.frames if f.image_id in split_ids) == 0
    assert {s: len(meta.empties[s]) for s in SPLITS} == {
        "train": 0,
        "cis_val": 995,
        "trans_val": 61,
        "cis_test": 1180,
        "trans_test": 1778,
    }
    by = Counter((f.night, f.location) for f in pool.frames)
    total = Counter(f.night for f in pool.frames)
    boxed = [r for r in records["train"] if r.boxes]
    fewest = min(total[r.night] - by[(r.night, r.location)] for r in boxed)
    assert (len(boxed), fewest) == (11633, 350)


# --- the paste geometry ---


def solid(size: tuple[int, int], colour: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", size, colour)


def noise(size: tuple[int, int]) -> Image.Image:
    """Colour noise, so a pasted patch's pixels show exactly where they came from."""
    width, height = size
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))


def rounded(box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x, y, w, h = box
    return (
        max(math.floor(x * width), 0),
        max(math.floor(y * height), 0),
        min(math.ceil((x + w) * width), width),
        min(math.ceil((y + h) * height), height),
    )


# Source size, box, background size. The cache's frames are 512 x 373 or 512 x 384.
GEOMETRY = {
    "centre, same size": ((512, 373), (0.4, 0.4, 0.15, 0.2), (512, 373)),
    "top-left corner": ((512, 373), (0.0, 0.0, 0.1, 0.1), (512, 373)),
    "bottom-right corner": ((512, 373), (0.9, 0.9, 0.1, 0.1), (512, 373)),
    "touching the right edge": ((512, 373), (0.7, 0.3, 0.3, 0.2), (512, 373)),
    "tall, over half the height": ((512, 373), (0.3, 0.1, 0.2, 0.8), (512, 373)),
    "the whole frame": ((512, 373), (0.0, 0.0, 1.0, 1.0), (512, 373)),
    "small, off the pixel grid": ((512, 373), (0.3331, 0.6667, 0.0123, 0.0171), (512, 373)),
    "under a pixel": ((512, 373), (0.5003, 0.5003, 0.0005, 0.0005), (512, 373)),
    "373 onto 384 rows": ((512, 373), (0.2, 0.5, 0.3, 0.25), (512, 384)),
    "384 onto 373 rows": ((512, 384), (0.1, 0.05, 0.5, 0.9), (512, 373)),
    "synthetic size": ((96, 72), (12 / 96, 10 / 72, 30 / 96, 20 / 72), (96, 72)),
}


@pytest.mark.parametrize(("size", "box", "back"), GEOMETRY.values(), ids=GEOMETRY.keys())
def test_the_paste_lies_inside_the_frame_at_the_boxs_size(
    size: tuple[int, int], box: Box, back: tuple[int, int]
) -> None:
    """Over 300 draws per case: the composite is the background's size; the pasted rectangle is
    the box rounded out to whole pixels, scaled by the frames' width and height ratios (the same
    fraction of the frame), wholly inside the frame with its bottom edge in the lower half. On
    every tenth draw, its pixels are the image's box region and every other pixel is the
    background's."""
    image = noise(size)
    background = solid(back, (10, 200, 30))
    left, top, right, bottom = rounded(box, size)
    w = min(max(round((right - left) * back[0] / size[0]), 1), back[0])
    h = min(max(round((bottom - top) * back[1] / size[1]), 1), back[1])
    patch = image.crop((left, top, right, bottom))
    patch = patch if patch.size == (w, h) else patch.resize((w, h), Image.Resampling.BILINEAR)
    expected_patch = np.asarray(patch)
    plain = np.asarray(background)
    torch.manual_seed(0)
    for draw in range(300):
        composite, (pasted,) = paste_box(image, box, background)
        assert composite.size == back and composite.mode == "RGB"
        x, y = round(pasted[0] * back[0]), round(pasted[1] * back[1])
        assert pasted == (x / back[0], y / back[1], w / back[0], h / back[1])
        assert 0 <= x and x + w <= back[0] and 0 <= y and y + h <= back[1]
        assert y + h >= back[1] / 2
        if draw % 10:
            continue  # every pixel on every tenth draw
        pixels = np.asarray(composite)
        assert np.array_equal(pixels[y : y + h, x : x + w], expected_patch)
        outside = np.ones(pixels.shape[:2], dtype=bool)
        outside[y : y + h, x : x + w] = False
        assert np.array_equal(pixels[outside], plain[outside])
    # The fraction of the frame: within half a pixel a side of the rounded-out box's.
    assert abs(w / back[0] - (right - left) / size[0]) <= 0.5 / back[0] + 1e-12 or w == 1
    assert abs(h / back[1] - (bottom - top) / size[1]) <= 0.5 / back[1] + 1e-12 or h == 1
    assert background.getpixel((0, 0)) == (10, 200, 30)  # the background is not changed


def test_the_position_is_uniform_with_the_bottom_edge_in_the_lower_half() -> None:
    """Mesnage et al. 2025 (section 2.4): a random x and y with the bottom of the paste in the
    bottom half. x spans the frame, the bottom edge spans the lower half, both about uniformly; a
    paste taller than half the frame may start at the top."""
    image, background = solid((512, 373), (0, 0, 0)), solid((512, 373), (255, 255, 255))
    box: Box = (0.4, 0.4, 0.1, 0.1)  # 52 x 38 px with the rounding out
    torch.manual_seed(1)
    places = [paste_box(image, box, background)[1][0] for _ in range(SAMPLES)]
    xs = [round(p[0] * 512) for p in places]
    bottoms = [round(p[1] * 373) + round(p[3] * 373) for p in places]
    w = round(places[0][2] * 512)
    assert min(xs) <= 5 and max(xs) >= 512 - w - 5
    assert 187 <= min(bottoms) <= 190 and 370 <= max(bottoms) <= 373  # ceil(373 / 2) = 187
    assert abs(sum(xs) / SAMPLES - (512 - w) / 2) < 15
    assert abs(sum(bottoms) / SAMPLES - (187 + 373) / 2) < 8
    tall: Box = (0.3, 0.1, 0.2, 0.8)  # 299 px of 373
    tops = [round(paste_box(image, tall, background)[1][0][1] * 373) for _ in range(200)]
    assert min(tops) == 0 and max(tops) == 373 - 299


def test_the_box_is_a_fraction_and_one_in_pixels_is_refused() -> None:
    """As arm B (R24): a box is a fraction of the image, scaled once by `to_pixels`."""
    image, background = solid((512, 373), (0, 0, 0)), solid((512, 373), (255, 255, 255))
    for pixels in [(100.0, 50.0, 40.0, 30.0), (0.5, 0.5, 40.0, 30.0)]:
        with pytest.raises(ValueError, match=r"^a box must be fractions of the image"):
            paste_box(image, pixels, background)


# --- the paste step: draws, day/night, location, probability ---


LOCATIONS = (1, 2, 3)


def hand_pool(root: Path, locations: Sequence[int] = LOCATIONS) -> BackgroundPool:
    """Two day and two night frames per location, as JPEGs under `root` (512 x 373, a solid
    colour per location and flag), built by hand to drive the draws."""
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for location in locations:
        for night in (False, True):
            for k in range(2):
                name = f"hand-{location}-{'n' if night else 'd'}{k}"
                f = EmptyFrame(name, f"{name}.jpg", "cis_val", location, f"seq-{name}", night)
                grey = 40 * location
                colour = (grey, grey, grey) if night else (grey, 255 - grey, 90)
                solid((512, 373), colour).save(root / f.file_name, quality=95)
                frames.append(f)
    return BackgroundPool(frames=tuple(frames), hash=pool_hash(frames))


class SpyPaste(CopyPaste):
    """Records every background the step loads."""

    def __init__(self, pool: BackgroundPool, root: Path, p: float) -> None:
        super().__init__(pool, root, p)
        self.loaded: list[EmptyFrame] = []

    def load_background(self, frame: EmptyFrame) -> Image.Image:
        self.loaded.append(frame)
        return super().load_background(frame)


BOX: Box = (0.4, 0.4, 0.15, 0.2)


def test_the_background_has_the_images_day_night_flag(tmp_path: Path) -> None:
    """Night onto night, day onto day (the sidecar's flags), for every location."""
    pool = hand_pool(tmp_path)
    spy = SpyPaste(pool, tmp_path, p=1.0)
    image = solid((512, 373), (1, 2, 3))
    torch.manual_seed(2)
    for night in (False, True):
        for location in LOCATIONS:
            spy.loaded.clear()
            for _ in range(100):
                spy(image, (BOX,), night, location)
            assert len(spy.loaded) == 100
            assert {f.night for f in spy.loaded} == {night}


def test_the_background_comes_from_another_location_uniformly(tmp_path: Path) -> None:
    """Never the image's own camera; every other frame of its flag about equally often."""
    pool = hand_pool(tmp_path)
    spy = SpyPaste(pool, tmp_path, p=1.0)
    image = solid((512, 373), (1, 2, 3))
    torch.manual_seed(3)
    for _ in range(800):
        spy(image, (BOX,), True, 2)
    assert {f.location for f in spy.loaded} == {1, 3}
    counts = Counter(f.image_id for f in spy.loaded)
    assert len(counts) == 4 and min(counts.values()) > 150
    assert spy.candidates(True, 2) == tuple(f for f in pool.frames if f.night and f.location != 2)
    assert spy.candidates(False, 9) == tuple(f for f in pool.frames if not f.night)


def test_no_background_of_the_flag_elsewhere_leaves_the_image_as_it_is(tmp_path: Path) -> None:
    """A pool with one location only: its own images get no background and stay unchanged."""
    pool = hand_pool(tmp_path, locations=(1,))
    spy = SpyPaste(pool, tmp_path, p=1.0)
    image = solid((512, 373), (1, 2, 3))
    out, boxes = spy(image, (BOX,), False, 1)
    assert out is image and boxes == (BOX,) and spy.loaded == []
    assert spy.draw_background(False, 1) is None
    out, boxes = spy(image, (BOX,), False, 2)
    assert out is not image and len(spy.loaded) == 1


def test_about_half_of_the_boxed_images_are_pasted(tmp_path: Path) -> None:
    spy = SpyPaste(hand_pool(tmp_path), tmp_path, p=PASTE_P)
    image = solid((512, 373), (1, 2, 3))
    torch.manual_seed(4)
    pasted = 0
    for _ in range(SAMPLES):
        out, boxes = spy(image, (BOX,), False, 1)
        pasted += out is not image
        assert (out is image) == (boxes == (BOX,))
    assert PASTE_P == 0.5
    assert 430 < pasted < 570 and pasted == len(spy.loaded)


def test_images_without_boxes_are_never_pasted(tmp_path: Path) -> None:
    """Even at p=1: the image comes back as it is and nothing is drawn from torch's generator."""
    spy = SpyPaste(hand_pool(tmp_path), tmp_path, p=1.0)
    image = solid((512, 373), (1, 2, 3))
    torch.manual_seed(5)
    state = torch.get_rng_state()
    for _ in range(50):
        out, boxes = spy(image, (), False, 1)
        assert out is image and boxes == ()
    assert torch.equal(torch.get_rng_state(), state)
    assert spy.loaded == []


def test_with_several_boxes_one_is_drawn_uniformly(tmp_path: Path) -> None:
    """Two animals: one rectangle per paste, each about half the time, never their union."""
    spy = SpyPaste(hand_pool(tmp_path), tmp_path, p=1.0)
    image = solid((512, 373), (0, 0, 0))
    first: Box = (0.05, 0.05, 0.1, 0.1)
    second: Box = (0.7, 0.6, 0.2, 0.3)
    image.paste((255, 0, 0), rounded(first, image.size))
    image.paste((0, 0, 255), rounded(second, image.size))
    torch.manual_seed(6)
    sizes = Counter[int]()
    for _ in range(SAMPLES):
        out, (pasted,) = spy(image, (first, second), False, 1)
        sizes[round(pasted[2] * 512)] += 1
    assert set(sizes) == {52, 103}  # rounded out: 52 and 103 px wide
    assert 400 < sizes[52] < 600


def test_the_paste_draws_from_torch_only(tmp_path: Path) -> None:
    """The same torch seed gives the same composites whatever Python's `random` state is, so the
    run seed and the DataLoader's per-worker seeds govern arm C as they govern arm A."""
    step = CopyPaste(hand_pool(tmp_path), tmp_path, p=PASTE_P)
    image = noise((512, 373))
    runs = []
    for python_seed in (1, 2):
        torch.manual_seed(7)
        random.seed(python_seed)
        np.random.seed(python_seed)
        runs.append([step(image, (BOX,), bool(k % 2), 1 + k % 3) for k in range(20)])
    for (a, a_boxes), (b, b_boxes) in zip(*runs, strict=True):
        assert a_boxes == b_boxes and a.tobytes() == b.tobytes()


# --- arm C's transform and the dataset ---


@pytest.fixture(scope="module")
def arm_c(fx: Fixture) -> TrainTransform:
    """Arm C as the training script builds it from the p2_arm_c config."""
    config = load_config("p2_arm_c")
    return train.train_transform(config, fx.planted, fx.split, fx.root, log=lambda _: None)


def test_the_training_script_builds_arm_c_from_its_config(fx: Fixture) -> None:
    """`train_transform` builds the pool from the run's metadata and split, logs its counts and
    hash, and hands the paste step the config's probability and the image folder."""
    lines: list[str] = []
    config = load_config("p2_arm_c")
    transform = train.train_transform(config, fx.planted, fx.split, fx.root, log=lines.append)
    assert isinstance(transform.paste, CopyPaste)
    assert transform.paste.p == config.paste_p == 0.5
    assert transform.paste.pool == background_pool(fx.planted, fx.split)
    assert transform.paste.image_root == fx.root
    assert lines == [transform.paste.pool.summary()]
    with pytest.raises(ValueError, match="arm C needs paste_p"):
        train.train_transform(replace(config, paste_p=None), fx.planted, fx.split, fx.root)
    for name in ("resnet18_arm_a", "resnet18_arm_b"):
        built = train.train_transform(load_config(name), fx.planted, fx.split, fx.root)
        assert built.paste is None and built.arm == load_config(name).arm


def test_the_arms_differ_only_in_the_sampler(arm_c: TrainTransform) -> None:
    """R25 for arm C: arm A's crop sampler, output size and after-crop steps, plus the paste step
    before the crop; arms A and B have none."""
    arm_a, arm_b = TrainTransform("A"), TrainTransform("B")
    assert set(vars(arm_a)) == set(vars(arm_b)) == set(vars(arm_c))
    assert set(vars(arm_c)) == {"arm", "sampler", "paste", "after_crop", "size"}
    assert arm_c.size == arm_a.size == load_config("p2_arm_c").input_size == 224
    assert arm_c.arm == "C" and type(arm_c.sampler) is ControlCropSampler
    assert repr(arm_c.after_crop) == repr(arm_a.after_crop)
    assert arm_a.paste is None and arm_b.paste is None
    assert type(arm_c.paste) is CopyPaste and arm_c.paste.p == PASTE_P


def test_arm_c_with_the_paste_off_is_arm_a(fx: Fixture, arm_c: TrainTransform) -> None:
    """With p=0 no coin is drawn: arm C gives arm A's tensor for every train image and seed. With
    p=1 the unboxed images still get arm A's tensor and the boxed ones change."""
    assert isinstance(arm_c.paste, CopyPaste)
    pool = arm_c.paste.pool
    records = fx.split.records["train"]
    arm_a = CCT20Dataset(records, fx.root, TrainTransform("A"))
    off = CCT20Dataset(records, fx.root, TrainTransform("C", paste=CopyPaste(pool, fx.root, 0.0)))
    on = CCT20Dataset(records, fx.root, TrainTransform("C", paste=CopyPaste(pool, fx.root, 1.0)))
    changed = 0
    for index, record in enumerate(records):
        torch.manual_seed(index)
        expected = arm_a[index][0]
        torch.manual_seed(index)
        assert torch.equal(off[index][0], expected)
        torch.manual_seed(index)
        same = torch.equal(on[index][0], expected)
        assert same or record.boxes
        changed += not same
    boxed = sum(1 for r in records if r.boxes)
    assert 0 < boxed < len(records) and changed > boxed // 4


def test_arm_c_crops_the_composite_at_the_configs_input_size(fx: Fixture) -> None:
    """Item 13.02's input size reaches arm C as it reaches arm A: the paste step works on the
    full image, then arm A's crop is resized to `size`. At 320 px every item is 3 x 320 x 320,
    and with the paste off arm C gives arm A's 320 px tensor draw for draw; `train_transform`
    passes the config's size to arm C's transform."""
    pool = background_pool(fx.planted, fx.split)
    records = fx.split.records["train"]
    off = TrainTransform("C", paste=CopyPaste(pool, fx.root, 0.0), size=320)
    on = TrainTransform("C", paste=CopyPaste(pool, fx.root, 1.0), size=320)
    arm_a = CCT20Dataset(records, fx.root, TrainTransform("A", size=320))
    for index in range(0, len(records), 7):
        torch.manual_seed(index)
        expected = arm_a[index][0]
        torch.manual_seed(index)
        assert torch.equal(CCT20Dataset(records, fx.root, off)[index][0], expected)
        tensor, _ = CCT20Dataset(records, fx.root, on)[index]
        assert tuple(tensor.shape) == (3, 320, 320)
    config = replace(load_config("p2_arm_c"), input_size=320)
    built = train.train_transform(config, fx.planted, fx.split, fx.root, log=lambda _: None)
    assert built.size == 320 and isinstance(built.paste, CopyPaste)


def test_the_composite_goes_through_arm_as_transform(fx: Fixture, arm_c: TrainTransform) -> None:
    """Arm C is the paste step, then arm A's transform on the composite and its box, draw for
    draw."""
    assert isinstance(arm_c.paste, CopyPaste)
    step = CopyPaste(arm_c.paste.pool, fx.root, 1.0)
    records = fx.split.records["train"]
    dataset = CCT20Dataset(records, fx.root, TrainTransform("A"))
    arm_a = TrainTransform("A")
    whole = TrainTransform("C", paste=step)
    checked = 0
    for index, record in enumerate(records):
        if not record.boxes:
            continue
        image = dataset.load_image(index)
        torch.manual_seed(index)
        composite, boxes = step(image, record.boxes, record.night, record.location)
        expected = arm_a(composite, boxes)
        torch.manual_seed(index)
        assert torch.equal(whole(image, record.boxes, record.night, record.location), expected)
        checked += composite is not image
    assert checked > 10


class SpyStep:
    """A paste step that records what the dataset hands it and pastes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Box, ...], bool | None, int | None]] = []

    def __call__(
        self, image: Image.Image, boxes: Sequence[Box], night: bool | None, location: int | None
    ) -> tuple[Image.Image, tuple[Box, ...]]:
        self.calls.append((tuple(boxes), night, location))
        return image, tuple(boxes)


def test_the_dataset_hands_the_paste_step_each_records_boxes_flag_and_location(
    fx: Fixture,
) -> None:
    records = fx.split.records["train"]
    spy = SpyStep()
    dataset = CCT20Dataset(
        records, fx.root, TrainTransform(sampler=ControlCropSampler(), paste=spy)
    )
    for index in range(len(dataset)):
        image, label = dataset[index]
        assert tuple(image.shape) == (3, 224, 224) and label == records[index].label
    assert spy.calls == [(r.boxes, r.night, r.location) for r in records]
    assert {night for _, night, _ in spy.calls} == {True, False}


def test_arm_c_is_refused_outside_train(fx: Fixture, arm_c: TrainTransform) -> None:
    """The training transform, paste step and all, is refused for any other split's records
    (R20): the pool and the paste never touch an evaluation image."""
    for name in EVAL_SPLITS:
        with pytest.raises(ValueError, match=r"^the training transform is for train records only"):
            CCT20Dataset(fx.split.records[name], fx.root, arm_c)


def test_arm_c_transform_survives_pickling(fx: Fixture, arm_c: TrainTransform) -> None:
    """DataLoader workers started with spawn (macOS) receive the transform pickled; a copy gives
    the same tensor for the same seed."""
    records = fx.split.records["train"]
    dataset = CCT20Dataset(records, fx.root, arm_c)
    copy = pickle.loads(pickle.dumps(dataset))
    assert isinstance(copy.transform.paste, CopyPaste)
    assert copy.transform.paste.pool == arm_c.paste.pool  # type: ignore[union-attr]
    for index in (i for i, r in enumerate(records) if r.boxes):
        torch.manual_seed(11 + index)
        expected = dataset[index][0]
        torch.manual_seed(11 + index)
        assert torch.equal(copy[index][0], expected)
        if index > 40:
            break


def test_bad_arguments_are_refused(fx: Fixture, arm_c: TrainTransform, tmp_path: Path) -> None:
    pool = hand_pool(tmp_path / "pool")
    for p in (-0.1, 1.5, math.nan):
        with pytest.raises(ValueError, match=r"^p must be between 0 and 1"):
            CopyPaste(pool, tmp_path / "pool", p=p)
    with pytest.raises(TypeError, match="BackgroundPool"):
        CopyPaste(list(pool.frames), tmp_path / "pool")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="the background pool is empty"):
        CopyPaste(BackgroundPool(frames=(), hash=pool_hash(())), tmp_path / "pool")
    (tmp_path / "pool" / pool.frames[0].file_name).unlink()
    (tmp_path / "pool" / pool.frames[3].file_name).write_bytes(b"")
    with pytest.raises(FileNotFoundError, match=r"^2 of 12 background images") as missing:
        CopyPaste(pool, tmp_path / "pool")
    assert "hand-" not in str(missing.value) and ".jpg" not in str(missing.value)
    step = CopyPaste(hand_pool(tmp_path / "ok"), tmp_path / "ok", p=1.0)
    image = solid((512, 373), (1, 2, 3))
    for night, location in ((None, 1), (True, None)):
        with pytest.raises(ValueError, match=r"^arm C needs each training image's day/night"):
            step(image, (BOX,), night, location)
        with pytest.raises(ValueError, match=r"^arm C needs each training image's day/night"):
            step(image, (), night, location)
    (tmp_path / "ok" / step.pool.frames[0].file_name).write_bytes(b"not a jpeg")
    with pytest.raises(OSError, match=r"^a day background of the pool could not be read") as bad:
        step.load_background(step.pool.frames[0])
    assert "hand-" not in str(bad.value)
    with pytest.raises(ValueError, match=r"^arm C needs its paste step"):
        TrainTransform("C")
    for arm in ("A", "B"):
        with pytest.raises(ValueError, match=rf"^only arm C pastes; arm {arm}"):
            TrainTransform(arm, paste=step)
    with pytest.raises(ValueError, match=r"^pass an arm or a sampler, not both"):
        TrainTransform("C", sampler=ControlCropSampler(), paste=step)
    unflagged = [replace(r, night=None) for r in fx.split.records["train"]]
    dataset = CCT20Dataset(unflagged, fx.root, arm_c)
    with pytest.raises(ValueError, match=r"^arm C needs each training image's day/night"):
        dataset[0]


# --- never at evaluation (R23, R28) ---


def test_the_eval_transform_has_no_paste_path() -> None:
    """The eval transform takes the image alone and names no paste step, pool or background."""
    assert list(inspect.signature(EvalTransform.__call__).parameters) == ["self", "image"]
    tree = ast.parse(textwrap.dedent(inspect.getsource(EvalTransform)))
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names |= {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    assert "pipeline" in names
    banned = ("paste", "pool", "background", "empt", "night", "location")
    assert not [n for n in names if any(word in n.lower() for word in banned)]


def _imports_of(path: Path, module: str) -> list[tuple[int, bool]]:
    """Each import of `module` in `path`: its line and whether it sits inside a function."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[tuple[int, bool]] = []

    def visit(node: ast.AST, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            inside = in_function or isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
            names: list[str] = []
            if isinstance(child, ast.Import):
                names = [alias.name for alias in child.names]
            elif isinstance(child, ast.ImportFrom) and child.module:
                names = [child.module] + [f"{child.module}.{a.name}" for a in child.names]
            if module in names:
                found.append((child.lineno, inside))
            visit(child, inside)

    visit(tree, False)
    return found


def test_only_the_training_script_imports_the_paste_module_and_only_for_arm_c() -> None:
    """No module of the package, the Space or the scripts imports `copypaste` but `train.py`,
    once, inside `train_transform`; and no notebook but the look-at-the-data one names it (its
    eye check of item 13.04), so the evaluation and scoring notebooks never reach it."""
    sources = [*REPO.glob("trailcam/**/*.py"), *REPO.glob("space/*.py"), *REPO.glob("scripts/*.py")]
    users = {}
    for path in sources:
        found = _imports_of(path, "trailcam.data.copypaste")
        if found:
            users[path.relative_to(REPO).as_posix()] = found
    assert list(users) == ["trailcam/train/train.py"]
    ((line, inside),) = users["trailcam/train/train.py"]
    assert inside
    source = inspect.getsource(train.train_transform)
    assert "from trailcam.data.copypaste import" in source
    notebooks = sorted(
        p.name for p in REPO.glob("notebooks/*.ipynb") if "copypaste" in p.read_text()
    )
    assert notebooks == ["kaggle_look_at_data.ipynb"]


def test_no_evaluation_path_loads_the_paste_module() -> None:
    """In a fresh interpreter, the data, evaluation, scoring and demo modules, the training
    script and an eval pass leave `copypaste` unloaded, as do arms A and B; building arm C's
    transform loads it."""
    code = textwrap.dedent(
        """
        import sys, tempfile
        from pathlib import Path
        from PIL import Image
        import trailcam.data.loaders, trailcam.data.cache, trailcam.data.split
        import trailcam.evaluate.evaluate, trailcam.evaluate.score_test
        import trailcam.evaluate.attention, trailcam.evaluate.slices
        import trailcam.evaluate.figures, trailcam.evaluate.gradcam
        import trailcam.demo.predict, trailcam.demo.sort, trailcam.demo.app
        import trailcam.train.train, trailcam.train.ledger
        from trailcam.data import synthetic
        from trailcam.data.dataset import EvalTransform, TrainTransform
        from trailcam.data.metadata import load_metadata
        from trailcam.data.split import make_split
        from trailcam.train.config import load_config
        EvalTransform()(Image.new("RGB", (300, 200)))
        TrainTransform("A"), TrainTransform("B")
        print("trailcam.data.copypaste" in sys.modules)
        root = Path(tempfile.mkdtemp())
        written = synthetic.write_synthetic_fixture(root / "fx")
        meta = load_metadata(root / "fx", written.paths["sidecar"])
        split = make_split(meta, 42)
        for name in ("resnet18_arm_a", "resnet18_arm_b"):
            trailcam.train.train.train_transform(load_config(name), meta, split, written.image_root)
        print("trailcam.data.copypaste" in sys.modules)
        trailcam.train.train.train_transform(
            load_config("p2_arm_c"), meta, split, written.image_root, log=lambda _: None
        )
        print("trailcam.data.copypaste" in sys.modules)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["False", "False", "True"]
