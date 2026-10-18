"""The arm C paste step: copy-paste onto other cameras' empty frames (docs/phase2-plan.md section
6, step 1, v1 rectangles).

Arm C attacks the background shortcut that arm B's crops could not: a crop around the box still
shows the animal on its own camera's scene, while a paste puts it on another camera's. Training
only: `CopyPaste` is the paste step of `TrainTransform("C", paste=...)` in
`trailcam/data/dataset.py`, built by `trailcam/train/train.py` for an arm C config; no other module
of the package imports this one, and loading the eval transform or any evaluation path never loads
it (`tests/test_copypaste.py`).

**The background pool is a leakage boundary**, like the split. `background_pool(meta, split)`
takes the empty frames the metadata drops (`Metadata.empties`) and keeps a frame only if:

1. it comes from a pooled file, official train or cis_val (`POOL_FILES`): the empties of the
   cis_test, trans_val and trans_test files are frames of evaluation files, never backgrounds, even
   at a training camera (in CCT20 official train has none, so the pool is cis_val's 995);
2. its location is a train location (a location of `split.records["train"]`), so never a
   location of val, trans_val, cis_test or trans_test that is not also a train location;
3. its sequence (`seq_id`) has no image in val, trans_val, cis_test or trans_test: val shares its
   cameras with train, and an empty frame of a val burst shows the val photo's scene seconds apart.
   A frame of a train burst may stay: it is the scene of a train photo.

`check_pool` then re-checks the result independently (each rule, no image id of any split, a
day/night flag on every frame) and refuses with counts only; `background_pool` returns nothing it
has not passed. The pool is a function of the metadata and the split seed, so its hash
(`pool_hash`) reproduces it; its repr and `summary` print counts per day and night and the hash,
never a frame. An all-empty burst at a train camera may lie minutes from a val burst at that camera,
as train bursts already do: the split is by sequence, so the pool shows val nothing train does not.

**The paste.** For a training image with at least one box, with probability `p` (`PASTE_P`, 0.5,
the config's `paste_p`):

1. One box is drawn uniformly when there are several, as arm B does. Its rectangle, rounded out to
   whole pixels (v1: no segmentation mask), is cut from the image.
2. A background is drawn uniformly among the pool's frames with the image's day/night flag (the
   sidecar's) at another location than the image's. Mesnage et al. 2025 pasted night animals onto
   day backgrounds and found mostly nocturnal species worse; their backgrounds came from the test
   locations, which rule 2 forbids here. No frame qualifies: the image is left as it is.
3. The rectangle keeps its size as a fraction of the frame, width and height each (the cache's
   frames are 512 x 373 or 512 x 384, so a paste between them stretches by 3% at most), and goes at
   a random position wholly inside the frame with its bottom edge in the lower half, the rule of
   Mesnage et al. 2025 (section 2.4), drawn uniformly among those placements.

The composite then goes through arm A's training transform unchanged, crop sampler included;
images without boxes, and the other draws, get arm A's transform on the image itself. The step
returns the image and its boxes: the pasted rectangle's box on the composite, the image's own
boxes otherwise. All randomness comes from torch's generator, as in arm A and arm B; with `p=0` no
coin is drawn, so arm C then gives arm A's tensors exactly for the same seed.

Data: Caltech Camera Traps (CCT20), Beery et al. 2018, CDLA-Permissive 1.0.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch
from PIL import Image

from trailcam.data.boxcrop import to_pixels
from trailcam.data.metadata import Box, EmptyFrame, Metadata
from trailcam.data.split import POOLED_FILES, SPLIT_NAMES, Split, split_hash

PASTE_P: Final = 0.5  # share of boxed training images pasted onto a background
POOL_FILES: Final = POOLED_FILES  # official train and cis_val, the files train is drawn from
EVAL_SPLITS: Final = tuple(name for name in SPLIT_NAMES if name != "train")


class PoolError(ValueError):
    """The background pool breaks one of its rules (module docstring)."""


def pool_hash(frames: Iterable[EmptyFrame]) -> str:
    """SHA-256 over (image id, location, night flag) of every frame, sorted by image id."""
    body = sorted([f.image_id, f.location, f.night] for f in frames)
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True, repr=False)
class BackgroundPool:
    """The empty frames arm C pastes onto, sorted by image id, and their hash. Built by
    `background_pool`; the repr shows counts per day and night and the hash, never a frame."""

    frames: tuple[EmptyFrame, ...]
    hash: str

    @property
    def day(self) -> int:
        return sum(1 for f in self.frames if f.night is False)

    @property
    def night(self) -> int:
        return sum(1 for f in self.frames if f.night is True)

    def __repr__(self) -> str:
        return f"BackgroundPool(day={self.day}, night={self.night}, hash={self.hash})"

    def summary(self) -> str:
        """One line for the run log: counts per day and night and the hash."""
        return (
            f"background pool (arm C, training only): {len(self.frames)} empty frames, "
            f"day {self.day}, night {self.night}, hash {self.hash}"
        )


def check_pool(frames: Sequence[EmptyFrame], split: Split) -> None:
    """Raise PoolError, with counts only, when a frame breaks a pool rule: from a file other than
    `POOL_FILES`, at a location that is not a train location or is a location of an evaluation
    split only, in a sequence with an image in an evaluation split, with an image id of any split,
    repeated, or without a day/night flag."""
    train_locations = {r.location for r in split.records["train"]}
    eval_records = [r for name in EVAL_SPLITS for r in split.records[name]]
    eval_only_locations = {r.location for r in eval_records} - train_locations
    eval_sequences = {r.seq_id for r in eval_records}
    split_ids = {r.image_id for name in SPLIT_NAMES for r in split.records[name]}
    counts = {
        f"from a file other than {', '.join(POOL_FILES)}": sum(
            1 for f in frames if f.split not in POOL_FILES
        ),
        "not at a train location": sum(1 for f in frames if f.location not in train_locations),
        "at a location of an evaluation split only": sum(
            1 for f in frames if f.location in eval_only_locations
        ),
        "in a sequence with an image in val, trans_val, cis_test or trans_test": sum(
            1 for f in frames if f.seq_id in eval_sequences
        ),
        "with the image id of a split's image": sum(1 for f in frames if f.image_id in split_ids),
        "repeated": len(frames) - len({f.image_id for f in frames}),
        "without a day/night flag (load the metadata with the sidecar)": sum(
            1 for f in frames if f.night is None
        ),
    }
    problems = [f"{n} frames {why}" for why, n in counts.items() if n]
    if problems:
        raise PoolError("background pool: " + "; ".join(problems))


def background_pool(meta: Metadata, split: Split) -> BackgroundPool:
    """The pool of the module docstring's rules 1 to 3 for `split`, made from `meta`, checked by
    `check_pool`. Refuses a split whose records no longer match its hash."""
    if split_hash(split.records) != split.hash:
        raise PoolError("the split's records no longer match its hash; pass make_split's output")
    train_locations = {r.location for r in split.records["train"]}
    eval_sequences = {r.seq_id for name in EVAL_SPLITS for r in split.records[name]}
    frames = [
        frame
        for name in POOL_FILES
        for frame in meta.empties.get(name, ())
        if frame.location in train_locations and frame.seq_id not in eval_sequences
    ]
    frames.sort(key=lambda f: f.image_id)
    check_pool(frames, split)
    return BackgroundPool(frames=tuple(frames), hash=pool_hash(frames))


def _randint(low: int, high: int) -> int:
    """Uniform over `low` to `high`, both included, from torch's generator."""
    return int(torch.randint(low, high + 1, (1,)))


def paste_box(
    image: Image.Image, box: Box, background: Image.Image
) -> tuple[Image.Image, tuple[Box]]:
    """`box`'s rectangle of `image`, rounded out to whole pixels, pasted onto a copy of
    `background` at the same width and height fractions of the frame, at a uniform position inside
    it with its bottom edge in the lower half; and the pasted rectangle's box on the composite."""
    width, height = image.size
    back_w, back_h = background.size
    x0, y0, x1, y1 = to_pixels(box, width, height)
    left, top = max(math.floor(x0), 0), max(math.floor(y0), 0)
    right, bottom = min(math.ceil(x1), width), min(math.ceil(y1), height)
    patch = image.crop((left, top, right, bottom))
    w = min(max(round((right - left) * back_w / width), 1), back_w)
    h = min(max(round((bottom - top) * back_h / height), 1), back_h)
    if patch.size != (w, h):
        patch = patch.resize((w, h), Image.Resampling.BILINEAR)
    if patch.mode != background.mode:
        patch = patch.convert(background.mode)
    # Wholly inside the frame, the bottom edge (y + h) at or below the middle.
    x = _randint(0, back_w - w)
    y = _randint(max(math.ceil(back_h / 2) - h, 0), back_h - h)
    composite = background.copy()
    composite.paste(patch, (x, y))
    return composite, ((x / back_w, y / back_h, w / back_w, h / back_h),)


class CopyPaste:
    """Arm C's paste step: with probability `p` per image with a box, one of its boxes pasted onto
    a pool background of its day/night flag from another location (module docstring). The pool's
    frames are read from `image_root`, the cache folder the training images come from; every one
    must be there. Training only; the boxes are fractions of the image."""

    def __init__(self, pool: BackgroundPool, image_root: Path, p: float = PASTE_P) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p must be between 0 and 1, got {p}")
        if not isinstance(pool, BackgroundPool):
            raise TypeError("pool must be the BackgroundPool of background_pool")
        if not pool.frames:
            raise ValueError("the background pool is empty")
        missing = Counter(
            "night" if f.night else "day"
            for f in pool.frames
            if not _is_image_file(Path(image_root) / f.file_name)
        )
        if missing:
            raise FileNotFoundError(
                f"{sum(missing.values())} of {len(pool.frames)} background images are missing "
                f"or empty under the image root (day {missing['day']}, night {missing['night']})"
            )
        self.pool = pool
        self.image_root = Path(image_root)
        self.p = p
        self._choices: dict[tuple[bool, int], tuple[EmptyFrame, ...]] = {}

    def __call__(
        self, image: Image.Image, boxes: Sequence[Box], night: bool | None, location: int | None
    ) -> tuple[Image.Image, tuple[Box, ...]]:
        if night is None or location is None:
            raise ValueError(
                "arm C needs each training image's day/night flag and location: load the "
                "metadata with the day/night sidecar"
            )
        if not boxes or not self._coin():
            return image, tuple(boxes)
        box = boxes[_randint(0, len(boxes) - 1)] if len(boxes) > 1 else boxes[0]
        frame = self.draw_background(night, location)
        if frame is None:
            return image, tuple(boxes)
        return paste_box(image, box, self.load_background(frame))

    def _coin(self) -> bool:
        """True with probability `p`. At 0 and 1 nothing is drawn."""
        if self.p == 0.0 or self.p == 1.0:
            return self.p == 1.0
        return float(torch.rand(1)) < self.p

    def candidates(self, night: bool, location: int) -> tuple[EmptyFrame, ...]:
        """The pool's frames with day/night flag `night` at a location other than `location`."""
        key = (night, location)
        if key not in self._choices:
            self._choices[key] = tuple(
                f for f in self.pool.frames if f.night is night and f.location != location
            )
        return self._choices[key]

    def draw_background(self, night: bool, location: int) -> EmptyFrame | None:
        """A frame drawn uniformly among `candidates(night, location)`; None when there is none."""
        choices = self.candidates(night, location)
        if not choices:
            return None
        return choices[_randint(0, len(choices) - 1)] if len(choices) > 1 else choices[0]

    def load_background(self, frame: EmptyFrame) -> Image.Image:
        """The frame as RGB. A read failure names the day/night flag, never the file (R03)."""
        try:
            with Image.open(self.image_root / frame.file_name) as background:
                return background.convert("RGB")
        except OSError as error:
            when = "night" if frame.night else "day"
            raise OSError(
                f"a {when} background of the pool could not be read: {type(error).__name__}"
            ) from None


def _is_image_file(path: Path) -> bool:
    """A regular file (or a link to one) with at least one byte, as the cache check counts one."""
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size > 0
