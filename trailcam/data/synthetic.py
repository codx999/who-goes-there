"""Synthetic CCT20-shaped fixture for the tests and the dry run (plan.md section 5).

This module writes only into the directory it is given (a pytest `tmp_path` or the dry run's
temporary directory); nothing it writes is committed. The metadata part (item 3.02) writes the five
`<split>_annotations.json` files and a day/night sidecar; the images part (item 4.03) adds one
small JPEG per image of those files.

Layout: the 12 kept classes; 2 cis locations (train, cis_val, cis_test) and 1 trans location
(trans_val, trans_test); every sequence has 3 frames; about 90% of animal annotations carry a box.
Each class has `TRAIN_SEQUENCES + 1` sequences per cis location in the train and cis_val pool, so
a re-split stratified by class and location has sequences to put on both sides. The cases the
loader must filter are planted on purpose: empty sequences outside train, one badger, deer and fox
sequence, a bobcat + rabbit frame in train and trans_test, a two-opossum frame in train and
cis_test (kept), and one sequence split across train and cis_val as in the released files.

Images: `write_synthetic_fixture` writes the metadata into the directory and the JPEGs into its
`images/` folder, each under the image's `file_name` at its `width` x `height`, read back from the
written files. A JPEG is a dark background (one colour per location, as each camera sees one
scene) with mild noise, and a bright rectangle in the annotation's class colour, outlined in
white, where each box is. A night image (the sidecar's flag) is saved as a grayscale (mode L) JPEG.
`python -m trailcam.data.synthetic <dir>` (or `--root <dir>`) writes the fixture into a new or
empty directory and prints counts only.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from PIL import Image, ImageDraw

from trailcam.data.classes import CLASSES, DROPPED_CCT_IDS, INDEX_TO_CCT_ID, index_of_name
from trailcam.data.metadata import (
    DAY_NIGHT_SIDECAR,
    SPLITS,
    annotation_file_name,
    load_night_flags,
)

CIS_LOCATIONS: Final = (101, 102)
TRANS_LOCATIONS: Final = (201,)
TRAIN_SEQUENCES: Final = 6  # per class per cis location; cis_val gets one more
FRAMES: Final = 3
WIDTH: Final = 96
HEIGHT: Final = 72
BOX_PROBABILITY: Final = 0.9
NIGHT_HOURS: Final = frozenset((*range(0, 6), *range(19, 24)))

EMPTY: Final = 30
BADGER: Final = 21
DEER: Final = 34
FOX: Final = 51

IMAGE_DIR: Final = "images"
JPEG_QUALITY: Final = 90
NOISE: Final = 0.25  # share of uniform noise blended into the background
GOLDEN: Final = 0.618033988749895  # hue step, so neighbouring ids get distant hues
OUTLINE: Final = (255, 255, 255)


def _categories() -> list[dict[str, Any]]:
    kept = [
        {"id": cct_id, "name": name} for name, cct_id in zip(CLASSES, INDEX_TO_CCT_ID, strict=True)
    ]
    dropped = [{"id": cct_id, "name": name} for cct_id, name in DROPPED_CCT_IDS.items()]
    return sorted(kept + dropped, key=lambda c: c["id"])


@dataclass
class _Writer:
    rng: random.Random
    files: dict[str, dict[str, list[dict[str, Any]]]] = field(
        default_factory=lambda: {split: {"images": [], "annotations": []} for split in SPLITS}
    )
    night: dict[str, dict[str, bool | int]] = field(default_factory=dict)
    n_images: int = 0
    n_sequences: int = 0
    n_annotations: int = 0

    def box(self) -> list[float] | None:
        if self.rng.random() >= BOX_PROBABILITY:
            return None
        w = float(self.rng.randint(12, WIDTH // 2))
        h = float(self.rng.randint(10, HEIGHT // 2))
        return [
            float(self.rng.randint(0, WIDTH - int(w))),
            float(self.rng.randint(0, HEIGHT - int(h))),
            w,
            h,
        ]

    def annotate(self, split: str, image_id: str, cct_id: int) -> None:
        self.n_annotations += 1
        ann: dict[str, Any] = {
            "id": f"syn-ann-{self.n_annotations:05d}",
            "image_id": image_id,
            "category_id": cct_id,
        }
        box = None if cct_id == EMPTY else self.box()  # CCT20 empties carry no box
        if box is not None:
            ann["bbox"] = box
        self.files[split]["annotations"].append(ann)

    def sequence(
        self,
        cct_id: int,
        location: int,
        splits: tuple[str, ...] = (),
        extra: dict[int, list[int]] | None = None,
    ) -> None:
        """One 3-frame sequence; `splits` gives each frame's split, `extra` adds annotations."""
        self.n_sequences += 1
        seq_id = f"syn-seq-{self.n_sequences:04d}"
        hour = (self.n_sequences * 5) % 24
        day = 1 + self.n_sequences % 28
        for frame in range(1, FRAMES + 1):
            split = splits[frame - 1]
            self.n_images += 1
            image_id = f"syn-img-{self.n_images:05d}"
            self.files[split]["images"].append(
                {
                    "id": image_id,
                    "file_name": f"{image_id}.jpg",
                    "width": WIDTH,
                    "height": HEIGHT,
                    "location": location,
                    "seq_id": seq_id,
                    "seq_num_frames": FRAMES,
                    "frame_num": frame,
                    "date_captured": f"2026-03-{day:02d} {hour:02d}:{frame * 7:02d}:{frame:02d}",
                    "rights_holder": "synthetic",
                }
            )
            self.night[image_id] = {"night": hour in NIGHT_HOURS, "hour": hour}
            self.annotate(split, image_id, cct_id)
            for other in (extra or {}).get(frame, []):
                self.annotate(split, image_id, other)


def write_synthetic_metadata(out_dir: Path, seed: int = 0) -> dict[str, Path]:
    """Write the five annotation files and the day/night sidecar into `out_dir`.

    Returns the path of each file by split, plus `"sidecar"`. Same seed, same files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    w = _Writer(rng=random.Random(seed))
    opossum, rabbit, bobcat = (
        INDEX_TO_CCT_ID[index_of_name(n)] for n in ("opossum", "rabbit", "bobcat")
    )

    for cct_id in INDEX_TO_CCT_ID:
        for location in CIS_LOCATIONS:
            for _ in range(TRAIN_SEQUENCES):
                w.sequence(cct_id, location, ("train",) * FRAMES)
            w.sequence(cct_id, location, ("cis_val",) * FRAMES)
            w.sequence(cct_id, location, ("cis_test",) * FRAMES)
        for location in TRANS_LOCATIONS:
            w.sequence(cct_id, location, ("trans_val",) * FRAMES)
            w.sequence(cct_id, location, ("trans_test",) * FRAMES)

    loc, trans = CIS_LOCATIONS[0], TRANS_LOCATIONS[0]
    # Filtered out: empties (none in train, as in CCT20), the rare classes, two-species frames.
    for split in ("cis_val", "cis_test"):
        w.sequence(EMPTY, loc, (split,) * FRAMES)
    for split in ("trans_val", "trans_test"):
        w.sequence(EMPTY, trans, (split,) * FRAMES)
    w.sequence(BADGER, loc, ("train",) * FRAMES)
    w.sequence(DEER, loc, ("cis_test",) * FRAMES)
    w.sequence(FOX, trans, ("trans_test",) * FRAMES)
    w.sequence(bobcat, loc, ("train",) * FRAMES, extra={2: [rabbit]})
    w.sequence(bobcat, trans, ("trans_test",) * FRAMES, extra={2: [rabbit]})
    # Kept: one species annotated twice on one frame.
    w.sequence(opossum, loc, ("train",) * FRAMES, extra={1: [opossum]})
    w.sequence(opossum, CIS_LOCATIONS[1], ("cis_test",) * FRAMES, extra={1: [opossum]})
    # One burst split across train and cis_val, like the 224 shared sequences in CCT20.
    w.sequence(rabbit, CIS_LOCATIONS[1], ("train", "train", "cis_val"))

    paths: dict[str, Path] = {}
    info = {"description": "synthetic CCT20-shaped fixture", "version": "synthetic", "year": 2026}
    for split in SPLITS:
        body = {"info": info, "categories": _categories(), **w.files[split]}
        paths[split] = out_dir / annotation_file_name(split)
        paths[split].write_text(json.dumps(body))
    paths["sidecar"] = out_dir / DAY_NIGHT_SIDECAR
    paths["sidecar"].write_text(json.dumps(w.night))
    return paths


def _colour(hue: float, saturation: float, value: float) -> tuple[int, int, int]:
    r, g, b = colorsys.hsv_to_rgb(hue % 1.0, saturation, value)
    return round(r * 255), round(g * 255), round(b * 255)


def background_colour(location: int) -> tuple[int, int, int]:
    """Dark (value 0.3, at most 77 per channel), one colour per location."""
    return _colour(location * GOLDEN, 0.5, 0.3)


def box_colour(cct_id: int) -> tuple[int, int, int]:
    """Bright (value 1, saturation 0.5: every channel at least 127), one colour per category, so
    a rectangle stays brighter than the background in a grayscale JPEG too."""
    return _colour(cct_id * GOLDEN, 0.5, 1.0)


@dataclass(frozen=True)
class ImageCounts:
    """The JPEGs written: per official split, how many are grayscale (night), how many carry at
    least one rectangle, the rectangles drawn and their bytes."""

    per_split: Mapping[str, int]
    night: int
    boxed: int
    rectangles: int
    jpeg_bytes: int

    @property
    def images(self) -> int:
        return sum(self.per_split.values())


@dataclass(frozen=True)
class Fixture:
    """What `write_synthetic_fixture` wrote: `paths` as `write_synthetic_metadata` returns them
    (one annotation file per split, plus `"sidecar"`), the folder holding the JPEGs (each record's
    `file_name` under it) and the counts."""

    paths: Mapping[str, Path]
    image_root: Path
    counts: ImageCounts


def _draw_image(
    image: Mapping[str, Any], anns: Sequence[Mapping[str, Any]], night: bool, seed: int
) -> Image.Image:
    size = (int(image["width"]), int(image["height"]))
    rng = random.Random(f"{seed}:{image['id']}")  # an image depends on its id and the seed only
    base = Image.new("RGB", size, background_colour(int(image["location"])))
    noise = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
    canvas = Image.blend(base, noise, NOISE)
    draw = ImageDraw.Draw(canvas)
    for ann in anns:
        if "bbox" in ann:
            x, y, w, h = (float(v) for v in ann["bbox"])
            corners = (round(x), round(y), round(x + w) - 1, round(y + h) - 1)  # inclusive
            draw.rectangle(corners, fill=box_colour(int(ann["category_id"])), outline=OUTLINE)
    return canvas.convert("L") if night else canvas


def _write_images(paths: Mapping[str, Path], image_dir: Path, seed: int = 0) -> ImageCounts:
    """One JPEG per image of the five annotation files in `paths` (as `write_synthetic_metadata`
    returns them), drawn from those files and the sidecar's night flags, into `image_dir`."""
    image_dir.mkdir(parents=True, exist_ok=True)
    flags = load_night_flags(paths["sidecar"])
    per_split = dict.fromkeys(SPLITS, 0)
    night = boxed = rectangles = written = 0
    for split in SPLITS:
        data = json.loads(paths[split].read_text())
        anns: dict[str, list[Mapping[str, Any]]] = {}
        for ann in data["annotations"]:
            anns.setdefault(ann["image_id"], []).append(ann)
        missing = sum(1 for image in data["images"] if image["id"] not in flags)
        if missing:
            raise ValueError(f"{split}: {missing} images missing from the day/night sidecar")
        for image in data["images"]:
            own = anns.get(image["id"], [])
            is_night = flags[image["id"]]
            path = image_dir / image["file_name"]
            _draw_image(image, own, is_night, seed).save(path, "JPEG", quality=JPEG_QUALITY)
            per_split[split] += 1
            night += is_night
            drawn = sum(1 for ann in own if "bbox" in ann)
            boxed += drawn > 0
            rectangles += drawn
            written += path.stat().st_size
    return ImageCounts(per_split, night, boxed, rectangles, written)


def write_synthetic_fixture(out_dir: Path, seed: int = 0) -> Fixture:
    """Write the metadata (`write_synthetic_metadata`) into `out_dir` and its JPEGs into
    `out_dir / IMAGE_DIR`. `out_dir` must be new or empty, so a mistyped path cannot mix the
    fixture into real data. Same seed, same files."""
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError("the synthetic fixture is written into a new or empty directory only")
    paths = write_synthetic_metadata(out_dir, seed)
    counts = _write_images(paths, out_dir / IMAGE_DIR, seed)
    return Fixture(paths=paths, image_root=out_dir / IMAGE_DIR, counts=counts)


def summary(fixture: Fixture, seed: int) -> str:
    """Counts only: files written, JPEGs per official split, night and boxed JPEGs, bytes."""
    counts = fixture.counts
    lines = [
        f"Synthetic CCT20 fixture, seed {seed}",
        f"Annotation files: {len(SPLITS)}, day/night sidecar: 1",
        "JPEGs per official split",
    ]
    lines += [f"  {split:<12}{counts.per_split[split]:>6}" for split in SPLITS]
    lines += [
        f"  {'total':<12}{counts.images:>6}",
        f"Grayscale (night) JPEGs: {counts.night} of {counts.images}",
        f"JPEGs with a drawn box: {counts.boxed} of {counts.images} "
        f"({counts.rectangles} rectangles)",
        f"JPEG bytes: {counts.jpeg_bytes}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("out_dir", nargs="?", type=Path, help="new or empty directory")
    parser.add_argument("--root", type=Path, default=None, help="the same, as an option")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if (args.out_dir is None) == (args.root is None):
        parser.error("give the output directory once: as OUT_DIR or as --root")
    out_dir: Path = args.root if args.out_dir is None else args.out_dir
    try:
        fixture = write_synthetic_fixture(out_dir, args.seed)
    except FileExistsError as error:
        parser.error(str(error))
    print(summary(fixture, args.seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
