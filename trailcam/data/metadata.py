"""Load the five CCT20 COCO-style annotation files into one record per image (plan.md section 2).

Each kept image becomes a `Record` with its label index from `trailcam.data.classes`, location,
sequence id and frame, capture hour, every bounding box (possibly none) and an optional day/night
flag from the cache's sidecar. Boxes are stored as fractions of the image (0 to 1), clipped to the
frame, so they hold at every resolution the image is read at (1024 px tarball, 512 px cache, crop).
Images are dropped by these rules, checked in this order:

1. `two_species`: annotations with more than one distinct category (78 images in CCT20).
   Several annotations of one species keep that label and all their boxes.
2. `empty`: the `empty` category (30).
3. `rare`: badger, fox and deer.
4. `unannotated`: no annotation at all (none in CCT20; counted so it cannot pass silently).

The five files keep their official split names here; pooling and re-splitting train and cis_val is
`trailcam/data/split.py`. `python -m trailcam.data.metadata --root <dir>` prints counts only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from trailcam.data.classes import CLASSES, DROPPED_CCT_IDS, NUM_CLASSES, index_of_cct_id

SPLITS: Final = ("train", "cis_val", "trans_val", "cis_test", "trans_test")
DROP_RULES: Final = ("two_species", "empty", "rare", "unannotated")
EMPTY_CCT_ID: Final = 30
DATE_FORMAT: Final = "%Y-%m-%d %H:%M:%S"

# Written next to the image cache by scripts/kaggle_prepare_cct20.py (item 6.01): a JSON object
# mapping image id to {"night": bool, "hour": int, ...} (plan.md section 2). Only "night" is read,
# only from a path passed explicitly, and every kept image must be in it.
DAY_NIGHT_SIDECAR: Final = "day_night.json"

# (x, y, width, height) as fractions of the image width and height: the COCO pixel box divided by
# the image's `width` and `height`, after clipping to the frame.
Box = tuple[float, float, float, float]


@dataclass(frozen=True, repr=False)
class Record:
    """One kept image. `split` is the official file it came from; in a `trailcam.data.split.Split`
    it is the split the image is assigned to (train or val for the pooled files). The repr shows
    the class, split, location, frame, box count and night flag, never the image id, file name or
    sequence id (R03); equality still compares every field. `__eq__` is written out so pytest
    explains a failed `==` with the repr instead of printing each differing field (review 3.05).
    """

    image_id: str
    file_name: str
    split: str
    label: int
    location: int
    seq_id: str
    frame_num: int
    seq_num_frames: int
    width: int
    height: int
    hour: int
    boxes: tuple[Box, ...]
    night: bool | None

    def __eq__(self, other: object) -> bool:
        """Every field in order, as the generated `__eq__` compared; the dataclass still adds the
        frozen `__hash__` over the same fields."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        names = [f.name for f in fields(self)]
        return [getattr(self, n) for n in names] == [getattr(other, n) for n in names]

    def __repr__(self) -> str:
        name = CLASSES[self.label] if 0 <= self.label < NUM_CLASSES else f"label {self.label}"
        return (
            f"Record(class={name}, split={self.split}, location={self.location}, "
            f"frame={self.frame_num}/{self.seq_num_frames}, boxes={len(self.boxes)}, "
            f"night={self.night})"
        )


@dataclass(frozen=True, repr=False)
class Metadata:
    """Kept records per official split, in file order, images dropped per split per rule, and
    the number of kept boxes per split that ran past the frame and were clipped. The repr shows
    records per split, drops per rule and clipped boxes summed over the splits, never a record
    (R03).
    """

    records: Mapping[str, tuple[Record, ...]]
    dropped: Mapping[str, Mapping[str, int]]
    clipped_boxes: Mapping[str, int]

    def __repr__(self) -> str:
        sizes = ", ".join(f"{split}={len(rs)}" for split, rs in self.records.items())
        rules = Counter[str]()
        for per_rule in self.dropped.values():
            rules.update(per_rule)
        drops = ", ".join(f"{rule}={n}" for rule, n in rules.items())
        clipped = sum(self.clipped_boxes.values())
        return f"Metadata(records: {sizes}; dropped: {drops}; clipped_boxes={clipped})"


def annotation_file_name(split: str) -> str:
    return f"{split}_annotations.json"


def find_annotation_files(root: Path) -> dict[str, Path]:
    """The one `<split>_annotations.json` under `root` (searched recursively) for every split."""
    if not root.is_dir():
        raise FileNotFoundError(f"metadata root is not a directory: {root}")
    found: dict[str, Path] = {}
    for split in SPLITS:
        matches = sorted(root.rglob(annotation_file_name(split)))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {annotation_file_name(split)} under {root}, found {len(matches)}"
            )
        found[split] = matches[0]
    return found


def capture_hour(date_captured: str) -> int:
    """Hour 0 to 23 of a CCT `date_captured` string ("YYYY-MM-DD HH:MM:SS")."""
    return datetime.strptime(date_captured, DATE_FORMAT).hour


def normalise_box(raw: Any, width: int, height: int) -> tuple[Box, bool]:
    """A COCO pixel box as fractions of `width` x `height`, clipped to the frame.

    Returns the box and whether clipping changed it. Raises ValueError for a malformed box or one
    with no area left inside the frame.
    """
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("a bbox is not a list of four numbers")
    x, y, w, h = (float(v) for v in raw)
    if w <= 0 or h <= 0:
        raise ValueError("a bbox has a non-positive width or height")
    if width <= 0 or height <= 0:
        raise ValueError("an image has a non-positive width or height")
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("a bbox lies outside its image")
    clipped = (x0, y0, x1, y1) != (x, y, x + w, y + h)
    return (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height), clipped


def _check_categories(categories: Sequence[Mapping[str, Any]], split: str) -> None:
    """Every category in the file is one the class map knows, under the same name."""
    for category in categories:
        cct_id = int(category["id"])
        name = category["name"]
        expected = (
            DROPPED_CCT_IDS[cct_id]
            if cct_id in DROPPED_CCT_IDS
            else CLASSES[index_of_cct_id(cct_id)]
        )
        if name != expected:
            raise ValueError(f"{split}: category {cct_id} is named {name!r}, expected {expected!r}")


def load_night_flags(path: Path) -> dict[str, bool]:
    """Image id to night flag from a day/night sidecar."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError("the day/night sidecar is not a JSON object")
    flags: dict[str, bool] = {}
    for image_id, entry in raw.items():
        night = entry.get("night") if isinstance(entry, dict) else None
        if not isinstance(night, bool):
            raise ValueError("a day/night sidecar entry has no boolean 'night'")
        flags[image_id] = night
    return flags


def load_split(
    path: Path, split: str, night_flags: Mapping[str, bool] | None = None
) -> tuple[tuple[Record, ...], dict[str, int], int]:
    """Kept records of one annotation file, images dropped per rule and boxes clipped."""
    data = json.loads(path.read_text())
    _check_categories(data["categories"], split)

    annotations: dict[str, list[Mapping[str, Any]]] = {}
    for ann in data["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    image_ids = [image["id"] for image in data["images"]]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"{split}: {len(image_ids) - len(set(image_ids))} repeated image ids")
    orphans = len(annotations.keys() - set(image_ids))
    if orphans:
        raise ValueError(f"{split}: annotations for {orphans} image ids not in the file")

    dropped = dict.fromkeys(DROP_RULES, 0)
    clipped = 0
    records: list[Record] = []
    for image in data["images"]:
        anns = annotations.get(image["id"], [])
        categories = {int(ann["category_id"]) for ann in anns}
        if len(categories) > 1:
            dropped["two_species"] += 1
            continue
        if not categories:
            dropped["unannotated"] += 1
            continue
        (cct_id,) = categories
        if cct_id == EMPTY_CCT_ID:
            dropped["empty"] += 1
            continue
        if cct_id in DROPPED_CCT_IDS:
            dropped["rare"] += 1
            continue
        location = image["location"]
        if not isinstance(location, int) or isinstance(location, bool):
            raise ValueError(f"{split}: a location is not an integer")
        width, height = int(image["width"]), int(image["height"])
        boxes: list[Box] = []
        for ann in anns:
            if "bbox" in ann:
                box, was_clipped = normalise_box(ann["bbox"], width, height)
                boxes.append(box)
                clipped += was_clipped
        records.append(
            Record(
                image_id=image["id"],
                file_name=image["file_name"],
                split=split,
                label=index_of_cct_id(cct_id),
                location=location,
                seq_id=image["seq_id"],
                frame_num=int(image["frame_num"]),
                seq_num_frames=int(image["seq_num_frames"]),
                width=width,
                height=height,
                hour=capture_hour(image["date_captured"]),
                boxes=tuple(boxes),
                night=None if night_flags is None else night_flags.get(image["id"]),
            )
        )
    if night_flags is not None:
        missing = sum(1 for r in records if r.night is None)
        if missing:
            raise ValueError(f"{split}: {missing} kept images missing from the day/night sidecar")
    return tuple(records), dropped, clipped


def load_metadata(root: Path, sidecar: Path | None = None) -> Metadata:
    """Load and filter all five files under `root`.

    The day/night flags come from `sidecar` only when it is passed (plan.md section 2): a
    `DAY_NIGHT_SIDECAR` next to the annotation files is not picked up on its own. Without one every
    record has `night=None`; with one, every kept image must be in it or loading fails.
    """
    files = find_annotation_files(root)
    night_flags = None if sidecar is None else load_night_flags(sidecar)
    records: dict[str, tuple[Record, ...]] = {}
    dropped: dict[str, Mapping[str, int]] = {}
    clipped: dict[str, int] = {}
    for split in SPLITS:
        records[split], dropped[split], clipped[split] = load_split(
            files[split], split, night_flags
        )
    return Metadata(records=records, dropped=dropped, clipped_boxes=clipped)


def summary(meta: Metadata) -> str:
    """Counts only: images per class per split, boxed fraction, hours, night share, drops."""
    width = max(len(name) for name in (*CLASSES, *DROP_RULES, "total")) + 2
    head = "".join(f"{split:>12}" for split in SPLITS) + f"{'all':>12}"
    lines = ["Images per class per split (kept)", f"{'class':<{width}}{head}"]
    per_split = {split: Counter(r.label for r in meta.records[split]) for split in SPLITS}
    for index, name in enumerate(CLASSES):
        row = [per_split[split][index] for split in SPLITS]
        lines.append(f"{name:<{width}}" + "".join(f"{n:>12}" for n in row) + f"{sum(row):>12}")
    totals = [len(meta.records[split]) for split in SPLITS]
    lines.append(f"{'total':<{width}}" + "".join(f"{n:>12}" for n in totals) + f"{sum(totals):>12}")

    lines += ["", "Images dropped per rule", f"{'rule':<{width}}{head}"]
    for rule in DROP_RULES:
        row = [meta.dropped[split][rule] for split in SPLITS]
        lines.append(f"{rule:<{width}}" + "".join(f"{n:>12}" for n in row) + f"{sum(row):>12}")

    lines += ["", "Boxed fraction (kept images with at least one box)"]
    for split in SPLITS:
        recs = meta.records[split]
        boxed = sum(1 for r in recs if r.boxes)
        share = boxed / len(recs) if recs else 0.0
        lines.append(f"  {split:<12}{boxed:>7} of {len(recs):>6}  {share:.3f}")

    lines += ["", "Boxes clipped to the frame (kept images)"]
    for split in SPLITS:
        n_boxes = sum(len(r.boxes) for r in meta.records[split])
        lines.append(f"  {split:<12}{meta.clipped_boxes[split]:>7} of {n_boxes:>6} boxes")

    every = [r for split in SPLITS for r in meta.records[split]]
    hours = Counter(r.hour for r in every)
    lines += ["", "Capture hour histogram (kept images, all splits)"]
    lines += [f"  {hour:02d}  {hours[hour]:>6}" for hour in range(24)]

    flagged = [r.night for r in every if r.night is not None]
    lines.append("")
    if flagged:
        lines.append(
            f"Night flag: {len(flagged)} of {len(every)} images flagged, "
            f"{sum(flagged)} night ({sum(flagged) / len(flagged):.3f})"
        )
    else:
        lines.append("Night flag: no day/night sidecar")
    lines.append(f"Classes: {NUM_CLASSES}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--root", type=Path, required=True, help="directory holding the 5 files")
    parser.add_argument("--sidecar", type=Path, default=None, help="day/night sidecar JSON")
    args = parser.parse_args(argv)
    print(summary(load_metadata(args.root, args.sidecar)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
