"""Synthetic CCT20-shaped fixture for the tests and the dry run (plan.md section 5).

This module writes only into the directory it is given (a pytest `tmp_path` or the dry run's
temporary directory); nothing it writes is committed. The metadata part (item 3.02) writes the five
`<split>_annotations.json` files and a day/night sidecar; the images part is item 4.03.

Layout: the 12 kept classes; 2 cis locations (train, cis_val, cis_test) and 1 trans location
(trans_val, trans_test); every sequence has 3 frames; about 90% of animal annotations carry a box.
Each class has `TRAIN_SEQUENCES + 1` sequences per cis location in the train and cis_val pool, so
a re-split stratified by class and location has sequences to put on both sides. The cases the
loader must filter are planted on purpose: empty sequences outside train, one badger, deer and fox
sequence, a bobcat + rabbit frame in train and trans_test, a two-opossum frame in train and
cis_test (kept), and one sequence split across train and cis_val as in the released files.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from trailcam.data.classes import CLASSES, DROPPED_CCT_IDS, INDEX_TO_CCT_ID, index_of_name
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, annotation_file_name

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
