"""Checks on the 512 px image cache before training reads it (item 6.07).

The cache of item 6.01 is a Kaggle dataset: `annotations/`, `images/` and the day/night sidecar
under one folder. Two things went wrong in the user's Kaggle rehearsal (2026-10-14/15), and each
check here makes one fail in seconds instead of at the first DataLoader batch:

- Kaggle mounted the dataset at `/kaggle/input/datasets/<user>/<slug>/<folder>`, deeper than the
  notebook assumed. `sidecar_folders` lists every folder under a root that holds a
  `DAY_NIGHT_SIDECAR`, at any depth up to `MAX_DEPTH`, so a wrong `DATA_ROOT` names the right one.
- One session start loaded a small part of the cache (645 files, 64.52 MiB of 2.92 GiB), so
  whether every image is there must be checked, not assumed. `check_images` stats the image of
  every kept record of the five splits (one `stat` per image, no decoding) and refuses when any is
  missing or empty, with counts per split only: never a file name or an image id (R03).

Images dropped by the metadata rules (empty, two species, rare) are never read, so they are not
checked. `trailcam.train.train.train_run` calls `check_images` before the split is drawn, so every
run is protected, not only notebook runs; the notebook calls it in its cache cell too.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Final

from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, Metadata

MAX_DEPTH: Final = 8  # folder levels searched below the root; the Kaggle mount needs 4


def sidecar_folders(root: Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    """Every folder at most `max_depth` levels below `root` (itself included) that holds a
    `DAY_NIGHT_SIDECAR`, sorted. Symlinked folders are followed; a missing root gives []."""
    found: list[Path] = []
    for folder, subfolders, files in os.walk(root, followlinks=True):
        if DAY_NIGHT_SIDECAR in files:
            found.append(Path(folder))
        if len(Path(folder).relative_to(root).parts) >= max_depth:
            subfolders.clear()
    return sorted(found)


def _is_image_file(path: Path) -> bool:
    """A regular file (or a link to one) with at least one byte."""
    try:
        info = path.stat()
    except OSError:
        return False
    return stat.S_ISREG(info.st_mode) and info.st_size > 0


def missing_images(meta: Metadata, image_root: Path) -> dict[str, tuple[int, int]]:
    """Per split, (kept records, of those whose image under `image_root` is missing or empty)."""
    return {
        split: (
            len(meta.records[split]),
            sum(1 for r in meta.records[split] if not _is_image_file(image_root / r.file_name)),
        )
        for split in SPLITS
    }


def check_images(meta: Metadata, image_root: Path) -> str:
    """Refuse a cache that lacks the image of any kept record; return a line of counts if none.

    Raises FileNotFoundError naming the expected and found counts in all and per split, and the
    folder; never a file name or an image id (R03).
    """
    if not image_root.is_dir():
        raise FileNotFoundError(f"the image folder {image_root} does not exist")
    counts = missing_images(meta, image_root)
    expected = sum(kept for kept, _ in counts.values())
    missing = sum(lost for _, lost in counts.values())
    if missing:
        per_split = ", ".join(
            f"{split} {kept - lost} of {kept}" for split, (kept, lost) in counts.items()
        )
        raise FileNotFoundError(
            f"the image cache is incomplete: {expected - missing} of {expected} kept images found "
            f"under {image_root} ({missing} missing or empty; found per split: {per_split}). "
            "Is the whole dataset attached? Restart the session and check again before training."
        )
    return f"all {expected} kept images of the {len(SPLITS)} splits found under {image_root}"
