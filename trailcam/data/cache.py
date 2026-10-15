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

Item 6.08: a committed GPU run of the 6.04 small CNN sat at "Starting..." for about 48 minutes
with an empty log while the GPU quota fell by about that much (2026-10-15). Why is not known: a
CPU run with the same input of 57,864 small files logged its first line in about 5 minutes. One
file is still a lighter input than 57,864, so the cache also ships as one uncompressed tar
(`scripts/kaggle_pack_cache.py`) with the tree under `CACHE_TOP`, attached as the packing
notebook's output (made a dataset, Kaggle extracts it), and `unpack_cache` extracts it to local
disk at the start of a run (about 35 s for the 2.99 GiB archive). It checks every member before
writing anything: only regular files and folders, every name relative, without `..` and under
`CACHE_TOP`; links of any kind are refused. The extraction itself uses tarfile's `data` filter
too. The destination may not lie under `/kaggle/working`, so the copy is not saved as output.
"""

from __future__ import annotations

import os
import shutil
import stat
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, Metadata

MAX_DEPTH: Final = 8  # folder levels searched below the root; the Kaggle mount needs 4
CACHE_TOP: Final = "cct20_512"  # the archive's one top folder: annotations/, images/, sidecar
ARCHIVE_SUFFIX: Final = ".tar"
KAGGLE_WORKING: Final = Path("/kaggle/working")  # saved as the version's output
UNPACK_MARGIN: Final = 64 << 20  # bytes kept free beyond the files' blocks, as the packer keeps


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


def archive_files(root: Path, max_depth: int = MAX_DEPTH) -> list[Path]:
    """Every `ARCHIVE_SUFFIX` file at most `max_depth` levels below `root`, sorted; a missing
    root gives []. Lists the candidates when the notebook's archive path is wrong."""
    found: list[Path] = []
    for folder, subfolders, files in os.walk(root, followlinks=True):
        found += [Path(folder) / name for name in files if name.endswith(ARCHIVE_SUFFIX)]
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


class ArchiveError(ValueError):
    """The archive holds a member `unpack_cache` refuses; nothing was extracted."""


@dataclass(frozen=True)
class Unpacked:
    root: Path  # the data root: dest / CACHE_TOP
    files: int
    size: int  # bytes of the regular files
    seconds: float

    def __str__(self) -> str:
        return (
            f"unpacked {self.files} files ({self.size / (1 << 30):.2f} GiB) into {self.root} "
            f"in {self.seconds:.1f} s"
        )


def _checked_members(tar: tarfile.TarFile, top: str) -> list[tarfile.TarInfo]:
    """Every member, refusing any that is not a regular file or a folder, or whose name is
    absolute, holds `..` or lies outside `top`. Counts and kinds only in the message (R03)."""
    members = tar.getmembers()
    refused: dict[str, int] = {}
    for member in members:
        path = PurePosixPath(member.name)
        if not (member.isfile() or member.isdir()):
            reason = "links or special files"
        elif path.is_absolute():
            reason = "absolute names"
        elif ".." in path.parts:
            reason = "names with .."
        elif not path.parts or path.parts[0] != top:
            reason = f"names outside {top}/"
        else:
            continue
        refused[reason] = refused.get(reason, 0) + 1
    if refused:
        found = ", ".join(f"{n} {reason}" for reason, n in refused.items())
        raise ArchiveError(f"the archive holds {found}; nothing was extracted")
    return members


def space_needed(members: list[tarfile.TarInfo], block: int) -> int:
    """Bytes the members take on disk, about: each file rounded up to whole blocks, one block per
    folder, plus `UNPACK_MARGIN` for inodes and folder entries (review 6.08: about 100 to 150 MB
    of rounding over 57,864 files)."""
    blocks = sum(-(-m.size // block) if m.isfile() else 1 for m in members)
    return blocks * block + UNPACK_MARGIN


def unpack_cache(archive: Path, dest: Path, top: str = CACHE_TOP) -> Unpacked:
    """Extract the cache archive into `dest` and return the data root, `dest / top`, with counts
    and the seconds taken. Refuses a `dest` under `KAGGLE_WORKING`, an existing `dest / top`, too
    little free space and any member `_checked_members` refuses, all before writing anything."""
    started = time.perf_counter()
    if dest.resolve().is_relative_to(KAGGLE_WORKING):
        raise ValueError(f"{dest} is under {KAGGLE_WORKING}, which is saved as output")
    root = dest / top
    if root.exists():
        raise FileExistsError(f"{root} already exists: restart the session and run again")
    with tarfile.open(archive, "r:") as tar:  # uncompressed: JPEGs do not compress
        members = _checked_members(tar, top)
        files = [m for m in members if m.isfile()]
        size = sum(m.size for m in files)
        disk = next(p for p in (dest, *dest.parents) if p.exists())  # dest may not exist yet
        needed = space_needed(members, os.statvfs(disk).f_frsize or 4096)
        free = shutil.disk_usage(disk).free
        if free < needed:
            raise OSError(
                f"{dest} has {free} bytes free, the archive needs about {needed} "
                f"({size} bytes of files, rounded to blocks, and a margin); nothing was extracted"
            )
        dest.mkdir(parents=True, exist_ok=True)
        tar.extractall(dest, members=members, filter="data")
    if not root.is_dir():
        raise ArchiveError(f"the archive holds no {top}/ folder")
    return Unpacked(root, len(files), size, time.perf_counter() - started)
