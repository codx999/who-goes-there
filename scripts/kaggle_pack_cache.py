"""Pack the attached 512 px cache dataset into one uncompressed tar on Kaggle (item 6.08).

A committed GPU run that attached the 6.03 cache (57,864 small JPEGs) sat at "Starting..." for
about 48 minutes with an empty log, billed (2026-10-15). Why is not known: a CPU run with the same
input logged its first line in about 5 minutes. One 2.9 GiB file is still a lighter input, and
`trailcam.data.cache.unpack_cache` extracts it to local disk at the start of each run (about
35 s). Run once, in a CPU session with the `cct20-512` dataset attached and
`trailcam` installed from a clone (as for 6.03), as its own process:

    python scripts/kaggle_pack_cache.py   # defaults: --cache <the 6.03 dataset's folder>
                                          #           --out /kaggle/working/cct20_512.tar

Packing the dataset, not rebuilding the cache, keeps the exact bytes the sidecar's day/night
flags were measured on and the rehearsals checked: no 6.5 GB download, no re-encode under another
Pillow. Before anything is written, the cache must be whole: the five annotation files, the
sidecar with one entry per annotated image, and in `images/` exactly the annotated JPEGs, none
empty. An interactive session can load the dataset lazily, so run it with Save & Run All (the
2026-10-15 packing run found every file); a partial cache is refused either way. The tar holds
`annotations/`, `images/` and the sidecar under `CACHE_TOP`, in sorted order with zeroed owners
and times, so the same cache gives the same bytes; it is written to a `.part` name and renamed
only when complete.

Steps on Kaggle (the user, once):

1. New Notebook. Session options: Accelerator None (a CPU session, no GPU quota), Internet on
   (the clone and pip).
2. Add Input: the private `cct20-512` dataset of item 6.03.
3. One code cell, with `<COMMIT>` the `main` commit the coordinator gives (at or after 6.08). The
   clone goes to `/tmp`, so `/kaggle/working` holds the archive alone and the output is one file:

       !git clone --quiet https://github.com/codx999/who-goes-there /tmp/who-goes-there
       !git -C /tmp/who-goes-there checkout --quiet --detach <COMMIT>
       !pip install --quiet /tmp/who-goes-there
       !python /tmp/who-goes-there/scripts/kaggle_pack_cache.py

4. Save Version > Save & Run All (Commit), and wait for the version to finish. Its log ends with
   the file counts, the size, the seconds and the archive's SHA-256.
5. Keep the version: the training notebook attaches its output with Add Input > Notebook, at
   `/kaggle/input/notebooks/<user>/<notebook>/cct20_512.tar`, one file of about 2.9 GiB. Do not
   make the output a dataset: Kaggle then extracts the `.tar` (the `cct20-512-tar` dataset came
   out as folders, 2026-10-15). Note the SHA-256 and the size in the progress log.

Output is counts only (R03): files per folder, bytes, seconds and the archive's SHA-256; never a
file name or an id. Exits 1 when a check fails. Refuses to start outside Kaggle unless
`--off-kaggle` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Final

from trailcam.data.cache import ARCHIVE_SUFFIX, CACHE_TOP, sidecar_folders
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, annotation_file_name

KAGGLE: Final = Path("/kaggle")
KAGGLE_INPUT: Final = KAGGLE / "input"
DEFAULT_CACHE: Final = KAGGLE_INPUT / "datasets" / "codx888" / "cct20-512" / "cct20_512"
DEFAULT_OUT: Final = Path("/kaggle/working") / f"{CACHE_TOP}{ARCHIVE_SUFFIX}"
ANNOTATION_DIR: Final = "annotations"  # the layout of scripts/kaggle_prepare_cct20.py
IMAGE_DIR: Final = "images"
EXPECTED_IMAGES: Final = 57_864  # as in scripts/kaggle_prepare_cct20.py
PROGRESS_EVERY: Final = 10_000
GIB: Final = 1 << 30


class PackError(Exception):
    """A check failed; the message holds counts only."""


@dataclass(frozen=True)
class Packed:
    files: dict[str, int]  # regular files per part: annotations, images, sidecar
    size: int  # bytes of the archive
    sha256: str
    seconds: float


def cache_files(cache: Path, expected: int = EXPECTED_IMAGES) -> list[Path]:
    """The cache's files, relative to `cache` and sorted, after checking it is whole."""
    annotations = [Path(ANNOTATION_DIR) / annotation_file_name(split) for split in SPLITS]
    missing = [str(p) for p in annotations if not (cache / p).is_file()]
    if missing or not (cache / DAY_NIGHT_SIDECAR).is_file():
        raise PackError(
            f"{cache} lacks {len(missing)} of {len(SPLITS)} annotation files or the sidecar; "
            f"folders under {KAGGLE_INPUT} holding a {DAY_NIGHT_SIDECAR}: "
            f"{[str(f) for f in sidecar_folders(KAGGLE_INPUT)] or 'none'}"
        )
    names: set[str] = set()
    for path in annotations:
        names |= {
            str(image["file_name"]) for image in json.loads((cache / path).read_bytes())["images"]
        }
    entries = len(json.loads((cache / DAY_NIGHT_SIDECAR).read_bytes()))
    folder = cache / IMAGE_DIR
    present = {p.name: p for p in folder.iterdir()} if folder.is_dir() else {}
    absent = sum(1 for name in names if name not in present)
    empty = sum(1 for name in names if name in present and not _non_empty_file(present[name]))
    extra = len(present.keys() - names)
    problems = [
        f"{n} {what}"
        for n, what in (
            (absent, "annotated images missing from images/"),
            (empty, "annotated images empty or not regular files"),
            (extra, "entries in images/ not in the annotations"),
        )
        if n
    ]
    if len(names) != expected:
        problems.append(f"the annotations name {len(names)} images, expected {expected}")
    if entries != len(names):
        problems.append(f"the sidecar has {entries} entries for {len(names)} annotated images")
    if problems:
        raise PackError(
            "; ".join(problems) + f" ({len(present)} entries in images/). Is the whole dataset "
            "attached? Run the notebook with Save & Run All; nothing was written"
        )
    images = [Path(IMAGE_DIR) / name for name in sorted(names)]
    return [*annotations, Path(DAY_NIGHT_SIDECAR), *images]


def _non_empty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _info(name: str, size: int = 0, folder: bool = False) -> tarfile.TarInfo:
    """A member with zeroed owner and time, so the archive depends on the cache's bytes only."""
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if folder else tarfile.REGTYPE
    info.mode = 0o755 if folder else 0o644
    info.size, info.mtime, info.uid, info.gid, info.uname, info.gname = size, 0, 0, 0, "", ""
    return info


def write_archive(cache: Path, files: Sequence[Path], out: BinaryIO) -> None:
    """`files` (relative to `cache`) into `out` as a tar under `CACHE_TOP`, folders first."""
    with tarfile.open(fileobj=out, mode="w:", format=tarfile.PAX_FORMAT) as tar:
        folders = sorted({str(p.parent) for p in files} - {"."})
        for name in ("", *folders):
            tar.addfile(_info(f"{CACHE_TOP}/{name}".rstrip("/"), folder=True))
        for count, path in enumerate(files, start=1):
            source = cache / path
            with source.open("rb") as body:
                tar.addfile(_info(f"{CACHE_TOP}/{path.as_posix()}", source.stat().st_size), body)
            if count % PROGRESS_EVERY == 0:
                print(f"  {count} of {len(files)} files packed", flush=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack(cache: Path, out: Path, expected: int = EXPECTED_IMAGES) -> Packed:
    """Check the cache, then write it as one tar to `out` through a `.part` file."""
    started = time.perf_counter()
    files = cache_files(cache, expected)
    size = sum((cache / p).stat().st_size for p in files)
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < size + (64 << 20):  # tar headers: 1.5 KiB per file at most, with PAX
        raise PackError(f"{out.parent} has {free} bytes free, about {size} needed")
    print(f"Packing {len(files)} files ({size / GIB:.2f} GiB) from {cache}", flush=True)
    part = out.with_name(out.name + ".part")
    try:
        with part.open("wb") as handle:
            write_archive(cache, files, handle)
        part.replace(out)
    finally:
        part.unlink(missing_ok=True)
    parts = {
        ANNOTATION_DIR: sum(p.parts[0] == ANNOTATION_DIR for p in files),
        IMAGE_DIR: sum(p.parts[0] == IMAGE_DIR for p in files),
        DAY_NIGHT_SIDECAR: 1,
    }
    return Packed(parts, out.stat().st_size, file_sha256(out), time.perf_counter() - started)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="the attached cache")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the tar to write")
    parser.add_argument("--off-kaggle", action="store_true", help="allow a run outside Kaggle")
    args = parser.parse_args(argv)
    if not KAGGLE.is_dir() and not args.off_kaggle:
        parser.error(f"{KAGGLE} not found: this runs on Kaggle (pass --off-kaggle to override)")
    if args.out.exists():
        parser.error(f"{args.out} exists: the archive is written to a new path")
    try:
        packed = pack(args.cache, args.out, EXPECTED_IMAGES)
    except PackError as error:
        print(f"kaggle_pack_cache: {error}", file=sys.stderr)
        return 1
    files = ", ".join(f"{part} {n}" for part, n in packed.files.items())
    print(f"Archive {args.out}: {sum(packed.files.values())} files ({files}) under {CACHE_TOP}/")
    print(f"  {packed.size} bytes ({packed.size / GIB:.2f} GiB), {packed.seconds:.0f} s")
    print(f"  sha256 {packed.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
