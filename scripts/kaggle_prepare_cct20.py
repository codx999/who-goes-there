"""Build the 512 px CCT20 image cache and its day/night sidecar on Kaggle (item 6.01).

The cache and sidecar formats are set in plan.md section 2. Run once by item 6.03, in a Kaggle
notebook with internet on, as its own process (`!python ...`, not `%run`: the encoder check
starts fresh processes that import this file):

    python scripts/kaggle_prepare_cct20.py   # defaults: --download-dir /tmp/cct20
                                             #           --out /kaggle/working/cct20_512

1. Download `eccv_18_annotations.tar.gz` and `eccv_18_all_images_sm.tar.gz` (6.5 GB, at most 1024 px
   per side) from LILA's GCP mirror into `--download-dir`. A file already there at the published
   size is reused; a partial one is resumed.
2. Print the Python, Pillow and libjpeg versions, then check that the JPEG encoder is reproducible:
   the same images (synthetic ones and the first real ones of the tarball) are resized and encoded
   twice in each of two separate processes and must give identical bytes every time; otherwise the
   script stops before writing anything (the 4.03 handoff: Pillow 10.1 on macOS arm64 was not).
3. Stream the image tarball once, never extracting it to disk. Each JPEG named in the annotations is
   decoded, resized so its longer side is 512 px (Lanczos, never enlarged), saved as a JPEG under
   `--out/images/<file_name>` and flagged day or night from its resized pixels
   (`trailcam.data.daynight`). The five annotation files are copied to `--out/annotations/`.
4. Only if every count checks, write `--out/day_night.json` (`DAY_NIGHT_SIDECAR`): image id to
   `{"night": bool, "hour": int, "spread": float}` for all 57,864 images, the hour from
   `date_captured`. It is written last, so a failed run leaves no sidecar and training (which
   requires one entry per kept image) refuses the cache.

Output is counts only (R03): the tarball layout as member types, file extensions and path depths,
images per split, the night share per split and per hour, the channel-spread histogram, sizes and
bytes; never a file name, id or path from inside an archive. Exits 1 when the annotations or the
tarball do not hold exactly 57,864 images, an annotated image is missing, repeated or unreadable,
or the encoder check fails. It refuses to start outside Kaggle unless `--off-kaggle` is given.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import platform
import random
import shutil
import sys
import tarfile
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Final

import PIL
from PIL import Image, ImageDraw, ImageFilter, features

from trailcam.data.daynight import NIGHT_MAX_SPREAD, channel_spread
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, annotation_file_name, capture_hour

LILA: Final = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
ANNOTATIONS_URL: Final = LILA + "eccv_18_annotations.tar.gz"
IMAGES_URL: Final = LILA + "eccv_18_all_images_sm.tar.gz"
EXPECTED_IMAGES: Final = 57_864

KAGGLE: Final = Path("/kaggle")
DEFAULT_DOWNLOAD_DIR: Final = Path("/tmp/cct20")
DEFAULT_OUT: Final = Path("/kaggle/working/cct20_512")
IMAGE_DIR: Final = "images"
ANNOTATION_DIR: Final = "annotations"

LONGER_SIDE: Final = 512
JPEG_QUALITY: Final = 90
PROBE_REAL: Final = 16  # real images from the start of the tarball in the encoder check
PROBE_SIZES: Final = ((1024, 747), (1023, 769), (1024, 576), (600, 450), (300, 201))
SPREAD_EDGES: Final = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
PROGRESS_EVERY: Final = 5_000

CHUNK: Final = 8 << 20
GIB: Final = 1 << 30
ATTEMPTS: Final = 5
TIMEOUT: Final = 60


class PrepareError(Exception):
    """A check failed; the message holds counts only."""


@dataclass(frozen=True)
class ImageInfo:
    """What the cache needs of one annotated image, keyed by its `file_name`."""

    image_id: str
    split: str
    hour: int


@dataclass(frozen=True)
class Cached:
    """One image as the cache stores it: the JPEG bytes and what was measured on the way."""

    jpeg: bytes
    source_mode: str
    source_size: tuple[int, int]
    size: tuple[int, int]
    spread: float


# ---------------------------------------------------------------------------------------------
# Download


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return int(response.headers["Content-Length"])


def download(url: str, dest: Path) -> Path:
    """`url` into `dest`, resuming a `.part` file with an HTTP range request after a failure."""
    expected = remote_size(url)
    if dest.is_file() and dest.stat().st_size == expected:
        print(f"Reusing {dest} ({expected} bytes)")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, ATTEMPTS + 1):
        have = part.stat().st_size if part.is_file() else 0
        if have >= expected:
            break
        free = shutil.disk_usage(dest.parent).free
        if free < expected - have:
            raise PrepareError(f"{dest.parent} has {free} bytes free, {expected - have} needed")
        headers = {"Range": f"bytes={have}-"} if have else {}
        print(f"Downloading {url} (attempt {attempt}, {have} of {expected} bytes present)")
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                resumed = response.status == 206
                with part.open("ab" if resumed else "wb") as out:
                    done = have if resumed else 0
                    next_report = (done // GIB + 1) * GIB
                    while chunk := response.read(CHUNK):
                        out.write(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            print(f"  {done / GIB:.1f} of {expected / GIB:.1f} GiB")
                            next_report += GIB
        except OSError as error:  # URLError, HTTPError and timeouts are OSErrors
            print(f"  attempt {attempt} failed: {type(error).__name__}")
    size = part.stat().st_size if part.is_file() else 0
    if size != expected:
        raise PrepareError(f"{dest.name}: {size} bytes downloaded, {expected} published")
    part.replace(dest)
    return dest


# ---------------------------------------------------------------------------------------------
# Annotations


def read_annotations(archive: Path) -> tuple[dict[str, ImageInfo], dict[str, bytes]]:
    """Every image of the five annotation files in `archive`, by file name, and the raw files by
    split. Refuses a missing or repeated file, a repeated image id or file name, and a file name
    with a folder part (the cache stores each image flat under its file name)."""
    raw: dict[str, bytes] = {}
    names = {annotation_file_name(split): split for split in SPLITS}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            split = names.get(PurePosixPath(member.name).name)
            if split is None or not member.isfile():
                continue
            if split in raw:
                raise PrepareError(f"two {annotation_file_name(split)} in the annotations archive")
            body = tar.extractfile(member)
            if body is None:
                raise PrepareError(f"{annotation_file_name(split)} could not be read")
            raw[split] = body.read()
    missing = [annotation_file_name(split) for split in SPLITS if split not in raw]
    if missing:
        raise PrepareError(f"the annotations archive lacks {', '.join(missing)}")

    images: dict[str, ImageInfo] = {}
    ids: set[str] = set()
    for split in SPLITS:
        for image in json.loads(raw[split])["images"]:
            file_name, image_id = str(image["file_name"]), str(image["id"])
            if file_name in images or image_id in ids:
                raise PrepareError(f"{split}: an image id or file name is repeated")
            if PurePosixPath(file_name).name != file_name or not file_name.endswith(".jpg"):
                raise PrepareError(f"{split}: a file name has a folder part or is not a .jpg")
            images[file_name] = ImageInfo(image_id, split, capture_hour(image["date_captured"]))
            ids.add(image_id)
    return images, raw


# ---------------------------------------------------------------------------------------------
# One image


def resized(image: Image.Image) -> Image.Image:
    """`image` (as L or RGB) scaled so its longer side is `LONGER_SIDE`; a smaller one as is."""
    if image.mode not in ("L", "RGB"):
        image = image.convert("RGB")
    width, height = image.size
    scale = LONGER_SIDE / max(width, height)
    if scale >= 1:
        return image
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def process(data: bytes) -> Cached:
    """Decode a source JPEG, resize it and encode it for the cache."""
    with Image.open(io.BytesIO(data)) as source:
        source.load()
        mode, source_size = source.mode, source.size
        small = resized(source)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=JPEG_QUALITY)
    return Cached(buffer.getvalue(), mode, source_size, small.size, channel_spread(small))


# ---------------------------------------------------------------------------------------------
# Encoder check


def synthetic_sources() -> list[bytes]:
    """Deterministic RGB and grayscale test JPEGs at camera-trap sizes, some not multiples of 8."""
    sources = []
    for index, size in enumerate(PROBE_SIZES):
        rng = random.Random(index)
        noise = Image.frombytes("RGB", size, rng.randbytes(size[0] * size[1] * 3))
        image = Image.blend(Image.new("RGB", size, (90, 120, 60)), noise, 0.4)
        image = image.filter(ImageFilter.GaussianBlur(1))
        ImageDraw.Draw(image).rectangle((size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2))
        for variant in (image, image.convert("L"), image.convert("L").convert("RGB")):
            buffer = io.BytesIO()
            variant.save(buffer, "JPEG", quality=95)
            sources.append(buffer.getvalue())
    return sources


def real_sources(images_tar: Path, annotations: Mapping[str, ImageInfo]) -> list[bytes]:
    """The first `PROBE_REAL` annotated JPEGs of the tarball (read from its start only)."""
    sources: list[bytes] = []
    with tarfile.open(images_tar, "r|gz") as tar:
        for member in tar:
            if member.isfile() and PurePosixPath(member.name).name in annotations:
                body = tar.extractfile(member)
                if body is not None:
                    sources.append(body.read())
                if len(sources) == PROBE_REAL:
                    break
    return sources


def fingerprints(sources: Sequence[bytes]) -> tuple[int, list[tuple[str, str]]]:
    """This process's id and, per source, the SHA-256 of its cache JPEG encoded twice."""
    hashes = []
    for data in sources:
        first, second = (hashlib.sha256(process(data).jpeg).hexdigest() for _ in range(2))
        hashes.append((first, second))
    return os.getpid(), hashes


def count_mismatches(runs: Sequence[Sequence[tuple[str, str]]]) -> int:
    """Sources whose encodings differ anywhere: between the two in one run or between runs."""
    if len({len(run) for run in runs}) != 1:
        raise PrepareError("the encoder check runs covered different numbers of images")
    return sum(len({h for run in runs for h in run[i]}) != 1 for i in range(len(runs[0])))


def check_encoder(sources: Sequence[bytes]) -> None:
    """Encode `sources` in two separate fresh processes and require identical bytes."""
    context = multiprocessing.get_context("spawn")
    pids, runs = [], []
    for _ in range(2):
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
                pid, hashes = pool.submit(fingerprints, sources).result()
        except BrokenProcessPool:
            raise PrepareError("an encoder check process died before it answered") from None
        pids.append(pid)
        runs.append(hashes)
    if len({*pids, os.getpid()}) != 3:
        raise PrepareError("the encoder check did not run in two separate processes")
    mismatched = count_mismatches(runs)
    print(f"Encoder check: {len(sources)} images, 2 processes x 2 encodings, {mismatched} differ")
    if mismatched:
        raise PrepareError(
            f"the JPEG encoder is not reproducible ({mismatched} of {len(sources)} images): "
            "upgrade Pillow in the notebook and run again"
        )


def versions() -> str:
    return (
        f"Python {platform.python_version()} on {platform.system()} {platform.machine()}; "
        f"Pillow {PIL.__version__}; libjpeg-turbo {features.version('libjpeg_turbo')}, "
        f"libjpeg {features.version('jpg')}"
    )


# ---------------------------------------------------------------------------------------------
# The cache


@dataclass
class Report:
    """Counts gathered over one pass of the image tarball, and the sidecar entries."""

    member_types: Counter[str] = field(default_factory=Counter)
    extensions: Counter[str] = field(default_factory=Counter)
    depths: Counter[int] = field(default_factory=Counter)
    top_level: set[str] = field(default_factory=set)
    jpeg_folders: set[str] = field(default_factory=set)
    jpegs: int = 0
    unannotated: int = 0
    repeated: int = 0
    unreadable: int = 0
    written: Counter[str] = field(default_factory=Counter)
    night: Counter[str] = field(default_factory=Counter)
    hours: Counter[int] = field(default_factory=Counter)
    night_hours: Counter[int] = field(default_factory=Counter)
    spreads: Counter[int] = field(default_factory=Counter)  # bin index over SPREAD_EDGES
    near_threshold: int = 0
    source_modes: Counter[str] = field(default_factory=Counter)
    source_sides: Counter[int] = field(default_factory=Counter)  # longer side
    sides: Counter[int] = field(default_factory=Counter)
    jpeg_bytes: int = 0
    entries: dict[str, dict[str, bool | int | float]] = field(default_factory=dict)

    def add_member(self, member: tarfile.TarInfo) -> PurePosixPath:
        path = PurePosixPath(member.name.removeprefix("./"))
        kind = "file" if member.isfile() else "directory" if member.isdir() else "other"
        self.member_types[kind] += 1
        if path.parts:
            self.top_level.add(path.parts[0])
        if member.isfile():
            self.extensions[path.suffix.lower() or "(none)"] += 1
            self.depths[len(path.parts) - 1] += 1
        return path

    def add_image(self, info: ImageInfo, cached: Cached) -> None:
        night = cached.spread <= NIGHT_MAX_SPREAD
        self.written[info.split] += 1
        self.night[info.split] += night
        self.hours[info.hour] += 1
        self.night_hours[info.hour] += night
        self.spreads[sum(cached.spread >= edge for edge in SPREAD_EDGES)] += 1
        self.near_threshold += NIGHT_MAX_SPREAD / 2 <= cached.spread <= 2 * NIGHT_MAX_SPREAD
        self.source_modes[cached.source_mode] += 1
        self.source_sides[max(cached.source_size)] += 1
        self.sides[max(cached.size)] += 1
        self.jpeg_bytes += len(cached.jpeg)
        self.entries[info.image_id] = {
            "night": night,
            "hour": info.hour,
            "spread": round(cached.spread, 3),
        }

    @property
    def total_written(self) -> int:
        return sum(self.written.values())

    def problems(self, annotated: int, expected: int) -> list[str]:
        found = []
        if annotated != expected:
            found.append(f"the annotations hold {annotated} images, expected {expected}")
        if self.jpegs != expected:
            found.append(f"the tarball holds {self.jpegs} JPEGs, expected {expected}")
        missing = annotated - self.total_written - self.unreadable
        for count, what in (
            (missing, "annotated images not in the tarball"),
            (self.repeated, "annotated images repeated in the tarball"),
            (self.unreadable, "images that could not be decoded"),
            (self.unannotated, "JPEGs in the tarball not in the annotations"),
        ):
            if count:
                found.append(f"{count} {what}")
        if self.total_written != expected:
            found.append(f"{self.total_written} images written, expected {expected}")
        return found


def stream_cache(images_tar: Path, annotations: Mapping[str, ImageInfo], image_dir: Path) -> Report:
    """One pass over the tarball: every annotated JPEG resized into `image_dir`."""
    image_dir.mkdir(parents=True, exist_ok=True)
    report = Report()
    with tarfile.open(images_tar, "r|gz") as tar:
        for member in tar:
            path = report.add_member(member)
            if not member.isfile() or path.suffix.lower() != ".jpg":
                continue
            report.jpegs += 1
            report.jpeg_folders.add(str(path.parent))
            info = annotations.get(path.name)
            if info is None:
                report.unannotated += 1
                continue
            if info.image_id in report.entries:
                report.repeated += 1
                continue
            body = tar.extractfile(member)
            try:
                if body is None:
                    raise OSError("no data")
                cached = process(body.read())
            except (OSError, ValueError, Image.DecompressionBombError):
                report.unreadable += 1
                continue
            (image_dir / path.name).write_bytes(cached.jpeg)
            report.add_image(info, cached)
            if report.jpegs % PROGRESS_EVERY == 0:
                print(f"  {report.jpegs} JPEGs read, {report.total_written} written")
    return report


def share(part: int, whole: int) -> str:
    return f"{part / whole:.3f}" if whole else "-"


def summary(report: Report, annotated: Mapping[str, int]) -> str:
    """Counts only: the tarball layout, images per split, night shares, spreads, sizes, bytes."""
    types = ", ".join(f"{kind} {n}" for kind, n in sorted(report.member_types.items()))
    extensions = ", ".join(f"{ext} {n}" for ext, n in report.extensions.most_common())
    depths = ", ".join(f"{depth}: {n}" for depth, n in sorted(report.depths.items()))
    lines = [
        "Tarball layout (counts only)",
        f"  members: {types}",
        f"  file extensions: {extensions}",
        f"  file depth (folders above the file): {depths}",
        f"  top-level entries: {len(report.top_level)}; "
        f"folders holding .jpg files: {len(report.jpeg_folders)}",
        "",
        f"JPEGs in the tarball: {report.jpegs}; not annotated: {report.unannotated}; "
        f"repeated: {report.repeated}; unreadable: {report.unreadable}",
        f"{'split':<12}{'annotated':>10}{'written':>10}{'night':>8}{'share':>7}",
    ]
    for split in SPLITS:
        written, night = report.written[split], report.night[split]
        lines.append(
            f"{split:<12}{annotated[split]:>10}{written:>10}{night:>8}{share(night, written):>7}"
        )
    total, nights = report.total_written, sum(report.night.values())
    lines.append(
        f"{'all':<12}{sum(annotated.values()):>10}{total:>10}{nights:>8}{share(nights, total):>7}"
    )

    lines += ["", "Night share per capture hour (written images)"]
    for hour in range(24):
        n, night = report.hours[hour], report.night_hours[hour]
        lines.append(f"  {hour:02d}  {n:>6}  night {night:>6}  {share(night, n)}")

    edges = (0.0, *SPREAD_EDGES)
    lines += ["", f"Channel spread (grey levels; night when <= {NIGHT_MAX_SPREAD})"]
    for index, low in enumerate(edges):
        high = f"{edges[index + 1]:g}" if index + 1 < len(edges) else "255"
        lines.append(f"  [{low:g}, {high}){report.spreads[index]:>8}")
    lines.append(
        f"  within [{NIGHT_MAX_SPREAD / 2:g}, {2 * NIGHT_MAX_SPREAD:g}] of the threshold: "
        f"{report.near_threshold}"
    )

    modes = ", ".join(f"{mode} {n}" for mode, n in sorted(report.source_modes.items()))
    lines += [
        "",
        f"Source modes: {modes}",
        f"Source longer side: {min(report.source_sides, default=0)} to "
        f"{max(report.source_sides, default=0)} px; cache longer side: "
        f"{min(report.sides, default=0)} to {max(report.sides, default=0)} px",
        f"Cache JPEG bytes: {report.jpeg_bytes} ({report.jpeg_bytes / GIB:.2f} GiB)",
    ]
    return "\n".join(lines)


def write_sidecar(report: Report, path: Path, annotations: Mapping[str, ImageInfo]) -> None:
    """The day/night sidecar, written to a temporary name and renamed, then read back with the
    keys every annotated image id must have."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report.entries, sort_keys=True))
    temporary.replace(path)
    back = json.loads(path.read_text())
    ids = {info.image_id for info in annotations.values()}
    if set(back) != ids or any(
        set(entry) != {"night", "hour", "spread"} for entry in back.values()
    ):
        raise PrepareError("the sidecar read back does not cover every annotated image")


def prepare(
    annotations_tar: Path, images_tar: Path, out: Path, expected: int = EXPECTED_IMAGES
) -> Report:
    """Steps 2 to 4 of the module docstring on two downloaded archives; raises PrepareError."""
    annotations, raw = read_annotations(annotations_tar)
    annotated = Counter(info.split for info in annotations.values())
    print("Annotated images: " + ", ".join(f"{s} {annotated[s]}" for s in SPLITS))
    if len(annotations) != expected:
        raise PrepareError(f"the annotations hold {len(annotations)} images, expected {expected}")

    check_encoder(synthetic_sources() + real_sources(images_tar, annotations))

    annotation_dir = out / ANNOTATION_DIR
    annotation_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (annotation_dir / annotation_file_name(split)).write_bytes(raw[split])
    print(f"Writing the cache into {out / IMAGE_DIR}")
    report = stream_cache(images_tar, annotations, out / IMAGE_DIR)
    print(summary(report, annotated))
    problems = report.problems(len(annotations), expected)
    if problems:
        raise PrepareError("; ".join(problems) + "; no sidecar written")
    write_sidecar(report, out / DAY_NIGHT_SIDECAR, annotations)
    print(f"Sidecar: {out / DAY_NIGHT_SIDECAR} ({len(report.entries)} images)")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="new or empty directory")
    parser.add_argument(
        "--off-kaggle", action="store_true", help="allow a run outside Kaggle (6.5 GB download)"
    )
    args = parser.parse_args(argv)
    if not KAGGLE.is_dir() and not args.off_kaggle:
        parser.error(f"{KAGGLE} not found: this runs on Kaggle (pass --off-kaggle to override)")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"{args.out} is not empty: the cache is written into a new directory")

    print(versions())
    try:
        annotations_tar = download(
            ANNOTATIONS_URL, args.download_dir / "eccv_18_annotations.tar.gz"
        )
        images_tar = download(IMAGES_URL, args.download_dir / "eccv_18_all_images_sm.tar.gz")
        prepare(annotations_tar, images_tar, args.out)
    except PrepareError as error:
        print(f"kaggle_prepare_cct20: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
