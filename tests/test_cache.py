"""trailcam/data/cache.py: the cache's sidecar search and image completeness check (item 6.07),
and the archive finder and safe unpacking (item 6.08).

Run on the synthetic fixture, which writes one JPEG per image of its five annotation files, the
way the 512 px cache of item 6.01 holds one per image of CCT20. The unpacking tests build tiny
tars in memory: a normal one, and ones with a traversal, an absolute name, a link or a member
outside the top folder, each refused before anything is written.
"""

import io
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any

import pytest

from trailcam.data import cache as cache_module
from trailcam.data import synthetic
from trailcam.data.cache import (
    CACHE_TOP,
    MAX_DEPTH,
    UNPACK_MARGIN,
    ArchiveError,
    archive_files,
    check_images,
    missing_images,
    sidecar_folders,
    space_needed,
    unpack_cache,
)
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, Metadata, load_metadata
from trailcam.train import train
from trailcam.train.config import load_config


@pytest.fixture
def cache(tmp_path: Path) -> tuple[Metadata, Path]:
    written = synthetic.write_synthetic_fixture(tmp_path / "fixture")
    return load_metadata(tmp_path / "fixture", written.paths["sidecar"]), written.image_root


def _identifiers(meta: Metadata) -> set[str]:
    """Every image id, file name and sequence id of the kept records."""
    return {
        value
        for split in SPLITS
        for r in meta.records[split]
        for value in (r.image_id, r.file_name, r.seq_id)
    }


def test_a_complete_cache_passes(cache: tuple[Metadata, Path]) -> None:
    meta, images = cache
    total = sum(len(meta.records[split]) for split in SPLITS)
    assert missing_images(meta, images) == {s: (len(meta.records[s]), 0) for s in SPLITS}
    assert check_images(meta, images) == (
        f"all {total} kept images of the 5 splits found under {images}"
    )


@pytest.mark.parametrize("damage", ["missing", "empty"])
def test_one_missing_or_empty_image_is_refused_with_counts_only(
    cache: tuple[Metadata, Path], damage: str
) -> None:
    meta, images = cache
    record = meta.records["cis_val"][0]
    path = images / record.file_name
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"")
    total = sum(len(meta.records[split]) for split in SPLITS)
    kept = len(meta.records["cis_val"])
    assert missing_images(meta, images)["cis_val"] == (kept, 1)

    with pytest.raises(FileNotFoundError) as refused:
        check_images(meta, images)
    message = str(refused.value)
    assert f"{total - 1} of {total} kept images found" in message
    assert f"cis_val {kept - 1} of {kept}" in message
    assert f"train {len(meta.records['train'])} of {len(meta.records['train'])}" in message
    assert "1 missing or empty" in message
    assert not [value for value in _identifiers(meta) if value in message]  # R03


def test_a_missing_image_folder_is_refused(cache: tuple[Metadata, Path], tmp_path: Path) -> None:
    meta, _ = cache
    with pytest.raises(FileNotFoundError, match="does not exist"):
        check_images(meta, tmp_path / "no-images")


def test_dropped_images_are_not_required(cache: tuple[Metadata, Path]) -> None:
    """Only kept records are checked: an extra file under images/ changes nothing."""
    meta, images = cache
    (images / "not-a-record.jpg").write_bytes(b"\xff\xd8")
    assert missing_images(meta, images) == {s: (len(meta.records[s]), 0) for s in SPLITS}


def test_train_run_refuses_an_incomplete_cache_before_the_split(
    cache: tuple[Metadata, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every run is protected, not only notebook runs: nothing is split, built or trained."""
    meta, images = cache

    def no_split(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("the split was drawn before the images were checked")

    monkeypatch.setattr(train, "make_split", no_split)
    (images / meta.records["train"][0].file_name).unlink()
    with pytest.raises(FileNotFoundError, match="the image cache is incomplete"):
        train.train_run("small_cnn", load_config("small_cnn"), meta, images, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_the_sidecar_search_finds_kaggles_dataset_mount(tmp_path: Path) -> None:
    """Kaggle mounts a private dataset at datasets/<user>/<slug>/<folder>, four levels down."""
    mount = tmp_path / "datasets" / "codx888" / "cct20-512" / "cct20_512"
    (mount / "images").mkdir(parents=True)
    (mount / DAY_NIGHT_SIDECAR).write_text("{}")
    (tmp_path / "other" / "notes.json").parent.mkdir()
    (tmp_path / "other" / "notes.json").write_text("{}")
    assert sidecar_folders(tmp_path) == [mount]


def test_the_sidecar_search_depth(tmp_path: Path) -> None:
    (tmp_path / DAY_NIGHT_SIDECAR).write_text("{}")
    deepest = tmp_path.joinpath(*[f"d{i}" for i in range(MAX_DEPTH)])
    too_deep = deepest / "beyond"
    too_deep.mkdir(parents=True)
    (deepest / DAY_NIGHT_SIDECAR).write_text("{}")
    (too_deep / DAY_NIGHT_SIDECAR).write_text("{}")
    assert sidecar_folders(tmp_path) == [tmp_path, deepest]
    assert sidecar_folders(tmp_path, max_depth=0) == [tmp_path]
    assert sidecar_folders(tmp_path / "absent") == []


def test_the_sidecar_search_follows_a_linked_folder(tmp_path: Path) -> None:
    real = tmp_path / "real" / "cct20_512"
    real.mkdir(parents=True)
    (real / DAY_NIGHT_SIDECAR).write_text("{}")
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "cct20-512").symlink_to(tmp_path / "real")
    assert sidecar_folders(tmp_path / "input") == [tmp_path / "input" / "cct20-512" / "cct20_512"]


def _tar(path: Path, members: list[tuple[str, bytes | str | None]]) -> Path:
    """An uncompressed tar of (name, content): bytes a file, None a folder, a str a symlink to
    it, "hard:<name>" a hard link to it."""
    with tarfile.open(path, "w:") as tar:
        for name, content in members:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif isinstance(content, str):
                hard = content.startswith("hard:")
                info.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
                info.linkname = content.removeprefix("hard:")
                tar.addfile(info)
            else:
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
    return path


NORMAL: list[tuple[str, bytes | str | None]] = [
    (CACHE_TOP, None),
    (f"{CACHE_TOP}/annotations", None),
    (f"{CACHE_TOP}/annotations/train_annotations.json", b"{}"),
    (f"{CACHE_TOP}/images", None),
    (f"{CACHE_TOP}/images/a.jpg", b"\xff\xd8jpeg"),
    (f"{CACHE_TOP}/{DAY_NIGHT_SIDECAR}", b"{}"),
]


def test_a_normal_archive_is_unpacked_with_counts_and_seconds(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "cache.tar", NORMAL)
    unpacked = unpack_cache(archive, tmp_path / "local")
    assert unpacked.root == tmp_path / "local" / CACHE_TOP
    assert (unpacked.files, unpacked.size) == (3, 2 + 6 + 2)
    assert (unpacked.root / "images" / "a.jpg").read_bytes() == b"\xff\xd8jpeg"
    assert (unpacked.root / DAY_NIGHT_SIDECAR).is_file()
    assert unpacked.seconds >= 0
    assert str(unpacked).startswith(f"unpacked 3 files (0.00 GiB) into {unpacked.root} in ")


@pytest.mark.parametrize(
    ("member", "reason"),
    [
        ((f"{CACHE_TOP}/../escape.jpg", b"x"), "names with .."),
        ((f"{CACHE_TOP}/images/../../escape.jpg", b"x"), "names with .."),
        (("/tmp/escape.jpg", b"x"), "absolute names"),
        (("elsewhere/escape.jpg", b"x"), f"names outside {CACHE_TOP}/"),
        ((f"{CACHE_TOP}/images/link.jpg", "a.jpg"), "links or special files"),
        ((f"{CACHE_TOP}/images/out.jpg", "/etc/passwd"), "links or special files"),
        ((f"{CACHE_TOP}/images/hard.jpg", f"hard:{CACHE_TOP}/images/a.jpg"), "links or special"),
    ],
)
def test_a_traversal_an_absolute_name_or_a_link_is_refused_before_writing(
    tmp_path: Path, member: tuple[str, bytes | str | None], reason: str
) -> None:
    archive = _tar(tmp_path / "bad.tar", [*NORMAL, member])
    dest = tmp_path / "local"
    with pytest.raises(ArchiveError, match=reason) as refused:
        unpack_cache(archive, dest)
    assert "nothing was extracted" in str(refused.value)
    assert "escape" not in str(refused.value) and "link.jpg" not in str(refused.value)  # R03
    assert not dest.exists()
    assert not (tmp_path / "escape.jpg").exists()


def test_a_destination_under_kaggle_working_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy would be saved as the version's output."""
    monkeypatch.setattr(cache_module, "KAGGLE_WORKING", tmp_path / "working")
    archive = _tar(tmp_path / "cache.tar", NORMAL)
    with pytest.raises(ValueError, match="saved as output"):
        unpack_cache(archive, tmp_path / "working" / "cache")
    assert not (tmp_path / "working").exists()


def test_an_existing_tree_is_refused(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "cache.tar", NORMAL)
    unpack_cache(archive, tmp_path / "local")
    with pytest.raises(FileExistsError, match="already exists"):
        unpack_cache(archive, tmp_path / "local")


def test_an_archive_without_the_top_folder_is_refused(tmp_path: Path) -> None:
    archive = _tar(tmp_path / "empty.tar", [])
    with pytest.raises(ArchiveError, match=f"no {CACHE_TOP}/ folder"):
        unpack_cache(archive, tmp_path / "local")


def test_the_archive_search_finds_kaggles_dataset_mount(tmp_path: Path) -> None:
    mount = tmp_path / "datasets" / "codx888" / "cct20-512-tar"
    mount.mkdir(parents=True)
    (mount / f"{CACHE_TOP}.tar").write_bytes(b"")
    (mount / "notes.json").write_text("{}")
    assert archive_files(tmp_path) == [mount / f"{CACHE_TOP}.tar"]
    assert archive_files(tmp_path, max_depth=2) == []
    assert archive_files(tmp_path / "absent") == []


def test_the_space_needed_rounds_each_file_to_blocks_and_adds_a_margin() -> None:
    """Review 6.08: 57,864 files lose about 100 to 150 MB to block rounding."""
    small, large, folder = (tarfile.TarInfo(n) for n in ("a", "b", "c"))
    small.size, large.size = 1, 4097
    folder.type = tarfile.DIRTYPE
    assert space_needed([small, large, folder], 4096) == (1 + 2 + 1) * 4096 + UNPACK_MARGIN
    assert space_needed([], 4096) == UNPACK_MARGIN


def test_too_little_space_for_the_blocks_is_refused_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Free space above the files' bytes but below their blocks and the margin is refused, and
    nothing is written; exactly the blocks and the margin is enough."""
    archive = _tar(tmp_path / "cache.tar", NORMAL)
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
    size = sum(m.size for m in members if m.isfile())
    needed = space_needed(members, os.statvfs(tmp_path).f_frsize or 4096)
    free = {"bytes": size + 1}
    real = shutil.disk_usage

    def disk_usage(path: Path | str) -> Any:
        return real(path)._replace(free=free["bytes"])

    monkeypatch.setattr(cache_module.shutil, "disk_usage", disk_usage)
    dest = tmp_path / "local"
    with pytest.raises(OSError, match="nothing was extracted"):
        unpack_cache(archive, dest)
    assert not dest.exists()
    free["bytes"] = needed - 1
    with pytest.raises(OSError, match=f"needs about {needed} "):
        unpack_cache(archive, dest)
    free["bytes"] = needed
    assert unpack_cache(archive, dest).files == 3
