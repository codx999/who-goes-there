"""trailcam/data/cache.py: the cache's sidecar search and image completeness check (item 6.07).

Run on the synthetic fixture, which writes one JPEG per image of its five annotation files, the
way the 512 px cache of item 6.01 holds one per image of CCT20.
"""

from pathlib import Path
from typing import Any

import pytest

from trailcam.data import synthetic
from trailcam.data.cache import MAX_DEPTH, check_images, missing_images, sidecar_folders
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
