"""scripts/kaggle_pack_cache.py on a cache built from the synthetic fixture (item 6.08; R03).

The fixture's five annotation files go under `annotations/`, its JPEGs under `images/` and its
sidecar next to them: the layout of the 6.03 dataset. The archive is packed with the fixture's own
image count as the expected one, unpacked with `trailcam.data.cache.unpack_cache`, and must pass
`check_images` with the same bytes as the source.
"""

import importlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

from trailcam.data import synthetic
from trailcam.data.cache import CACHE_TOP, check_images, unpack_cache
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, annotation_file_name, load_metadata

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pack() -> ModuleType:
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("kaggle_pack_cache")


@pytest.fixture
def cache(tmp_path: Path) -> tuple[Path, int]:
    """The fixture in the 6.03 layout, and its image count."""
    written = synthetic.write_synthetic_fixture(tmp_path / "fixture")
    root = tmp_path / "cache"
    (root / "annotations").mkdir(parents=True)
    for split in SPLITS:
        shutil.copy(written.paths[split], root / "annotations" / annotation_file_name(split))
    shutil.copy(written.paths["sidecar"], root / DAY_NIGHT_SIDECAR)
    shutil.copytree(written.image_root, root / "images")
    return root, written.counts.images


def _names(root: Path) -> set[str]:
    """Every file name and image id of the cache's annotations."""
    names: set[str] = set()
    for split in SPLITS:
        data = json.loads((root / "annotations" / annotation_file_name(split)).read_bytes())
        names |= {str(i["file_name"]) for i in data["images"]} | {
            str(i["id"]) for i in data["images"]
        }
    return names


def test_the_archive_unpacks_to_the_same_cache(
    pack: ModuleType, cache: tuple[Path, int], tmp_path: Path
) -> None:
    root, images = cache
    packed = pack.pack(root, tmp_path / "out" / "cct20_512.tar", expected=images)
    assert packed.files == {"annotations": 5, "images": images, DAY_NIGHT_SIDECAR: 1}
    assert not (tmp_path / "out" / "cct20_512.tar.part").exists()

    unpacked = unpack_cache(tmp_path / "out" / "cct20_512.tar", tmp_path / "local")
    assert unpacked.root == tmp_path / "local" / CACHE_TOP
    assert unpacked.files == images + 6
    sources = sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())
    copies = sorted(p.relative_to(unpacked.root) for p in unpacked.root.rglob("*") if p.is_file())
    assert copies == sources
    assert all((root / p).read_bytes() == (unpacked.root / p).read_bytes() for p in sources)
    meta = load_metadata(unpacked.root / "annotations", unpacked.root / DAY_NIGHT_SIDECAR)
    assert check_images(meta, unpacked.root / "images").startswith("all ")


def test_the_same_cache_gives_the_same_archive(
    pack: ModuleType, cache: tuple[Path, int], tmp_path: Path
) -> None:
    """Owners and times are zeroed and the order sorted, so only the bytes matter."""
    root, images = cache
    first = pack.pack(root, tmp_path / "a.tar", expected=images)
    for path in root.rglob("*"):
        os.utime(path, (1_000_000, 1_000_000))
    second = pack.pack(root, tmp_path / "b.tar", expected=images)
    assert first.sha256 == second.sha256 and first.size == second.size


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("missing", "1 annotated images missing from images/"),
        ("empty", "1 annotated images empty or not regular files"),
        ("extra", "1 entries in images/ not in the annotations"),
        ("sidecar", "the sidecar has"),
        ("expected", "expected"),
    ],
)
def test_an_incomplete_cache_is_refused_with_counts_only(
    pack: ModuleType, cache: tuple[Path, int], tmp_path: Path, damage: str, message: str
) -> None:
    root, images = cache
    first = sorted((root / "images").iterdir())[0]
    expected = images
    if damage == "missing":
        first.unlink()
    elif damage == "empty":
        first.write_bytes(b"")
    elif damage == "extra":
        (root / "images" / "extra.jpg").write_bytes(b"\xff\xd8")
    elif damage == "sidecar":
        sidecar = json.loads((root / DAY_NIGHT_SIDECAR).read_text())
        sidecar.pop(sorted(sidecar)[0])
        (root / DAY_NIGHT_SIDECAR).write_text(json.dumps(sidecar))
    else:
        expected = images + 1
    out = tmp_path / "out.tar"
    with pytest.raises(pack.PackError, match=message) as refused:
        pack.pack(root, out, expected=expected)
    text = str(refused.value)
    assert "nothing was written" in text
    assert not [name for name in _names(root) | {"extra.jpg"} if name in text]  # R03
    assert not out.exists() and not out.with_name("out.tar.part").exists()


def test_a_missing_annotation_file_is_refused(
    pack: ModuleType, cache: tuple[Path, int], tmp_path: Path
) -> None:
    root, images = cache
    (root / "annotations" / annotation_file_name("trans_val")).unlink()
    with pytest.raises(pack.PackError, match="lacks 1 of 5 annotation files"):
        pack.pack(root, tmp_path / "out.tar", expected=images)


def test_main_refuses_outside_kaggle_and_an_existing_archive(
    pack: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(pack, "KAGGLE", tmp_path / "no-kaggle")
    with pytest.raises(SystemExit):
        pack.main(["--cache", str(tmp_path), "--out", str(tmp_path / "x.tar")])
    assert "this runs on Kaggle" in capsys.readouterr().err
    (tmp_path / "x.tar").write_bytes(b"")
    with pytest.raises(SystemExit):
        pack.main(["--off-kaggle", "--cache", str(tmp_path), "--out", str(tmp_path / "x.tar")])
    assert "exists" in capsys.readouterr().err


def test_main_prints_counts_and_the_hash(
    pack: ModuleType,
    cache: tuple[Path, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, images = cache
    monkeypatch.setattr(pack, "EXPECTED_IMAGES", images)
    out = tmp_path / "cct20_512.tar"
    assert pack.main(["--off-kaggle", "--cache", str(root), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert f"{images + 6} files (annotations 5, images {images}, {DAY_NIGHT_SIDECAR} 1)" in printed
    assert f"sha256 {pack.file_sha256(out)}" in printed
    assert not [name for name in _names(root) if name in printed]  # R03
    assert (
        pack.main(["--off-kaggle", "--cache", str(root / "absent"), "--out", str(out) + "2"]) == 1
    )
