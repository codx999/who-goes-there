"""scripts/kaggle_prepare_cct20.py on synthetic archives (item 6.01; R03 counts only).

The 4.03 fixture is packed into two small tar.gz files shaped like LILA's (annotations in one
folder, JPEGs in another) and run through `prepare` with its own image count as the expected one.
Nothing is downloaded from LILA: `download` runs against a local HTTP server, and the two URLs are
checked with `curl -sI` (the item's steps).
"""

import functools
import http.server
import importlib
import io
import json
import sys
import tarfile
import threading
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest
from PIL import Image

from trailcam.data import synthetic
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, SPLITS, load_metadata, load_night_flags

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def prep() -> ModuleType:
    # Imported by name from scripts/ (not by file path) so the encoder check's spawned processes,
    # which inherit sys.path, can import it too.
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("kaggle_prepare_cct20")


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> synthetic.Fixture:
    return synthetic.write_synthetic_fixture(tmp_path_factory.mktemp("fixture"))


def annotated_count(fixture: synthetic.Fixture) -> int:
    return sum(len(json.loads(fixture.paths[s].read_text())["images"]) for s in SPLITS)


def pack(
    fixture: synthetic.Fixture,
    out: Path,
    *,
    skip: int = 0,
    repeat: int = 0,
    extra: int = 0,
    corrupt: int = 0,
) -> tuple[Path, Path]:
    """The fixture as an annotations and an images tar.gz. The image tarball can leave out the
    first `skip` JPEGs, add `repeat` copies of the next ones, `extra` JPEGs no annotation names,
    and replace `corrupt` JPEGs with bytes that are not an image."""
    out.mkdir(parents=True)
    annotations = out / "annotations.tar.gz"
    with tarfile.open(annotations, "w:gz") as tar:
        tar.add(out.parent, arcname="eccv_18_annotation_files", recursive=False)
        for split in SPLITS:
            tar.add(
                fixture.paths[split], arcname=f"eccv_18_annotation_files/{split}_annotations.json"
            )
    jpegs = sorted(fixture.image_root.iterdir())
    images = out / "images.tar.gz"
    with tarfile.open(images, "w:gz") as tar:
        tar.add(fixture.image_root, arcname="./eccv_18_all_images_sm", recursive=False)
        for index, path in enumerate(jpegs[skip:]):
            data = b"not a jpeg" if index < corrupt else path.read_bytes()
            add_bytes(tar, f"./eccv_18_all_images_sm/{path.name}", data)
        for path in jpegs[skip : skip + repeat]:
            add_bytes(tar, f"./eccv_18_all_images_sm/copy/{path.name}", path.read_bytes())
        for index in range(extra):
            add_bytes(tar, f"./eccv_18_all_images_sm/stray-{index}.jpg", jpegs[0].read_bytes())
    return annotations, images


def add_bytes(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def test_cache_matches_the_fixture(
    prep: ModuleType, fixture: synthetic.Fixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    annotations, images = pack(fixture, tmp_path / "archives")
    out = tmp_path / "cache"
    expected = annotated_count(fixture)
    report = prep.prepare(annotations, images, out, expected=expected)
    assert report.total_written == expected

    # Every image is in the cache under its file name, and the sidecar's night flag and hour are
    # the fixture's: its night frames are grayscale JPEGs, its day frames colour.
    written = json.loads((out / DAY_NIGHT_SIDECAR).read_text())
    truth = json.loads(fixture.paths["sidecar"].read_text())
    assert set(written) == set(truth)
    for image_id, entry in written.items():
        assert set(entry) == {"night", "hour", "spread"}
        assert entry["night"] == truth[image_id]["night"]
        assert entry["hour"] == truth[image_id]["hour"]
    assert len(list((out / prep.IMAGE_DIR).iterdir())) == expected

    # The consumers read it: metadata from the copied annotations with the sidecar, every image.
    meta = load_metadata(out / prep.ANNOTATION_DIR, sidecar=out / DAY_NIGHT_SIDECAR)
    flags = load_night_flags(out / DAY_NIGHT_SIDECAR)
    for records in meta.records.values():
        for record in records:
            assert record.night == flags[record.image_id]
            with Image.open(out / prep.IMAGE_DIR / record.file_name) as image:
                assert image.size == (record.width, record.height)  # 96 x 72: never enlarged

    # Counts only (R03): no image id, file name or archive folder name in the output.
    printed = capsys.readouterr()
    text = printed.out + printed.err
    assert "Encoder check:" in text and ", 0 differ" in text
    assert "file depth (folders above the file): 1: " in text
    assert "folders holding .jpg files: 1" in text
    assert "eccv_18_all_images_sm" not in text and "eccv_18_annotation_files" not in text
    assert "syn-img" not in text and "syn-seq" not in text


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"skip": 2}, "2 annotated images not in the tarball"),
        ({"repeat": 1}, "1 annotated images repeated in the tarball"),
        ({"extra": 1}, "1 JPEGs in the tarball not in the annotations"),
        ({"extra": 1}, "the tarball holds 685 JPEGs, expected 684"),
        ({"corrupt": 1}, "1 images that could not be decoded"),
    ],
)
def test_a_count_mismatch_fails_with_no_sidecar(
    prep: ModuleType,
    fixture: synthetic.Fixture,
    tmp_path: Path,
    options: dict[str, int],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    annotations, images = pack(fixture, tmp_path / "archives", **options)
    out = tmp_path / "cache"
    monkeypatch.setattr(prep, "check_encoder", lambda sources: None)  # run in the test above
    with pytest.raises(prep.PrepareError, match=message):
        prep.prepare(annotations, images, out, expected=annotated_count(fixture))
    assert not (out / DAY_NIGHT_SIDECAR).exists()


def test_the_expected_count_is_enforced(
    prep: ModuleType, fixture: synthetic.Fixture, tmp_path: Path
) -> None:
    annotations, images = pack(fixture, tmp_path / "archives")
    out = tmp_path / "cache"
    with pytest.raises(prep.PrepareError, match="expected 57864"):
        prep.prepare(annotations, images, out)
    assert not out.exists()  # refused before the encoder check and before any write
    assert prep.EXPECTED_IMAGES == 57_864


def test_longer_side_becomes_512(prep: ModuleType) -> None:
    for size, expected in [
        ((1024, 747), (512, 374)),
        ((747, 1024), (374, 512)),
        ((2048, 1494), (512, 374)),
    ]:
        assert prep.resized(Image.new("RGB", size)).size == expected
    assert prep.resized(Image.new("RGB", (512, 300))).size == (512, 300)
    assert prep.resized(Image.new("L", (300, 200))).size == (300, 200)
    assert prep.resized(Image.new("CMYK", (1024, 768))).mode == "RGB"
    assert prep.resized(Image.new("L", (1024, 768))).mode == "L"


def test_mismatched_encodings_are_counted(prep: ModuleType) -> None:
    same = [("a", "a"), ("b", "b")]
    assert prep.count_mismatches([same, same]) == 0
    assert prep.count_mismatches([same, [("a", "a"), ("b", "c")]]) == 1  # within one process
    assert prep.count_mismatches([same, [("x", "x"), ("b", "b")]]) == 1  # between processes
    with pytest.raises(prep.PrepareError):
        prep.count_mismatches([same, same[:1]])


def test_encoder_check_passes_on_the_locked_pillow(prep: ModuleType) -> None:
    prep.check_encoder(prep.synthetic_sources())


def test_encoder_check_refuses_a_mismatch(
    prep: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prep, "count_mismatches", lambda runs: 1)
    with pytest.raises(prep.PrepareError, match="not reproducible"):
        prep.check_encoder(prep.synthetic_sources()[:1])


def test_refuses_to_start_off_kaggle(prep: ModuleType, tmp_path: Path) -> None:
    if prep.KAGGLE.is_dir():
        pytest.skip("running on Kaggle")
    with pytest.raises(SystemExit) as raised:
        prep.main(["--out", str(tmp_path / "cache"), "--download-dir", str(tmp_path / "dl")])
    assert raised.value.code == 2
    assert not (tmp_path / "dl").exists()


def test_urls_are_lila_gcp(prep: ModuleType) -> None:
    base = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/"
    assert prep.ANNOTATIONS_URL == base + "eccv_18_annotations.tar.gz"
    assert prep.IMAGES_URL == base + "eccv_18_all_images_sm.tar.gz"


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A local HTTP server over a folder (it ignores Range headers and answers 200 in full)."""
    root = tmp_path / "served"
    root.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *args: None  # type: ignore[attr-defined]
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/", root
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_download_writes_reuses_and_restarts(
    prep: ModuleType, server: tuple[str, Path], tmp_path: Path
) -> None:
    base, root = server
    body = bytes(range(256)) * 4096  # 1 MiB
    (root / "a.tar.gz").write_bytes(body)
    dest = tmp_path / "dl" / "a.tar.gz"

    assert prep.download(base + "a.tar.gz", dest) == dest
    assert dest.read_bytes() == body
    assert not dest.with_name("a.tar.gz.part").exists()

    mtime = dest.stat().st_mtime_ns
    prep.download(base + "a.tar.gz", dest)  # already complete: reused, not fetched again
    assert dest.stat().st_mtime_ns == mtime

    # A partial file from an interrupted run, and a server that ignores the range request: the
    # download starts over instead of appending the whole body to the partial one.
    dest.unlink()
    dest.with_name("a.tar.gz.part").write_bytes(body[:1000])
    prep.download(base + "a.tar.gz", dest)
    assert dest.read_bytes() == body


def test_download_refuses_a_short_file(
    prep: ModuleType, server: tuple[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, root = server
    (root / "a.tar.gz").write_bytes(b"x" * 1000)
    monkeypatch.setattr(prep, "remote_size", lambda url: 2000)  # the published size
    with pytest.raises(prep.PrepareError, match="1000 bytes downloaded, 2000 published"):
        prep.download(base + "a.tar.gz", tmp_path / "a.tar.gz")
