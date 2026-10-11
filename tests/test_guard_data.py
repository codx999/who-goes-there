"""Rules of scripts/guard_data.py: paths, contents, budgets, git-output parsing. No git needed."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("guard_data", ROOT / "scripts" / "guard_data.py")
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard_data"] = guard
_spec.loader.exec_module(guard)


@pytest.mark.parametrize(
    "path",
    [
        "data/cct20/train_annotations.json",
        "models/best.pth",
        "outputs/run1/log.txt",
        "trailcam/demo/sample.jpg",
        "docs/figure.PNG",
        "notebooks/weights.pt",
        "export.safetensors",
        "features.npy",
        "images.tar.gz",
        "split.csv",
        "kaggle.json",
        ".env.local",
        "prod.env",
        "assets/weights.pt",
        "assets/photo.jpg.zip",
        "assets/demo/a.gif",  # only the four web image types are allowed under assets/
        "assets/demo/a.tif",
        "docs/backup.mp4",
        "assets/demo/clip.mov",
    ],
)
def test_blocked_paths(path: str) -> None:
    assert guard.path_problems(path)


@pytest.mark.parametrize(
    "path",
    [
        "assets/demo/coyote_night.jpg",
        "assets/README.png",
        "assets/demo/a.webp",
        "trailcam/data/splits.py",
        "docs/plan.md",
        ".env.example",
        "tests/fixtures/tiny_annotations.json",
        "notebooks/train.ipynb",
    ],
)
def test_allowed_paths(path: str) -> None:
    assert guard.path_problems(path) == []


def test_magic_bytes_catch_renamed_files() -> None:
    assert "JPEG file" in guard.content_problems(b"\xff\xd8\xff\xe0rest", "docs/notes.txt")
    assert "PNG file" in guard.content_problems(b"\x89PNG\r\n\x1a\nrest", "x.md")
    assert "NumPy array file" in guard.content_problems(b"\x93NUMPY\x01", "x.md")
    assert "zip (also torch .pt/.pth and .npz) file" in guard.content_problems(b"PK\x03\x04", "m")
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\0" * 32
    assert "MP4/MOV video file" in guard.content_problems(mp4, "docs/demo.bin.txt")


def test_allowed_asset_image_passes_content_check() -> None:
    assert guard.content_problems(b"\xff\xd8\xff\xe0rest", "assets/demo/a.jpg") == []


def _notebook(outputs: list[dict[str, object]]) -> bytes:
    return json.dumps({"cells": [{"cell_type": "code", "outputs": outputs}]}).encode()


def test_notebook_image_outputs_are_refused() -> None:
    nb = _notebook([{"data": {"image/png": "iVBORw0KGgo" + "A" * 400, "text/plain": "<Figure>"}}])
    assert guard.content_problems(nb, "notebooks/explore.ipynb") == [
        "notebook with 1 image outputs (clear outputs first)"
    ]


def test_notebook_without_image_outputs_passes() -> None:
    nb = _notebook([{"data": {"text/plain": "accuracy 0.83"}}, {"name": "stdout", "text": "ok"}])
    assert guard.content_problems(nb, "notebooks/explore.ipynb") == []
    assert guard.content_problems(b"not json at all", "notebooks/broken.ipynb") == []


def _metadata(n_images: int) -> bytes:
    images = [{"id": str(i), "seq_id": "s", "location": 1} for i in range(n_images)]
    return json.dumps({"images": images, "annotations": [], "categories": []}).encode()


def test_metadata_dump_is_refused_but_small_fixture_passes() -> None:
    assert guard.content_problems(_metadata(3000), "tests/fixtures/cis_val_annotations.json") == [
        "dataset metadata (3000 images records)"
    ]
    assert guard.content_problems(_metadata(guard.MAX_METADATA_RECORDS), "tests/f.json") == []
    assert guard.content_problems(b'{"a": 1}', "config.json") == []


def test_secret_patterns() -> None:
    assert "Hugging Face token" in guard.content_problems(b"token=hf_" + b"a" * 34, "app.py")
    # Built at runtime so this file itself never contains a key-shaped literal.
    fake_key = b"abc123" * 4
    assert "key, salt or token assignment" in guard.content_problems(
        b'API_KEY="' + fake_key + b'"', "notebook.ipynb"
    )
    assert guard.content_problems(b"api_key = os.environ['API_KEY']", "x.py") == []


def test_size_limit() -> None:
    assert "larger than 5 MB" in guard.content_problems(b"0" * (guard.MAX_BYTES + 1), "big.md")


def test_asset_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    jpeg = b"\xff\xd8\xff\xe0" + b"\0" * 100
    n = guard.MAX_ASSET_IMAGES + 1
    entries = [guard.Entry(f"assets/live/{i:03d}.jpg", f"blob{i}") for i in range(n)]
    monkeypatch.setattr(guard, "read_blobs", lambda blobs: dict.fromkeys(blobs, jpeg))
    findings = guard.check(entries)
    assert [p for _, p in findings] == [f"{n} images under assets/ (cap {guard.MAX_ASSET_IMAGES})"]
    assert guard.check(entries[:-1]) == []
    big = (
        b"\xff\xd8\xff\xe0" + b"\0" * 4_500_000
    )  # under the per-file cap, 3 of them over the budget
    monkeypatch.setattr(guard, "read_blobs", lambda blobs: dict.fromkeys(blobs, big))
    findings = guard.check(entries[:3])
    assert len(findings) == 1 and findings[0][1].startswith("assets/ images total")


def test_gitignore_matches_guard() -> None:
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    missing = sorted(s for s in guard.BLOCKED_SUFFIXES if f"*{s}" not in lines)
    assert missing == [], f".gitignore lacks patterns for {missing}"
    for directory in guard.BLOCKED_DIRS:
        assert f"/{directory}" in lines
    allowed = sorted(f"!assets/**/*{s}" for s in guard.ALLOWED_IMAGE_SUFFIXES)
    assert [p for p in allowed if p not in lines] == []
    extra = sorted(p for p in lines if p.startswith("!assets/") and p not in allowed)
    assert extra == [], f".gitignore re-includes types the guard refuses: {extra}"


def test_parse_raw() -> None:
    out = (
        b":000000 100644 " + b"0" * 40 + b" " + b"a" * 40 + b" A\0docs/plan.md\0"
        b":100644 100644 " + b"b" * 40 + b" " + b"c" * 40 + b" M\0sp ace/\xc3\xa9.py\0"
        b":000000 160000 " + b"0" * 40 + b" " + b"d" * 40 + b" A\0vendor/sub\0"
    )
    entries = guard._parse_raw(out, "abc123")
    assert entries == [
        guard.Entry("docs/plan.md", "a" * 40, "abc123"),
        guard.Entry("sp ace/é.py", "c" * 40, "abc123"),
    ]
    assert guard._parse_raw(b"") == []


def test_pre_push_revs() -> None:
    push = "refs/heads/s1 " + "1" * 40 + " refs/heads/s1 " + "0" * 40
    delete = "refs/heads/old " + "0" * 40 + " refs/heads/old " + "2" * 40
    assert guard.pre_push_revs([push, delete]) == ["1" * 40, "--not", "--remotes"]
    assert guard.pre_push_revs([delete]) == []
    assert guard.pre_push_revs([]) == []


def test_read_blobs(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = b"a" * 40 + b" blob 3\nxyz\n" + b"b" * 40 + b" blob 0\n\n"
    monkeypatch.setattr(guard, "_git", lambda *a, **k: stream)
    assert guard.read_blobs(["a" * 40, "b" * 40, "a" * 40]) == {"a" * 40: b"xyz", "b" * 40: b""}
    monkeypatch.setattr(guard, "_git", lambda *a, **k: b"c" * 40 + b" missing\n")
    with pytest.raises(RuntimeError):
        guard.read_blobs(["c" * 40])
    assert guard.read_blobs([]) == {}
