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

# Markers are assembled at runtime so this file never contains them literally (the guard scans
# every text blob, this one included).
PNG_MIME = "image/" + "png"
DATA_URI = "data:" + "image/png;base64,"
PAYLOAD = "iVBORw0KGgo" + "A" * 1200  # a real embedded image is tens of thousands of characters
JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 100


@pytest.mark.parametrize(
    "path",
    [
        "data/cct20/train_annotations.json",
        "models/best.pth",
        "outputs/run1/log.txt",
        "trailcam/demo/sample.jpg",
        "docs/figure.PNG",  # figures go under docs/figures/
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
        "assets/demo/a.gif",  # only the four web image types are allowed in image folders
        "assets/demo/a.tif",
        "assets/demo/photo.heic",
        "docs/figures/curve.bmp",
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
        "docs/figures/confusion_matrix.png",
        "docs/figures/gradcam/trans_001.jpg",
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
    assert "JPEG file" in guard.content_problems(JPEG, "docs/notes.txt")
    assert "PNG file" in guard.content_problems(b"\x89PNG\r\n\x1a\nrest", "x.md")
    assert "NumPy array file" in guard.content_problems(b"\x93NUMPY\x01", "x.md")
    assert "zip (also torch .pt/.pth and .npz) file" in guard.content_problems(b"PK\x03\x04", "m")
    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\0" * 32
    assert "MP4/MOV video file" in guard.content_problems(mp4, "docs/demo.bin.txt")
    heic = b"\x00\x00\x00\x18ftypheic" + b"\0" * 32
    assert guard.content_problems(heic, "assets/demo/photo.heic") == ["HEIC/AVIF image file"]


def test_allowed_asset_image_passes_content_check() -> None:
    assert guard.content_problems(JPEG, "assets/demo/a.jpg") == []
    assert guard.content_problems(JPEG, "docs/figures/a.jpg") == []


def _notebook(cells: list[dict[str, object]]) -> bytes:
    return json.dumps({"cells": cells}).encode()


EMBEDDED = "1 embedded images (notebook outputs, attachments or data URIs)"
NOT_CLEARED = "notebook outputs not cleared (1 cells)"


def test_notebook_image_outputs_are_refused() -> None:
    out = {"cell_type": "code", "outputs": [{"data": {PNG_MIME: PAYLOAD}}]}
    assert guard.content_problems(_notebook([out]), "notebooks/explore.ipynb") == [
        EMBEDDED,
        NOT_CLEARED,
    ]
    # Under another suffix the embedded-image check alone still catches it.
    assert guard.content_problems(_notebook([out]), "notes/explore.json") == [EMBEDDED]


def test_notebook_attachments_and_html_data_uris_are_refused() -> None:
    attached = {"cell_type": "markdown", "attachments": {"x.png": {PNG_MIME: PAYLOAD}}}
    html = {
        "cell_type": "code",
        "outputs": [{"data": {"text/html": f"<img src='{DATA_URI}{PAYLOAD}'>"}}],
    }
    assert guard.content_problems(_notebook([attached]), "nb.ipynb") == [
        EMBEDDED,
        "notebook attachments (1 cells)",
    ]
    assert guard.content_problems(_notebook([html]), "nb.ipynb") == [EMBEDDED, NOT_CLEARED]
    assert guard.content_problems(f"<img src='{DATA_URI}{PAYLOAD}'>".encode(), "docs/x.md") == [
        EMBEDDED
    ]
    # Upper case and parameters before ;base64 are still a data URI.
    odd_uri = DATA_URI.upper().replace(";BASE64,", ";charset=utf-8;base64,")
    assert guard.content_problems(f"<img src='{odd_uri}{PAYLOAD}'>".encode(), "x.md") == [EMBEDDED]
    # Notebook JSON may split the payload into a list of lines.
    split = {"cell_type": "code", "outputs": [{"data": {PNG_MIME: [PAYLOAD[:600], PAYLOAD[600:]]}}]}
    assert guard.content_problems(_notebook([split]), "nb.ipynb") == [EMBEDDED, NOT_CLEARED]


def test_mentioning_the_key_or_a_tiny_icon_passes() -> None:
    doc = f'The guard refuses any "{PNG_MIME}": output key in a notebook.'.encode()
    assert guard.content_problems(doc, "docs/plan.md") == []
    icon = {"cell_type": "code", "outputs": [{"data": {PNG_MIME: "iVBOR" + "A" * 400}}]}
    assert guard.embedded_problems(_notebook([icon])) == []


def test_notebook_with_cleared_outputs_passes() -> None:
    cell = {"cell_type": "code", "source": ["acc = 0.83\n"], "outputs": []}
    markdown = {"cell_type": "markdown", "source": ["# Train"]}
    assert guard.content_problems(_notebook([cell, markdown]), "notebooks/explore.ipynb") == []
    assert guard.content_problems(b"not json at all", "notebooks/broken.ipynb") == []
    assert guard.content_problems(b"[1, 2]", "notebooks/odd.ipynb") == []


def test_notebook_text_and_svg_outputs_are_refused() -> None:
    text = {"cell_type": "code", "outputs": [{"name": "stdout", "text": ["accuracy 0.83"]}]}
    svg = {"cell_type": "code", "outputs": [{"data": {"image/" + "svg+xml": ["<svg/>"]}}]}
    for cell in (text, svg):
        assert guard.content_problems(_notebook([cell]), "nb.IPYNB") == [NOT_CLEARED]


def test_metadata_dump_in_notebook_output_source_or_repr_is_refused() -> None:
    dump = json.dumps(_records(3000))
    expected = "dataset metadata (6000 record keys)"
    stream = {"cell_type": "code", "outputs": [{"name": "stdout", "text": [dump]}]}
    source = {"cell_type": "code", "source": [f"records = {dump}"], "outputs": []}
    # Under a non-notebook name only the record check applies.
    assert guard.content_problems(_notebook([stream]), "notes/peek.json") == [expected]
    assert guard.content_problems(_notebook([source]), "notebooks/peek.ipynb") == [expected]
    assert guard.content_problems(repr(_records(3000)).encode(), "notes/peek.txt") == [expected]


def _records(n: int) -> list[dict[str, object]]:
    return [{"id": str(i), "seq_id": "s", "location": 1, "file_name": f"{i}.jpg"} for i in range(n)]


def test_metadata_dump_is_refused_whatever_its_shape_or_suffix() -> None:
    expected = ["dataset metadata (6000 record keys)"]
    as_dict = json.dumps({"images": _records(3000), "annotations": []}).encode()
    as_list = json.dumps(_records(3000)).encode()
    nested = json.dumps({"splits": {"train": {"images": _records(3000)}}}).encode()
    assert guard.content_problems(as_dict, "tests/fixtures/cis_val_annotations.json") == expected
    assert guard.content_problems(as_list, "tests/fixtures/ids.json") == expected
    assert guard.content_problems(nested, "tests/fixtures/splits.json") == expected
    assert guard.content_problems(as_list, "notes/dump.txt") == expected
    assert guard.content_problems(as_list, "notebooks/peek.ipynb") == expected


def test_small_fixture_and_plain_json_pass() -> None:
    tiny = json.dumps({"images": _records(guard.MAX_METADATA_RECORDS // 2)}).encode()
    assert guard.content_problems(tiny, "tests/fixtures/tiny_annotations.json") == []
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


def _images(directory: str, n: int, commit: str = "") -> list[guard.Entry]:
    return [
        guard.Entry(f"{directory}live/{i:03d}.jpg", f"{commit}blob{i}", commit) for i in range(n)
    ]


def test_image_budgets_per_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "read_blobs", lambda blobs: dict.fromkeys(blobs, JPEG))
    for directory, (max_files, _) in guard.IMAGE_BUDGETS.items():
        over = _images(directory, max_files + 1)
        assert [p for _, p in guard.check(over)] == [
            f"{max_files + 1} images under {directory} (cap {max_files})"
        ]
        assert guard.check(over[:-1]) == []
    # 24 in assets/ plus 24 in docs/figures/ is fine: the budgets are separate.
    assert guard.check(_images("assets/", 24) + _images("docs/figures/", 24)) == []


def test_image_byte_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    big = b"\xff\xd8\xff\xe0" + b"\0" * 4_500_000  # under the per-file cap, 3 of them over budget
    monkeypatch.setattr(guard, "read_blobs", lambda blobs: dict.fromkeys(blobs, big))
    findings = guard.check(_images("assets/", 3))
    assert len(findings) == 1 and findings[0][1].startswith("images under assets/ total")


def test_history_budget_is_per_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(guard, "read_blobs", lambda blobs: dict.fromkeys(blobs, JPEG))
    two_commits = _images("assets/", 20, "c1") + _images("assets/", 20, "c2")
    assert guard.check(two_commits) == []
    findings = guard.check(_images("assets/", 25, "c1") + _images("assets/", 3, "c2"))
    assert [(e.commit, p) for e, p in findings] == [("c1", "25 images under assets/ (cap 24)")]


def test_gitignore_matches_guard() -> None:
    lines = {line.strip() for line in (ROOT / ".gitignore").read_text().splitlines()}
    missing = sorted(s for s in guard.BLOCKED_SUFFIXES if f"*{s}" not in lines)
    assert missing == [], f".gitignore lacks patterns for {missing}"
    for directory in guard.BLOCKED_DIRS:
        assert f"/{directory}" in lines
    allowed = sorted(
        f"!{d}**/*{s}" for d in guard.IMAGE_BUDGETS for s in guard.ALLOWED_IMAGE_SUFFIXES
    )
    assert [p for p in allowed if p not in lines] == []
    extra = sorted(p for p in lines if p.startswith("!") and p not in allowed)
    assert extra == [], f".gitignore re-includes paths the guard refuses: {extra}"


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
