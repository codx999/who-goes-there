"""Path and content rules of scripts/guard_data.py (no git needed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "guard_data", Path(__file__).resolve().parents[1] / "scripts" / "guard_data.py"
)
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
        "assets/weights.pt",
        "assets/photo.jpg.zip",
    ],
)
def test_blocked_paths(path: str) -> None:
    assert guard.path_problems(path)


@pytest.mark.parametrize(
    "path",
    [
        "assets/demo/coyote_night.jpg",
        "assets/README.png",
        "trailcam/data/splits.py",
        "docs/plan.md",
        ".env.example",
        "tests/fixtures/tiny_annotations.json",
    ],
)
def test_allowed_paths(path: str) -> None:
    assert guard.path_problems(path) == []


def test_magic_bytes_catch_renamed_files() -> None:
    assert "JPEG file" in guard.content_problems(b"\xff\xd8\xff\xe0rest", "docs/notes.txt")
    assert "PNG file" in guard.content_problems(b"\x89PNG\r\n\x1a\nrest", "x.md")
    assert "NumPy array file" in guard.content_problems(b"\x93NUMPY\x01", "x.md")
    assert "zip (also torch .pt/.pth and .npz) file" in guard.content_problems(b"PK\x03\x04", "m")


def test_allowed_asset_image_passes_content_check() -> None:
    assert guard.content_problems(b"\xff\xd8\xff\xe0rest", "assets/demo/a.jpg") == []


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
