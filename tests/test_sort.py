"""The live sort (item 10.02; R01, R03, R17).

The model is the small CNN with random weights, saved as `train.py` saves a checkpoint, and the
photos are the synthetic fixture's JPEGs: no trained run and no dataset image is needed.
"""

import ast
import builtins
import io
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from trailcam.data import synthetic
from trailcam.data.classes import CLASSES
from trailcam.demo import sort
from trailcam.demo.predict import PHOTO_SUFFIXES, load_model
from trailcam.demo.sort import (
    NOT_COPIED,
    UNREADABLE,
    DestinationError,
    SourceError,
    check_destination,
    check_source,
    find_photos,
    main,
    sort_photos,
)
from trailcam.model.factory import build_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The whole synthetic fixture: its JPEGs under `IMAGE_DIR` and its metadata beside them."""
    out = tmp_path_factory.mktemp("fixture")
    synthetic.write_synthetic_fixture(out)
    return out


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("run") / "small_cnn.pt"
    state = build_model("small_cnn").state_dict()
    torch.save({"model": "small_cnn", "classes": list(CLASSES), "state_dict": state}, path)
    return path


@pytest.fixture(scope="module")
def model(checkpoint: Path) -> nn.Module:
    return load_model(checkpoint, "small_cnn")


def _photos(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in PHOTO_SUFFIXES)


def _snapshot(folder: Path) -> dict[Path, bytes]:
    return {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}


def test_the_cli_sorts_every_fixture_photo_into_a_class_folder(
    fixture_dir: Path, checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dst = tmp_path / "sorted"
    before = _snapshot(fixture_dir)
    argv = ["--src", str(fixture_dir), "--dst", str(dst)]
    assert main([*argv, "--checkpoint", str(checkpoint), "--config-name", "small_cnn"]) == 0
    photos = _photos(fixture_dir)
    assert photos
    assert sorted(p.name for p in dst.iterdir()) == sorted(CLASSES)
    copied = _photos(dst)
    assert len(copied) == len(photos)
    assert all(p.parent.parent == dst and p.parent.name in CLASSES for p in copied)
    assert sorted(p.read_bytes() for p in copied) == sorted(p.read_bytes() for p in photos)
    assert _snapshot(fixture_dir) == before  # copied, never moved or changed
    out = capsys.readouterr().out
    assert f"{len(photos)} photos in " in out
    assert " s (" in out and "photos per second" in out


def test_without_a_checkpoint_it_says_the_weights_are_random(
    fixture_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dst = tmp_path / "sorted"
    assert main(["--src", str(fixture_dir), "--dst", str(dst), "--config-name", "small_cnn"]) == 0
    assert "random weights" in capsys.readouterr().out
    assert len(_photos(dst)) == len(_photos(fixture_dir))


def test_the_output_has_counts_and_seconds_but_no_file_name_or_path(
    fixture_dir: Path, checkpoint: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dst = tmp_path / "sorted"
    argv = ["--src", str(fixture_dir), "--dst", str(dst)]
    main([*argv, "--checkpoint", str(checkpoint), "--config-name", "small_cnn"])
    out = capsys.readouterr().out
    for photo in _photos(fixture_dir):
        assert photo.stem not in out
    for folder in (fixture_dir, dst, tmp_path):
        assert str(folder) not in out and folder.name not in out
    assert "accuracy" not in out.lower()
    lines = out.splitlines()
    per_photo = [line for line in lines if "/" in line and " s " in line]
    assert len(per_photo) == len(_photos(fixture_dir))
    first_label = per_photo[0].split()[-2].rstrip(":")
    assert first_label in CLASSES and per_photo[0].endswith(": 1")
    for name in CLASSES:
        assert any(line.split()[:1] == [name] for line in lines)


def test_the_model_is_warmed_up_once_then_one_forward_per_photo(
    fixture_dir: Path, checkpoint: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_model(checkpoint, "small_cnn")
    calls: list[int] = []
    model.register_forward_hook(lambda *_: calls.append(1))
    monkeypatch.setattr(sort, "load_model", lambda *_: model)
    main(["--src", str(fixture_dir), "--dst", str(tmp_path / "sorted"), "--checkpoint", "x.pt"])
    assert len(calls) == 1 + len(_photos(fixture_dir))


def test_only_photos_are_opened_never_labels_or_metadata(
    fixture_dir: Path, model: nn.Module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixture's metadata sits beside its photos in `--src`; the sort opens none of it, through
    `open` or through `Path.open`, `read_text` and `read_bytes`, which call `io.open`."""
    assert any(p.is_file() and p.suffix == ".json" for p in fixture_dir.rglob("*"))
    opened: list[Path] = []
    real_open = builtins.open

    def spy(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, str | Path):
            opened.append(Path(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    monkeypatch.setattr(io, "open", spy)
    sort_photos(model, fixture_dir, tmp_path / "sorted")
    monkeypatch.undo()
    assert opened
    assert all(path.suffix.lower() in PHOTO_SUFFIXES for path in opened)


def _imported_names(tree: ast.Module) -> set[str]:
    """Every dotted part of every module `tree` imports, and every name it imports from one."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            names.update((node.module or "").split("."))
            names.update(alias.name for alias in node.names)
    return names


def test_sort_py_names_no_metadata_split_json_ledger_or_metrics_module() -> None:
    """`sort.py`'s own imports, in any form, name none of them, and it imports nothing by a
    string name. The package loads these modules anyway (the dataset and predict imports bring
    them in), so this guards the file, not the process; the open spy above checks behaviour."""
    tree = ast.parse(Path(sort.__file__).read_text())
    forbidden = {"metadata", "split", "json", "ledger", "metrics", "importlib"}
    assert not _imported_names(tree) & forbidden
    called = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "__import__" not in called


@pytest.mark.parametrize(
    "line",
    [
        "from trailcam.data import metadata",
        "from trailcam.data.metadata import load",
        "import json as anything",
        "import trailcam.data.metadata",
        "from json import loads",
        "from trailcam.evaluate import metrics as m",
        "import importlib",
        "from importlib import import_module",
    ],
)
def test_the_import_check_sees_every_form(line: str) -> None:
    assert _imported_names(ast.parse(line)) & {"metadata", "json", "metrics", "importlib"}


def test_a_source_inside_the_repo_or_any_git_repository_is_refused(tmp_path: Path) -> None:
    (tmp_path / "other" / ".git").mkdir(parents=True)
    (tmp_path / "other" / "photos").mkdir()
    for src in (ROOT, ROOT / "data", ROOT / "assets", tmp_path / "other" / "photos"):
        with pytest.raises(SourceError, match="git repository"):
            check_source(src)
    check_source(tmp_path)


def test_a_source_above_a_repository_or_linking_into_one_is_refused(
    fixture_dir: Path, tmp_path: Path
) -> None:
    """A folder above a repository, or a link to a photo inside one, would reach `data/`."""
    photo = _photos(fixture_dir)[0]
    (tmp_path / "outer" / "repo" / ".git").mkdir(parents=True)
    (tmp_path / "outer" / "repo" / "data").mkdir()
    shutil.copy2(photo, tmp_path / "outer" / "repo" / "data" / "x.jpg")
    (tmp_path / "linked").mkdir()
    (tmp_path / "linked" / "x.jpg").symlink_to(tmp_path / "outer" / "repo" / "data" / "x.jpg")
    for src in (tmp_path / "outer", tmp_path / "linked"):
        with pytest.raises(SourceError, match="holds photos inside a git repository"):
            check_source(src)
    (tmp_path / "plain").mkdir()
    shutil.copy2(photo, tmp_path / "plain" / "x.jpg")
    check_source(tmp_path / "plain")


def test_the_cli_refuses_a_source_above_a_repository_before_loading_the_model(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: object) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(sort, "load_model", no_model)
    monkeypatch.setattr(sort, "build_model", no_model)
    (tmp_path / "outer" / "repo" / ".git").mkdir(parents=True)
    shutil.copy2(_photos(fixture_dir)[0], tmp_path / "outer" / "repo" / "x.jpg")
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--src",
                str(tmp_path / "outer"),
                "--dst",
                str(tmp_path / "sorted"),
                "--checkpoint",
                "x",
            ]
        )
    assert exit_info.value.code == 2
    assert not (tmp_path / "sorted").exists()


def test_a_photo_that_cannot_be_copied_is_counted_without_its_name(
    fixture_dir: Path,
    model: nn.Module,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A full disk raises an `OSError` whose message holds the photo's path; it is never shown."""
    src = tmp_path / "src"
    src.mkdir()
    for name in ("first_photo.jpg", "second_photo.jpg"):
        shutil.copy2(_photos(fixture_dir)[0], src / name)

    def full_disk(source: Path, target: Path) -> None:
        raise OSError(28, "No space left on device", str(source))

    monkeypatch.setattr(sort.shutil, "copy2", full_disk)
    counts = sort_photos(model, src, tmp_path / "sorted")
    assert counts[NOT_COPIED] == 2 and sum(counts.values()) == 2
    out = capsys.readouterr().out
    assert f"{NOT_COPIED}: 2" in out and "2 photos in " in out
    assert "_photo" not in out and str(src) not in out and "No space" not in out


def test_the_cli_refuses_a_source_in_the_repo_before_loading_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: object) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(sort, "load_model", no_model)
    monkeypatch.setattr(sort, "build_model", no_model)
    with pytest.raises(SystemExit) as exit_info:
        main(["--src", str(ROOT / "tests"), "--dst", str(tmp_path / "sorted"), "--checkpoint", "x"])
    assert exit_info.value.code == 2
    assert not (tmp_path / "sorted").exists()


def test_sort_photos_refuses_a_source_in_the_repo_too(model: nn.Module, tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        sort_photos(model, ROOT / "tests", tmp_path / "sorted")
    assert not (tmp_path / "sorted").exists()


def test_a_destination_inside_the_repo_is_refused(fixture_dir: Path) -> None:
    for dst in (ROOT, ROOT / "sorted", ROOT / "docs" / "sorted"):
        with pytest.raises(DestinationError, match="git repository"):
            check_destination(fixture_dir, dst)


def test_a_destination_inside_any_git_repository_is_refused(
    fixture_dir: Path, tmp_path: Path
) -> None:
    (tmp_path / "other" / ".git").mkdir(parents=True)
    with pytest.raises(DestinationError, match="git repository"):
        check_destination(fixture_dir, tmp_path / "other" / "sorted")


def test_the_source_or_a_folder_inside_it_is_refused(fixture_dir: Path) -> None:
    for dst in (fixture_dir, fixture_dir / "sorted", fixture_dir / synthetic.IMAGE_DIR / "x"):
        with pytest.raises(DestinationError, match="--src"):
            check_destination(fixture_dir, dst)


def test_a_non_empty_destination_or_a_file_is_refused(fixture_dir: Path, tmp_path: Path) -> None:
    full = tmp_path / "full"
    full.mkdir()
    (full / "earlier.txt").write_text("a rehearsal")
    with pytest.raises(DestinationError, match="not an empty folder"):
        check_destination(fixture_dir, full)
    file = tmp_path / "file"
    file.write_text("")
    with pytest.raises(DestinationError, match="not an empty folder"):
        check_destination(fixture_dir, file)
    empty = tmp_path / "empty"
    empty.mkdir()
    check_destination(fixture_dir, empty)
    check_destination(fixture_dir, tmp_path / "new")


def test_the_cli_refuses_a_bad_destination_before_loading_the_model(
    fixture_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_model(*_: object) -> None:
        raise AssertionError("the model was loaded")

    monkeypatch.setattr(sort, "load_model", no_model)
    monkeypatch.setattr(sort, "build_model", no_model)
    for dst in (ROOT / "sorted", fixture_dir / "sorted"):
        with pytest.raises(SystemExit) as exit_info:
            main(["--src", str(fixture_dir), "--dst", str(dst), "--checkpoint", "x.pt"])
        assert exit_info.value.code == 2
    assert not (ROOT / "sorted").exists() and not (fixture_dir / "sorted").exists()


def test_sort_photos_refuses_a_bad_destination_too(
    fixture_dir: Path, model: nn.Module, tmp_path: Path
) -> None:
    with pytest.raises(DestinationError):
        sort_photos(model, fixture_dir, fixture_dir / "sorted")
    assert not (fixture_dir / "sorted").exists()


def test_a_missing_source_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--src", str(tmp_path / "missing"), "--dst", str(tmp_path / "sorted")])


def test_unreadable_photos_are_counted_and_skipped_and_names_never_clash(
    fixture_dir: Path, model: nn.Module, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = tmp_path / "src"
    photo = _photos(fixture_dir)[0]
    for sub in ("a", "b"):
        (src / sub).mkdir(parents=True)
        shutil.copy2(photo, src / sub / "same_name.jpg")
    (src / "broken_photo.jpg").write_bytes(b"not a jpeg")
    (src / ".hidden.jpg").write_bytes(photo.read_bytes())
    (src / "notes.txt").write_text("not a photo")
    assert len(find_photos(src)) == 3
    counts = sort_photos(model, src, tmp_path / "sorted")
    assert counts[UNREADABLE] == 1 and sum(counts.values()) == 3
    copied = _photos(tmp_path / "sorted")
    assert sorted(p.name for p in copied) == ["same_name.jpg", "same_name_1.jpg"]
    out = capsys.readouterr().out
    assert f"{UNREADABLE}: 1" in out and "broken_photo" not in out and "same_name" not in out
