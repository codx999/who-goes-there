"""scripts/guard_data.py against a real git repository in a temporary directory: the modes the
hooks and check.sh call (--staged, --history, --pre-push) read what git gives them."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).resolve().parents[1] / "scripts" / "guard_data.py"
JPEG = b"\xff\xd8\xff\xe0" + b"\0" * 100
GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=GIT_ENV, check=True, capture_output=True, text=True
    ).stdout.strip()


def _guard(repo: Path, mode: str, stdin: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GUARD), mode],
        cwd=repo,
        env=GIT_ENV,
        input=stdin,
        capture_output=True,
        text=True,
    )


def _commit(repo: Path, files: dict[str, bytes], message: str) -> str:
    for name, data in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_bytes(data)
    _git(repo, "add", "-f", *files)
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    _commit(tmp_path, {"README.md": b"# t\n"}, "root")
    return tmp_path


def test_staged_refuses_a_data_file(repo: Path) -> None:
    (repo / "data").mkdir()
    (repo / "data" / "x.json").write_bytes(b"{}")
    _git(repo, "add", "-f", "data/x.json")
    result = _guard(repo, "--staged")
    assert result.returncode == 1
    assert "data/x.json: file under data/" in result.stderr


def test_history_sees_a_removed_file_and_a_file_added_by_a_merge(repo: Path) -> None:
    _commit(repo, {"models/best.pth": b"weights"}, "add")
    _git(repo, "rm", "-q", "--cached", "models/best.pth")
    _git(repo, "commit", "-q", "-m", "remove")
    _git(repo, "checkout", "-q", "-b", "side")
    _commit(repo, {"side.md": b"x\n"}, "side")
    _git(repo, "checkout", "-q", "main")
    _git(repo, "merge", "-q", "--no-commit", "side")
    (repo / "split.csv").write_bytes(b"a,b\n")
    _git(repo, "add", "-f", "split.csv")
    _git(repo, "commit", "-q", "-m", "merge")
    result = _guard(repo, "--history")
    assert result.returncode == 1
    assert "models/best.pth" in result.stderr
    assert "split.csv" in result.stderr


def test_pre_push_counts_images_over_the_whole_pushed_tree(repo: Path) -> None:
    _commit(repo, {f"assets/a{i:02d}.jpg": JPEG for i in range(20)}, "first 20")
    tip = _commit(repo, {f"assets/b{i:02d}.jpg": JPEG for i in range(20)}, "next 20")
    push = f"refs/heads/main {tip} refs/heads/main {'0' * 40}\n"
    assert _guard(repo, "--history").returncode == 0  # per commit, each is under the cap
    result = _guard(repo, "--pre-push", push)
    assert result.returncode == 1
    assert "40 images under assets/ (cap 24)" in result.stderr


def test_pre_push_within_budget_passes(repo: Path) -> None:
    tip = _commit(repo, {f"assets/a{i:02d}.jpg": JPEG for i in range(20)}, "20")
    push = f"refs/heads/main {tip} refs/heads/main {'0' * 40}\n"
    assert _guard(repo, "--pre-push", push).returncode == 0
    assert _guard(repo, "--pre-push", "").returncode == 0  # nothing pushed
