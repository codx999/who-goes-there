"""The control plane keeps its rules (invariants R07, R42, R43, R45): the hooks run the guard in
the right mode, CI runs only check.sh and needs no secrets, check.sh runs every check and fails
closed, the Read deny lines stay, the reviewer agents stay read-only."""

import json
import os
import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD_CALL = "exec uv run --no-project --quiet python scripts/guard_data.py"


@pytest.mark.parametrize(
    ("hook", "mode"),
    [("pre-commit", "--staged"), ("pre-merge-commit", "--staged"), ("pre-push", "--pre-push")],
)
def test_hooks_run_the_guard(hook: str, mode: str) -> None:
    path = ROOT / ".githooks" / hook
    assert os.access(path, os.X_OK), f"{hook} is not executable"
    lines = path.read_text().splitlines()
    assert "set -euo pipefail" in lines
    assert lines[-1] == f"{GUARD_CALL} {mode}"


def test_workflow_runs_only_check_sh_and_needs_no_secrets() -> None:
    workflows = ROOT / ".github" / "workflows"
    assert sorted(p.name for p in workflows.iterdir()) == ["check.yml"]  # no second workflow
    text = (workflows / "check.yml").read_text()
    runs = re.findall(r"^\s*-?\s*run:\s*(.+)$", text, re.MULTILINE)
    assert runs == ["bash scripts/check.sh"]
    # secrets.X, secrets['X'], toJSON(secrets) inside an expression, or a secrets: block
    assert not re.search(r"\$\{\{[^}]*\bsecrets\b|^\s*secrets:", text, re.MULTILINE)
    assert re.search(r"^permissions:\n\s+contents: read$", text, re.MULTILINE)
    assert "fetch-depth: 0" in text  # --history needs every commit


def test_check_sh_runs_every_check() -> None:
    lines = (ROOT / "scripts" / "check.sh").read_text().splitlines()
    assert "set -euo pipefail" in lines
    for command in (
        "uv sync --locked --quiet",
        "uv run pytest --tb=short -q",
        "uv run mypy trailcam/ scripts/guard_data.py",
        "uv run ruff check .",
        "uv run ruff format --check .",
        "uv run python scripts/guard_data.py --all",
        "uv run python scripts/guard_data.py --history",
    ):
        assert command in lines, command


def test_check_sh_fails_when_hooks_folder_is_missing(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    shutil.copy(ROOT / "scripts" / "check.sh", tmp_path / "scripts" / "check.sh")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    env = {k: v for k, v in os.environ.items() if k != "CI"}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    result = subprocess.run(
        ["bash", "scripts/check.sh"], cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "git hooks are not enabled" in result.stderr


def _git(cwd: Path, *args: str) -> None:
    env = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
    env |= {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com"}
    env |= {"GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)


def _data_root(cwd: Path) -> str:
    return subprocess.run(
        ["bash", str(ROOT / "scripts" / "data_root.sh")],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_data_root_finds_the_main_checkout_data_from_a_worktree(tmp_path: Path) -> None:
    main = tmp_path / "main"
    main.mkdir()
    _git(main, "init", "-q", "-b", "main")
    (main / "README.md").write_text("t\n")
    _git(main, "add", "README.md")
    _git(main, "commit", "-q", "-m", "root")
    _git(main, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "wt")
    assert _data_root(main) == "" and _data_root(tmp_path / "wt") == ""  # no data/: nothing
    (main / "data").mkdir()
    expected = str((main / "data").resolve())
    assert _data_root(main) == expected
    assert _data_root(tmp_path / "wt") == expected


def test_check_sh_sets_the_data_root_outside_ci() -> None:
    lines = (ROOT / "scripts" / "check.sh").read_text().splitlines()
    assert 'if [ -z "${CI:-}" ] && [ -z "${TRAILCAM_DATA_ROOT:-}" ]; then' in lines
    assert '  root="$(bash scripts/data_root.sh)"' in lines
    assert '    export TRAILCAM_DATA_ROOT="$root"' in lines


def test_pytest_config_is_not_narrowed() -> None:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["pytest"]["ini_options"]
    assert config.get("testpaths") == ["tests"]
    assert "addopts" not in config
    # pytest reads pytest.toml, .pytest.toml, pytest.ini and .pytest.ini before pyproject.toml,
    # even when empty; mypy reads mypy.ini and .mypy.ini first.
    overrides = (
        *("pytest.toml", ".pytest.toml", "pytest.ini", ".pytest.ini", "tox.ini", "setup.cfg"),
        *("mypy.ini", ".mypy.ini", "ruff.toml", ".ruff.toml"),
    )
    for name in overrides:
        assert not (ROOT / name).exists(), f"{name} would override pyproject.toml"


def test_no_conftest() -> None:
    """A conftest.py can deselect or skip tests without touching them. None exists yet; adding
    one means changing this test in the same, reviewed commit."""
    tracked = subprocess.run(
        ["git", "ls-files", "*conftest.py"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    assert tracked == []


def test_settings_keep_the_read_deny_lines() -> None:
    deny = json.loads((ROOT / ".claude" / "settings.json").read_text())["permissions"]["deny"]
    for rule in ("Read(./data/**)", "Read(./models/**)", "Read(./outputs/**)", "Read(./.env)"):
        assert rule in deny


@pytest.mark.parametrize("agent", ["code-reviewer", "risk-reviewer"])
def test_reviewer_agents_are_read_only(agent: str) -> None:
    text = (ROOT / ".claude" / "agents" / f"{agent}.md").read_text()
    front = text.split("---")[1]
    assert re.search(r"^tools: Read, Grep, Glob, Bash$", front, re.MULTILINE)


def test_review_session_is_user_started() -> None:
    text = (ROOT / ".claude" / "commands" / "review-session.md").read_text()
    assert re.search(r"^disable-model-invocation: true$", text.split("---")[1], re.MULTILINE)
    assert "Never use `--fix`" in text
