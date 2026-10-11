"""scripts/risk_files.sh: the high-stakes path regex, range checks, and its agreement with the
"where" column of docs/review/invariants.md (R44). Runs the script in a throwaway git repo."""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "risk_files.sh"
INVARIANTS = ROOT / "docs" / "review" / "invariants.md"

RISKY = [
    # The package: splits, transforms, the box sampler, training, evaluation, the demo.
    "trailcam/__init__.py",
    "trailcam/data/classes.py",
    "trailcam/data/split.py",
    "trailcam/data/boxcrop.py",
    "trailcam/data/loaders.py",
    "trailcam/model/backbone.py",
    "trailcam/train/train.py",
    "trailcam/train/ledger.py",
    "trailcam/evaluate/score_test.py",
    "trailcam/demo/app.py",
    # The Space and the notebooks.
    "space/app.py",
    "space/requirements.txt",
    "space/README.md",
    "notebooks/kaggle_train.ipynb",
    "kaggle_prepare.ipynb",
    # Root training configs: they set the arms (R25).
    "config.yaml",
    "config.yml",
    "configs/arm_b.yaml",
    # Control plane: a change here can switch off a check the invariant map relies on.
    "CLAUDE.md",
    "tests/test_split.py",
    "tests/conftest.py",
    "conftest.py",
    "pytest.ini",
    ".pytest.ini",
    "pytest.toml",
    ".pytest.toml",
    "tox.ini",
    "setup.cfg",
    "mypy.ini",
    ".mypy.ini",
    "ruff.toml",
    ".ruff.toml",
    "pyproject.toml",
    "uv.lock",
    "scripts/check.sh",
    "scripts/guard_data.py",
    "scripts/risk_files.sh",
    "scripts/deploy_space.py",
    "scripts/kaggle_prepare_cct20.py",
    ".githooks/pre-commit",
    ".githooks/pre-push",
    ".github/workflows/check.yml",
    ".claude/settings.json",
    ".claude/rules/data.md",
    ".claude/commands/review-session.md",
    ".claude/agents/code-reviewer.md",
]
SAFE = [
    "README.md",
    "LICENSE",
    "docs/plan.md",
    "docs/progress-log.md",
    "docs/progress.json",
    "docs/ledger.md",
    "docs/review/invariants.md",
    "docs/review/playbook-reusable.md",
    "docs/uv.lock",
    "docs/conftest.md",
    "docs/trailcam/notes.py",
    "docs/space/notes.md",
    "docs/notebook.ipynb.md",
    "pyproject.toml.orig",
    "docs/pytest.toml",
    "mypy.ini.example",
    "docs/config.yaml",
    "config.yaml.example",
    "assets/trans_001.jpg",
    "docs/figures/confusion_matrix.png",
    ".github/CODEOWNERS",
    ".gitignore",
    ".python-version",
]

ROW = re.compile(r"^R\d+$")


def _where_samples() -> list[str]:
    """One sample path per path-like entry in the "where" column of invariants.md."""
    samples: list[str] = []
    for line in INVARIANTS.read_text().splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 6 or not ROW.match(cells[0]):
            continue
        for entry in re.findall(r"`([^`]+)`", cells[3]):
            if "/" not in entry and "." not in entry:
                continue  # an item id or a word, not a path
            samples.append(entry.replace("**", "sub/leaf.py").replace("*", "x"))
    assert samples, "no path entries parsed from invariants.md"
    return samples


def test_invariants_table_shape() -> None:
    """Item 2.01: one table of 30 to 40 rows with six columns, every row id unique."""
    lines = [ln for ln in INVARIANTS.read_text().splitlines() if ln.startswith("|")]
    rows = [ln for ln in lines if ROW.match(ln.strip("| ").split("|")[0].strip())]
    assert len(lines) == len(rows) + 2, "one header and one separator line, nothing else"
    assert 30 <= len(rows) <= 40
    ids = [ln.strip("| ").split("|")[0].strip() for ln in rows]
    assert len(set(ids)) == len(ids)
    header = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    assert header == ["id", "rule", "source", "where", "scenario to trace", "test"]
    for ln in rows:
        assert len(ln.strip().strip("|").split("|")) == 6, ln[:40]


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@example.com",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    ).stdout.strip()


def _run(rng: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPT), rng], capture_output=True, text=True, cwd=cwd)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git("init", "-q", "-b", "main", cwd=tmp_path)
    (tmp_path / "README.md").write_text("x\n")
    _git("add", ".", cwd=tmp_path)
    _git("commit", "-q", "-m", "base", cwd=tmp_path)
    return tmp_path


def _commit_paths(repo: Path, paths: list[str]) -> None:
    for rel in paths:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("x\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "change", cwd=repo)


def test_prints_only_risky_paths(repo: Path) -> None:
    _commit_paths(repo, [p for p in RISKY + SAFE if p != "README.md"])
    (repo / "README.md").write_text("y\n")
    _git("commit", "-q", "-am", "readme", cwd=repo)
    result = _run("main~2..main", repo)
    assert result.returncode == 0
    assert sorted(result.stdout.split()) == sorted(RISKY)


def test_every_invariant_where_path_is_risky(repo: Path) -> None:
    """R44: every path in the "where" column of docs/review/invariants.md matches the regex."""
    samples = sorted(set(_where_samples()))
    _commit_paths(repo, samples)
    result = _run("main~1..main", repo)
    assert result.returncode == 0
    missed = sorted(set(samples) - set(result.stdout.split()))
    assert not missed, f"invariants.md paths not matched by risk_files.sh: {missed}"


def test_no_risky_changes_prints_nothing(repo: Path) -> None:
    _commit_paths(repo, ["docs/notes.md", "docs/figures/curve.png"])
    result = _run("main~1..main", repo)
    assert (result.returncode, result.stdout) == (0, "")


@pytest.mark.parametrize("rng", ["nosuchref..main", "..main", "main..", "main", "main~1...main"])
def test_bad_or_open_range_fails(repo: Path, rng: str) -> None:
    # An empty end ("$base" lost between shell calls) must not read as "no risky files".
    (repo / "README.md").write_text("y\n")
    _git("commit", "-q", "-am", "change", cwd=repo)
    result = _run(rng, repo)
    assert (result.returncode, result.stdout) == (2, "")


def test_missing_argument_fails(repo: Path) -> None:
    assert subprocess.run(["bash", str(SCRIPT)], capture_output=True, cwd=repo).returncode == 2


def test_move_out_of_risky_path_lists_old_path(repo: Path) -> None:
    src = repo / "trailcam" / "data" / "split.py"
    src.parent.mkdir(parents=True)
    src.write_text("".join(f"line {i}\n" for i in range(50)))
    _git("add", ".", cwd=repo)
    _git("commit", "-q", "-m", "add", cwd=repo)
    (repo / "docs").mkdir()
    _git("mv", "trailcam/data/split.py", "docs/split.py", cwd=repo)
    _git("commit", "-q", "-m", "move", cwd=repo)
    assert _run("main~1..main", repo).stdout.split() == ["trailcam/data/split.py"]


def test_non_ascii_path_is_listed_unquoted(repo: Path) -> None:
    _commit_paths(repo, ["trailcam/data/café.py"])
    assert _run("main~1..main", repo).stdout == "trailcam/data/café.py\n"
