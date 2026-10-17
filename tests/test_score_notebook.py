"""notebooks/kaggle_score_test.ipynb, the final scoring's notebook (item 9.03).

The notebook runs only on Kaggle, for item 9.02, the user's one run. Here it is read as JSON and
its code cells as Python: these tests check its shape (R41 outputs cleared; R38 the one pip
install, of the clone; R17 the scoring runs once, through one call of `score_test.py`'s command
line, after the go-ahead cell, and no cell reaches the training or evaluation code; R06 the
licence and citation; R07 no credentials). They also run its cells as written on the synthetic
fixture of `tests/test_score_test.py`: random weights, never a real checkpoint, and never the real
cis_test or trans_test. The scoring cell runs `python -m trailcam.evaluate.score_test` on that
fixture once.
"""

import ast
import importlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest
import torch
from PIL import Image

from tests.test_score_test import APPROVER, World, _build_world, _figures_snapshot
from trailcam.data.cache import CACHE_TOP
from trailcam.evaluate import evaluate, score_test
from trailcam.evaluate.score_test import (
    ALARM_EXIT,
    TEST_LEDGER_COLUMNS,
    TEST_LEDGER_PATH,
    Refusal,
    check_complete,
    check_run_seeds,
    check_unique,
    ledger_line,
    read_ledger,
    read_test_ledger,
    scoring_header,
)
from trailcam.train import device

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_score_test.ipynb"
TRAIN_NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
PUBLIC_REPO = "https://github.com/codx999/who-goes-there"
SCORING_MODULE = "trailcam.evaluate.score_test"
PARAMETERS = {"COMMIT", "CACHE_ARCHIVE", "CHECKPOINTS_ROOT", "APPROVED_BY", "RUN_SEEDS"}
# Everything the notebook takes from `trailcam`, exactly: a new name (a second route to the
# scoring, an evaluation or training call, a checkpoint loader) fails here first. From
# score_test.py only its checks before anything is loaded and its test ledger reader and writer
# of one line; from train.py two folder names.
TRAILCAM_IMPORTS = {
    "trailcam.data.cache": {"archive_files", "check_images", "sidecar_folders", "unpack_cache"},
    "trailcam.data.metadata": {"DAY_NIGHT_SIDECAR", "load_metadata"},
    "trailcam.demo.predict": {"PHOTO_SUFFIXES"},
    SCORING_MODULE: {
        "ALARM_EXIT",
        "Refusal",
        "check_approver",
        "check_run_seeds",
        "ledger_line",
        "read_ledger",
        "read_test_ledger",
        "resolve_checkpoints",
        "scoring_header",
    },
    "trailcam.train.device": {"check_gpu"},
    "trailcam.train.train": {"CACHE_ANNOTATIONS", "CACHE_IMAGES"},
}
OTHER_IMPORTS = {
    "os",
    "re",
    "sys",
    "subprocess",
    "shlex",
    "shutil",
    "pathlib",
    "importlib.metadata",
}
# The training and evaluation code: no cell may run it (item 9.03).
NEVER_RUN = (
    "trailcam.evaluate.evaluate",
    "trailcam.evaluate.attention",
    "trailcam.evaluate.slices",
    "trailcam.train.train",
)
PACKAGES = ("trailcam.evaluate", "trailcam.train")  # whose modules those are
DYNAMIC = {"exec", "eval", "compile", "__import__", "getattr", "setattr", "globals", "locals"}
DYNAMIC |= {"vars", "get_ipython", "os.system", "os.popen", "importlib.import_module"}
DYNAMIC |= {"import_module", "runpy.run_module", "runpy.run_path"}

_spec = importlib.util.spec_from_file_location("guard_data", ROOT / "scripts" / "guard_data.py")
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard_data"] = guard
_spec.loader.exec_module(guard)

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
MARKDOWN = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")
FLAT = " ".join(MARKDOWN.split())  # the markdown with its line breaks as spaces
ALL_CODE = "\n".join(CODE)
TREES = [ast.parse(source) for source in CODE]  # plain Python: a magic fails here


def _calls_in(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def _cell(name: str) -> int:
    """The one code cell that calls `name`."""
    (cell,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, name)]
    return cell


def _scoring_cell() -> int:
    (cell,) = [i for i, tree in enumerate(TREES) if f"'{SCORING_MODULE}'" in ast.unparse(tree)]
    return cell


def _imports_cell() -> int:
    return next(i for i, tree in enumerate(TREES) if "trailcam" in _imported(tree))


def _imported(tree: ast.AST) -> set[str]:
    """Top-level package names a cell imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _parameters() -> dict[str, object]:
    """The literal assignments of the first code cell; others as their source."""
    values: dict[str, object] = {}
    for node in TREES[0].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                values[node.targets[0].id] = ast.unparse(node.value)
    return values


def _run(cell: int, names: dict[str, Any]) -> dict[str, Any]:
    exec(compile(TREES[cell], f"code cell {cell}", "exec"), names)
    return names


def _imports() -> dict[str, Any]:
    """The names the notebook imports, by its own import statements (the import cell's guard
    against a kernel that already holds `trailcam` is left out: these tests hold it)."""
    statements = [
        node
        for tree in TREES
        for node in tree.body
        if isinstance(node, ast.Import | ast.ImportFrom)
    ]
    names: dict[str, Any] = {}
    exec(compile(ast.Module(statements, type_ignores=[]), "imports", "exec"), names)
    return names


def _namespace(work: Path, kaggle_input: Path, **values: Any) -> dict[str, Any]:
    """The parameter cell run with `work` and `kaggle_input` for Kaggle's two folders, then the
    notebook's imports, then `values`."""
    source = CODE[0]
    for kaggle, local in (
        ('Path("/kaggle/working")', work),
        ('Path("/kaggle/input")', kaggle_input),
    ):
        assert source.count(kaggle) == 1
        source = source.replace(kaggle, f"Path({str(local)!r})")
    names: dict[str, Any] = {}
    exec(compile(source, "parameter cell", "exec"), names)
    names.update(_imports())
    names.update(values)
    return names


# The notebook as a file.


def test_outputs_and_attachments_are_cleared_and_the_guard_passes() -> None:
    """R41 and the data guard: no output, execution count or attachment, no image, record or
    token; the path is allowed."""
    data = NOTEBOOK.read_bytes()
    path = "notebooks/kaggle_score_test.ipynb"
    assert guard.content_problems(data, path) == []
    assert guard.path_problems(path) == []
    for cell in NB["cells"]:
        assert "attachments" not in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


def test_code_cells_define_nothing() -> None:
    """Every code cell is plain Python that calls `trailcam` or the standard library."""
    defining = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    for tree in TREES:
        assert not [n for n in ast.walk(tree) if isinstance(n, defining)]


def test_trailcam_names_exist() -> None:
    """Every name the notebook takes from `trailcam` is there, so a rename fails here first."""
    for module, names in TRAILCAM_IMPORTS.items():
        for name in names:
            assert hasattr(importlib.import_module(module), name), f"{module}.{name}"


# Parameters.


def test_parameters() -> None:
    p = _parameters()
    assert isinstance(p["COMMIT"], str) and len(p["COMMIT"]) == 40
    assert p["APPROVED_BY"] == ""  # the go-ahead is the user's, filled in only for the run
    assert p["RUN_SEEDS"] == (0, 1, 2)
    assert p["CHECKPOINTS_ROOT"] is None and p["CHECKPOINTS_DATASET"] == "trailcam-checkpoints"
    assert p["REPO_URL"] == PUBLIC_REPO  # the public repo, not a fork (R38)
    assert p["INPUT"] == "Path('/kaggle/input')" and p["WORK"] == "Path('/kaggle/working')"
    assert p["CLONE"] == "WORK / 'who-goes-there'"
    assert p["CONFIG_FILE"] == "CLONE / 'config.yaml'"
    assert p["LEDGER"] == "CLONE / 'docs' / 'ledger.md'"
    assert p["TEST_LEDGER"] == "CLONE / 'docs' / 'test-ledger.md'"
    assert p["ASSETS"] == "CLONE / 'assets'"
    assert p["OUT"] == "WORK / 'score'"  # --out /kaggle/working/score
    assert p["LOCAL_CACHE"] == "Path('/tmp/trailcam-cache')"  # local disk, not /kaggle/working


def test_the_cache_archive_is_the_training_notebook_s() -> None:
    """The same packing notebook output as training (item 6.09)."""
    train = json.loads(TRAIN_NOTEBOOK.read_text(encoding="utf-8"))
    first = ast.parse(
        "".join(next(c for c in train["cells"] if c["cell_type"] == "code")["source"])
    )
    (theirs,) = [
        ast.unparse(node.value)
        for node in first.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "CACHE_ARCHIVE"
    ]
    assert _parameters()["CACHE_ARCHIVE"] == theirs


def test_the_parameter_cell_is_the_only_one_that_sets_a_parameter() -> None:
    """A stale second parameter cell overrode a hand edit on Kaggle once (item 6.07). The first
    code cell binds each parameter once, and no later cell binds, rebinds or deletes any name the
    first one binds: the paths are fixed there too."""
    first: dict[str, int] = {}
    for node in ast.walk(TREES[0]):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            first[node.id] = first.get(node.id, 0) + 1
    assert {name: first.get(name) for name in PARAMETERS} == dict.fromkeys(PARAMETERS, 1)
    bound_first = set(first) | {"Path"}
    later = [
        (i, node.id)
        for i, tree in enumerate(TREES[1:], start=1)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load)
        if node.id in bound_first
    ]
    assert later == []
    (check,) = [source for source in CODE if "COMMIT must be" in source]
    for shown in ("COMMIT {COMMIT}", "RUN_SEEDS {RUN_SEEDS}", "APPROVED_BY {APPROVED_BY!r}"):
        assert shown in check
    for shown in ("CACHE_ARCHIVE {CACHE_ARCHIVE}", "CHECKPOINTS_ROOT {CHECKPOINTS_ROOT}"):
        assert shown in check


def test_the_default_commit_holds_the_ledger_the_scoring_accepts(tmp_path: Path) -> None:
    """The default COMMIT is on this branch's history, and score_test.py's checks pass on its
    ledger, test ledger and configs for the default RUN_SEEDS: the 7 rows of phase 1, complete for
    seeds 0, 1 and 2, no test row (CI checks out the full history)."""
    p = _parameters()
    commit = str(p["COMMIT"])
    git = ["git", "-C", str(ROOT)]
    ancestor = subprocess.run([*git, "merge-base", "--is-ancestor", commit, "HEAD"], check=False)
    assert ancestor.returncode == 0, f"COMMIT {commit} is not in this branch's history"
    for name in ("docs/ledger.md", "docs/test-ledger.md", "config.yaml"):
        shown = subprocess.run(
            [*git, "show", f"{commit}:{name}"], check=True, capture_output=True
        ).stdout
        (tmp_path / Path(name).name).write_bytes(shown)
    rows = read_ledger(tmp_path / "ledger.md")
    assert len(rows) == 7
    check_unique(rows)
    check_complete(rows, check_run_seeds(p["RUN_SEEDS"]))
    assert read_test_ledger(tmp_path / "test-ledger.md") == []
    for row in rows:
        score_test._config_of(row, tmp_path / "config.yaml")  # the config still hashes to its row


# The go-ahead: APPROVED_BY.


@pytest.mark.parametrize("approver", ["", "   ", None, "a | b", "Jane\u2028Doe", "Jane\x0bDoe"])
def test_the_go_ahead_cell_refuses_an_empty_or_broken_approver(approver: str | None) -> None:
    """The refusal: the user's go-ahead is needed, and nothing has been scored yet."""
    names = {**_imports(), "APPROVED_BY": approver}
    with pytest.raises(RuntimeError, match="needs the user's go-ahead") as refused:
        _run(_cell("check_approver"), names)
    assert "nothing has been scored" in str(refused.value)
    assert "approver" not in names


def test_the_go_ahead_cell_passes_a_name(capsys: pytest.CaptureFixture[str]) -> None:
    names = _run(_cell("check_approver"), {**_imports(), "APPROVED_BY": f" {APPROVER} "})
    assert names["approver"] == APPROVER
    assert f"approved by {APPROVER}" in capsys.readouterr().out


def test_the_go_ahead_comes_right_before_the_scoring() -> None:
    """The go-ahead cell is the last code cell before the one that runs the scoring, so a Run All
    with APPROVED_BY empty stops there, after every check."""
    assert _cell("check_approver") == _scoring_cell() - 1


def test_the_cells_run_in_order() -> None:
    """Parameters, clone, install, the code, GPU, unpacking, cache, checkpoints and their checks,
    the go-ahead, the scoring, the rows, the zip."""
    order = [
        next(i for i, source in enumerate(CODE) if "COMMIT must be" in source),
        next(i for i, tree in enumerate(TREES) if "'git', 'clone'" in ast.unparse(tree)),
        next(i for i, tree in enumerate(TREES) if "'pip', 'install'" in ast.unparse(tree)),
        _imports_cell(),
        _cell("check_gpu"),
        _cell("unpack_cache"),
        _cell("check_images"),
        _cell("os.walk"),
        _cell("resolve_checkpoints"),
        _cell("check_approver"),
        _scoring_cell(),
        _cell("ledger_line"),
        _cell("shutil.make_archive"),
    ]
    assert order == sorted(order) and len(set(order)) == len(order), order


# One scoring, through score_test.py's command line; nothing else reached.


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def test_exactly_one_code_cell_runs_the_scoring() -> None:
    """R17: the module name appears once, in the command of the one `subprocess.Popen`, which
    sits at the top of its cell (no loop around it); no other process is started but git and
    pip."""
    named = [
        (i, node)
        for i, tree in enumerate(TREES)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        if "score_test" in node.value
    ]
    assert [(i, node.value) for i, node in named] == [(_scoring_cell(), SCORING_MODULE)]
    cell = TREES[_scoring_cell()]
    (command,) = [
        node
        for node in ast.walk(cell)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "command"
    ]
    assert ast.unparse(command.value) == f"[sys.executable, '-m', '{SCORING_MODULE}', *argv]"
    popens = [
        (i, call) for i, tree in enumerate(TREES) for call in _calls_in(tree, "subprocess.Popen")
    ]
    assert [(i, ast.unparse(call.args[0])) for i, call in popens] == [(_scoring_cell(), "command")]
    parents = _parents(cell)
    node: ast.AST = popens[0][1]
    while node in parents:
        node = parents[node]
        assert not isinstance(node, ast.For | ast.While | ast.comprehension), "a loop runs it"
    runs = [
        ast.unparse(call.args[0]) for tree in TREES for call in _calls_in(tree, "subprocess.run")
    ]
    assert all(run.startswith("['git'") or "'pip'" in run for run in runs), runs
    for other in ("call", "check_call", "check_output", "getoutput", "getstatusoutput"):
        assert "subprocess." + other not in ALL_CODE


def test_only_the_listed_names_are_imported() -> None:
    """The scoring has no second route: from score_test.py only the checks that run before
    anything is loaded and the test ledger's reader and line writer (no `main`, `prepare`,
    `load_models` or `score_checkpoint`); no module of trailcam imported whole but the package,
    whose only attribute used is `__file__`; no dynamic import or code."""
    taken: dict[str, set[str]] = {}
    plain: set[str] = set()
    for tree in TREES:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module and node.level == 0
                taken.setdefault(node.module, set()).update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                plain |= {alias.name for alias in node.names}
            elif isinstance(node, ast.Call):
                assert ast.unparse(node.func) not in DYNAMIC, ast.unparse(node)
            elif isinstance(node, ast.Attribute) and ast.unparse(node.value) == "trailcam":
                assert node.attr == "__file__"
    assert {m: n for m, n in taken.items() if m.startswith("trailcam")} == TRAILCAM_IMPORTS
    assert plain | {m for m in taken if not m.startswith("trailcam")} == OTHER_IMPORTS | {
        "trailcam"
    }
    assert taken["pathlib"] == {"Path"} and taken["importlib.metadata"] == {"version"}


def test_no_cell_calls_the_evaluation_or_training_code() -> None:
    """Item 9.03: no cell runs evaluate, attention, slices or train. Their modules are never
    imported whole or named in a string (a `-m` command), what is taken from train.py holds no
    callable, and no cell calls a `main`."""
    for i, tree in enumerate(TREES):
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in NEVER_RUN:
                module = importlib.import_module(node.module)
                assert not [a.name for a in node.names if callable(getattr(module, a.name))]
            if isinstance(node, ast.ImportFrom) and node.module in PACKAGES:
                assert not {a.name for a in node.names} & {m.split(".")[-1] for m in NEVER_RUN}
            if isinstance(node, ast.Import):
                assert not [a.name for a in node.names if a.name.startswith(NEVER_RUN)]
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not [m for m in NEVER_RUN if m in node.value], (i, node.value)
            if isinstance(node, ast.Call):
                assert ast.unparse(node.func).split(".")[-1] != "main", (i, ast.unparse(node))


def test_no_split_is_named_in_code() -> None:
    """R17: the notebook names no split; only score_test.py reads cis_test and trans_test."""
    for name in ("cis_test", "trans_test", "trans_val", "cis_val", "'val'", '"val"', "--split"):
        assert name not in ALL_CODE


# The scoring cell's command.


def _score_with(exit_code: int, **values: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The scoring cell run as written with a stand-in `subprocess.Popen` that records its
    arguments, prints one line and exits `exit_code`."""
    calls: list[dict[str, Any]] = []

    class Popen:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            calls.append({"command": command, **kwargs})
            self.stdout = io.StringIO("a line of the scoring\n")
            self.returncode = exit_code

        def __enter__(self) -> "Popen":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    work = Path("/kaggle/working")
    names = _namespace(work, Path("/kaggle/input"), APPROVED_BY=APPROVER)
    names["subprocess"] = types.SimpleNamespace(
        Popen=Popen, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT
    )
    names["checked"] = [
        types.SimpleNamespace(path=Path(f"/kaggle/input/trailcam-checkpoints/{n}.pt"))
        for n in ("small", "arm_a", "arm_b")
    ]
    names["overlay_photos"] = [work / "who-goes-there" / "assets" / "photo.jpg"]
    names["data_root"] = Path("/tmp/trailcam-cache/cct20_512")
    names.update(values)
    return _run(_scoring_cell(), names), calls


@pytest.mark.parametrize("seeds", [(0, 1, 2), (0,), (2, 0)])
def test_run_seeds_and_the_paths_reach_the_command_line(
    seeds: tuple[int, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    """RUN_SEEDS reaches --run-seeds, APPROVED_BY --approved-by (a name starting with a dash
    too), and the clone's config, ledger and test ledger, OUT, the cache, every checked
    checkpoint and every assets/ photo reach theirs, as score_test.py's own parser reads them."""
    for approver in (APPROVER, "-dash name"):
        names, calls = _score_with(0, RUN_SEEDS=seeds, APPROVED_BY=approver)
        ((call,),) = [calls]
        command = call["command"]
        assert command[:3] == [sys.executable, "-m", SCORING_MODULE]
        args = score_test.build_parser().parse_args(command[3:])
        assert args.run_seeds == list(seeds)
        assert args.approved_by == approver
        assert args.checkpoint == [c.path for c in names["checked"]]
        assert args.data_root == names["data_root"]
        assert args.config == names["CONFIG_FILE"]
        assert (
            args.ledger == names["LEDGER"] == Path("/kaggle/working/who-goes-there/docs/ledger.md")
        )
        assert args.test_ledger == names["TEST_LEDGER"]
        assert args.out == names["OUT"] == Path("/kaggle/working/score")
        assert args.assets == names["ASSETS"]
        assert args.overlay == [(photo, None) for photo in names["overlay_photos"]]
        assert call["env"]["PYTHONUNBUFFERED"] == "1"
        assert call["stderr"] == subprocess.STDOUT and call["errors"] == "replace"
        assert "a line of the scoring" in capsys.readouterr().out


def test_the_scoring_cell_run_alone_with_approved_by_empty_is_refused_by_score_test() -> None:
    """The go-ahead cell skipped, APPROVED_BY empty and an `approver` left from an earlier run in
    the kernel: the command still carries APPROVED_BY itself, so score_test.py's first check
    refuses it (exit 1 before anything is loaded, tests/test_score_test.py)."""
    _, calls = _score_with(0, APPROVED_BY="", approver="Someone Earlier")
    ((call,),) = [calls]
    args = score_test.build_parser().parse_args(call["command"][3:])
    assert args.approved_by == ""
    with pytest.raises(Refusal, match="--approved-by is required"):
        score_test.check_approver(args.approved_by)


@pytest.mark.parametrize("code", [1, 2, -9])
def test_a_scoring_that_fails_fails_its_cell(code: int) -> None:
    """A refusal, a crash or a kill: the cell raises, so Save & Run All stops before the rows and
    the zip, and says how to recover."""
    with pytest.raises(RuntimeError, match=f"exited {code}") as failed:
        _score_with(code)
    assert "nothing was appended" in str(failed.value) and "new OUT" in str(failed.value)


def test_an_alarm_is_reported_and_the_outputs_are_kept(capsys: pytest.CaptureFixture[str]) -> None:
    """R16: exit 3 has appended the rows, and the scoring cannot run again, so the cell goes on
    (the rows are printed and the outputs zipped next) and reports the alarm; exit 0 is quiet."""
    names, _ = _score_with(ALARM_EXIT)
    assert names["exit_code"] == ALARM_EXIT
    assert "LEAKAGE ALARM (R16)" in capsys.readouterr().err
    _score_with(0)
    assert "LEAKAGE ALARM" not in capsys.readouterr().err


# The checkpoints' folder.


def _find(kaggle_input: Path, root: Path | None = None) -> dict[str, Any]:
    names = _namespace(Path("/kaggle/working"), kaggle_input, CHECKPOINTS_ROOT=root)
    return _run(_cell("os.walk"), names)


def test_the_checkpoints_folder_is_found_under_the_input(tmp_path: Path) -> None:
    """At any depth, links followed and counted once, every `.pt` under it; none or two folders
    of the dataset's name are refused with the folders that hold `.pt` files; CHECKPOINTS_ROOT,
    when set, is used as given."""
    mount = tmp_path / "input" / "datasets" / "someone" / "trailcam-checkpoints"
    (mount / "models").mkdir(parents=True)
    for name in ("b.pt", "models/a.pt", "notes.txt"):
        (mount / name).write_bytes(b"x")
    (tmp_path / "input" / "trailcam-checkpoints").symlink_to(mount)  # a second mount, one folder
    names = _find(tmp_path / "input")
    assert names["checkpoint_dir"] == mount.resolve()
    assert names["checkpoint_files"] == [mount.resolve() / "b.pt", mount.resolve() / "models/a.pt"]

    other = tmp_path / "other-input"
    (other / "elsewhere").mkdir(parents=True)
    (other / "elsewhere" / "c.pt").write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="0 folders named trailcam-checkpoints") as none:
        _find(other)
    assert str(other / "elsewhere") in str(none.value) and "set CHECKPOINTS_ROOT" in str(none.value)
    assert _find(other, other / "elsewhere")["checkpoint_files"] == [other / "elsewhere" / "c.pt"]
    with pytest.raises(FileNotFoundError, match="is not a folder"):
        _find(other, other / "elsewhere" / "c.pt")

    (other / "a" / "trailcam-checkpoints").mkdir(parents=True)
    (other / "b" / "trailcam-checkpoints").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="2 folders named"):
        _find(other)
    with pytest.raises(FileNotFoundError, match="is the dataset attached"):
        _find(tmp_path / "empty-input")


# The cells on the synthetic fixture: Kaggle's folders under tmp_path.


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> World:
    """The fixture of tests/test_score_test.py: a cache, three random-weight checkpoints (the
    small CNN and ResNet-18 arm A and arm B, seed 0), their ledger rows and an assets/ photo."""
    return _build_world(tmp_path_factory.mktemp("score_notebook"))


def _kaggle(world: World, root: Path) -> tuple[Path, Path]:
    """A clone with the fixture's config, ledger, an empty test ledger and the photo, and an
    input folder holding the checkpoints as Kaggle mounts a dataset (links to the files)."""
    work, kaggle_input = root / "working", root / "input"
    clone = work / "who-goes-there"
    (clone / "docs").mkdir(parents=True)
    (clone / "assets").mkdir()
    shutil.copy(world.config_file, clone / "config.yaml")
    shutil.copy(world.ledger, clone / "docs" / "ledger.md")
    shutil.copy(TEST_LEDGER_PATH, clone / "docs" / "test-ledger.md")
    shutil.copy(world.photo, clone / "assets" / "photo.jpg")
    mount = kaggle_input / "datasets" / "someone" / "trailcam-checkpoints" / "models"
    mount.mkdir(parents=True)
    for path in world.checkpoints.values():
        (mount / path.name).symlink_to(path)
    return work, kaggle_input


def _fixture_names(world: World, root: Path, **values: Any) -> dict[str, Any]:
    work, kaggle_input = _kaggle(world, root)
    return _namespace(work, kaggle_input, data_root=world.cache, head="0" * 40, **values)


@pytest.fixture
def forbid_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """No checkpoint loaded, no image opened, no process started."""

    def fail(*_: Any, **__: Any) -> None:
        raise AssertionError("loaded, opened or started before the go-ahead")

    monkeypatch.setattr(torch, "load", fail)
    monkeypatch.setattr(Image, "open", fail)
    monkeypatch.setattr(subprocess, "Popen", fail)


@pytest.mark.usefixtures("forbid_loading")
def test_the_rehearsal_checks_everything_and_stops_before_anything_is_loaded(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """APPROVED_BY empty, its default: the cache and the checkpoints pass their checks, then the
    go-ahead cell refuses, before any checkpoint is loaded, any image opened or the scoring
    started; OUT is never made."""
    names = _fixture_names(world, tmp_path, RUN_SEEDS=(0,))
    assert names["APPROVED_BY"] == ""
    for cell in (_cell("check_images"), _cell("os.walk"), _cell("resolve_checkpoints")):
        _run(cell, names)
    out = capsys.readouterr().out
    assert "3 checkpoints, one per ledger row" in out and "1 assets/ photos" in out
    with pytest.raises(RuntimeError, match="needs the user's go-ahead"):
        _run(_cell("check_approver"), names)
    assert not names["OUT"].exists()


def _broken(names: dict[str, Any], how: str) -> None:
    mount = names["INPUT"] / "datasets" / "someone" / "trailcam-checkpoints" / "models"
    link = sorted(mount.glob("*.pt"))[0]  # arm_a-seed0.pt
    if how == "missing":
        link.unlink()
    elif how == "changed":
        data = link.read_bytes()
        link.unlink()
        link.write_bytes(data + b"\0")
    elif how == "scored":
        names["TEST_LEDGER"].write_text(
            names["TEST_LEDGER"].read_text()
            + ledger_line(dict.fromkeys(TEST_LEDGER_COLUMNS, "x"))
            + "\n"
        )
    elif how == "seeds":
        names["RUN_SEEDS"] = (0, 1)


@pytest.mark.usefixtures("forbid_loading")
@pytest.mark.parametrize(
    ("how", "why"),
    [
        ("missing", "no checkpoint given for arm_a-seed0"),
        ("changed", "has no row in the ledger: an unknown checkpoint"),
        ("scored", "the test ledger already holds 1 rows"),
        ("seeds", "no resnet18 arm A row for run seed 1"),
    ],
)
def test_the_checkpoint_checks_refuse_before_anything_is_loaded(
    world: World, tmp_path: Path, how: str, why: str
) -> None:
    """score_test.py's own checks, called by section 4: a checkpoint missing from the dataset,
    one whose bytes changed (its sha256 is no ledger row's), a test ledger that holds a row (the
    scoring has run), a seed the ledger lacks."""
    names = _fixture_names(world, tmp_path, RUN_SEEDS=(0,))
    _broken(names, how)
    _run(_cell("os.walk"), names)
    with pytest.raises(Refusal, match=why):
        _run(_cell("resolve_checkpoints"), names)


def test_the_notebook_scores_the_fixture_once_and_prints_and_zips_the_rows(
    world: World, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Sections 3 to 7 as written, with the go-ahead, on the fixture: one run of score_test's
    command line appends one row per checkpoint to the clone's test ledger, in ledger order, with
    each file's sha256 and the approver; the cell after it prints them under the header, and the
    zip holds the tables and figures. Run again in the same session, section 4 refuses: the
    scoring has run. Nothing is written to the repo's docs."""
    figures_before = _figures_snapshot()
    repo_test_ledger = TEST_LEDGER_PATH.read_bytes()
    names = _fixture_names(world, tmp_path, RUN_SEEDS=(0,), APPROVED_BY=APPROVER)
    cells = ["check_images", "os.walk", "resolve_checkpoints", "check_approver"]
    for cell in [*map(_cell, cells), _scoring_cell(), _cell("ledger_line")]:
        _run(cell, names)
    _run(_cell("shutil.make_archive"), names)
    captured = capsys.readouterr()

    assert names["exit_code"] in (0, ALARM_EXIT), captured.out[-3000:]
    assert ("LEAKAGE ALARM (R16)" in captured.err) == (names["exit_code"] == ALARM_EXIT)
    rows = read_test_ledger(names["TEST_LEDGER"])
    assert [row["run id"] for row in rows] == [row["run id"] for row in world.rows.values()]
    assert [row["checkpoint sha256"] for row in rows] == [
        row["checkpoint sha256"] for row in world.rows.values()
    ]
    assert {row["approved by"] for row in rows} == {APPROVER}
    assert f"--approved-by={APPROVER}" in captured.out and "--run-seeds 0 " in captured.out
    printed = captured.out.split("for docs/test-ledger.md (paste once, below its header):\n", 1)[1]
    assert printed.startswith(scoring_header() + "\n")
    assert [ledger_line(row) for row in rows] == printed.splitlines()[2:5]

    with zipfile.ZipFile(names["WORK"] / "score.zip") as archive:
        held = set(archive.namelist())
    assert {"score/headline.md", "score/slices.md"} <= held
    for row in rows:
        run = f"score/{row['run id']}"
        assert {f"{run}/metrics.md", f"{run}/attention.md"} <= held
        assert f"{run}/{evaluate.OVERLAY_NAME.format(1)}" in held  # the assets/ photo's overlay
    assert len(held - {n for n in held if n.endswith("/")}) == sum(
        p.is_file() for p in names["OUT"].rglob("*")
    )

    with pytest.raises(Refusal, match="already holds 3 rows"):
        _run(_cell("resolve_checkpoints"), names)
    assert _figures_snapshot() == figures_before
    assert TEST_LEDGER_PATH.read_bytes() == repo_test_ledger


# Install and unpack: the training notebook's patterns.


def test_the_clone_may_differ_from_commit_in_the_test_ledger_only(tmp_path: Path) -> None:
    """The clone cell run as written, cloning this repo: a changed docs/test-ledger.md (the
    scoring appended to it) is let through, so section 4's run-once refusal is reached; any other
    change, docs/ledger.md included, is refused."""
    names = _namespace(tmp_path / "working", tmp_path / "input", REPO_URL=str(ROOT))
    (clone_cell,) = [i for i, tree in enumerate(TREES) if "'git', 'clone'" in ast.unparse(tree)]
    _run(clone_cell, names)
    docs = names["CLONE"] / "docs"
    with (docs / "test-ledger.md").open("a") as ledger:
        ledger.write(ledger_line(dict.fromkeys(TEST_LEDGER_COLUMNS, "x")) + "\n")
    _run(clone_cell, names)  # reused, the test ledger changed: passes
    with (docs / "ledger.md").open("a") as ledger:
        ledger.write("\n")
    with pytest.raises(RuntimeError, match="1 changed files, not a clean"):
        _run(clone_cell, names)


def test_the_only_pip_install_is_the_clone() -> None:
    """R38: one `pip install`, of the clone at COMMIT, that may not upgrade what Kaggle ships."""
    pip_calls = [
        call
        for tree in TREES
        for call in _calls_in(tree, "subprocess.run")
        if "'pip'" in ast.unparse(call.args[0])
    ]
    assert [ast.unparse(c.args[0]) for c in pip_calls] == [
        "[sys.executable, '-m', 'pip', 'install', '--quiet', str(CLONE)]"
    ]
    assert ALL_CODE.count('"pip"') == 1 and "%pip" not in ALL_CODE and "!pip" not in ALL_CODE
    for flag in ("git+", "--upgrade", "-U", "--force-reinstall", "--no-deps", "-e"):
        assert f'"{flag}' not in ALL_CODE
    assert 'KEPT = ("torch", "torchvision", "Pillow", "timm", "PyYAML")' in ALL_CODE
    assert "after_install != preinstalled" in ALL_CODE
    checkout = [ast.unparse(c.args[0]) for tree in TREES for c in _calls_in(tree, "subprocess.run")]
    assert "['git', 'clone', '--quiet', REPO_URL, str(CLONE)]" in checkout
    assert "['git', '-C', str(CLONE), 'checkout', '--quiet', '--detach', COMMIT]" in checkout


def test_trailcam_is_imported_once_per_kernel_and_matches_the_clone() -> None:
    """The first cell that imports `trailcam` refuses a kernel that already holds it, and every
    installed module is compared with the clone's byte for byte."""
    first = _imports_cell()
    assert not any("trailcam" in _imported(tree) for tree in TREES[:first])
    guard_stmt = TREES[first].body[0]
    assert isinstance(guard_stmt, ast.If)
    assert ast.unparse(guard_stmt.test) == "'trailcam' in sys.modules"
    assert isinstance(guard_stmt.body[0], ast.Raise)
    assert "(installed / m).read_bytes() != (CLONE / 'trailcam' / m).read_bytes()" in ast.unparse(
        TREES[first]
    )


def test_one_gpu_is_pinned_before_torch_and_checked_before_the_cache() -> None:
    """plan.md section 5: `CUDA_VISIBLE_DEVICES` is set once, to `VISIBLE_DEVICES`, before any
    cell imports torch or trailcam (the scoring's process inherits it); `check_gpu` fails a
    session without a usable GPU before the cache is unpacked."""
    pins = [
        (i, node)
        for i, tree in enumerate(TREES)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "os.environ['CUDA_VISIBLE_DEVICES']"
    ]
    assert ALL_CODE.count("CUDA_VISIBLE_DEVICES") == 1
    ((pin_cell, pin),) = pins
    assert ast.literal_eval(pin.value) == device.VISIBLE_DEVICES
    first = next(i for i, tree in enumerate(TREES) if _imported(tree) & {"torch", "trailcam"})
    assert pin_cell < first == _imports_cell()
    assert "print(check_gpu())" in CODE[_cell("check_gpu")]
    assert _cell("check_gpu") < _cell("unpack_cache")


def test_the_unpack_cell_sets_data_root_and_refuses_what_training_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The unpacking cell run as written: a tar becomes the local data_root; a wrong path lists
    the archives; the loose dataset attached too is refused in an interactive session and warned
    about in a committed one, before anything is unpacked."""
    archive = tmp_path / "cct20_512.tar"
    with tarfile.open(archive, "w:") as tar:
        top = tarfile.TarInfo(CACHE_TOP)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
    kaggle_input = tmp_path / "input"
    kaggle_input.mkdir()

    def unpack(names: dict[str, Any], local: str) -> dict[str, Any]:
        names["LOCAL_CACHE"] = tmp_path / local
        return _run(_cell("unpack_cache"), names)

    names = _namespace(tmp_path / "working", kaggle_input, CACHE_ARCHIVE=archive)
    assert unpack(names, "local")["data_root"] == tmp_path / "local" / CACHE_TOP
    names["CACHE_ARCHIVE"] = tmp_path / "absent.tar"
    with pytest.raises(FileNotFoundError, match="is the output of the packing notebook attached"):
        unpack(names, "absent")

    loose = kaggle_input / "datasets" / "codx888" / "cct20-512" / "cct20_512"
    loose.mkdir(parents=True)
    (loose / "day_night.json").write_text("{}")
    names["CACHE_ARCHIVE"] = archive
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    with pytest.raises(RuntimeError, match="still attached"):
        unpack(names, "interactive")
    assert not (tmp_path / "interactive").exists()
    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Batch")
    assert unpack(names, "batch")["data_root"] == tmp_path / "batch" / CACHE_TOP


# Markdown.


def test_markdown_says_once_part_way_and_never_save_and_run_all() -> None:
    """Item 9.03's three statements, and why the notebook is separate; the warning is said at
    the top and again over the go-ahead cell."""
    never = "Never Save & Run All with `APPROVED_BY` filled in until the user means it."
    for needle in (
        "## This runs once",
        "separate from the training and evaluation notebooks so the test sets can never be "
        "touched by accident",
        "`APPROVED_BY` starts empty",
        never,
        "## If it stops part way",
        "run again from the start with a new `--out`, the recovery `score_test.py` documents",
        "If it stopped after `appended ... rows`, the scoring has run: never run it again.",
        'set `APPROVED_BY` back to `""`',
    ):
        assert needle in FLAT, needle
    cells = NB["cells"]
    go_ahead = [
        i for i, c in enumerate(cells) if "check_approver(APPROVED_BY)" in "".join(c["source"])
    ]
    assert never in " ".join("".join(cells[go_ahead[0] - 1]["source"]).split())


def test_how_to_run_says_import_attach_rehearse_then_run_once() -> None:
    steps = [
        "File > Import Notebook",
        f"{PUBLIC_REPO}/blob/main/notebooks/kaggle_score_test.ipynb",
        "Add Input > Notebook",
        "Add Input > Dataset: the private dataset `trailcam-checkpoints`",
        "Keep exactly one parameter cell",
        "at or after the merge of item 9.03",
        "Accelerator GPU T4 x2",
        "Internet on",
        "Rehearse: with `APPROVED_BY` empty, Run All.",
        "Then stop the session",
        "The one run, when the user means it",
        "Save Version > Save & Run All (Commit)",
        "paste the rows section 7 prints into `docs/test-ledger.md`, once",
        "download `score.zip`",
    ]
    positions = [FLAT.find(step) for step in steps]
    assert -1 not in positions, [s for s, i in zip(steps, positions, strict=True) if i == -1]
    assert positions == sorted(positions)


def test_markdown_cites_the_data() -> None:
    """R06: licence and citation."""
    for needle in ("CDLA-Permissive 1.0", "Beery", "Recognition in Terra Incognita", "MIT"):
        assert needle in MARKDOWN


def test_no_credentials() -> None:
    """R07: no token or key file is read or pasted."""
    lowered = ALL_CODE.lower()
    for needle in ("token", "kaggle.json", "secret", "password", "api_key", "hf_"):
        assert needle not in lowered
