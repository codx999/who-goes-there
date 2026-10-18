"""notebooks/kaggle_evaluate.ipynb, read as JSON and its code cells as Python (item 8.08).

The notebook runs only on Kaggle, so its clone, install and GPU cells never run here. Its other
cells run as written: the checkpoint check on small made-up files and a ledger made for them, the
evaluation cells with the entry points replaced by recorders, and every cell from the checkpoint
check to the summary on a cache-shaped synthetic fixture with random-weight checkpoints. Every code
cell parses as plain Python and defines nothing, so the logic stays in `trailcam` (R17 val and
trans_val only, no test split or final scoring, the SHA-256 check before any evaluation; R14 the
split hash; R41 outputs cleared and the guard; R38 the one pip install; R03 counts, run ids and
hashes only; R06 the licence and citation; R07 no credentials).
"""

import ast
import importlib
import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import torch
import yaml

from trailcam.data import synthetic
from trailcam.data.cache import CACHE_TOP
from trailcam.data.classes import CLASSES
from trailcam.data.metadata import DAY_NIGHT_SIDECAR, load_metadata
from trailcam.data.split import make_split
from trailcam.evaluate import attention, evaluate, slices
from trailcam.model.factory import build_model
from trailcam.train import device
from trailcam.train.config import CONFIG_PATH, config_hash, load_config
from trailcam.train.ledger import check_ledger, read_table
from trailcam.train.train import (
    CACHE_ANNOTATIONS,
    CACHE_IMAGES,
    LEDGER_COLUMNS,
    LEDGER_PATH,
    file_sha256,
    ledger_header,
    ledger_row,
)

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_evaluate.ipynb"
TRAIN_NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
LOOK_NOTEBOOK = ROOT / "notebooks" / "kaggle_look_at_data.ipynb"
PUBLIC_REPO = "https://github.com/codx999/who-goes-there"
PARAMETERS = (
    "COMMIT",
    "LEDGER",
    "CACHE_ARCHIVE",
    "CHECKPOINTS_ROOT",
    "ASSETS_OVERLAY_RUN",
    "SLICE_PAIRS",
)
ENTRY_POINTS = ("evaluate.main", "attention.main", "slices.main")
CLI_SPLITS = ("val", "trans_val")
# Kaggle's folders in the parameter cell, and where the tests put them (`_rebase`).
KAGGLE_FOLDERS = ("/tmp/who-goes-there", "/tmp/trailcam-cache", "/kaggle/working", "/kaggle/input")
# The runs of the made-up ledgers: the small CNN, and arm A and arm B of two run seeds.
RUNS = (
    ("small_cnn", 0),
    ("resnet18_arm_a", 0),
    ("resnet18_arm_b", 0),
    ("resnet18_arm_a", 1),
    ("resnet18_arm_b", 1),
)
RUN_IDS = tuple(f"{name}-seed{seed}-20261016T0{n}0000Z" for n, (name, seed) in enumerate(RUNS))
# Item 13.05: the runs of the made-up phase-2 ledgers, and the configs they use.
P2_LEDGER = "docs/ledger-phase2.md"
P2_CONFIGS = ("p2_baseline", "p2_head3", "p2_res320")
P2_RUNS = (
    ("p2_baseline", 0),
    ("p2_head3", 0),
    ("p2_res320", 0),
    ("p2_baseline", 1),
    ("p2_res320", 1),
)
P2_RUN_IDS = tuple(
    f"{name}-seed{seed}-20261019T0{n}0000Z" for n, (name, seed) in enumerate(P2_RUNS)
)

_spec = importlib.util.spec_from_file_location("guard_data", ROOT / "scripts" / "guard_data.py")
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard_data"] = guard
_spec.loader.exec_module(guard)

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: "".join(cell["source"]) for cell in NB["cells"]}
CODE_IDS = [cell["id"] for cell in NB["cells"] if cell["cell_type"] == "code"]
TREES = {cell_id: ast.parse(CELLS[cell_id]) for cell_id in CODE_IDS}
MARKDOWN = "\n".join(CELLS[c["id"]] for c in NB["cells"] if c["cell_type"] == "markdown")
FLAT_MARKDOWN = " ".join(MARKDOWN.split())
ALL_CODE = "\n".join(CELLS[cell_id] for cell_id in CODE_IDS)


def _calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == name
    ]


def _cells_calling(name: str) -> list[str]:
    return [cell_id for cell_id in CODE_IDS if _calls(TREES[cell_id], name)]


def _imported(tree: ast.AST) -> set[str]:
    """Top-level package names a cell imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def _bound(tree: ast.AST) -> set[str]:
    """Every name a cell binds: assignments, loop and `with` targets, imports."""
    names = {
        n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            names |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    return names


def _parameters(tree: ast.Module) -> dict[str, object]:
    """The assignments of a parameter cell: literals as values, anything else as source."""
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                values[node.targets[0].id] = ast.unparse(node.value)
    return values


def _run(cell_ids: Sequence[str], names: dict[str, object]) -> dict[str, object]:
    """The cells run as written, in order, in one namespace (one kernel)."""
    for cell_id in cell_ids:
        exec(compile(TREES[cell_id], f"cell {cell_id}", "exec"), names)
    return names


def _rebase(value: object, root: Path) -> object:
    """A Kaggle path of the parameter cell moved under `root`, keeping its place below the folder
    (`/kaggle/input/x` becomes `<root>/kaggle/input/x`); any other value as it is."""
    if isinstance(value, Path) and any(value.is_relative_to(f) for f in KAGGLE_FOLDERS):
        return root / value.relative_to("/")
    return value


def _kaggle_names(root: Path, **overrides: object) -> dict[str, object]:
    """The namespace after the parameter cell and the imports cell, with every Kaggle path under
    `root` and `head` standing for the clone cell's commit. The imports cell runs as written
    against a clone whose `trailcam` is this checkout's package (a symbolic link), with a `sys`
    whose modules lack `trailcam`, as in a fresh kernel."""
    names = _run(["parameters"], {})
    names = {key: _rebase(value, root) for key, value in names.items()}
    clone = names["CLONE"]
    assert isinstance(clone, Path)
    clone.mkdir(parents=True, exist_ok=True)
    if not (clone / "trailcam").exists():
        (clone / "trailcam").symlink_to(ROOT / "trailcam", target_is_directory=True)
    names.update(os=os, sys=SimpleNamespace(modules={}), head="0" * 40)
    _run(["imports"], names)
    names["sys"] = sys
    names.update(overrides)
    return names


def _write_config(path: Path) -> None:
    """config.yaml with phase 1's three training configs and the phase-2 ones the tests use small
    enough for the CPU: batch 8, no workers, no pinned memory, no download."""
    settings = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    for name in ("small_cnn", "resnet18_arm_a", "resnet18_arm_b", *P2_CONFIGS):
        settings["configs"][name].update(
            batch_size=8, num_workers=0, pin_memory=False, pretrained=False
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(settings), encoding="utf-8")


def _write_ledger(
    names: Mapping[str, object],
    files: Mapping[str, Path],
    split_hash: str,
    score: str = "1.0000",
    runs: Mapping[str, tuple[str, int]] = dict(zip(RUN_IDS, RUNS, strict=True)),
) -> None:
    """A ledger at LEDGER_PATH with one row per run id of `files` (its config and run seed in
    `runs`), as train.py writes them, that passes `trailcam.train.ledger`'s check."""
    config_file, ledger = names["CONFIG_FILE"], names["LEDGER_PATH"]
    assert isinstance(config_file, Path) and isinstance(ledger, Path)
    lines = ["# Ledger", "", ledger_header()]
    for run_id, path in files.items():
        name, seed = runs[run_id]
        config = load_config(name, config_file)
        values = {
            "run id": run_id,
            "model": config.model,
            "arm": config.arm,
            "config": name,
            "config hash": config_hash(config),
            "split seed": str(config.split_seed),
            "run seed": str(seed),
            "split hash": split_hash,
            "epochs run": "2",
            "best epoch": "1",
            "scheduler": f"{config.scheduler} warmup {config.warmup_epochs}",
            "val accuracy": score,
            "val macro-F1": score,
            "trans_val accuracy": score,
            "trans_val macro-F1": score,
            "checkpoint sha256": file_sha256(path),
        }
        lines.append(ledger_row({column: values[column] for column in LEDGER_COLUMNS}))
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows, problems = check_ledger(ledger)
    assert problems == [] and len(rows) == len(files)


def _light_names(root: Path, run_ids: Sequence[str]) -> dict[str, object]:
    """The namespace of `_kaggle_names` over a clone with the small-batch config and a ledger of
    `run_ids`, each with a small made-up `.pt` file under CHECKPOINTS_ROOT (the checkpoint check
    only hashes them), and no cache."""
    names = _kaggle_names(root)
    config_file, checkpoints_root = names["CONFIG_FILE"], names["CHECKPOINTS_ROOT"]
    assert isinstance(config_file, Path) and isinstance(checkpoints_root, Path)
    _write_config(config_file)
    checkpoints_root.mkdir(parents=True)
    files = {}
    for run_id in run_ids:
        files[run_id] = checkpoints_root / f"{run_id}.pt"
        files[run_id].write_bytes(f"made-up checkpoint {run_id}".encode())
    _write_ledger(names, files, "0" * 64)
    return names


@pytest.fixture
def light(tmp_path: Path) -> dict[str, object]:
    """A Kaggle layout for the checkpoint check: the clone's config and ledger, five made-up
    checkpoints and no cache."""
    return _light_names(tmp_path, RUN_IDS)


# The notebook as committed


def test_outputs_and_attachments_are_cleared() -> None:
    """R41: no output, execution count or attachment."""
    for cell in NB["cells"]:
        assert "attachments" not in cell
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None


@pytest.mark.parametrize("notebook", [NOTEBOOK, LOOK_NOTEBOOK], ids=lambda p: p.name)
def test_both_notebooks_pass_the_data_guard(notebook: Path) -> None:
    """R41 and R04: the guard finds nothing in the evaluation notebook or in the look-at-the-data
    notebook with its crop check."""
    path = notebook.relative_to(ROOT).as_posix()
    assert guard.path_problems(path) == []
    assert guard.content_problems(notebook.read_bytes(), path) == []


def test_code_cells_define_nothing() -> None:
    """Plain Python that calls `trailcam` and the standard library: no function, class or lambda."""
    defining = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
    for tree in TREES.values():
        assert not [n for n in ast.walk(tree) if isinstance(n, defining)]


def test_trailcam_names_exist() -> None:
    """Every name the notebook takes from `trailcam` is there, so a rename fails here first."""
    modules = {"evaluate": evaluate, "attention": attention, "slices": slices}
    for tree in TREES.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("trail"):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
            if isinstance(node, ast.Attribute) and ast.unparse(node.value) in modules:
                assert hasattr(modules[ast.unparse(node.value)], node.attr), ast.unparse(node)


def test_no_test_split_or_final_scoring_is_named() -> None:
    """R17: no code cell names cis_test or trans_test or calls the final scoring, in any form."""
    for name in ("cis_test", "trans_test", "score_test", "TEST_SPLITS"):
        assert name not in ALL_CODE
    for tree in TREES.values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert not re.search(r"(^|_)test($|_)", node.value), node.value


def test_every_split_argument_is_val_or_trans_val() -> None:
    """R17, read from the source: every `--split` in a list literal is followed by val or
    trans_val, and the one split list is both of them."""
    lists = [n for tree in TREES.values() for n in ast.walk(tree) if isinstance(n, ast.List)]
    flagged = 0
    for node in lists:
        values = [e.value if isinstance(e, ast.Constant) else None for e in node.elts]
        for i, value in enumerate(values):
            if value == "--split":
                flagged += 1
                assert i + 1 < len(values) and values[i + 1] in CLI_SPLITS
    assert flagged == 2
    assert 'split_args = ["--split", "val", "--split", "trans_val"]' in CELLS["evaluate"]
    assert ALL_CODE.count('"--split"') == 2


def test_the_parameters() -> None:
    p = _parameters(TREES["parameters"])
    assert isinstance(p["COMMIT"], str) and re.fullmatch(r"[0-9a-f]{40}", p["COMMIT"])
    train = _parameters(
        ast.parse("".join(json.loads(TRAIN_NOTEBOOK.read_text())["cells"][1]["source"]))
    )
    assert p["CACHE_ARCHIVE"] == train["CACHE_ARCHIVE"]  # the training notebook's default
    assert p["CHECKPOINTS_ROOT"] == "Path('/kaggle/input/datasets/codx888/trailcam-checkpoints')"
    assert p["ASSETS_OVERLAY_RUN"] is None  # overlays only when the user names a run
    assert p["REPO_URL"] == PUBLIC_REPO  # the public repo, not a fork (R38)
    assert p["INPUT"] == "Path('/kaggle/input')" and p["WORK"] == "Path('/kaggle/working')"
    assert p["EVAL_DIR"] == "WORK / 'eval'"
    assert p["CLONE"] == "Path('/tmp/who-goes-there')"  # local disk, not the version's output
    assert p["LOCAL_CACHE"] == "Path('/tmp/trailcam-cache')"
    assert p["CONFIG_FILE"] == "CLONE / 'config.yaml'"
    assert p["LEDGER"] == "docs/ledger.md"  # phase 1 by default (item 13.05)
    assert p["LEDGER_PATH"] == "CLONE / LEDGER"
    assert p["SLICE_PAIRS"] is None  # the ledger's own pairs
    assert p["ASSETS"] == "CLONE / 'assets'"
    for value in p.values():  # every Kaggle path lies under a folder the tests move (_rebase)
        if isinstance(value, str) and value.startswith("Path('/"):
            path = PurePosixPath(ast.literal_eval(value[5:-1]))
            assert any(path.is_relative_to(folder) for folder in KAGGLE_FOLDERS), value


def test_the_parameter_cell_is_the_only_one_that_sets_the_parameters() -> None:
    """Item 6.07's lesson: a stale second parameter cell overrides a hand edit. The first code cell
    sets the four parameters and the paths; no later cell binds any of its names, and the check
    cell prints the four in effect."""
    assert CODE_IDS[0] == "parameters"
    first = _bound(TREES["parameters"])
    assert set(PARAMETERS) <= first
    for cell_id in CODE_IDS[1:]:
        assert not first & _bound(TREES[cell_id]), cell_id
    for name in PARAMETERS:
        assert f"{name} {{{name}}}" in CELLS["check"]


def test_the_check_cell_refuses_a_bad_commit_or_overlay_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unset by the test")  # restored afterwards
    names = _run(["parameters"], {})
    _run(["check"], names)
    out = capsys.readouterr().out
    for name in PARAMETERS:
        assert f"{name} {names[name]}" in out
    for bad in ({"COMMIT": "main"}, {"COMMIT": "B64C" * 10}, {"ASSETS_OVERLAY_RUN": 3}):
        with pytest.raises(ValueError, match="must be"):
            _run(["check"], {**names, **bad})


def test_the_default_commit_holds_the_ledger_and_the_code() -> None:
    """The default COMMIT is a commit of main's history whose ledger has exactly the rows of
    docs/ledger.md (the 7 runs of phase 1) and which holds every module the clone cell asks for
    (CI checks out the full history)."""
    commit = str(_parameters(TREES["parameters"])["COMMIT"])
    git = ["git", "-C", str(ROOT)]
    assert subprocess.run([*git, "merge-base", "--is-ancestor", commit, "HEAD"]).returncode == 0

    def show(path: str) -> str:
        shown = subprocess.run(
            [*git, "show", f"{commit}:{path}"], check=True, capture_output=True, text=True
        )
        return shown.stdout

    rows_then, _ = read_table(show("docs/ledger.md"))
    rows_now, _ = read_table(LEDGER_PATH.read_text(encoding="utf-8"))
    then = [(r["run id"], r["checkpoint sha256"]) for r in rows_then]
    assert then == [(r["run id"], r["checkpoint sha256"]) for r in rows_now] and len(then) == 7
    (needed,) = [
        node.value
        for node in ast.walk(TREES["clone"])
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "needed_code"
    ]
    for module, code in ast.literal_eval(needed).items():
        assert code in show(module), module
        assert code in (ROOT / module).read_text(encoding="utf-8"), module


def test_the_clone_is_checked_before_the_install() -> None:
    """The clone cell refuses a moved or edited clone and a COMMIT without the code it calls,
    before pip installs anything."""
    clone, install = CODE_IDS.index("clone"), CODE_IDS.index("install")
    assert clone < install
    assert "head != COMMIT or changed" in CELLS["clone"] and "Factory reset" in CELLS["clone"]
    assert "if lacking:" in CELLS["clone"] and "raise RuntimeError" in CELLS["clone"]


def test_the_only_pip_install_is_the_clone() -> None:
    """R38: one `pip install`, of the clone at COMMIT, that may not upgrade what Kaggle ships."""
    pip_calls = [
        call
        for tree in TREES.values()
        for call in _calls(tree, "subprocess.run")
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
    git = [ast.unparse(c.args[0]) for c in _calls(TREES["clone"], "subprocess.run")]
    assert "['git', 'clone', '--quiet', REPO_URL, str(CLONE)]" in git
    assert "['git', '-C', str(CLONE), 'checkout', '--quiet', '--detach', COMMIT]" in git


def test_one_gpu_is_pinned_before_torch_and_checked_before_the_checkpoints() -> None:
    """As in training (item 6.06): `CUDA_VISIBLE_DEVICES` is set once, to `VISIBLE_DEVICES`, before
    any cell imports torch or `trailcam`; `check_gpu` runs after the imports and before the
    checkpoint check."""
    pins = [
        (cell_id, node)
        for cell_id in CODE_IDS
        for node in ast.walk(TREES[cell_id])
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "os.environ['CUDA_VISIBLE_DEVICES']"
    ]
    assert ALL_CODE.count("CUDA_VISIBLE_DEVICES") == 1
    ((pin_cell, pin),) = pins
    assert ast.literal_eval(pin.value) == device.VISIBLE_DEVICES
    first_torch = min(
        i for i, cell_id in enumerate(CODE_IDS) if _imported(TREES[cell_id]) & {"torch", "trailcam"}
    )
    assert CODE_IDS.index(pin_cell) < first_torch
    assert _cells_calling("check_gpu") == ["gpu"]
    assert first_torch < CODE_IDS.index("gpu") < CODE_IDS.index("checkpoints")
    assert "print(check_gpu())" in CELLS["gpu"]


def test_trailcam_is_imported_once_per_kernel() -> None:
    """Review 6.02's guard: the first cell importing `trailcam` refuses a kernel that holds it."""
    first = next(c for c in CODE_IDS if "trailcam" in _imported(TREES[c]))
    assert first == "imports"
    guard_stmt = TREES[first].body[0]
    assert isinstance(guard_stmt, ast.If)
    assert ast.unparse(guard_stmt.test) == "'trailcam' in sys.modules"
    assert isinstance(guard_stmt.body[0], ast.Raise) and "Factory reset" in ast.unparse(guard_stmt)


def test_the_sha256_check_comes_before_the_cache_and_any_evaluation() -> None:
    """The cell that checks the ledger and hashes the checkpoints runs before the archive is
    unpacked and before the first call of an evaluation entry point."""
    (check,) = [c for c in _cells_calling("check_ledger") if c in _cells_calling("file_sha256")]
    assert check == "checkpoints"
    evaluating = [c for name in ENTRY_POINTS for c in _cells_calling(name)]
    assert sorted(set(evaluating), key=CODE_IDS.index) == ["evaluate", "slices"]
    first = min(CODE_IDS.index(c) for c in evaluating)
    assert CODE_IDS.index(check) < CODE_IDS.index("unpack") < CODE_IDS.index("cache") < first
    for name in ENTRY_POINTS:  # each entry point is called once, inside a try for SystemExit
        (call,) = [c for tree in TREES.values() for c in _calls(tree, name)]
        handlers = [
            n
            for tree in TREES.values()
            for n in ast.walk(tree)
            if isinstance(n, ast.Try) and call in [c for s in n.body for c in ast.walk(s)]
        ]
        assert [ast.unparse(h.handlers[0].type) for h in handlers] == ["SystemExit"]


def test_markdown_cites_the_data_and_says_how_to_run() -> None:
    """R06: licence and citation. The How-to-run steps of item 8.08, in order, ending at the
    files to copy into docs/figures/."""
    for needle in ("CDLA-Permissive 1.0", "Beery", "Recognition in Terra Incognita", "MIT"):
        assert needle in MARKDOWN
    steps = [
        "## How to run on Kaggle",
        "Put the checkpoints into one private dataset, once.",
        "Datasets > New Dataset, title `trailcam-checkpoints`",
        "keep the original versions",
        "File > Import Notebook",
        f"{PUBLIC_REPO}/blob/main/notebooks/kaggle_evaluate.ipynb",
        "Add Input > Notebook: the output of the packing notebook of item 6.08",
        "Add Input > Dataset: the private `trailcam-checkpoints`",
        "Keep exactly one parameter cell",
        "Accelerator GPU T4 x2",
        "Internet on",
        "Rehearse: run sections 1 to 5",
        "then stop the session",
        "Save Version > Save & Run All",
        'still shows "Starting...", cancel it and submit again',
        "download `eval.zip`",
        "A difference above 0.01 is printed as CHECK",
        "Copy these files from `eval.zip` into `docs/figures/`",
        "Commit this notebook with its outputs cleared",
    ]
    positions = [FLAT_MARKDOWN.find(step) for step in steps]
    assert -1 not in positions, [s for s, i in zip(steps, positions, strict=True) if i == -1]
    assert positions == sorted(positions)
    for needle in ("cis_test or trans_test", "R17", "R14", "R03"):
        assert needle in FLAT_MARKDOWN


def test_the_figures_to_copy_fit_the_guard_s_budget() -> None:
    """How to run, step 9: 11 named PNGs and four overlays, 15 in all as the step says; with the 5
    data charts of item 8.07 they stay within the guard's file budget for docs/figures/ with room
    for the final scoring's figures. Every target is a new name, none a raw photo's."""
    table = MARKDOWN.split("Copy these files from `eval.zip`", 1)[1].split("That is", 1)[0]
    targets = [
        name
        for line in table.splitlines()
        if line.strip().startswith("| `") or line.strip().startswith("| four")
        for name in re.findall(r"`([^`]+)`", line.split("|")[2])
    ]
    pngs = [t for t in targets if t.endswith(".png")]
    named = [t for t in pngs if "<" not in t]
    assert len(named) == len(set(named)) == 11
    assert pngs.count("gradcam_overlay_<nn>.png") == 1 and "four of" in table
    copied = len(named) + 4  # and four overlays
    assert copied == 15 and "That is 15 PNGs and one table" in FLAT_MARKDOWN
    budget, _ = guard.IMAGE_BUDGETS["docs/figures/"]
    charts = 5  # item 8.07's data charts, in docs/figures/ before item 8.06
    held = copied + charts
    assert held <= budget and f"holds {held} files, which leaves {budget - held}" in FLAT_MARKDOWN
    assert not any(t.endswith((".jpg", ".jpeg")) for t in targets)


def test_no_credentials() -> None:
    """R07: no token or key file is read or pasted."""
    lowered = ALL_CODE.lower()
    for needle in ("token", "kaggle.json", "secret", "password", "api_key"):
        assert needle not in lowered


# The checkpoint check, run as written on made-up files


def test_the_checkpoint_check_matches_every_ledger_row(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    names = _run(["checkpoints"], light)
    checkpoints = names["checkpoints"]
    assert isinstance(checkpoints, dict) and list(checkpoints) == list(RUN_IDS)
    assert names["pairs"] == {
        0: {"A": RUN_IDS[1], "B": RUN_IDS[2]},
        1: {"A": RUN_IDS[3], "B": RUN_IDS[4]},
    }
    assert names["photos"] == []
    out = capsys.readouterr().out
    assert "5 .pt files, 5 distinct, match the 5 ledger rows" in out
    assert "no photo under assets/ at this commit (item 8.05): no Grad-CAM overlays" in out
    assert ".pt:" not in out and "made-up" not in out


def test_a_checkpoint_attached_twice_is_one_checkpoint(light: dict[str, object]) -> None:
    root = light["CHECKPOINTS_ROOT"]
    assert isinstance(root, Path)
    (root / "copy").mkdir()
    shutil.copy(root / f"{RUN_IDS[0]}.pt", root / "copy" / "again.pt")
    checkpoints = _run(["checkpoints"], light)["checkpoints"]
    assert isinstance(checkpoints, dict) and len(checkpoints) == 5


def _refused(
    names: dict[str, object], error: type[Exception], capsys: pytest.CaptureFixture[str]
) -> str:
    """The checkpoint check refuses, and the evaluation cell after it never calls an entry point
    (recorders stand in for them); the message and everything printed."""
    calls: list[list[str]] = []
    recorder = SimpleNamespace(main=lambda argv: calls.append(argv) or 0)
    names.update(evaluate=recorder, attention=recorder, slices=recorder)
    with pytest.raises(error) as refused:
        _run(["checkpoints", "evaluate", "slices"], names)
    assert calls == []
    return f"{refused.value}\n{capsys.readouterr()}"


def test_an_unknown_checkpoint_is_refused_and_counted(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    root = light["CHECKPOINTS_ROOT"]
    assert isinstance(root, Path)
    (root / "mystery-run.pt").write_bytes(b"not a row of the ledger")
    message = _refused(light, RuntimeError, capsys)
    assert "1 match no row; 0 rows have no file (none)" in message
    assert "mystery" not in message


def test_a_missing_checkpoint_is_refused_by_run_id(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    root = light["CHECKPOINTS_ROOT"]
    assert isinstance(root, Path)
    (root / f"{RUN_IDS[3]}.pt").unlink()
    message = _refused(light, RuntimeError, capsys)
    assert f"0 match no row; 1 rows have no file ({RUN_IDS[3]})" in message


def test_no_checkpoint_under_the_root_lists_the_folders_holding_some(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """As the cache cell lists the `.tar` files: a wrong CHECKPOINTS_ROOT names the folders under
    /kaggle/input that hold `.pt` files, with counts."""
    root = light["CHECKPOINTS_ROOT"]
    assert isinstance(root, Path)
    light["CHECKPOINTS_ROOT"] = root.parent / "another-slug"
    message = _refused(light, FileNotFoundError, capsys)
    assert f"{root}': 5" in message
    shutil.rmtree(root)
    message = _refused(light, FileNotFoundError, capsys)
    assert "none (is the trailcam-checkpoints dataset attached?)" in message


def test_a_config_that_no_longer_matches_its_row_is_refused(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    config_file = light["CONFIG_FILE"]
    assert isinstance(config_file, Path)
    settings = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    settings["configs"]["resnet18_arm_b"]["lr_head"] = 5.0e-3
    config_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    message = _refused(light, RuntimeError, capsys)
    assert f"run {RUN_IDS[2]}: config resnet18_arm_b" in message
    assert "no longer matches its ledger row (config hash)" in message


def test_a_seed_without_both_arms_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    names = _light_names(tmp_path, RUN_IDS[:4])  # no arm B row of run seed 1
    message = _refused(names, RuntimeError, capsys)
    assert "run seeds [1] lack an arm A or an arm B row" in message


def test_an_overlay_run_outside_the_ledger_is_refused(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    light["ASSETS_OVERLAY_RUN"] = "resnet18_arm_b-seed7-20261016T000000Z"
    message = _refused(light, ValueError, capsys)
    assert "is not a run id of docs/ledger.md" in message


def test_assets_photos_are_found_and_checked(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """The photos directly under assets/, by the demo's suffixes, each checked by `assets_image`;
    a link out of the folder is refused before anything is evaluated."""
    assets = light["ASSETS"]
    assert isinstance(assets, Path)
    assets.mkdir()
    for name in ("b-photo.JPG", "a-photo.png", "notes.txt"):
        (assets / name).write_bytes(b"x")
    light["ASSETS_OVERLAY_RUN"] = RUN_IDS[2]
    names = _run(["checkpoints"], light)
    assert names["photos"] == [
        (assets / "a-photo.png").resolve(),
        (assets / "b-photo.JPG").resolve(),
    ]
    out = capsys.readouterr().out
    assert f"2 photos under assets/: overlays by run {RUN_IDS[2]}" in out
    assert "photo" not in out.replace("photos under", "")

    outside = assets.parent / "elsewhere.jpg"
    outside.write_bytes(b"x")
    (assets / "link.jpg").symlink_to(outside)
    _refused(light, ValueError, capsys)


# The evaluation cells, with recorders for the entry points


def _recorders() -> tuple[dict[str, list[list[str]]], dict[str, SimpleNamespace]]:
    calls: dict[str, list[list[str]]] = {"evaluate": [], "attention": [], "slices": []}

    def recorder(name: str) -> Callable[[list[str]], int]:
        def main(argv: list[str]) -> int:
            calls[name].append(list(argv))
            return 0

        return main

    return calls, {name: SimpleNamespace(main=recorder(name)) for name in calls}


@pytest.mark.parametrize(
    ("photo_count", "overlay_run"),
    [(0, None), (0, RUN_IDS[2]), (2, None), (2, RUN_IDS[2])],
    ids=["no-assets", "no-assets-run-set", "assets-no-run", "assets-and-run"],
)
def test_every_evaluation_call_passes_val_and_trans_val_only(
    light: dict[str, object], photo_count: int, overlay_run: str | None
) -> None:
    """R17, run as written: each call of the three entry points, parsed by its own command line,
    asks for val and trans_val and nothing else. Every checkpoint gets evaluate and attention,
    in ledger order; slices gets each run seed's arm A and arm B, never the small CNN; overlays go
    to the one run named, and only when assets/ holds photos (an empty assets/ still works)."""
    assets = light["ASSETS"]
    assert isinstance(assets, Path)
    assets.mkdir()
    for n in range(photo_count):
        (assets / f"photo{n}.jpg").write_bytes(b"x")
    light["ASSETS_OVERLAY_RUN"] = overlay_run
    calls, recorders = _recorders()
    names = _run(["checkpoints"], light)
    names.update(recorders, DATA_ROOT=Path("/unpacked/cct20_512"))
    _run(["evaluate", "slices"], names)

    parsers = {"evaluate": evaluate, "attention": attention, "slices": slices}
    parsed = {
        name: [parsers[name].build_parser().parse_args(argv) for argv in argvs]
        for name, argvs in calls.items()
    }
    for runs in parsed.values():
        for args in runs:
            assert args.split == list(CLI_SPLITS)
            assert getattr(args, "dry_run", False) is False
    eval_dir = names["EVAL_DIR"]
    assert [a.out for a in parsed["evaluate"]] == [eval_dir / run for run in RUN_IDS]
    assert [a.out for a in parsed["attention"]] == [eval_dir / run for run in RUN_IDS]
    assert [a.config_name for a in parsed["evaluate"]] == [name for name, _ in RUNS]
    assert [a.config_name for a in parsed["attention"]] == [name for name, _ in RUNS]
    assert [a.config for a in parsed["attention"]] == [names["CONFIG_FILE"]] * 5
    assert [a.config_file for a in parsed["evaluate"]] == [names["CONFIG_FILE"]] * 5
    files = names["checkpoints"]
    assert isinstance(files, dict)
    assert [a.checkpoint for a in parsed["evaluate"]] == list(files.values())
    assert [a.checkpoint for a in parsed["attention"]] == list(files.values())
    assert [a.out for a in parsed["slices"]] == [
        eval_dir / "slices-seed0",
        eval_dir / "slices-seed1",
    ]
    assert [a.checkpoint for a in parsed["slices"]] == [
        [["resnet18_arm_a", str(files[RUN_IDS[1]])], ["resnet18_arm_b", str(files[RUN_IDS[2]])]],
        [["resnet18_arm_a", str(files[RUN_IDS[3]])], ["resnet18_arm_b", str(files[RUN_IDS[4]])]],
    ]
    with_overlays = [run for run, a in zip(RUN_IDS, parsed["evaluate"], strict=True) if a.overlay]
    if photo_count and overlay_run:
        assert with_overlays == [overlay_run]
        (args,) = [a for a in parsed["evaluate"] if a.overlay]
        assert [path for path, _ in args.overlay] == names["photos"]
        assert args.assets == assets
    else:
        assert with_overlays == []
        assert all("--assets" not in argv for argv in calls["evaluate"])


@pytest.mark.parametrize("failing", ["evaluate", "attention", "slices"])
@pytest.mark.parametrize("outcome", [SystemExit(2), 1], ids=["refused", "exit-1"])
def test_a_refused_or_failing_entry_point_fails_its_cell(
    light: dict[str, object], failing: str, outcome: SystemExit | int
) -> None:
    """An argparse refusal (SystemExit) or a non-zero exit raises, so Save & Run All stops there."""
    names = _run(["checkpoints"], light)
    _, recorders = _recorders()

    def main(argv: list[str]) -> int:
        if isinstance(outcome, SystemExit):
            raise outcome
        return outcome

    recorders[failing] = SimpleNamespace(main=main)
    names.update(recorders, DATA_ROOT=Path("/unpacked/cct20_512"))
    with pytest.raises(RuntimeError, match=f"^{failing} (stopped|exited)"):
        _run(["evaluate", "slices"], names)


def test_an_eval_dir_holding_files_is_refused(light: dict[str, object]) -> None:
    """No output of two runs in one session is mixed: a non-empty EVAL_DIR is refused before any
    entry point is called."""
    names = _run(["checkpoints"], light)
    calls, recorders = _recorders()
    names.update(recorders, DATA_ROOT=Path("/unpacked/cct20_512"))
    eval_dir = names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    (eval_dir / "old").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="already holds files"):
        _run(["evaluate"], names)
    assert calls == {"evaluate": [], "attention": [], "slices": []}


# Every cell from the checkpoint check on, on a cache-shaped fixture


@pytest.fixture(scope="module")
def evaluated(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """A Kaggle session under a temporary root, and the notebook's cells from the checkpoint check
    to the zip run on it as written, with the real entry points. The inputs: the packing
    notebook's archive of a cache-shaped synthetic fixture; five random-weight checkpoints as
    train.py saves them, in the checkpoints dataset; a clone with a small-batch config, a ledger
    made for the checkpoints (every score 1.0000) and one assets/ photo named like a dataset
    image. `ASSETS_OVERLAY_RUN` is arm B of run seed 0. Returns the namespace and what was
    printed."""
    root = tmp_path_factory.mktemp("kaggle")
    names = _kaggle_names(root, ASSETS_OVERLAY_RUN=RUN_IDS[2])
    config_file = names["CONFIG_FILE"]
    assert isinstance(config_file, Path)
    _write_config(config_file)

    built = root / "built" / CACHE_TOP
    fixture = synthetic.write_synthetic_fixture(root / "fixture")
    built.mkdir(parents=True)
    fixture.image_root.rename(built / CACHE_IMAGES)
    Path(fixture.paths["sidecar"]).rename(built / DAY_NIGHT_SIDECAR)
    (root / "fixture").rename(built / CACHE_ANNOTATIONS)
    meta = load_metadata(built / CACHE_ANNOTATIONS, built / DAY_NIGHT_SIDECAR)
    split_hash = make_split(meta, 42).hash
    file_names = [r.file_name for records in meta.records.values() for r in records]
    archive = names["CACHE_ARCHIVE"]
    assets = names["ASSETS"]
    assert isinstance(archive, Path) and isinstance(assets, Path)
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as tar:
        tar.add(built, arcname=CACHE_TOP)
    assets.mkdir(parents=True)
    photo = assets / file_names[0]
    shutil.copy(built / CACHE_IMAGES / file_names[0], photo)
    shutil.rmtree(root / "built")

    checkpoints_root = names["CHECKPOINTS_ROOT"]
    assert isinstance(checkpoints_root, Path)
    checkpoints_root.mkdir(parents=True)
    files = {}
    for n, (run_id, (name, _)) in enumerate(zip(RUN_IDS, RUNS, strict=True)):
        torch.manual_seed(n)
        model_name = load_config(name, config_file).model
        state = build_model(model_name, pretrained=False).state_dict()
        saved = {"model": model_name, "classes": list(CLASSES), "state_dict": state}
        files[run_id] = checkpoints_root / f"{run_id}.pt"
        torch.save(saved, files[run_id])
    _write_ledger(names, files, split_hash)

    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        _run(["checkpoints", "unpack", "cache", "evaluate", "slices", "zip"], names)
    return SimpleNamespace(
        names=names,
        out=out.getvalue(),
        err=err.getvalue(),
        photo=photo,
        file_names=file_names,
        split_hash=split_hash,
    )


def _metrics_rows(path: Path) -> dict[str, tuple[str, str]]:
    """The accuracy and macro-F1 per split of an evaluate.py metrics table."""
    pattern = r"^\| (val|trans_val) \| (?:cis|trans) \| \d+ \| \d+ \| ([0-9.]+) \| ([0-9.]+) \|$"
    found = re.findall(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    return {split: (accuracy, f1) for split, accuracy, f1 in found}


def test_every_checkpoint_is_evaluated_and_every_seed_sliced(evaluated: SimpleNamespace) -> None:
    """The archive unpacks to LOCAL_CACHE and the cache and its split hash pass. Every run gets its
    metrics table and the six PNGs of val and trans_val, and its attention table and histogram;
    the overlay run alone gets the overlay of the one assets/ photo; each run seed gets its slice
    table and two feature maps, of the ResNet-18 arms only. Nothing else is written."""
    eval_dir = evaluated.names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    for run_id in RUN_IDS:
        found = sorted(p.name for p in (eval_dir / run_id).iterdir())
        expected = ["attention.md", attention.HISTOGRAM_NAME, evaluate.METRICS_NAME]
        expected += [f"{s}_{f}.png" for s in CLI_SPLITS for f in evaluate.FIGURES]
        expected += [evaluate.OVERLAY_NAME.format(1)] if run_id == RUN_IDS[2] else []
        assert found == sorted(expected), run_id
        assert set(_metrics_rows(eval_dir / run_id / evaluate.METRICS_NAME)) == set(CLI_SPLITS)
    for seed in (0, 1):
        folder = eval_dir / f"slices-seed{seed}"
        assert sorted(p.name for p in folder.iterdir()) == sorted(
            [slices.CAMERA_FIGURE, slices.SPECIES_FIGURE, slices.TABLE_NAME]
        )
        table = (folder / slices.TABLE_NAME).read_text(encoding="utf-8")
        assert "resnet18 arm A" in table and "resnet18 arm B" in table
        assert "small_cnn" not in table
    folders = sorted(p.name for p in eval_dir.iterdir())
    assert folders == sorted([*RUN_IDS, "slices-seed0", "slices-seed1"])
    assert f"split seed 42, split hash {evaluated.split_hash}: the ledger's" in evaluated.out
    assert "1 photos under assets/: overlays by run" in evaluated.out
    assert "and 1 --overlay photos of assets/" in evaluated.out
    assert evaluated.out.count("python -m trailcam.evaluate.evaluate ") == 5
    assert evaluated.out.count("python -m trailcam.evaluate.attention ") == 5
    assert evaluated.out.count("python -m trailcam.evaluate.slices ") == 2
    for table in eval_dir.rglob("*.md"):  # R17: no table scores a test split
        text = table.read_text(encoding="utf-8")
        assert "cis_test" not in text and "trans_test" not in text, table.name


@pytest.mark.parametrize(
    ("seeds", "hashes", "message"),
    [
        (["42"], ["f" * 64], "not the cache the runs were trained on"),
        (["42", "7"], ["f" * 64], "2 split seeds and 1 split hashes"),
    ],
    ids=["another-split-hash", "two-split-seeds"],
)
def test_a_cache_with_another_split_is_refused(
    evaluated: SimpleNamespace, seeds: list[str], hashes: list[str], message: str
) -> None:
    """R14, run as written on the unpacked fixture: the split drawn with the rows' one split seed
    must have their one split hash, or val is not the val the runs were selected on."""
    rows = [{"split seed": seed, "split hash": digest} for seed in seeds for digest in hashes]
    names = {**evaluated.names, "rows": rows}
    with pytest.raises(RuntimeError, match=message):
        _run(["cache"], names)


def test_the_zip_holds_every_output(evaluated: SimpleNamespace) -> None:
    eval_dir, work = evaluated.names["EVAL_DIR"], evaluated.names["WORK"]
    assert isinstance(eval_dir, Path) and isinstance(work, Path)
    files = sorted(p.relative_to(eval_dir).as_posix() for p in eval_dir.rglob("*") if p.is_file())
    with zipfile.ZipFile(work / "eval.zip") as archive:
        assert sorted(n for n in archive.namelist() if not n.endswith("/")) == files
    # Per run 6 PNGs, the histogram and the tables; one overlay; per seed 2 PNGs and a table.
    assert len(files) == 5 * 9 + 1 + 2 * 3
    assert f"eval.zip: {len(files)} files (40 PNG, 12 tables)" in evaluated.out


def test_nothing_printed_names_a_dataset_file_or_the_assets_photo(
    evaluated: SimpleNamespace,
) -> None:
    """R03: the printed command lines count the assets/ photos without naming them (a photo's name
    may be an image id), and no cell or entry point prints an image id, file name or sequence id
    of the cache."""
    text = evaluated.out + evaluated.err
    assert evaluated.photo.name not in text
    assert "syn-img-" not in text and "syn-seq-" not in text
    assert not [name for name in evaluated.file_names if name in text]


def _summary(
    evaluated: SimpleNamespace, rows: Mapping[str, Mapping[str, str]]
) -> dict[str, object]:
    names = {**evaluated.names, "rows_by_run": rows}
    return _run(["summary"], names)


def _rows_from_metrics(evaluated: SimpleNamespace) -> dict[str, dict[str, str]]:
    """Ledger-like rows holding the scores the metrics tables hold."""
    eval_dir = evaluated.names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    rows = {}
    for run_id in RUN_IDS:
        scores = _metrics_rows(eval_dir / run_id / evaluate.METRICS_NAME)
        rows[run_id] = {
            f"{split} {metric}": value
            for split, pair in scores.items()
            for metric, value in zip(("accuracy", "macro-F1"), pair, strict=True)
        }
    return rows


def test_the_summary_prints_scores_beside_the_ledger_s(
    evaluated: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ledger made for the fixture says 1.0000 everywhere and random weights score far below,
    so every split is flagged; the table shows run ids, splits, counts and scores only."""
    names = evaluated.names
    _run(["summary"], {**names})
    printed = capsys.readouterr()
    lines = [
        line for line in printed.out.splitlines() if line.startswith("| ") and "---" not in line
    ]
    assert lines[0].startswith("| run id | split | images |")
    run = r"[a-z0-9_]+-seed\d-\d{8}T\d{6}Z"
    score = r"[01]\.\d{4}"
    row = rf"^\| {run} \| (val|trans_val) \| \d+ \| {score} \| {score} \| 1\.0000 \| 1\.0000 \|$"
    assert len(lines[1:]) == 10 and all(re.match(row, line) for line in lines[1:])
    assert "CHECK: 10 of 10 splits differ from the ledger by more than 0.01" in printed.err


def test_the_summary_flags_only_a_score_beyond_the_tolerance(
    evaluated: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _rows_from_metrics(evaluated)
    _summary(evaluated, rows)
    printed = capsys.readouterr()
    assert "all 10 splits within 0.01 of the ledger's scores" in printed.out
    assert "CHECK" not in printed.err

    near = float(rows[RUN_IDS[3]]["val accuracy"]) + 0.009
    rows[RUN_IDS[3]]["val accuracy"] = f"{near:.4f}"
    far = float(rows[RUN_IDS[1]]["trans_val macro-F1"]) - 0.02
    rows[RUN_IDS[1]]["trans_val macro-F1"] = f"{far:.4f}"
    _summary(evaluated, rows)
    printed = capsys.readouterr()
    assert (
        f"CHECK: 1 of 10 splits differ from the ledger by more than 0.01: {RUN_IDS[1]} trans_val"
        in printed.err
    )


def test_the_summary_refuses_a_metrics_table_without_its_split_rows(
    evaluated: SimpleNamespace, tmp_path: Path
) -> None:
    eval_dir = evaluated.names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    for run_id in RUN_IDS:
        (tmp_path / run_id).mkdir()
        text = (eval_dir / run_id / evaluate.METRICS_NAME).read_text(encoding="utf-8")
        if run_id == RUN_IDS[0]:
            text = "\n".join(
                line for line in text.splitlines() if not line.startswith("| trans_val | trans |")
            )
        (tmp_path / run_id / evaluate.METRICS_NAME).write_text(text, encoding="utf-8")
    names = {**evaluated.names, "EVAL_DIR": tmp_path, "rows_by_run": _rows_from_metrics(evaluated)}
    with pytest.raises(RuntimeError, match="found 9 split rows in the metrics tables"):
        _run(["summary"], names)


# Item 13.05: the LEDGER parameter, phase 2's ledger and the pairs of the slices


def _phase2_names(
    root: Path, runs: Mapping[str, tuple[str, int]] = dict(zip(P2_RUN_IDS, P2_RUNS, strict=True))
) -> dict[str, object]:
    """`_light_names` for phase 2: `LEDGER = "docs/ledger-phase2.md"`, as the parameter cell
    derives LEDGER_PATH from it, and a phase-2 ledger of `runs` (run id: config and run seed)
    with made-up `.pt` files."""
    names = _kaggle_names(root)
    clone = names["CLONE"]
    assert isinstance(clone, Path)
    names.update(LEDGER=P2_LEDGER, LEDGER_PATH=clone / P2_LEDGER)
    config_file, checkpoints_root = names["CONFIG_FILE"], names["CHECKPOINTS_ROOT"]
    assert isinstance(config_file, Path) and isinstance(checkpoints_root, Path)
    _write_config(config_file)
    checkpoints_root.mkdir(parents=True)
    files = {}
    for run_id in runs:
        files[run_id] = checkpoints_root / f"{run_id}.pt"
        files[run_id].write_bytes(f"made-up checkpoint {run_id}".encode())
    _write_ledger(names, files, "0" * 64, runs=runs)
    return names


def test_the_check_cell_refuses_a_bad_ledger_or_slice_pairs(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`LEDGER` is one of the two ledgers, as in the training notebook; `SLICE_PAIRS` is None or
    a list of pairs of two different config names."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "unset by the test")
    names = _run(["parameters"], {})
    for good in (
        {"LEDGER": P2_LEDGER},
        {"SLICE_PAIRS": []},
        {"SLICE_PAIRS": [("p2_baseline", "p2_res320"), ("p2_head3", "p2_res320")]},
    ):
        _run(["check"], {**names, **good})
    assert f"LEDGER {P2_LEDGER}" in capsys.readouterr().out
    for bad in (
        {"LEDGER": "docs/ledger-phase3.md"},
        {"LEDGER": "ledger.md"},
        {"LEDGER": Path("docs/ledger.md")},
        {"SLICE_PAIRS": ("p2_baseline", "p2_res320")},
        {"SLICE_PAIRS": [["p2_baseline", "p2_res320"]]},
        {"SLICE_PAIRS": [("p2_baseline",)]},
        {"SLICE_PAIRS": [("p2_baseline", "p2_baseline")]},
        {"SLICE_PAIRS": [("p2_baseline", 0)]},
    ):
        with pytest.raises(ValueError, match="must be"):
            _run(["check"], {**names, **bad})


def test_the_ledger_parameter_picks_the_ledger_the_checkpoints_are_matched_to(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With `LEDGER = "docs/ledger-phase2.md"` the attached files are matched by SHA-256 to that
    ledger's rows; phase 1's ledger is not read. A phase-2 ledger holding phase-1 rows, or phase
    1's holding phase-2 rows, fails its check before anything is matched."""
    names = _phase2_names(tmp_path / "p2")
    checkpoints = _run(["checkpoints"], names)["checkpoints"]
    assert isinstance(checkpoints, dict) and list(checkpoints) == list(P2_RUN_IDS)
    assert "5 .pt files, 5 distinct, match the 5 ledger rows" in capsys.readouterr().out

    phase1 = {**names, "LEDGER": "docs/ledger.md", "LEDGER_PATH": tmp_path / "none" / "ledger.md"}
    message = _refused(phase1, RuntimeError, capsys)
    assert (
        "docs/ledger.md at commit" in message and "fails its check (1 problems, 0 rows)" in message
    )

    swapped = _light_names(tmp_path / "p1", RUN_IDS)  # phase 1's rows, under the phase-2 name
    phase1_ledger = swapped["LEDGER_PATH"]
    assert isinstance(phase1_ledger, Path)
    clone = swapped["CLONE"]
    assert isinstance(clone, Path)
    shutil.copy(phase1_ledger, clone / P2_LEDGER)
    swapped.update(LEDGER=P2_LEDGER, LEDGER_PATH=clone / P2_LEDGER)
    message = _refused(swapped, RuntimeError, capsys)
    assert f"{P2_LEDGER} at commit" in message and "fails its check" in message

    pasted = _phase2_names(tmp_path / "pasted")  # phase 2's rows, under docs/ledger.md
    clone = pasted["CLONE"]
    assert isinstance(clone, Path)
    shutil.copy(clone / P2_LEDGER, clone / "docs" / "ledger.md")
    pasted.update(LEDGER="docs/ledger.md", LEDGER_PATH=clone / "docs" / "ledger.md")
    message = _refused(pasted, RuntimeError, capsys)
    assert "docs/ledger.md at commit" in message and "fails its check" in message


def test_phase2_slices_pair_each_config_with_p2_baseline_at_the_same_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`SLICE_PAIRS = None` in phase 2: every other config beside `p2_baseline`, once per run
    seed both have, `p2_baseline` first. The evaluation and slices calls, parsed by their own
    command lines, ask for val and trans_val only and name each run's config."""
    names = _phase2_names(tmp_path)
    names = _run(["checkpoints"], names)
    base0, head0, res0, base1, res1 = P2_RUN_IDS
    assert names["slice_runs"] == {
        "slices-p2_baseline-p2_head3-seed0": [base0, head0],
        "slices-p2_baseline-p2_res320-seed0": [base0, res0],
        "slices-p2_baseline-p2_res320-seed1": [base1, res1],
    }
    assert names["pairs"] == {}  # phase 1's arm pairing is not used
    assert "slices of 3 pairs, same run seed: slices-p2_baseline-p2_head3-seed0" in (
        capsys.readouterr().out
    )

    calls, recorders = _recorders()
    names.update(recorders, DATA_ROOT=Path("/unpacked/cct20_512"))
    _run(["evaluate", "slices"], names)
    parsers = {"evaluate": evaluate, "attention": attention, "slices": slices}
    parsed = {
        name: [parsers[name].build_parser().parse_args(argv) for argv in argvs]
        for name, argvs in calls.items()
    }
    for runs in parsed.values():
        for args in runs:
            assert args.split == list(CLI_SPLITS)
    assert [a.config_name for a in parsed["evaluate"]] == [name for name, _ in P2_RUNS]
    assert [a.config_name for a in parsed["attention"]] == [name for name, _ in P2_RUNS]
    files = names["checkpoints"]
    assert isinstance(files, dict)
    eval_dir = names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    assert [a.out for a in parsed["slices"]] == [
        eval_dir / folder
        for folder in (
            "slices-p2_baseline-p2_head3-seed0",
            "slices-p2_baseline-p2_res320-seed0",
            "slices-p2_baseline-p2_res320-seed1",
        )
    ]
    assert [a.checkpoint for a in parsed["slices"]] == [
        [["p2_baseline", str(files[base0])], ["p2_head3", str(files[head0])]],
        [["p2_baseline", str(files[base0])], ["p2_res320", str(files[res0])]],
        [["p2_baseline", str(files[base1])], ["p2_res320", str(files[res1])]],
    ]


def test_named_slice_pairs_replace_the_default_in_either_phase(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pairs named, in their order, at every run seed both configs have; `[]` slices
    nothing. In phase 1 a named pair replaces arm A beside arm B."""
    names = _phase2_names(tmp_path / "p2")
    names["SLICE_PAIRS"] = [("p2_res320", "p2_head3")]
    _, head0, res0, _, _ = P2_RUN_IDS
    found = _run(["checkpoints"], names)["slice_runs"]
    assert found == {"slices-p2_res320-p2_head3-seed0": [res0, head0]}
    names["SLICE_PAIRS"] = []
    assert _run(["checkpoints"], names)["slice_runs"] == {}
    assert "slices of 0 pairs, same run seed: none" in capsys.readouterr().out

    light = _light_names(tmp_path / "p1", RUN_IDS)
    light["SLICE_PAIRS"] = [("resnet18_arm_b", "resnet18_arm_a")]
    names = _run(["checkpoints"], light)
    assert names["slice_runs"] == {
        "slices-resnet18_arm_b-resnet18_arm_a-seed0": [RUN_IDS[2], RUN_IDS[1]],
        "slices-resnet18_arm_b-resnet18_arm_a-seed1": [RUN_IDS[4], RUN_IDS[3]],
    }
    assert names["pairs"] == {}


def test_phase1_s_default_slices_are_unchanged(light: dict[str, object]) -> None:
    """`SLICE_PAIRS = None` with `docs/ledger.md`: arm A beside arm B per run seed, in folders
    `slices-seed<n>`, as before item 13.05."""
    names = _run(["checkpoints"], light)
    assert names["slice_runs"] == {
        "slices-seed0": [RUN_IDS[1], RUN_IDS[2]],
        "slices-seed1": [RUN_IDS[3], RUN_IDS[4]],
    }


@pytest.mark.parametrize(
    ("runs", "pairs", "error", "message"),
    [
        (P2_RUN_IDS, [("p2_baseline", "p2_r50")], ValueError, "p2_r50 has no row in"),
        (P2_RUN_IDS[1:3], None, ValueError, "p2_baseline has no row in"),
        (
            (P2_RUN_IDS[0], P2_RUN_IDS[1], P2_RUN_IDS[4]),
            [("p2_head3", "p2_res320")],
            RuntimeError,
            "p2_head3 and p2_res320 share no run seed in docs/ledger-phase2.md",
        ),
    ],
    ids=["named-config-absent", "no-baseline-row", "named-pair-shares-no-seed"],
)
def test_a_pair_that_cannot_be_sliced_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    runs: Sequence[str],
    pairs: list[tuple[str, str]] | None,
    error: type[Exception],
    message: str,
) -> None:
    by_id = dict(zip(P2_RUN_IDS, P2_RUNS, strict=True))
    names = _phase2_names(tmp_path, {run_id: by_id[run_id] for run_id in runs})
    names["SLICE_PAIRS"] = pairs
    assert message in _refused(names, error, capsys)


def test_a_default_pair_without_a_shared_seed_is_skipped_and_said(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    by_id = dict(zip(P2_RUN_IDS, P2_RUNS, strict=True))
    runs = {run_id: by_id[run_id] for run_id in (P2_RUN_IDS[0], P2_RUN_IDS[1], P2_RUN_IDS[4])}
    names = _run(["checkpoints"], _phase2_names(tmp_path, runs))
    assert names["slice_runs"] == {
        "slices-p2_baseline-p2_head3-seed0": [P2_RUN_IDS[0], P2_RUN_IDS[1]]
    }
    assert "p2_res320 shares no run seed with p2_baseline: not sliced" in capsys.readouterr().out


def test_two_runs_of_one_config_and_seed_are_refused_for_the_slices(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A re-run (`docs/phase2-plan.md` section 4: a `p2_baseline` seed re-run at a later commit)
    leaves two rows of one config and seed; the slices take one run per config and seed, so the
    pair is refused, naming both, before anything is evaluated."""
    again = "p2_baseline-seed0-20261020T000000Z"
    runs = {**dict(zip(P2_RUN_IDS, P2_RUNS, strict=True)), again: ("p2_baseline", 0)}
    names = _phase2_names(tmp_path, runs)
    message = _refused(names, RuntimeError, capsys)
    assert f"runs {P2_RUN_IDS[0]}, {again} are all p2_baseline of run seed 0" in message
    names["SLICE_PAIRS"] = [("p2_head3", "p2_res320")]  # a pair without p2_baseline still runs
    assert list(_run(["checkpoints"], names)["slice_runs"]) == ["slices-p2_head3-p2_res320-seed0"]


def test_a_named_pair_with_the_small_cnn_is_refused(
    light: dict[str, object], capsys: pytest.CaptureFixture[str]
) -> None:
    """Section 7: the small CNN's 25,088 features per image never reach the slices."""
    light["SLICE_PAIRS"] = [("small_cnn", "resnet18_arm_a")]
    assert "holds the small CNN" in _refused(light, ValueError, capsys)


def _clone_cell_names(root: Path, slices_text: str, ledger: str) -> dict[str, object]:
    """The clone cell's namespace over a fake clone at COMMIT: the modules the cell asks for,
    this checkout's but `trailcam/evaluate/slices.py`, which holds `slices_text`."""
    names = _run(["parameters"], {})
    names = {key: _rebase(value, root) for key, value in names.items()}
    clone = names["CLONE"]
    assert isinstance(clone, Path)
    for module in ("data/cache.py", "evaluate/evaluate.py", "evaluate/attention.py"):
        (clone / "trailcam" / module).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / "trailcam" / module, clone / "trailcam" / module)
    (clone / "trailcam" / "demo").mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "trailcam/demo/predict.py", clone / "trailcam/demo/predict.py")
    (clone / "trailcam/evaluate/slices.py").write_text(slices_text, encoding="utf-8")
    names["LEDGER"] = ledger
    return names


def test_a_phase2_ledger_needs_a_commit_with_item_13_05_s_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run as written with git replaced: with `LEDGER = "docs/ledger-phase2.md"` the clone cell
    refuses a COMMIT whose `slices.py` predates item 13.05 (no per-config labels, so no eval at
    each config's input size either), before the install; phase 1 still takes it. The default
    COMMIT, phase 1's, is such a commit."""
    commit = str(_parameters(TREES["parameters"])["COMMIT"])

    def git(argv: list[str], **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout=f"{commit}\n" if "rev-parse" in argv else "")

    monkeypatch.setattr(subprocess, "run", git)
    current = (ROOT / "trailcam/evaluate/slices.py").read_text(encoding="utf-8")
    assert "def model_label(" in current
    older = current.replace("def model_label(", "def other_label(")
    _run(["clone"], _clone_cell_names(tmp_path / "a", older, "docs/ledger.md"))
    with pytest.raises(RuntimeError, match="lacks trailcam/evaluate/slices.py"):
        _run(["clone"], _clone_cell_names(tmp_path / "b", older, P2_LEDGER))
    _run(["clone"], _clone_cell_names(tmp_path / "c", current, P2_LEDGER))
    assert capsys.readouterr().out.count(f"commit {commit}") == 2

    monkeypatch.undo()
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{commit}:trailcam/evaluate/slices.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "def model_label(" not in shown.stdout


@pytest.fixture(scope="module")
def evaluated_p2(tmp_path_factory: pytest.TempPathFactory) -> SimpleNamespace:
    """As `evaluated`, for phase 2: `LEDGER = "docs/ledger-phase2.md"`, three random-weight
    checkpoints saved as train.py saves them since item 13.02 (with their input size):
    `p2_baseline` and `p2_head3` at 224 and `p2_res320` at 320, run seed 0, and the notebook's
    cells from the checkpoint check to the zip run on the cache-shaped fixture. The side of every
    batch a loaded model is given is recorded."""
    root = tmp_path_factory.mktemp("kaggle-p2")
    names = _kaggle_names(root)
    clone = names["CLONE"]
    config_file = names["CONFIG_FILE"]
    assert isinstance(clone, Path) and isinstance(config_file, Path)
    names.update(LEDGER=P2_LEDGER, LEDGER_PATH=clone / P2_LEDGER)
    _write_config(config_file)

    built = root / "built" / CACHE_TOP
    fixture = synthetic.write_synthetic_fixture(root / "fixture")
    built.mkdir(parents=True)
    fixture.image_root.rename(built / CACHE_IMAGES)
    Path(fixture.paths["sidecar"]).rename(built / DAY_NIGHT_SIDECAR)
    (root / "fixture").rename(built / CACHE_ANNOTATIONS)
    meta = load_metadata(built / CACHE_ANNOTATIONS, built / DAY_NIGHT_SIDECAR)
    split_hash = make_split(meta, 42).hash
    archive = names["CACHE_ARCHIVE"]
    assert isinstance(archive, Path)
    archive.parent.mkdir(parents=True)
    with tarfile.open(archive, "w") as tar:
        tar.add(built, arcname=CACHE_TOP)
    shutil.rmtree(root / "built")

    runs = {run_id: P2_RUNS[n] for n, run_id in enumerate(P2_RUN_IDS[:3])}
    checkpoints_root = names["CHECKPOINTS_ROOT"]
    assert isinstance(checkpoints_root, Path)
    checkpoints_root.mkdir(parents=True)
    files = {}
    for n, (run_id, (name, _)) in enumerate(runs.items()):
        torch.manual_seed(n)
        config = load_config(name, config_file)
        state = build_model(config.model, pretrained=False).state_dict()
        saved = {"model": config.model, "classes": list(CLASSES), "state_dict": state}
        saved["input_size"] = config.input_size
        files[run_id] = checkpoints_root / f"{run_id}.pt"
        torch.save(saved, files[run_id])
    _write_ledger(names, files, split_hash, runs=runs)

    sides: list[tuple[int, int]] = []
    real = attention.build_model

    def spy(name: str, **kwargs: bool) -> torch.nn.Module:
        model = real(name, **kwargs)
        model.register_forward_pre_hook(
            lambda _m, inputs: sides.append(tuple(inputs[0].shape[-2:]))
        )
        return model

    out, err = io.StringIO(), io.StringIO()
    with pytest.MonkeyPatch.context() as patch, redirect_stdout(out), redirect_stderr(err):
        patch.setattr(attention, "build_model", spy)
        _run(["checkpoints", "unpack", "cache", "evaluate", "slices", "zip", "summary"], names)
    return SimpleNamespace(names=names, out=out.getvalue(), err=err.getvalue(), sides=sides)


def test_a_phase2_ledger_is_evaluated_each_run_at_its_input_size(
    evaluated_p2: SimpleNamespace,
) -> None:
    """Every phase-2 run gets its metrics, figures and attention outputs; `p2_res320`'s tables
    say 320 px and its batches are 320 x 320, the others' 224; each variant is sliced beside
    `p2_baseline`, labelled by config; the summary compares every run with its phase-2 row; no
    table scores a test split."""
    eval_dir = evaluated_p2.names["EVAL_DIR"]
    assert isinstance(eval_dir, Path)
    base, head, res = P2_RUN_IDS[:3]
    for run_id in (base, head, res):
        found = sorted(p.name for p in (eval_dir / run_id).iterdir())
        expected = ["attention.md", attention.HISTOGRAM_NAME, evaluate.METRICS_NAME]
        expected += [f"{s}_{f}.png" for s in CLI_SPLITS for f in evaluate.FIGURES]
        assert found == sorted(expected), run_id
        assert set(_metrics_rows(eval_dir / run_id / evaluate.METRICS_NAME)) == set(CLI_SPLITS)
    for run_id, size in ((base, ""), (head, ""), (res, ", input size 320 px")):
        name = run_id.split("-seed")[0]
        for table in (evaluate.METRICS_NAME, attention.TABLE_NAME):
            text = (eval_dir / run_id / table).read_text(encoding="utf-8")
            assert f"config {name}{size}, checkpoint sha256" in text, (run_id, table)
    assert "whole 320 x 320 input" in (eval_dir / res / attention.TABLE_NAME).read_text()
    assert set(evaluated_p2.sides) == {(224, 224), (320, 320)}
    folders = sorted(p.name for p in eval_dir.iterdir())
    pairs = ["slices-p2_baseline-p2_head3-seed0", "slices-p2_baseline-p2_res320-seed0"]
    assert folders == sorted([base, head, res, *pairs])
    for folder, other in zip(pairs, ("p2_head3", "p2_res320"), strict=True):
        table = (eval_dir / folder / slices.TABLE_NAME).read_text(encoding="utf-8")
        assert "## p2_baseline" in table and f"## {other}" in table
    assert "- p2_res320, input size 320 px: checkpoint sha256" in (
        eval_dir / pairs[1] / slices.TABLE_NAME
    ).read_text(encoding="utf-8")
    assert evaluated_p2.out.count("python -m trailcam.evaluate.slices ") == 2
    assert "CHECK: 6 of 6 splits differ from the ledger by more than 0.01" in evaluated_p2.err
    for table in eval_dir.rglob("*.md"):  # R17: no table scores a test split
        text = table.read_text(encoding="utf-8")
        assert "cis_test" not in text and "trans_test" not in text, table.name
