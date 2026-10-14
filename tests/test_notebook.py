"""notebooks/kaggle_train.ipynb, read as JSON and its code cells as Python (items 6.02 and 6.06).

The notebook runs only on Kaggle, so it is never executed here: these tests check its shape. Every
code cell parses as plain Python (no magics) and defines nothing, so the logic stays in `trailcam`
(R41 outputs cleared; R38 the one pip install, of the clone; R37 and R25 the real-run entry point
with no dry-run or recipe flags; R17 no test split; R16 an alarm fails the cell; R27 seeds only
through the config and `--run-seed`; R06 the licence and citation).
"""

import ast
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from trailcam.train import device, train
from trailcam.train.config import load_configs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
PUBLIC_REPO = "https://github.com/codx999/who-goes-there"

_spec = importlib.util.spec_from_file_location("guard_data", ROOT / "scripts" / "guard_data.py")
assert _spec is not None and _spec.loader is not None
guard = importlib.util.module_from_spec(_spec)
sys.modules["guard_data"] = guard
_spec.loader.exec_module(guard)

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CODE = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]
MARKDOWN = "\n".join("".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "markdown")
ALL_CODE = "\n".join(CODE)
TREES = [ast.parse(source) for source in CODE]


def _parameters() -> dict[str, object]:
    """The literal assignments of the first code cell."""
    values: dict[str, object] = {}
    for node in TREES[0].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                values[node.targets[0].id] = ast.unparse(node.value)
    return values


def _calls(attribute: str) -> list[ast.Call]:
    return [
        node
        for tree in TREES
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == attribute
    ]


def test_outputs_and_attachments_are_cleared() -> None:
    data = NOTEBOOK.read_bytes()
    assert guard.content_problems(data, "notebooks/kaggle_train.ipynb") == []
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
    for tree in TREES:
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("trailcam")
            ):
                module = importlib.import_module(node.module)
                for alias in node.names:
                    assert hasattr(module, alias.name), f"{node.module}.{alias.name}"
            if isinstance(node, ast.Attribute) and ast.unparse(node.value) == "train":
                assert hasattr(train, node.attr), f"train.{node.attr}"


def test_parameters() -> None:
    p = _parameters()
    assert set(p) >= {"CONFIG_NAME", "RUN_SEED", "COMMIT", "DATA_ROOT", "REPO_URL", "OUT_DIR"}
    assert isinstance(p["COMMIT"], str) and re.fullmatch(r"[0-9a-f]{40}", p["COMMIT"])
    assert p["REPO_URL"] == PUBLIC_REPO  # the public repo, not a fork (R38)
    assert p["RUN_SEED"] is None  # the config's run seed unless the user overrides it (R27)
    training = {name for name, c in load_configs().items() if not c.fixture}
    assert p["CONFIG_NAME"] in training
    # The check cell's list of config names is the training configs of config.yaml.
    (listed,) = [
        ast.literal_eval(n.comparators[0])
        for tree in TREES
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare) and ast.unparse(n.left) == "CONFIG_NAME"
    ]
    assert set(listed) == training


def test_the_only_pip_install_is_the_clone() -> None:
    """R38: one `pip install`, of the clone at COMMIT, that may not upgrade what Kaggle ships."""
    pip_calls = [call for call in _calls("subprocess.run") if "'pip'" in ast.unparse(call.args[0])]
    assert [ast.unparse(c.args[0]) for c in pip_calls] == [
        "[sys.executable, '-m', 'pip', 'install', '--quiet', str(CLONE)]"
    ]
    assert ALL_CODE.count('"pip"') == 1 and "%pip" not in ALL_CODE and "!pip" not in ALL_CODE
    for flag in ("git+", "--upgrade", "-U", "--force-reinstall", "--no-deps", "-e"):
        assert f'"{flag}' not in ALL_CODE
    assert 'KEPT = ("torch", "torchvision", "Pillow", "timm", "PyYAML")' in ALL_CODE
    assert "after_install != preinstalled" in ALL_CODE
    checkout = [ast.unparse(c.args[0]) for c in _calls("subprocess.run")]
    assert "['git', 'clone', '--quiet', REPO_URL, str(CLONE)]" in checkout
    assert "['git', '-C', str(CLONE), 'checkout', '--quiet', '--detach', COMMIT]" in checkout


def test_training_is_one_real_run_with_the_clone_paths() -> None:
    """R37 and R25: `train.main` once, config and ledger from the clone, the checkpoint under
    /kaggle/working, no dry-run or recipe flag; R27: the run seed only through `--run-seed`."""
    assert len(_calls("train.main")) == 1
    p = _parameters()
    assert p["WORK"] == "Path('/kaggle/working')" and p["OUT_DIR"] == "WORK / 'models'"
    assert 'LEDGER = CLONE / "docs" / "ledger.md"' in ALL_CODE
    assert '"--config", str(CLONE / "config.yaml")' in ALL_CODE
    assert '"--ledger", str(LEDGER)' in ALL_CODE
    assert '"--out-dir", str(OUT_DIR)' in ALL_CODE
    flags = set(re.findall(r'"(--[a-z-]+)"', ALL_CODE)) - {"--quiet", "--detach", "--porcelain"}
    flags -= {"--untracked-files=no"}
    assert flags == {
        "--config-name",
        "--config",
        "--data-root",
        "--out-dir",
        "--ledger",
        "--run-seed",
    }
    options = {o for action in train.build_parser()._actions for o in action.option_strings}
    assert flags <= options
    assert "split_seed" not in ALL_CODE and "seed_everything" not in ALL_CODE


def test_no_test_split_is_read() -> None:
    """R17: cis_test and trans_test are scored once, at the end, never in this notebook."""
    for name in ("cis_test", "trans_test", "evaluate("):
        assert name not in ALL_CODE


@pytest.mark.parametrize("needle", ["exit_code == train.ALARM_EXIT", "exit_code != 0"])
def test_a_failing_run_fails_its_cell(needle: str) -> None:
    """R16: an alarmed run (exit 3) or a refused one raises, so Save & Run All stops there."""
    (cell,) = [source for source in CODE if "train.main(" in source]
    block = cell.split(needle, 1)[1].split("\nif ", 1)[0]
    assert "raise RuntimeError" in block
    assert "except SystemExit" in cell


def test_markdown_cites_the_data_and_says_how_to_run() -> None:
    """R06: licence and citation; the How-to-run steps of item 6.02."""
    for needle in ("CDLA-Permissive 1.0", "Beery", "Recognition in Terra Incognita", "MIT"):
        assert needle in MARKDOWN
    assert "## How to run on Kaggle" in MARKDOWN
    for needle in ("Add Input", "Accelerator GPU", "Internet on", "Save & Run All", "ledger"):
        assert needle in MARKDOWN


def test_no_credentials() -> None:
    """R07: no token or key file is read or pasted."""
    lowered = ALL_CODE.lower()
    for needle in ("token", "kaggle.json", "secret", "password", "api_key"):
        assert needle not in lowered


def test_the_plan_s_t4_x2_is_named_and_one_gpu_is_pinned() -> None:
    """plan.md section 5 (item 6.06): GPU T4 x2, training on the first T4. The notebook names no
    P100 as the accelerator and no longer warns on two GPUs; `CUDA_VISIBLE_DEVICES` is set once,
    to `VISIBLE_DEVICES`, in a cell before any cell that imports torch or `trailcam`."""
    assert "Accelerator GPU T4 x2" in MARKDOWN and "training uses the first one" in MARKDOWN
    assert "30 hours a week" in MARKDOWN
    assert "GPU P100" not in ALL_CODE and "Accelerator GPU P100" not in MARKDOWN
    assert "device_count" not in ALL_CODE and "WARNING" not in ALL_CODE
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
    first_torch = min(i for i, tree in enumerate(TREES) if _imported(tree) & {"torch", "trailcam"})
    assert pin_cell < first_torch


def test_the_gpu_and_cpus_are_checked_before_the_cache_and_training() -> None:
    """Item 6.06: `check_gpu` (an architecture torch was not built for fails here, not at the
    first CUDA call of training) and the CPU count with `workers_warning` for the chosen config,
    after the stale-import guard and before the cache is loaded or `train.main` runs."""
    cells_calling = {
        name: [i for i, tree in enumerate(TREES) for _ in _calls_in(tree, name)]
        for name in ("check_gpu", "workers_warning", "load_metadata", "train.main")
    }
    assert all(len(found) == 1 for found in cells_calling.values()), cells_calling
    first = next(i for i, tree in enumerate(TREES) if "trailcam" in _imported(tree))
    gpu = cells_calling["check_gpu"][0]
    assert first <= gpu == cells_calling["workers_warning"][0]
    assert gpu < cells_calling["load_metadata"][0] < cells_calling["train.main"][0]
    cell = CODE[gpu]
    assert "print(check_gpu())" in cell
    assert 'load_config(CONFIG_NAME, CLONE / "config.yaml").num_workers' in cell
    assert "os.cpu_count()" in cell and "workers_warning(num_workers, os.cpu_count())" in cell


def _calls_in(tree: ast.Module, attribute: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func) == attribute
    ]


def test_trailcam_is_imported_once_per_kernel() -> None:
    """Review 6.02: Python keeps modules it has imported, so the first cell that imports
    `trailcam` refuses a kernel that already holds it, and the clone cell offers a fresh session
    only, never deleting the clone and running again in the same kernel."""
    first = next(i for i, tree in enumerate(TREES) if "trailcam" in _imported(tree))
    assert not any("trailcam" in _imported(tree) for tree in TREES[:first])
    guard_stmt = TREES[first].body[0]
    assert isinstance(guard_stmt, ast.If)
    assert ast.unparse(guard_stmt.test) == "'trailcam' in sys.modules"
    assert isinstance(guard_stmt.body[0], ast.Raise)
    assert "Factory reset" in ast.unparse(guard_stmt.body[0])
    (clone_cell,) = [source for source in CODE if "git" in source and "clone" in source]
    assert "Factory reset" in clone_cell and "delete" not in clone_cell.lower()


def _imported(tree: ast.Module) -> set[str]:
    """Top-level package names a cell imports."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names
