"""notebooks/kaggle_train.ipynb, read as JSON and its code cells as Python (items 6.02, 6.06-6.08).

The notebook runs only on Kaggle, so it is never executed here: these tests check its shape, and
run its one unpacking cell on a tiny tar (item 6.08). Every code cell parses as plain Python (no
magics) and defines nothing, so the logic stays in `trailcam`
(R41 outputs cleared; R38 the one pip install, of the clone; R37 and R25 the real-run entry point
with no dry-run or recipe flags; R17 no test split; R16 an alarm fails the cell; R27 seeds only
through the config and `--run-seed`; R06 the licence and citation; item 13.02: the `LEDGER`
parameter, and a config of the other phase than `LEDGER`'s refused in the first check cell).
"""

import ast
import importlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from trailcam.data.cache import CACHE_TOP, archive_files, sidecar_folders, unpack_cache
from trailcam.data.metadata import DAY_NIGHT_SIDECAR
from trailcam.train import device, train
from trailcam.train.config import PHASE2_PREFIX, is_phase2, load_configs

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_train.ipynb"
PUBLIC_REPO = "https://github.com/codx999/who-goes-there"
# Item 6.07: the user's private cache dataset as Kaggle mounts it (2026-10-14/15 rehearsal), and
# the main commit that holds trailcam/train/device.py (6.06): the default COMMIT descends from it.
KAGGLE_DATA_ROOT = "Path('/kaggle/input/datasets/codx888/cct20-512/cct20_512')"
OLDEST_COMMIT = "17ca508d420cccf10f4ed83d8df849a1dde153cd"
# Item 6.09: the archive of item 6.08, attached as the user's packing notebook's output (made a
# dataset, Kaggle extracted it); verified 2026-10-15, one file of 3,213,506,560 bytes.
KAGGLE_ARCHIVE = "Path('/kaggle/input/notebooks/codx888/notebook2b5a4bf6ad/cct20_512.tar')"
PARAMETERS = {"CONFIG_NAME", "RUN_SEED", "LEDGER", "COMMIT", "CACHE_ARCHIVE", "DATA_ROOT"}
LEDGERS = ("docs/ledger.md", f"docs/{train.PHASE2_LEDGER_NAME}")  # phase 1, phase 2

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
    assert p["CONFIG_NAME"] == "small_cnn" and p["CONFIG_NAME"] in training  # 6.04 first
    assert p["LEDGER"] == "docs/ledger.md" == LEDGERS[0]  # phase 1's, unless a p2_ config runs
    assert ROOT / str(p["LEDGER"]) == train.LEDGER_PATH
    assert p["DATA_ROOT"] == KAGGLE_DATA_ROOT
    assert p["CACHE_ARCHIVE"] == KAGGLE_ARCHIVE
    assert p["LOCAL_CACHE"] == "Path('/tmp/trailcam-cache')"  # local disk, not /kaggle/working
    # The check cell's list of config names is the phase-1 training configs of config.yaml; the
    # phase-2 ones are matched by their prefix, so a new p2_ config needs no notebook edit.
    (listed,) = [
        ast.literal_eval(n.comparators[0])
        for tree in TREES
        for n in ast.walk(tree)
        if isinstance(n, ast.Compare) and ast.unparse(n.left) == "CONFIG_NAME"
    ]
    assert set(listed) == {name for name in training if not is_phase2(name)}
    phase2 = {name for name in training if is_phase2(name)}
    assert phase2 and all(re.fullmatch(r"p2_[a-z0-9_]+", name) for name in phase2)


def _check_cell() -> int:
    (cell,) = [i for i, source in enumerate(CODE) if "CONFIG_NAME must be" in source]
    return cell


def _run_check_cell(
    monkeypatch: pytest.MonkeyPatch, config_name: str, ledger: str
) -> dict[str, object]:
    """The first check cell run as written with `CONFIG_NAME` and `LEDGER`; the GPU pin it sets
    is undone after the test."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", device.VISIBLE_DEVICES)
    names: dict[str, object] = {
        **{key: value for key, value in _parameters().items() if key in PARAMETERS},
        "Path": Path,
        "CONFIG_NAME": config_name,
        "LEDGER": ledger,
    }
    exec(compile(TREES[_check_cell()], "check cell", "exec"), names)
    return names


@pytest.mark.parametrize(
    ("config_name", "ledger"),
    [("small_cnn", LEDGERS[0]), ("resnet18_arm_b", LEDGERS[0]), ("p2_head3", LEDGERS[1])],
)
def test_the_check_cell_takes_a_config_with_the_ledger_of_its_phase(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_name: str,
    ledger: str,
) -> None:
    _run_check_cell(monkeypatch, config_name, ledger)
    assert f"CONFIG_NAME {config_name}, RUN_SEED None, LEDGER {ledger}," in capsys.readouterr().out


@pytest.mark.parametrize(
    ("config_name", "ledger", "message"),
    [
        ("p2_head3", LEDGERS[0], "are of different phases"),
        ("p2_baseline", LEDGERS[0], "are of different phases"),
        ("p2_arm_c_p25", LEDGERS[0], "are of different phases"),
        ("resnet18_arm_a", LEDGERS[1], "are of different phases"),
        ("small_cnn", LEDGERS[1], "are of different phases"),
        ("small_cnn", "docs/test-ledger.md", "LEDGER must be"),
        ("p2_head3", "/kaggle/working/ledger-phase2.md", "LEDGER must be"),
        ("resnet18", LEDGERS[0], "CONFIG_NAME must be"),
        ("p2_Head3", LEDGERS[1], "CONFIG_NAME must be"),
    ],
)
def test_the_check_cell_refuses_a_config_of_the_other_phase_than_the_ledger(
    monkeypatch: pytest.MonkeyPatch, config_name: str, ledger: str, message: str
) -> None:
    """Item 13.02: a `p2_` config with docs/ledger.md, or a phase-1 config with the phase-2
    ledger, fails in the first check cell, before the clone, the install and training;
    `train.py` refuses it again (tests/test_train.py)."""
    with pytest.raises(ValueError, match=message):
        _run_check_cell(monkeypatch, config_name, ledger)


def test_the_phase_check_comes_first_and_names_the_packages_ledgers() -> None:
    """The check cell runs before the clone and `train.main`, and its ledger names and prefix
    are the package's: `train.py`'s phase-2 ledger name and `config.py`'s `p2_`."""
    (clone,) = [i for i, tree in enumerate(TREES) if "'git', 'clone'" in ast.unparse(tree)]
    (training,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "train.main")]
    assert _check_cell() < clone < training
    source = CODE[_check_cell()]
    assert f'LEDGER not in ("{LEDGERS[0]}", "{LEDGERS[1]}")' in source
    assert f'CONFIG_NAME.startswith("{PHASE2_PREFIX}") != (LEDGER == "{LEDGERS[1]}")' in source
    assert 'r"p2_[a-z0-9_]+"' in source and PHASE2_PREFIX == "p2_"


def test_the_default_commit_has_the_gpu_check() -> None:
    """Item 6.07: the 6.06 rehearsal ran the default COMMIT dc05dad, which lacks
    trailcam/train/device.py. The default now descends from the commit that added it (CI checks
    out the full history)."""
    commit = str(_parameters()["COMMIT"])
    git = ["git", "-C", str(ROOT)]
    known = subprocess.run([*git, "cat-file", "-e", f"{commit}^{{commit}}"], check=False)
    assert known.returncode == 0, f"COMMIT {commit} is not a commit of this repo"
    ancestor = subprocess.run(
        [*git, "merge-base", "--is-ancestor", OLDEST_COMMIT, commit], check=False
    )
    assert ancestor.returncode == 0, f"COMMIT {commit} predates {OLDEST_COMMIT}"


def test_exactly_one_parameter_cell() -> None:
    """Item 6.07: a stale second parameter cell overrode a hand edit on Kaggle. Only the first
    code cell assigns a parameter, and the check cell prints the values in effect. The one other
    assignment is the unpacking cell's `DATA_ROOT = unpacked.root` (item 6.08)."""
    assigning = [
        (i, ast.unparse(node))
        for i, tree in enumerate(TREES)
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and {ast.unparse(target) for target in node.targets} & PARAMETERS
    ]
    first = [i for i, _ in assigning if i == 0]
    later = [(i, line) for i, line in assigning if i != 0]
    assert len(first) == len(PARAMETERS)
    assert [line for _, line in later] == ["DATA_ROOT = unpacked.root"]
    assert _calls_in(TREES[later[0][0]], "unpack_cache")
    (check,) = [source for source in CODE if "CONFIG_NAME must be" in source]
    for name in PARAMETERS:
        assert f"{name} {{{name}}}" in check


def test_a_commit_before_the_archive_unpacking_is_refused_before_the_install() -> None:
    """Items 6.07 and 6.08: the clone cell refuses a commit whose trailcam/data/cache.py is
    missing or has no `unpack_cache`, which section 3 imports, before pip installs it."""
    (clone,) = [i for i, tree in enumerate(TREES) if "'git', 'clone'" in ast.unparse(tree)]
    (pip,) = [i for i, tree in enumerate(TREES) if "'pip', 'install'" in ast.unparse(tree)]
    assert clone < pip
    assert 'cache_module = CLONE / "trailcam" / "data" / "cache.py"' in CODE[clone]
    assert (
        'if not cache_module.is_file() or "def unpack_cache(" not in '
        'cache_module.read_text("utf-8"):'
    ) in CODE[clone]
    assert "predates item 6.08" in CODE[clone]
    assert "def unpack_cache(" in (ROOT / "trailcam" / "data" / "cache.py").read_text("utf-8")
    default = str(_parameters()["COMMIT"])  # written against a main before 6.08: refused
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"{default}:trailcam/data/cache.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "def unpack_cache(" not in shown


def _unpack_cell() -> int:
    (cell,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "unpack_cache")]
    return cell


def test_the_archive_is_unpacked_after_the_gpu_check_and_before_the_cache_check() -> None:
    """Item 6.08: the GPU check fails in seconds first; then the archive is unpacked to
    LOCAL_CACHE and `DATA_ROOT` names the copy, so the cache cell and training read local disk."""
    cell = _unpack_cell()
    (gpu,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "check_gpu")]
    (cache,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "load_metadata")]
    assert gpu < cell < cache
    (call,) = _calls_in(TREES[cell], "unpack_cache")
    assert ast.unparse(call) == "unpack_cache(CACHE_ARCHIVE, LOCAL_CACHE)"
    assert "archive_files(INPUT)" in CODE[cell]
    # Review 6.08: the loose dataset attached too is refused before anything is unpacked.
    assert CODE[cell].index("sidecar_folders(INPUT)") < CODE[cell].index("unpack_cache(")
    assert "needed = {" in CODE[cache] and "DATA_ROOT / CACHE_IMAGES" in CODE[cache]


def _run_unpack_cell(
    archive: Path | None, local: Path, data_root: Path, attached: Path | None = None
) -> dict[str, object]:
    """The unpacking cell run as written, with `attached` (default: an empty folder) as
    Kaggle's input folder."""
    if attached is None:
        attached = local.parent / "empty-input"
        attached.mkdir(exist_ok=True)
    names: dict[str, object] = {
        "Path": Path,
        "os": os,
        "sys": sys,
        "INPUT": attached,
        "DAY_NIGHT_SIDECAR": DAY_NIGHT_SIDECAR,
        "sidecar_folders": sidecar_folders,
        "CACHE_ARCHIVE": archive,
        "LOCAL_CACHE": local,
        "DATA_ROOT": data_root,
        "archive_files": archive_files,
        "unpack_cache": unpack_cache,
    }
    exec(compile(TREES[_unpack_cell()], "unpack cell", "exec"), names)
    return names


def test_the_unpack_cell_sets_data_root_or_keeps_the_loose_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The cell run as written: a tar becomes the local DATA_ROOT; None keeps DATA_ROOT; a wrong
    path is refused."""
    archive = tmp_path / "cct20_512.tar"
    with tarfile.open(archive, "w:") as tar:
        top = tarfile.TarInfo(CACHE_TOP)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)
        info = tarfile.TarInfo(f"{CACHE_TOP}/images/a.jpg")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"ab"))
    loose = tmp_path / "loose"
    names = _run_unpack_cell(archive, tmp_path / "local", loose)
    assert names["DATA_ROOT"] == tmp_path / "local" / CACHE_TOP
    assert (tmp_path / "local" / CACHE_TOP / "images" / "a.jpg").read_bytes() == b"ab"
    assert "unpacked 1 files" in capsys.readouterr().out

    assert _run_unpack_cell(None, tmp_path / "other", loose)["DATA_ROOT"] == loose
    assert not (tmp_path / "other").exists()

    with pytest.raises(FileNotFoundError, match="is not a file") as absent:
        _run_unpack_cell(tmp_path / "absent.tar", tmp_path / "third", loose)
    assert "is the output of the packing notebook attached?" in str(absent.value)  # not a dataset


def test_the_unpack_cell_refuses_the_loose_dataset_only_in_an_interactive_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review 6.08: with the loose cct20-512 dataset still attached next to the archive, a
    rehearsal (run type Interactive) is refused with a count, before unpacking. A committed run
    (Batch, or any other value) has already paid for its start, so it warns on stderr and
    unpacks (fix round). With `CACHE_ARCHIVE = None` the loose dataset is the one read."""
    attached = tmp_path / "input"
    mount = attached / "datasets" / "codx888" / "cct20-512" / "cct20_512"
    mount.mkdir(parents=True)
    (mount / DAY_NIGHT_SIDECAR).write_text("{}")
    archive = tmp_path / "cct20_512.tar"
    with tarfile.open(archive, "w:") as tar:
        top = tarfile.TarInfo(CACHE_TOP)
        top.type = tarfile.DIRTYPE
        tar.addfile(top)

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    with pytest.raises(RuntimeError, match="1 folder\\(s\\) under .* still attached") as refused:
        _run_unpack_cell(archive, tmp_path / "local", mount, attached)
    assert "Remove it from the inputs" in str(refused.value)
    assert not (tmp_path / "local").exists()

    for run_type, local in (("Batch", "batch"), (None, "unset")):
        if run_type is None:
            monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE")
        else:
            monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", run_type)
        capsys.readouterr()
        names = _run_unpack_cell(archive, tmp_path / local, mount, attached)
        assert names["DATA_ROOT"] == tmp_path / local / CACHE_TOP
        err = capsys.readouterr().err
        assert f"{run_type or 'unknown'} run, continuing" in err and "still attached" in err

    monkeypatch.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    assert _run_unpack_cell(None, tmp_path / "none", mount, attached)["DATA_ROOT"] == mount


def test_the_cache_cell_searches_any_depth_and_checks_every_image() -> None:
    """Item 6.07: a wrong DATA_ROOT lists the folders holding a day_night.json at any depth
    under /kaggle/input (Kaggle's datasets/<user>/<slug>/<folder> mount is four levels down), and
    `check_images` refuses an incomplete cache after the metadata is loaded, before training."""
    (cell,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "load_metadata")]
    source = CODE[cell]
    assert "sidecar_folders(INPUT)" in source and _parameters()["INPUT"] == "Path('/kaggle/input')"
    assert ".glob(" not in source and "*/*" not in source
    (check,) = _calls_in(TREES[cell], "check_images")
    assert ast.unparse(check) == "check_images(meta, needed['images'])"
    assert source.index("load_metadata(") < source.index("check_images(")
    (training,) = [i for i, tree in enumerate(TREES) if _calls_in(tree, "train.main")]
    assert cell < training


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
    assert "ledger_path = CLONE / LEDGER" in ALL_CODE
    assert '"--config", str(CLONE / "config.yaml")' in ALL_CODE
    assert '"--ledger", str(ledger_path)' in ALL_CODE
    (clone,) = [source for source in CODE if "git" in source and "clone" in source]
    assert 'not line.endswith(f" {LEDGER}")' in clone  # a rerun's own ledger row, nothing else
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


def test_how_to_run_says_import_attach_one_cell_rehearse_then_run() -> None:
    """Item 6.07: the steps of the user's Kaggle rehearsal, in order."""
    steps = [
        "File > Import Notebook",
        f"{PUBLIC_REPO}/blob/main/notebooks/kaggle_train.ipynb",
        "Add Input > Notebook: attach the output of the packing notebook of item 6.08",
        "Not a dataset: when a notebook's output becomes a\n   dataset, Kaggle extracts a `.tar`",
        "/kaggle/input/notebooks/<user>/<notebook>/<file>",
        "another account or packing notebook\n   changes it",
        "Remove the `cct20-512` and `cct20-512-tar` inputs if the notebook still has them",
        "`CACHE_ARCHIVE = None` reads the loose 6.03 dataset",
        "Keep exactly one parameter cell",
        "Edit only\n   `COMMIT` (and `CACHE_ARCHIVE` for another packing notebook)",
        "at or after the merge of item 6.08",
        "Accelerator GPU T4 x2",
        "Internet on",
        "Rehearse: run sections 1 to 3",
        "then stop the session",
        "Save Version > Save & Run All",
        "one line per epoch with the epoch's seconds and training images per second",
        "About 10 minutes after Save & Run All, open the version's Logs tab",
        'still shows "Starting...", cancel it and submit again',
    ]
    positions = [MARKDOWN.find(step) for step in steps]
    assert -1 not in positions, [s for s, i in zip(steps, positions, strict=True) if i == -1]
    assert positions == sorted(positions)


def test_the_2026_10_15_stall_is_not_put_down_to_the_input_copy() -> None:
    """Item 6.09: why the 6.04 GPU version sat at "Starting..." is not known (a CPU version with
    the same 57,864-file input logged its first line in about 5 minutes), so neither the notebook
    nor the docstrings of the cache and packing modules name the input copy as the cause."""
    texts = {
        "notebook": "\n".join("".join(c["source"]) for c in NB["cells"]),
        "cache.py": (ROOT / "trailcam" / "data" / "cache.py").read_text(encoding="utf-8"),
        "kaggle_pack_cache.py": (ROOT / "scripts" / "kaggle_pack_cache.py").read_text("utf-8"),
    }
    for name, text in texts.items():
        for claim in ("copies every input file before", "copies its 57,864", "Kaggle copied"):
            assert claim not in text, (name, claim)
    assert "why is not known" in MARKDOWN


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
