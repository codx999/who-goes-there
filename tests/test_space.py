"""The Hugging Face Space's files (item 11.01; R02, R05, R06, R20, R21, R23, R33 to R36, R38).

`space/app.py` is loaded with `load_model` replaced by a small CNN with random weights, so no
trained run, no `space/model.pt` and no network are needed, and with `ASSETS_DIR` pointed at a
temporary folder, missing or holding synthetic photos, so no test depends on what the repo's
`assets/` holds (item 8.05 fills it). The requirements file is checked against `uv export` of the
lock and against the imports `app.py` loads at startup; the README's header against the lock.
"""

import ast
import importlib.util
import inspect
import os
import shutil
import subprocess
import sys
import tomllib
import types
from collections.abc import Callable
from importlib.metadata import packages_distributions
from pathlib import Path
from typing import Any

import gradio as gr
import pytest
import torch
import yaml
from PIL import Image
from torch import nn

from tests.test_demo import (
    button_handler,
    example_names,
    handlers_of_our_own,
    other_handlers,
    record_examples,
)
from trailcam.data import synthetic
from trailcam.demo import predict
from trailcam.demo.app import classify
from trailcam.evaluate import gradcam
from trailcam.evaluate.gradcam import ASSETS_DIR
from trailcam.model.factory import build_model
from trailcam.train.config import CONFIG_PATH

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "space"
APP = SPACE / "app.py"
PINNED = {"torch", "torchvision", "gradio"}
SPACE_PROVIDED = {"spaces"}  # on Hugging Face only; imported with a fallback


EXAMPLES = 3  # synthetic photos in the `assets` fixture


@pytest.fixture(scope="module")
def fixture_photos(tmp_path_factory: pytest.TempPathFactory) -> list[Path]:
    root = synthetic.write_synthetic_fixture(tmp_path_factory.mktemp("fixture")).image_root
    return sorted(root.rglob("*.jpg"))


@pytest.fixture(scope="module")
def photo(fixture_photos: list[Path]) -> Image.Image:
    with Image.open(fixture_photos[0]) as image:
        return image.convert("RGB")


@pytest.fixture
def no_assets(tmp_path: Path) -> Path:
    """A missing `assets/`: the Space is built without examples."""
    return tmp_path / "assets"


@pytest.fixture
def assets(tmp_path: Path, fixture_photos: list[Path]) -> Path:
    """An `assets/` folder holding synthetic fixture photos, never the repo's demo photos."""
    folder = tmp_path / "assets"
    folder.mkdir()
    for index, source in enumerate(fixture_photos[:EXAMPLES]):
        shutil.copyfile(source, folder / f"example_{index}.jpg")
    return folder


class Loaded:
    """What the fake `load_model` was called with, and the model it returned."""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str, Path]] = []
        self.devices: list[torch.device] = []
        torch.manual_seed(0)
        self.model = build_model("small_cnn").eval()


def load_space(
    monkeypatch: pytest.MonkeyPatch, *, spaces: types.ModuleType | None, assets: Path
) -> Any:
    """`space/app.py` as a fresh module, with `spaces` installed or not (`None`), and its
    examples taken from `assets` instead of the repo's `assets/`."""
    loaded = Loaded()
    to = loaded.model.to

    def record_to(device: torch.device) -> nn.Module:
        loaded.devices.append(device)
        return to("cpu")  # this machine may have no CUDA; the device asked for is what counts

    monkeypatch.setattr(loaded.model, "to", record_to)

    def fake_load_model(checkpoint: Path, config_name: str, config_path: Path) -> nn.Module:
        loaded.calls.append((checkpoint, config_name, config_path))
        return loaded.model

    monkeypatch.setattr(predict, "load_model", fake_load_model)
    monkeypatch.setattr(gradcam, "ASSETS_DIR", assets)  # app.py imports it when it is loaded
    monkeypatch.setitem(sys.modules, "spaces", spaces)  # None: the import raises ImportError
    spec = importlib.util.spec_from_file_location("space_app_under_test", APP)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.loaded = loaded
    return module


def fake_spaces(wrapped: list[Callable[..., Any]]) -> types.ModuleType:
    """A `spaces` module whose `GPU` records the function it wraps."""
    module = types.ModuleType("spaces")

    def gpu(fn: Callable[..., Any]) -> Callable[..., Any]:
        wrapped.append(fn)

        def on_gpu(*args: Any) -> Any:
            return fn(*args)

        on_gpu.on_gpu = True  # type: ignore[attr-defined]
        return on_gpu

    module.GPU = gpu  # type: ignore[attr-defined]
    return module


def test_the_space_loads_space_model_pt_with_the_default_config_on_the_cpu(
    monkeypatch: pytest.MonkeyPatch, no_assets: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    space = load_space(monkeypatch, spaces=None, assets=no_assets)
    assert space.loaded.calls == [(SPACE / "model.pt", predict.DEFAULT_CONFIG, CONFIG_PATH)]
    assert space.loaded.devices == [torch.device("cpu")]
    assert isinstance(space.demo, gr.Blocks)
    assert space.demo.analytics_enabled is False


def test_the_model_goes_to_cuda_at_module_level_when_cuda_is_available(
    monkeypatch: pytest.MonkeyPatch, no_assets: Path
) -> None:
    """ZeroGPU: the model is placed on cuda at startup, not inside the GPU call."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    space = load_space(monkeypatch, spaces=None, assets=no_assets)
    assert space.loaded.devices == [torch.device("cuda")]


def test_without_spaces_the_handler_is_classify_on_the_space_model(
    monkeypatch: pytest.MonkeyPatch, photo: Image.Image, no_assets: Path
) -> None:
    space = load_space(monkeypatch, spaces=None, assets=no_assets)
    assert space.spaces is None
    assert example_names(space.demo) == []
    assert button_handler(space.demo) is space.classify_on_device
    assert len(space.demo.fns) == 1  # no handler of its own besides the button's
    scores, seconds, overlay = space.classify_on_device(photo, False)
    expected, _, _ = classify(space.loaded.model, photo, False)
    assert list(scores) == list(expected)
    assert len(scores) == predict.TOP_K and 0 < sum(scores.values()) <= 1 + 1e-6
    assert seconds.endswith("s (forward pass only)")
    assert overlay is None


def test_grad_cam_is_in_the_space_and_off_by_default(
    monkeypatch: pytest.MonkeyPatch, photo: Image.Image, no_assets: Path
) -> None:
    """Open question 5, closed 2026-10-16: included, off by default."""
    space = load_space(monkeypatch, spaces=None, assets=no_assets)
    boxes = [c for c in space.demo.get_config_file()["components"] if c["type"] == "checkbox"]
    assert [box["props"]["value"] for box in boxes] == [False]
    _, _, overlay = space.classify_on_device(photo, True)
    assert isinstance(overlay, Image.Image)


def test_with_spaces_the_button_runs_the_handler_wrapped_in_spaces_gpu(
    monkeypatch: pytest.MonkeyPatch, photo: Image.Image, no_assets: Path
) -> None:
    """The one GPU-dependent function, prediction and Grad-CAM, is wrapped; it takes the photo
    and the switch only, so a ZeroGPU call never carries the model."""
    wrapped: list[Callable[..., Any]] = []
    space = load_space(monkeypatch, spaces=fake_spaces(wrapped), assets=no_assets)
    assert len(wrapped) == 1
    assert list(inspect.signature(wrapped[0]).parameters) == ["photo", "show_gradcam"]
    handler = button_handler(space.demo)
    assert handler is space.classify_on_device and getattr(handler, "on_gpu", False)
    assert len(space.demo.fns) == 1  # no handler outside spaces.GPU
    _, _, overlay = handler(photo, True)
    assert isinstance(overlay, Image.Image)


@pytest.mark.parametrize("installed", [False, True], ids=["without_spaces", "with_spaces"])
def test_with_photos_in_assets_the_space_shows_them_and_the_button_still_runs_its_handler(
    monkeypatch: pytest.MonkeyPatch,
    photo: Image.Image,
    assets: Path,
    tmp_path: Path,
    installed: bool,
) -> None:
    """Item 8.05: once `assets/` holds the demo photos, Gradio adds the examples' loader to the
    app's handlers; the button still runs `classify_on_device` (wrapped in `spaces.GPU` where
    `spaces` is installed), every other handler is Gradio's, and the examples are the folder's
    photos. They are not cached, though Hugging Face switches caching on, and loading the Space
    writes nothing into the working folder (the review's findings, 2026-10-18)."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("GRADIO_CACHE_EXAMPLES", "true")
    made = record_examples(monkeypatch)
    wrapped: list[Callable[..., Any]] = []
    space = load_space(
        monkeypatch, spaces=fake_spaces(wrapped) if installed else None, assets=assets
    )
    assert example_names(space.demo) == [f"example_{index}.jpg" for index in range(EXAMPLES)]
    assert other_handlers(space.demo)  # the examples' loader is there too
    assert handlers_of_our_own(other_handlers(space.demo)) == []
    assert [examples.cache_examples for examples in made] == [False]
    assert os.listdir(cwd) == []
    handler = button_handler(space.demo)
    assert handler is space.classify_on_device
    assert getattr(handler, "on_gpu", False) is installed and len(wrapped) == int(installed)
    scores, _, overlay = handler(photo, True)
    assert len(scores) == predict.TOP_K and isinstance(overlay, Image.Image)


def test_spaces_is_imported_before_torch() -> None:
    """Hugging Face asks for `spaces` first: on ZeroGPU it sets up torch's CUDA emulation."""
    line = {
        alias.name: node.lineno
        for node in ast.walk(ast.parse(APP.read_text()))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert line["spaces"] < line["torch"]


def test_the_space_examples_are_the_assets_folder() -> None:
    """R35: the Space shows the photos of the `assets/` folder beside the package, never data/."""
    source = APP.read_text()
    assert "example_photos(ASSETS_DIR)" in source
    assert ASSETS_DIR == ROOT / "assets"
    assert "data/" not in source and "data_root" not in source.lower()


def test_the_space_has_no_prediction_code_of_its_own() -> None:
    """R5, R20, R21, R33, R34, R36: the Space imports the demo's `classify` (which calls
    `predict`) and `load_model`, and copies none of their code: no class list, transform, timer
    or softmax of its own."""
    tree = ast.parse(APP.read_text())
    imported = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {("trailcam.demo.app", "classify"), ("trailcam.demo.app", "build_app")} <= imported
    assert ("trailcam.demo.predict", "load_model") in imported
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    forbidden = {"CLASSES", "EvalTransform", "Normalize", "Resize", "softmax", "topk"}
    forbidden |= {"perf_counter", "time", "timed_forward", "inference_mode", "no_grad"}
    assert not names & forbidden
    modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    modules |= {module for module, _ in imported if module is not None}
    assert "time" not in modules and "torchvision" not in modules


def requirements() -> dict[str, str]:
    lines = (SPACE / "requirements.txt").read_text().splitlines()
    pins = [line for line in lines if line and not line.startswith(("#", "--"))]
    return dict(line.split("==") for line in pins)


def module_level_imports(path: Path) -> set[str]:
    """The modules `path` imports when it is loaded: module-level statements, including those in
    `try` and `if` blocks, but not under `if TYPE_CHECKING`, and not inside functions."""
    found: set[str] = set()

    def visit(statements: list[ast.stmt]) -> None:
        for node in statements:
            if isinstance(node, ast.Import):
                found.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                found.add(node.module)
                found.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Try):
                visit(node.body)
                for handler in node.handlers:
                    visit(handler.body)
                visit(node.orelse)
                visit(node.finalbody)
            elif isinstance(node, ast.If) and "TYPE_CHECKING" not in ast.unparse(node.test):
                visit(node.body)
                visit(node.orelse)

    visit(ast.parse(path.read_text()).body)
    return found


def startup_imports() -> set[str]:
    """The top-level non-trailcam packages imported when `space/app.py` starts, following every
    trailcam module (and its packages' `__init__`) it loads."""
    third_party: set[str] = set()
    seen: set[Path] = set()
    queue = [APP]
    while queue:
        path = queue.pop()
        if path in seen:
            continue
        seen.add(path)
        for name in module_level_imports(path):
            top = name.split(".")[0]
            if top != "trailcam":
                if top not in sys.stdlib_module_names and top != "__future__":
                    third_party.add(top)
                continue
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                base = ROOT.joinpath(*parts[:depth])
                for candidate in (base / "__init__.py", base.with_suffix(".py")):
                    if candidate.is_file():
                        queue.append(candidate)
    return third_party


def test_requirements_are_trimmed_to_what_the_app_imports_at_startup() -> None:
    """R36: one line per distribution the startup imports need, nothing else (no dev group,
    no timm: EfficientNet is not the Space's model)."""
    distributions = packages_distributions()
    needed = {
        distributions[top][0].lower()
        for top in startup_imports() - SPACE_PROVIDED
        if top in distributions
    }
    assert startup_imports() - SPACE_PROVIDED - set(distributions) == set()
    assert set(requirements()) == needed
    assert {"pytest", "mypy", "ruff", "timm", "huggingface-hub", "trailcam"}.isdisjoint(needed)


def test_requirements_pin_torch_torchvision_and_gradio_at_the_locked_cpu_versions() -> None:
    """R36, R38: every line is a `uv export --group demo` line of the lock, the Linux (CPU-only
    wheel) one where the platforms differ, with the CPU index line pip needs for `+cpu`."""
    pins = requirements()
    assert PINNED <= set(pins)
    exported = subprocess.run(
        ["uv", "export", "--no-dev", "--no-hashes", "--group", "demo", "--frozen"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    linux = {
        line.split(" ;")[0]
        for line in exported
        if "==" in line and (" ; " not in line or line.endswith("sys_platform != 'darwin'"))
    }
    assert {f"{name}=={version}" for name, version in pins.items()} <= linux
    assert pins["torch"].endswith("+cpu") and pins["torchvision"].endswith("+cpu")
    options = [
        line for line in (SPACE / "requirements.txt").read_text().splitlines() if "--" in line
    ]
    assert [line for line in options if not line.startswith("#")] == [
        "--extra-index-url https://download.pytorch.org/whl/cpu"
    ]


def space_header() -> dict[str, Any]:
    text = (SPACE / "README.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    header: dict[str, Any] = yaml.safe_load(text.split("---\n")[1])
    return header


def test_the_readme_header_is_a_gradio_space_on_the_locked_versions() -> None:
    header = space_header()
    assert header["sdk"] == "gradio"
    assert header["sdk_version"] == requirements()["gradio"]
    python = (ROOT / ".python-version").read_text().strip()
    requires = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"]
    assert header["python_version"] == python and requires == f">={python}"
    assert header["app_file"] == "app.py"
    assert header["suggested_hardware"] == "cpu-basic"
    assert header["license"] == "mit"
    assert len(header["short_description"]) <= 60  # the Hub refuses a longer card text


def test_the_readme_cites_the_data_and_its_licence() -> None:
    """R06: the Space shows dataset photos, so its card carries the citation and the licence."""
    body = (SPACE / "README.md").read_text(encoding="utf-8").split("---\n", 2)[2]
    assert "Beery" in body and "2018" in body and "Recognition in Terra Incognita" in body
    assert "CDLA-Permissive 1.0" in body
    assert "MIT" in body


def test_the_space_weights_path_is_ignored_by_git() -> None:
    """R02: `space/model.pt`, the local copy app.py loads, never enters git."""
    done = subprocess.run(
        ["git", "check-ignore", "space/model.pt"], cwd=ROOT, capture_output=True, text=True
    )
    assert done.returncode == 0 and done.stdout.strip() == "space/model.pt"
