"""Runtime dependency minimums (invariant R38; review 4.01).

The Kaggle notebook installs `trailcam` with pip, which ignores `[tool.uv.sources]`. A minimum
above Kaggle's preinstalled torch would make pip replace it with PyPI's default CUDA build, so the
minimums are the oldest versions `tests/test_dataset.py` was run on (2026-10-12: torch 2.3.0,
torchvision 0.18.0, Pillow 10.1.0, Python 3.12, NumPy 2.5), not the locked ones. timm's minimum is
the oldest version `tests/test_models.py` was run on with those (2026-10-13: timm 1.0.3, the first
1.0 release; Kaggle's image has 1.0.30). Raising one needs a reason, such as an API the code starts
to use.
"""

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VERIFIED_MINIMUMS = {
    "pillow": ">=10.1",
    "timm": ">=1.0.3",
    "torch": ">=2.3",
    "torchvision": ">=0.18",
}


def _locked_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    versions: dict[str, set[str]] = {}
    for package in lock["package"]:
        versions.setdefault(package["name"], set()).add(package["version"].split("+")[0])
    return versions


def _specifiers() -> dict[str, str]:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    specifiers = {}
    for dependency in dependencies:
        name, _, rest = dependency.partition(">=")
        specifiers[name.strip()] = f">={rest.strip()}"
    return specifiers


def test_runtime_minimums_are_the_verified_ones() -> None:
    specifiers = _specifiers()
    for name, minimum in VERIFIED_MINIMUMS.items():
        assert specifiers.get(name) == minimum, name


def test_minimums_sit_below_the_locked_versions() -> None:
    """A minimum equal to the version locked today is the defect of review 4.01."""
    locked = _locked_versions()
    specifiers = _specifiers()
    for name in VERIFIED_MINIMUMS:
        floor = tuple(int(part) for part in specifiers[name].removeprefix(">=").split("."))
        for version in locked[name]:
            assert floor < tuple(int(part) for part in version.split(".")), name


def test_the_lock_holds_no_cuda_build() -> None:
    """timm depends on torch and torchvision too (item 5.01); off macOS both still come from the
    CPU-only index, PyPI's builds are macOS wheels only, and no CUDA runtime package is locked."""
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    names = {package["name"] for package in lock["package"]}
    assert not {n for n in names if n.startswith(("nvidia-", "triton", "cuda-"))}
    for package in lock["package"]:
        if package["name"] not in ("torch", "torchvision"):
            continue
        wheels = [wheel["url"].rsplit("/", 1)[-1] for wheel in package["wheels"]]
        if package["source"]["registry"] == "https://download.pytorch.org/whl/cpu":
            assert package["version"].endswith("+cpu"), package["name"]
        else:
            assert all("macosx" in wheel for wheel in wheels), package["name"]
