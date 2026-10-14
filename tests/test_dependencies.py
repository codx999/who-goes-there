"""Runtime dependency minimums and the package build (invariant R38; review 4.01, item 5.04).

The Kaggle notebook installs `trailcam` with pip, which ignores `[tool.uv.sources]`. A minimum
above Kaggle's preinstalled torch would make pip replace it with PyPI's default CUDA build, so the
minimums are the oldest versions `tests/test_dataset.py` was run on (2026-10-12: torch 2.3.0,
torchvision 0.18.0, Pillow 10.1.0, Python 3.12, NumPy 2.5), not the locked ones. timm's minimum is
the oldest version `tests/test_models.py` was run on with those (2026-10-13: timm 1.0.3, the first
1.0 release; Kaggle's image has 1.0.30). Raising one needs a reason, such as an API the code starts
to use.

The build tests run `uv build` the way the item does: the sdist first, then the wheel from it.
pip installs the wheel's files and its `Requires-Dist` lines, so those are what is checked.
"""

import subprocess
import tarfile
import tomllib
import zipfile
from email.parser import Parser
from pathlib import Path

import pytest

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


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    out = tmp_path_factory.mktemp("dist")
    subprocess.run(
        ["uv", "build", "--quiet", "--out-dir", str(out), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = out.glob("*.whl")
    (sdist,) = out.glob("*.tar.gz")
    return wheel, sdist


def _package_files() -> set[str]:
    return {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "trailcam").rglob("*.py")
        if "__pycache__" not in path.parts
    }


def test_the_wheel_holds_the_trailcam_package_alone(built: tuple[Path, Path]) -> None:
    wheel, _ = built
    names = zipfile.ZipFile(wheel).namelist()
    dist_info = "-".join(wheel.name.split("-")[:2]) + ".dist-info/"
    assert {n for n in names if not n.startswith(dist_info)} == _package_files()
    assert {n.removeprefix(dist_info) for n in names if n.startswith(dist_info)} == {
        "METADATA",
        "WHEEL",
        "RECORD",
        "licenses/LICENSE",
    }


def test_the_wheel_requires_the_runtime_dependencies_and_no_index(
    built: tuple[Path, Path],
) -> None:
    """pip ignores `[tool.uv.sources]`; the CPU index must not reach the metadata either."""
    wheel, _ = built
    archive = zipfile.ZipFile(wheel)
    (metadata_name,) = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
    text = archive.read(metadata_name).decode()
    metadata = Parser().parsestr(text)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert sorted(metadata.get_all("Requires-Dist") or []) == sorted(project["dependencies"])
    assert metadata["Requires-Python"] == project["requires-python"]
    assert "download.pytorch.org" not in text
    assert "@" not in "".join(metadata.get_all("Requires-Dist") or [])


def test_the_sdist_holds_the_package_readme_and_licence_only(built: tuple[Path, Path]) -> None:
    """An untracked file in a checkout (a data symlink, a scratch CSV) never enters the sdist."""
    _, sdist = built
    root = sdist.name.removesuffix(".tar.gz") + "/"
    names = {n.removeprefix(root) for n in tarfile.open(sdist).getnames()}
    expected = _package_files() | {"README.md", "LICENSE", "pyproject.toml", "PKG-INFO"}
    # hatchling adds the .gitignore it read the exclusions from
    assert names - {".gitignore"} == expected
