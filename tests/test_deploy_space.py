"""The Space's deploy script (item 11.01; R02, R03, R07, R35, R36).

Each test builds a small git repository in a temporary folder with the real `space/` files,
`config.yaml` and `.gitignore`, two trailcam modules, one example photo and an ignored `data/`.
The weights are random ResNet-18 and small-CNN checkpoints saved as `train.py` saves them.
huggingface_hub's `HfApi` is replaced by a fake and sockets refuse to connect: no test touches the
network, and no real token is used.
"""

import importlib.util
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import huggingface_hub
import pytest
import torch
from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from PIL import Image

from trailcam.data.classes import CLASSES
from trailcam.model.factory import build_model

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("deploy_space", ROOT / "scripts" / "deploy_space.py")
assert _spec is not None and _spec.loader is not None
deploy = importlib.util.module_from_spec(_spec)
sys.modules["deploy_space"] = deploy  # dataclasses look their module up there
_spec.loader.exec_module(deploy)

# Built at run time, so this file holds no string the guard's token pattern matches.
SENTINEL_TOKEN = "hf_" + "Q7" * 18
EXPECTED = {
    "app.py",
    "requirements.txt",
    "README.md",
    "config.yaml",
    "model.pt",
    "assets/a.jpg",
    "trailcam/__init__.py",
    "trailcam/demo/__init__.py",
}


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _commit(root: Path) -> None:
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "fixture")


def _checkpoint(path: Path, model: str) -> Path:
    torch.manual_seed(0)
    state = build_model(model, pretrained=False).state_dict()
    torch.save({"model": model, "classes": list(CLASSES), "state_dict": state}, path)
    return path


@pytest.fixture(scope="module")
def weights(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ResNet-18 checkpoint, the model of `DEFAULT_CONFIG`, outside any repository."""
    return _checkpoint(tmp_path_factory.mktemp("models") / "run.pt", "resnet18")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for name in ("space/app.py", "space/requirements.txt", "space/README.md", "config.yaml"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, root / name)
    shutil.copyfile(ROOT / ".gitignore", root / ".gitignore")
    (root / "trailcam" / "demo").mkdir(parents=True)
    (root / "trailcam" / "__init__.py").write_text("")
    (root / "trailcam" / "demo" / "__init__.py").write_text("")
    (root / "trailcam" / "notes.txt").write_text("not a module")
    (root / "assets").mkdir()
    Image.new("RGB", (16, 16), "green").save(root / "assets" / "a.jpg")
    (root / "assets" / "LOCATIONS.md").write_text("not a photo")
    (root / "data").mkdir()
    Image.new("RGB", (16, 16), "red").save(root / "data" / "secret_frame.jpg")  # ignored
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "test")
    _commit(root)
    return root


class FakeApi:
    """Stands in for `huggingface_hub.HfApi`: records the token and the commit, and lists
    `space_files` as the Space's files."""

    instances: list["FakeApi"] = []
    space_files: list[str] = []

    def __init__(self, token: str | None = None) -> None:
        self.token = token
        self.commits: list[dict[str, Any]] = []
        self.listed: list[tuple[str, str | None]] = []
        FakeApi.instances.append(self)

    def list_repo_files(self, repo_id: str, *, repo_type: str | None = None) -> list[str]:
        self.listed.append((repo_id, repo_type))
        return list(FakeApi.space_files)

    def create_commit(self, **kwargs: Any) -> Any:
        self.commits.append(kwargs)
        return type("Commit", (), {"commit_url": "https://huggingface.co/spaces/u/s/commit/1"})


@pytest.fixture
def fake_api(monkeypatch: pytest.MonkeyPatch) -> type[FakeApi]:
    FakeApi.instances = []
    FakeApi.space_files = []
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    # The library's switch, read at each request; set to "on" here and restored after the test.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_DISABLE_TELEMETRY", False)
    return FakeApi


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("network call")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def _paths(uploads: list[Any]) -> set[str]:
    return {upload.path_in_space for upload in uploads}


def test_the_plan_is_exactly_the_space_files_examples_package_and_weights(
    repo: Path, weights: Path
) -> None:
    uploads = deploy.plan(weights, repo)
    assert _paths(uploads) == EXPECTED
    sources = {upload.path_in_space: upload.source for upload in uploads}
    assert sources["app.py"] == repo / "space" / "app.py"
    assert sources["model.pt"] == weights
    in_repo = [upload for upload in uploads if upload.path_in_space != "model.pt"]
    assert all("data" not in upload.source.relative_to(repo).parts for upload in in_repo)


def test_dry_run_lists_the_files_without_a_token_or_the_network(
    repo: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    no_network: None,
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("HfApi built in a dry run")

    monkeypatch.setattr(huggingface_hub, "HfApi", refuse)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(deploy, "ROOT", repo)
    assert deploy.main(["--dry-run", "--weights", str(weights)]) == 0
    out = capsys.readouterr().out
    listed = {line.split()[0] for line in out.splitlines()[1:] if line.startswith("  ")}
    assert listed == EXPECTED
    assert "not listed are deleted" in out
    assert "data/" not in out and "secret_frame" not in out


def test_upload_is_one_space_commit_with_the_token_from_the_environment(
    repo: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
    fake_api: type[FakeApi],
) -> None:
    monkeypatch.setenv("HF_TOKEN", SENTINEL_TOKEN)
    monkeypatch.setattr(deploy, "ROOT", repo)
    assert deploy.main(["--space", "user/who-goes-there", "--weights", str(weights)]) == 0
    (api,) = fake_api.instances
    assert api.token == SENTINEL_TOKEN
    (commit,) = api.commits
    assert commit["repo_id"] == "user/who-goes-there" and commit["repo_type"] == "space"
    added = {op.path_in_repo for op in commit["operations"] if isinstance(op, CommitOperationAdd)}
    assert added == EXPECTED and len(commit["operations"]) == len(EXPECTED)
    assert api.listed == [("user/who-goes-there", "space")]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
    ).stdout
    assert head[:12] in commit["commit_message"]
    assert huggingface_hub.constants.HF_HUB_DISABLE_TELEMETRY is True
    captured = capsys.readouterr()
    piece = SENTINEL_TOKEN[3:9]  # any six characters of it: a partial print is a leak too
    assert piece not in captured.out + captured.err + caplog.text
    assert piece not in str(commit)


def test_upload_deletes_the_spaces_stale_examples_and_modules_only(
    repo: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_api: type[FakeApi],
) -> None:
    """R35: a photo taken out of `assets/` stops being shown; files outside the deploy's folders
    (the Space's `.gitattributes`, a file added on the Hub) are left alone."""
    fake_api.space_files = [
        ".gitattributes",
        "README.md",
        "notes.md",
        "assets/a.jpg",
        "assets/removed.jpg",
        "trailcam/__init__.py",
        "trailcam/old_module.py",
    ]
    monkeypatch.setenv("HF_TOKEN", SENTINEL_TOKEN)
    monkeypatch.setattr(deploy, "ROOT", repo)
    assert deploy.main(["--space", "user/s", "--weights", str(weights)]) == 0
    (commit,) = fake_api.instances[0].commits
    operations = commit["operations"]
    deleted = {op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)}
    assert deleted == {"assets/removed.jpg", "trailcam/old_module.py"}
    added = {op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)}
    assert added == EXPECTED
    out = capsys.readouterr().out
    assert "deleted 2 stale files" in out and "assets/removed.jpg" in out


def test_stale_paths_keep_planned_files_and_other_folders() -> None:
    uploads = [deploy.Upload("assets/a.jpg", Path("a")), deploy.Upload("app.py", Path("b"))]
    space = ["assets/a.jpg", "assets/b.png", "assetsx/c.jpg", "app.py", "old.py", "trailcam/x.py"]
    assert deploy.stale_paths(space, uploads) == ["assets/b.png", "trailcam/x.py"]


def test_an_upper_case_photo_suffix_is_uploaded(repo: Path, weights: Path) -> None:
    """The demo and the guard read suffixes in any case; so does the allowed list."""
    Image.new("RGB", (16, 16), "blue").save(repo / "assets" / "B.JPG", format="JPEG")
    _commit(repo)
    assert _paths(deploy.plan(weights, repo)) == EXPECTED | {"assets/B.JPG"}


def test_upload_without_hf_token_stops_before_huggingface_hub(
    repo: Path,
    weights: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    fake_api: type[FakeApi],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(deploy, "ROOT", repo)
    assert deploy.main(["--space", "user/s", "--weights", str(weights)]) == 1
    assert fake_api.instances == []
    assert "HF_TOKEN is not set" in capsys.readouterr().err


def test_the_token_has_no_command_line_option_or_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """R07: only the environment; a `--token` would land in the shell history."""
    with pytest.raises(SystemExit):
        deploy.build_parser().parse_args(["--weights", "w.pt", "--token", "x"])
    source = (ROOT / "scripts" / "deploy_space.py").read_text()
    for name in ("get_token", "HfFolder", "login(", "token.txt", "open(", "read_text", ".write"):
        assert name not in source, name


def test_upload_needs_a_space_id(weights: Path) -> None:
    with pytest.raises(SystemExit):
        deploy.main(["--weights", str(weights)])
    with pytest.raises(SystemExit):
        deploy.main(["--space", "no-owner", "--weights", str(weights)])


def test_weights_under_data_are_refused_before_they_are_loaded(
    repo: Path, weights: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise AssertionError("a data/ file was loaded")

    monkeypatch.setattr(deploy, "load_model", refuse)
    inside = repo / "data" / "run.pt"
    shutil.copyfile(weights, inside)
    with pytest.raises(deploy.DeployError, match="data/"):
        deploy.plan(inside, repo)
    link = repo.parent / "linked.pt"
    link.symlink_to(inside)
    with pytest.raises(deploy.DeployError, match="data/"):
        deploy.plan(link, repo)


def test_weights_under_the_data_root_variable_are_refused(
    repo: Path, weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The main checkout's data/ in a worktree, or wherever TRAILCAM_DATA_ROOT points."""
    elsewhere = tmp_path / "cct20"
    elsewhere.mkdir()
    shutil.copyfile(weights, elsewhere / "run.pt")
    monkeypatch.setenv("TRAILCAM_DATA_ROOT", str(elsewhere))
    with pytest.raises(deploy.DeployError, match="data/"):
        deploy.plan(elsewhere / "run.pt", repo)


def test_the_main_checkouts_data_is_refused_from_a_worktree(
    repo: Path, weights: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("TRAILCAM_DATA_ROOT", raising=False)
    worktree = tmp_path / "worktree"
    _git(repo, "worktree", "add", "-q", str(worktree))
    shutil.copyfile(weights, repo / "data" / "run.pt")
    with pytest.raises(deploy.DeployError, match="data/"):
        deploy.plan(repo / "data" / "run.pt", worktree)


@pytest.mark.parametrize("content", [b"not a checkpoint", None])
def test_weights_that_are_not_a_default_config_checkpoint_are_refused(
    repo: Path, tmp_path: Path, content: bytes | None
) -> None:
    path = tmp_path / "other.pt"
    if content is None:
        _checkpoint(path, "small_cnn")  # another model than the Space's config names
    else:
        path.write_bytes(content)
    with pytest.raises(deploy.DeployError, match="checkpoint"):
        deploy.plan(path, repo)


def test_weights_of_another_input_size_are_refused(repo: Path, tmp_path: Path) -> None:
    """Item 13.02, the review's low finding: a ResNet-18 checkpoint trained at 320 px (a
    `p2_res320` run) has the Space model's weights, but records its input size, so the deploy's
    load under the Space's 224 px config refuses it instead of serving it at 224."""
    path = tmp_path / "res320.pt"
    torch.manual_seed(0)
    state = build_model("resnet18", pretrained=False).state_dict()
    torch.save(
        {"model": "resnet18", "classes": list(CLASSES), "state_dict": state, "input_size": 320},
        path,
    )
    with pytest.raises(deploy.DeployError, match="trained at 320 px"):
        deploy.plan(path, repo)


def test_weights_must_be_a_pt_file(repo: Path, weights: Path, tmp_path: Path) -> None:
    other = tmp_path / "run.safetensors"
    shutil.copyfile(weights, other)
    with pytest.raises(deploy.DeployError, match=r"\.pt"):
        deploy.plan(other, repo)
    with pytest.raises(deploy.DeployError, match="does not exist"):
        deploy.plan(tmp_path / "missing.pt", repo)


def test_an_example_linking_into_data_is_refused(repo: Path, weights: Path) -> None:
    (repo / "assets" / "b.jpg").symlink_to(repo / "data" / "secret_frame.jpg")
    _commit(repo)
    with pytest.raises(deploy.DeployError) as refused:
        deploy.plan(weights, repo)
    assert "secret_frame" not in str(refused.value)


@pytest.mark.parametrize(
    "change",
    ["edit space/app.py", "untracked trailcam/extra.py", "untracked assets/c.jpg"],
)
def test_uncommitted_sources_are_refused(
    repo: Path, weights: Path, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Space always matches a commit; the refusal counts files, it names none."""
    if change == "edit space/app.py":
        with (repo / "space" / "app.py").open("a") as file:
            file.write("# changed\n")
    elif change == "untracked trailcam/extra.py":
        (repo / "trailcam" / "extra.py").write_text("")
    else:
        Image.new("RGB", (16, 16)).save(repo / "assets" / "c.jpg")
    with pytest.raises(deploy.DeployError, match="1 uncommitted or untracked") as refused:
        deploy.plan(weights, repo)
    assert "c.jpg" not in str(refused.value) and "extra.py" not in str(refused.value)


def test_ignored_files_such_as_space_model_pt_do_not_block(repo: Path, weights: Path) -> None:
    shutil.copyfile(weights, repo / "space" / "model.pt")
    assert _paths(deploy.plan(weights, repo)) == EXPECTED


@pytest.mark.parametrize(
    "path_in_space",
    [
        "data/secret_frame.jpg",
        "notes.txt",
        "space/app.py",
        "assets/sub/a.jpg",
        "assets/a.heic",
        "trailcam/notes.txt",
        "trailcam/../app.py",
        "model.pth",
        ".env",
    ],
)
def test_paths_off_the_allowed_list_are_refused(repo: Path, path_in_space: str) -> None:
    upload = deploy.Upload(path_in_space, repo / "space" / "app.py")
    with pytest.raises(deploy.DeployError, match="allowed list"):
        deploy.check_upload(upload, repo, [repo / "data"])


def test_a_file_outside_the_repository_is_refused_except_the_weights(
    repo: Path, weights: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "app.py"
    outside.write_text("")
    with pytest.raises(deploy.DeployError, match="outside the repository"):
        deploy.check_upload(deploy.Upload("app.py", outside), repo, [repo / "data"])
    deploy.check_upload(deploy.Upload("model.pt", weights), repo, [repo / "data"])


def test_every_real_source_is_on_the_allowed_list() -> None:
    """The repo's own package modules and fixed files pass the list a real deploy checks."""
    tracked = subprocess.run(
        ["git", "ls-files", "trailcam"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    modules = [path for path in tracked if path.endswith(".py")]
    assert "trailcam/demo/predict.py" in modules
    assert all(deploy.ALLOWED.fullmatch(path) for path in modules)
    assert all((ROOT / source).is_file() for source in deploy.FIXED_FILES.values())
