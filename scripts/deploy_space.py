"""Upload the Hugging Face Space (plan.md section 8; item 11.01; R02, R07, R35, R36).

The Space gets, in one commit, these files and nothing else (`ALLOWED`):

- `app.py`, `requirements.txt` and `README.md` from `space/`;
- `config.yaml`, which names the model `load_model` builds;
- the example photos directly under `assets/` (`example_photos` of the demo, so the Space shows
  the same examples as the local app);
- the `trailcam` package's modules, `trailcam/**/*.py` as git tracks them, at the same paths, so
  `import trailcam` works beside `app.py` and the package finds `config.yaml` and `assets/` at the
  Space's root as it finds them at the repo's root here;
- the weights, `--weights <path>`, as `model.pt`. The arm is chosen after the final scoring (open
  question 7), so no run is named here. The file must be a `train.py` checkpoint of the model
  `config.yaml` gives `DEFAULT_CONFIG`; anything else is refused.

Refused before anything is read for upload or any token is looked at: a file under `data/` (this
checkout's, the main checkout's, or `TRAILCAM_DATA_ROOT`), links followed; a file outside this
repository other than the weights; a path off the allowed list; and uncommitted or untracked
changes under `space/`, `trailcam/`, `config.yaml` or `assets/`, so the Space always matches a
commit (named in the Space's commit message). The weights never enter git: `space/model.pt` and
every `.pt` are ignored.

The token is read from the `HF_TOKEN` environment variable only, never from a file or the
command line (there is no `--token`), and is never printed, logged or written. `--dry-run` lists
the files and stops: it reads no token, builds no Hugging Face client and makes no network call.
(`huggingface_hub` is loaded in every run anyway, by gradio through the demo's imports; loading
it reads nothing and calls nothing.) Telemetry is off for the upload: the library's own switch
is set before the client is built, since it read the environment when gradio imported it.

    uv run python scripts/deploy_space.py --dry-run --weights models/<run>.pt
    uv run python scripts/deploy_space.py --space <user>/<space> --weights models/<run>.pt

The second needs `HF_TOKEN` exported in the shell beforehand (a write token, item 11.02) and the
Space created on Hugging Face. The same commit deletes every file under `assets/` and `trailcam/`
in the Space that the list no longer holds (`STALE_PREFIXES`), so a photo taken out of `assets/`
stops being shown; other files in the Space are left as they are. The dry run cannot see the
Space without a network call, so it says that, and lists nothing to delete.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from trailcam.demo.app import example_photos
from trailcam.demo.predict import DEFAULT_CONFIG, load_model

ROOT: Final = Path(__file__).resolve().parents[1]
TOKEN_VARIABLE: Final = "HF_TOKEN"
DATA_ROOT_VARIABLE: Final = "TRAILCAM_DATA_ROOT"
WEIGHTS_NAME: Final = "model.pt"
# Path in the Space -> path in the repo, for the fixed files.
FIXED_FILES: Final = {
    "app.py": "space/app.py",
    "requirements.txt": "space/requirements.txt",
    "README.md": "space/README.md",
    "config.yaml": "config.yaml",
}
# Every path the Space may receive; anything else is refused.
ALLOWED: Final = re.compile(
    r"app\.py|requirements\.txt|README\.md|config\.yaml|model\.pt"
    r"|assets/[^/]+\.(?i:jpg|jpeg|png)"  # any case, as `example_photos` and the guard read it
    r"|trailcam/(?:[A-Za-z_][A-Za-z0-9_]*/)*[A-Za-z_][A-Za-z0-9_]*\.py"
)
# What the Space is built from: must match the commit before an upload.
SOURCES: Final = ("space", "trailcam", "config.yaml", "assets")
# Space folders the deploy owns: a file there that is not in the plan is deleted on upload.
STALE_PREFIXES: Final = ("assets/", "trailcam/")
SPACE_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*")


class DeployError(Exception):
    """A refusal; the message names no file under `data/`."""


@dataclass(frozen=True)
class Upload:
    """One file: its path in the Space and the local file it comes from."""

    path_in_space: str
    source: Path


def git(root: Path, *args: str) -> str:
    """The output of `git -C root <args>`; a failure is a `DeployError`."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise DeployError(f"git {args[0]} failed in {root}") from error
    return done.stdout


def data_folders(root: Path) -> list[Path]:
    """Every `data/` the upload must stay out of, resolved: this checkout's, the main checkout's
    (a worktree reaches it by absolute path) and `TRAILCAM_DATA_ROOT` if set."""
    common = Path(git(root, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    folders = [root / "data", common.parent / "data"]
    if os.environ.get(DATA_ROOT_VARIABLE):
        folders.append(Path(os.environ[DATA_ROOT_VARIABLE]))
    return [folder.resolve() for folder in folders]


def check_upload(upload: Upload, root: Path, data: Sequence[Path]) -> None:
    """Refuse a path off the allowed list, a missing file, a file under `data/` (links
    followed) and, except the weights, a file outside the repository."""
    if not ALLOWED.fullmatch(upload.path_in_space):
        raise DeployError(f"{upload.path_in_space} is not on the Space's allowed list")
    resolved = upload.source.resolve()
    if any(resolved == folder or resolved.is_relative_to(folder) for folder in data):
        raise DeployError(f"the file for {upload.path_in_space} lies under data/; refused")
    if not resolved.is_file():
        raise DeployError(f"the file for {upload.path_in_space} does not exist")
    if upload.path_in_space != WEIGHTS_NAME and not resolved.is_relative_to(root.resolve()):
        raise DeployError(f"the file for {upload.path_in_space} lies outside the repository")


def check_committed(root: Path) -> None:
    """Refuse uncommitted or untracked files under the Space's sources (ignored ones, such as
    `space/model.pt`, are not counted)."""
    status = git(root, "status", "--porcelain", "-z", "--untracked-files=all", "--", *SOURCES)
    changed = [entry for entry in status.split("\0") if entry]
    if changed:
        raise DeployError(
            f"{len(changed)} uncommitted or untracked files under space/, trailcam/, config.yaml "
            "or assets/; commit them first (git status), so the Space matches a commit"
        )


def check_weights(weights: Path, config_path: Path) -> None:
    """Refuse weights that are not a `train.py` checkpoint of `DEFAULT_CONFIG`'s model."""
    if weights.suffix != ".pt":
        raise DeployError("--weights must be a .pt checkpoint written by trailcam.train.train")
    try:
        load_model(weights, DEFAULT_CONFIG, config_path)
    except Exception as error:
        raise DeployError(f"--weights is not a {DEFAULT_CONFIG} checkpoint: {error}") from error


def plan(weights: Path, root: Path = ROOT) -> list[Upload]:
    """The files to upload, sorted by their path in the Space, each checked; see the module
    docstring for what is refused."""
    check_committed(root)
    data = data_folders(root)
    tracked = set(git(root, "ls-files", "-z", "--", *SOURCES).split("\0")) - {""}
    uploads = [Upload(name, root / source) for name, source in FIXED_FILES.items()]
    uploads += [
        Upload(path, root / path)
        for path in sorted(tracked)
        if path.startswith("trailcam/") and path.endswith(".py")
    ]
    assets = (root / "assets").resolve()
    try:
        photos = example_photos(root / "assets")
    except (ValueError, FileNotFoundError) as error:
        raise DeployError(f"assets/: {error}") from error
    for photo in photos:
        path = f"assets/{photo.relative_to(assets).as_posix()}"
        if path not in tracked:
            raise DeployError(f"{path} is not tracked by git; commit it so the guard checks it")
        uploads.append(Upload(path, photo))
    uploads.append(Upload(WEIGHTS_NAME, weights))
    for upload in uploads:
        check_upload(upload, root, data)
    check_weights(weights, root / "config.yaml")
    if len({upload.path_in_space for upload in uploads}) != len(uploads):
        raise DeployError("two files would land on one path in the Space")
    return sorted(uploads, key=lambda upload: upload.path_in_space)


def stale_paths(space_files: Sequence[str], uploads: Sequence[Upload]) -> list[str]:
    """The Space's files under `STALE_PREFIXES` that `uploads` no longer holds, sorted."""
    planned = {upload.path_in_space for upload in uploads}
    return sorted(
        path for path in space_files if path.startswith(STALE_PREFIXES) and path not in planned
    )


def upload_to_space(
    uploads: Sequence[Upload], space: str, token: str, message: str
) -> tuple[str, list[str]]:
    """One commit to the Space that adds `uploads` and deletes its stale files; the commit's URL
    and the deleted paths. Telemetry is off: the library reads its switch at each request."""
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, constants

    constants.HF_HUB_DISABLE_TELEMETRY = True
    api = HfApi(token=token)
    stale = stale_paths(api.list_repo_files(space, repo_type="space"), uploads)
    operations: list[CommitOperationAdd | CommitOperationDelete] = [
        CommitOperationAdd(path_in_repo=upload.path_in_space, path_or_fileobj=str(upload.source))
        for upload in uploads
    ]
    operations += [CommitOperationDelete(path_in_repo=path) for path in stale]
    commit = api.create_commit(
        repo_id=space, repo_type="space", operations=operations, commit_message=message
    )
    return str(commit.commit_url), stale


def describe(upload: Upload) -> str:
    size = upload.source.stat().st_size
    if upload.path_in_space == WEIGHTS_NAME:
        return f"  {WEIGHTS_NAME:<40} {size / 1e6:8.1f} MB  <- {upload.source}"
    return f"  {upload.path_in_space:<40} {size / 1e3:8.1f} kB"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts/deploy_space.py",
        description="Upload the Space: app, requirements, README, config, examples, trailcam "
        "and the weights. The token comes from the HF_TOKEN environment variable only.",
        allow_abbrev=False,
    )
    parser.add_argument("--weights", type=Path, required=True, help="the chosen run's .pt")
    parser.add_argument("--space", help="the Space, <user>/<name>; not needed with --dry-run")
    parser.add_argument("--dry-run", action="store_true", help="list the files and stop")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.dry_run and args.space is None:
        parser.error("--space is required unless --dry-run is given")
    if args.space is not None and not SPACE_ID.fullmatch(args.space):
        parser.error("--space must be <user>/<name>")
    try:
        uploads = plan(args.weights, ROOT)
    except DeployError as error:
        print(f"deploy_space.py: refused: {error}", file=sys.stderr)
        return 1
    if args.dry_run:
        print(f"dry run: would upload {len(uploads)} files (no token read, no network call):")
        print("\n".join(describe(upload) for upload in uploads))
        print(
            "on upload, files in the Space under assets/ or trailcam/ that are not listed are "
            "deleted (the dry run does not look: that needs the network)"
        )
        return 0
    token = os.environ.get(TOKEN_VARIABLE, "").strip()
    if not token:
        print(f"deploy_space.py: {TOKEN_VARIABLE} is not set; export it first", file=sys.stderr)
        return 1
    commit = git(ROOT, "rev-parse", "HEAD").strip()
    url, stale = upload_to_space(uploads, args.space, token, f"Deploy who-goes-there {commit[:12]}")
    print(f"uploaded {len(uploads)} files to https://huggingface.co/spaces/{args.space}: {url}")
    print(f"deleted {len(stale)} stale files from the Space" + "".join(f"\n  {p}" for p in stale))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
