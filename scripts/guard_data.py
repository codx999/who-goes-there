"""Refuse data files, images, model weights and keys in git (CLAUDE.md ground rules: CCT20 images
and metadata, checkpoints and credentials never reach the repo).

    uv run python scripts/guard_data.py --staged     # pre-commit / pre-merge-commit hooks
    uv run python scripts/guard_data.py --pre-push   # pre-push hook: commits not on a remote yet
    uv run python scripts/guard_data.py --all        # check.sh and CI: every file in the index
    uv run python scripts/guard_data.py --history    # check.sh and CI: every commit from HEAD

Contents are read by blob ID from git (the index or the commits), never from the working tree.
Findings name the path and the rule only, never the matching text. Standard library only, so the
hooks can run it with `uv run --no-project`.

Demo sample images are allowed under `assets/` only (ALLOWED_IMAGE_SUFFIXES, ALLOWED_IMAGE_DIRS),
within a count and total-size budget that `--all` (check.sh, CI) enforces over the whole index.
Notebooks must have no image outputs, and a JSON file with more than MAX_METADATA_RECORDS image
records is treated as dataset metadata.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

BLOCKED_DIRS = ("data/", "models/", "outputs/")
IMAGE_SUFFIXES = frozenset({".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
ALLOWED_IMAGE_DIRS = ("assets/",)  # demo samples from trans-test locations, a few files only
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})  # mirrored in .gitignore
MAX_ASSET_IMAGES = 24
MAX_ASSET_BYTES = 12_000_000
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
MAX_METADATA_RECORDS = 100  # a JSON "images"/"annotations" list longer than this is a dataset dump
BLOCKED_SUFFIXES = (
    IMAGE_SUFFIXES
    | VIDEO_SUFFIXES
    | frozenset(
        {
            # model weights and tensors
            ".bin",
            ".ckpt",
            ".h5",
            ".hdf5",
            ".npy",
            ".npz",
            ".onnx",
            ".pt",
            ".pth",
            ".safetensors",
            # SQLite stores and their side files
            ".db",
            ".db-journal",
            ".db-shm",
            ".db-wal",
            ".sqlite",
            ".sqlite3",
            ".sqlite-journal",
            ".sqlite-shm",
            ".sqlite-wal",
            # tables and exports
            ".arrow",
            ".csv",
            ".feather",
            ".jsonl",
            ".ndjson",
            ".parquet",
            ".tsv",
            ".xls",
            ".xlsx",
            # pickled objects
            ".pickle",
            ".pkl",
            # archives and backups, which hide their contents from the checks below
            ".7z",
            ".bak",
            ".bz2",
            ".gz",
            ".rar",
            ".tar",
            ".tgz",
            ".xz",
            ".zip",
            ".zst",
            # env files with another name, e.g. prod.env
            ".env",
        }
    )
)
ALLOWED_ENV_FILES = frozenset({".env.example"})
SECRET_FILES = frozenset({"secrets.toml", "kaggle.json"})  # Streamlit and Kaggle credentials
MAX_BYTES = 5_000_000

# File signatures, for data files under an innocent name.
MAGIC = {
    b"SQLite format 3\x00": "SQLite",
    b"PAR1": "Parquet",
    b"\x1f\x8b": "gzip",
    b"PK\x03\x04": "zip (also torch .pt/.pth and .npz)",
    b"(\xb5/\xfd": "zstd",
    b"\xfd7zXZ\x00": "xz",
    b"7z\xbc\xaf'\x1c": "7z",
    b"\xff\xd8\xff": "JPEG",
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"GIF8": "GIF",
    b"\x89HDF\r\n\x1a\n": "HDF5",
    b"\x93NUMPY": "NumPy array",
    b"\x80\x02": "pickle",
    b"\x80\x03": "pickle",
    b"\x80\x04": "pickle",
    b"\x80\x05": "pickle",
}

SECRET_PATTERNS = {
    "Anthropic API key": re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    "Google API key": re.compile(rb"AIza[0-9A-Za-z_-]{35}"),
    "Hugging Face token": re.compile(rb"hf_[A-Za-z0-9]{30,}"),
    "GitHub token": re.compile(rb"gh[pousr]_[A-Za-z0-9]{36}"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    # A key/salt/token/secret/password name assigned a long value with letters and digits, in
    # .env, shell, Python, JSON/YAML or a notebook's JSON-escaped source: API_KEY=<hex>,
    # API_KEY="<hex>", api_key = '<hex>', "key": "<hex>", os.environ["API_KEY"] = ...
    "key, salt or token assignment": re.compile(
        rb"(?:key|salt|token|secret|password)\\?[\"']?\]?[ \t]*[:=][ \t]*\\?[\"']?"
        rb"(?=[A-Za-z0-9_+/=-]*[0-9])(?=[A-Za-z0-9_+/=-]*[A-Za-z])[A-Za-z0-9_+/=-]{16,}",
        re.IGNORECASE,
    ),
}

GITLINK = "160000"  # a submodule commit, not a file

ADVICE = {
    "staged": "Unstage with `git restore --staged <path>`; data belongs in data/, weights in "
    "models/, keys in .env; demo images only under assets/.",
    "all": "The file is in the index. If it is only staged, unstage it with `git restore --staged "
    "<path>`. If it is committed, `git rm --cached` alone leaves it in history: rewrite the "
    "commit (amend or rebase) before pushing.",
    "history": "These commits contain refused files. A later `git rm` leaves them in history: "
    "rewrite the commits (amend or rebase) before pushing. If they were pushed, rotate any "
    "exposed key and purge the history (git filter-repo), then force-push.",
}


@dataclass(frozen=True)
class Entry:
    path: str
    blob: str
    commit: str = ""


def _git(*args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], input=stdin, check=True, capture_output=True).stdout


def _decode(path: bytes) -> str:
    return path.decode("utf-8", errors="surrogateescape")


def _parse_raw(out: bytes, commit: str = "") -> list[Entry]:
    """Parse `--raw -z --no-renames` output: ":<mode> <mode> <sha> <sha> <status>\\0<path>\\0"."""
    parts = out.split(b"\0")
    entries: list[Entry] = []
    for meta, path in zip(parts[0:-1:2], parts[1::2], strict=True):
        _, new_mode, _, new_blob, _ = meta.decode().lstrip(":").split()
        if new_mode != GITLINK:
            entries.append(Entry(_decode(path), new_blob, commit))
    return entries


def staged_entries() -> list[Entry]:
    """Files added or changed in the index, compared with HEAD."""
    return _parse_raw(
        _git("diff", "--cached", "--raw", "-z", "--no-abbrev", "--no-renames", "--diff-filter=ACMT")
    )


def index_entries() -> list[Entry]:
    """Every file in the index."""
    entries: list[Entry] = []
    for record in _git("ls-files", "-s", "-z").split(b"\0"):
        if record:
            meta, path = record.split(b"\t", 1)
            mode, blob, _ = meta.decode().split()
            if mode != GITLINK:
                entries.append(Entry(_decode(path), blob))
    return entries


def commit_entries(*revs: str) -> list[Entry]:
    """Files added or changed by each commit `git rev-list <revs>` lists (merges against each
    parent, so a file a merge adds is seen too)."""
    entries: list[Entry] = []
    for commit in _git("rev-list", *revs).decode().split():
        out = _git(
            "diff-tree", "-r", "-z", "-m", "--root", "--no-commit-id", "--raw", "--no-abbrev",
            "--no-renames", "--diff-filter=ACMT", commit,
        )  # fmt: skip
        entries.extend(_parse_raw(out, commit[:12]))
    return list(dict.fromkeys(entries))


def pre_push_revs(lines: list[str]) -> list[str]:
    """rev-list arguments for the refs a pre-push hook gets on stdin: the pushed commits that no
    remote-tracking branch has yet."""
    tips = []
    for line in lines:
        fields = line.split()  # <local ref> <local sha> <remote ref> <remote sha>
        if len(fields) == 4 and set(fields[1]) != {"0"}:  # all zeros: a branch deletion
            tips.append(fields[1])
    return [*tips, "--not", "--remotes"] if tips else []


def read_blobs(blobs: list[str]) -> dict[str, bytes]:
    unique = list(dict.fromkeys(blobs))
    if not unique:
        return {}
    out = _git("cat-file", "--batch", stdin="".join(f"{b}\n" for b in unique).encode())
    contents, pos = {}, 0
    for blob in unique:
        end = out.index(b"\n", pos)
        header = out[pos:end].split()
        if len(header) != 3 or header[1] != b"blob":
            raise RuntimeError(f"guard_data: cannot read blob {blob}")
        size = int(header[2])
        contents[blob] = out[end + 1 : end + 1 + size]
        pos = end + 1 + size + 1
    return contents


def is_allowed_image(path: str) -> bool:
    """A demo sample image under assets/: a web image suffix only, no other blocked suffix."""
    lower = path.lower()
    suffixes = PurePosixPath(lower).suffixes
    return (
        lower.startswith(ALLOWED_IMAGE_DIRS)
        and bool(suffixes)
        and suffixes[-1] in ALLOWED_IMAGE_SUFFIXES
        and not any(s in BLOCKED_SUFFIXES - ALLOWED_IMAGE_SUFFIXES for s in suffixes)
    )


def _load_json(data: bytes) -> Any:
    try:
        return json.loads(data)
    except ValueError:  # includes UnicodeDecodeError
        return None


def notebook_problems(data: bytes) -> list[str]:
    """Cell outputs carrying image data (a rendered dataset sample ends up as base64 in git)."""
    nb = _load_json(data)
    if not isinstance(nb, dict):
        return []
    n_images = 0
    for cell in nb.get("cells", []):
        if not isinstance(cell, dict):
            continue
        for out in cell.get("outputs", []):
            mime = out.get("data", {}) if isinstance(out, dict) else {}
            if isinstance(mime, dict) and any(str(k).startswith("image/") for k in mime):
                n_images += 1
    return [f"notebook with {n_images} image outputs (clear outputs first)"] if n_images else []


def metadata_problems(data: bytes) -> list[str]:
    """A COCO-style dump: more than MAX_METADATA_RECORDS image or annotation records."""
    obj = _load_json(data)
    if not isinstance(obj, dict):
        return []
    for key in ("images", "annotations"):
        records = obj.get(key)
        if isinstance(records, list) and len(records) > MAX_METADATA_RECORDS:
            return [f"dataset metadata ({len(records)} {key} records)"]
    return []


def path_problems(path: str) -> list[str]:
    lower = path.lower()
    pure = PurePosixPath(lower)
    problems = []
    if lower.startswith(BLOCKED_DIRS):
        problems.append("file under data/, models/ or outputs/")
    blocked = [s for s in pure.suffixes if s in BLOCKED_SUFFIXES]
    if blocked and not is_allowed_image(path):
        problems.append(f"data, image, weights or archive file type ({''.join(blocked)})")
    name = pure.name
    if (name in (".env", ".envrc") or name.startswith(".env.")) and name not in ALLOWED_ENV_FILES:
        problems.append("env file")
    if name in SECRET_FILES:
        problems.append("credentials file")
    return problems


def content_problems(data: bytes, path: str = "") -> list[str]:
    problems = [f"{kind} file" for magic, kind in MAGIC.items() if data.startswith(magic)]
    if data[4:8] == b"ftyp":
        problems.append("MP4/MOV video file")
    if is_allowed_image(path):
        problems = [p for p in problems if p not in ("JPEG file", "PNG file")]
    lower = path.lower()
    if lower.endswith(".ipynb"):
        problems += notebook_problems(data)
    elif lower.endswith(".json"):
        problems += metadata_problems(data)
    problems += [rule for rule, pattern in SECRET_PATTERNS.items() if pattern.search(data)]
    if len(data) > MAX_BYTES:
        problems.append(f"larger than {MAX_BYTES // 1_000_000} MB")
    return problems


def check(entries: list[Entry]) -> list[tuple[Entry, str]]:
    contents = read_blobs([e.blob for e in entries])
    findings: list[tuple[Entry, str]] = []
    for entry in entries:
        problems = path_problems(entry.path) + content_problems(contents[entry.blob], entry.path)
        findings.extend((entry, problem) for problem in problems)
    assets = {e.path: e for e in entries if is_allowed_image(e.path)}
    if len(assets) > MAX_ASSET_IMAGES:
        findings.append(
            (Entry("assets/", ""), f"{len(assets)} images under assets/ (cap {MAX_ASSET_IMAGES})")
        )
    total = sum(len(contents[e.blob]) for e in assets.values())
    if total > MAX_ASSET_BYTES:
        findings.append(
            (
                Entry("assets/", ""),
                f"assets/ images total {total:,} bytes (cap {MAX_ASSET_BYTES:,})",
            )
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refuse data files, images, weights and keys.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--staged", action="store_true", help="files staged for commit")
    mode.add_argument("--pre-push", action="store_true", help="pushed commits (hook stdin)")
    mode.add_argument("--all", action="store_true", help="every file in the index")
    mode.add_argument("--history", action="store_true", help="every commit reachable from HEAD")
    args = parser.parse_args(argv)

    if args.staged:
        entries, advice = staged_entries(), ADVICE["staged"]
    elif args.all:
        entries, advice = index_entries(), ADVICE["all"]
    else:
        revs = pre_push_revs(sys.stdin.readlines()) if args.pre_push else ["HEAD"]
        entries, advice = (commit_entries(*revs) if revs else []), ADVICE["history"]

    findings = check(entries)
    for entry, problem in findings:
        where = f" (commit {entry.commit})" if entry.commit else ""
        print(f"guard_data: {entry.path}{where}: {problem}", file=sys.stderr)
    if findings:
        print(f"guard_data: refused. {advice}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
