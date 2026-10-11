"""Refuse data files, images, model weights and keys in git (CLAUDE.md ground rules: CCT20 images
and metadata, checkpoints and credentials never reach the repo).

    uv run python scripts/guard_data.py --staged     # pre-commit / pre-merge-commit hooks
    uv run python scripts/guard_data.py --pre-push   # pre-push hook: commits not on a remote yet
    uv run python scripts/guard_data.py --all        # check.sh and CI: every file in the index
    uv run python scripts/guard_data.py --history    # check.sh and CI: every commit from HEAD

Contents are read by blob ID from git (the index or the commits), never from the working tree.
Findings name the path and the rule only, never the matching text. Standard library only, so the
hooks can run it with `uv run --no-project`.

Web images (ALLOWED_IMAGE_SUFFIXES) are allowed only under the IMAGE_BUDGETS directories, each with
a file-count and byte budget: `--all` (check.sh, CI) enforces it over the whole index, `--history`
and `--pre-push` per commit, `--staged` over the staged files. Every text blob is scanned for
embedded images (notebook outputs and attachments, data URIs) and for dataset record keys, whatever
its suffix, so a dump renamed to .txt or pasted into a notebook is still refused. A notebook
(.ipynb) with any output or attachment left in it is refused: outputs are cleared before a commit.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import PurePosixPath

BLOCKED_DIRS = ("data/", "models/", "outputs/")
IMAGE_SUFFIXES = frozenset(
    {".avif", ".bmp", ".gif", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
ALLOWED_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})  # mirrored in .gitignore
# Directories where web images may live, with (max files, max total bytes). Mirrored in .gitignore.
IMAGE_BUDGETS: dict[str, tuple[int, int]] = {
    "assets/": (24, 12_000_000),  # demo samples from trans-test locations
    "docs/figures/": (40, 20_000_000),  # generated plots: curves, confusion matrix, Grad-CAM
}
ALLOWED_IMAGE_DIRS = tuple(IMAGE_BUDGETS)
VIDEO_SUFFIXES = frozenset({".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"})
MAX_METADATA_RECORDS = 100  # more dataset record keys than this in one blob is a dataset dump
HEIF_BRANDS = (b"avif", b"avis", b"heic", b"heix", b"heif", b"hevc", b"mif1", b"msf1")
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
NON_IMAGE_BLOCKED = BLOCKED_SUFFIXES - ALLOWED_IMAGE_SUFFIXES
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

# Embedded images in any text blob: a notebook output or attachment MIME key, or a data URI (HTML
# outputs, markdown), followed by a base64 payload of at least MIN_IMAGE_PAYLOAD characters, so a
# doc that mentions the key or a tiny icon does not count. Split literals: this file must not
# match itself.
IMAGE_MARKERS = re.compile(
    rb'"image/(?:png|jpe?g|gif|webp|bmp|tiff?|heic|avif)"\s*:|data:'
    rb"image/[a-z+.-]+(?:;[a-z0-9_.+=-]+)*;base64,",  # parameters such as ;charset=utf-8
    re.IGNORECASE,
)
MIN_IMAGE_PAYLOAD = 1000  # base64 characters, about 750 bytes; a camera-trap photo is 100x that
_PAYLOAD = re.compile(rb'[\sA-Za-z0-9+/=",\[\]\\]{0,2000000}')  # base64 plus JSON list noise
_NOT_BASE64 = re.compile(rb"[^A-Za-z0-9+/=]")
# Dataset record keys (CCT20 metadata and anything derived from it): JSON, a notebook's
# JSON-escaped source or output (\"seq_id\":) and a printed Python dict ('seq_id':).
RECORD_KEYS = re.compile(rb"\\?[\"'](?:seq_id|file_name|image_id)\\?[\"']\s*:")
# CCT20 image and sequence ids are UUIDs (every one of the 57,864 images, docs/data-inspection.md
# file name pattern): more than MAX_METADATA_RECORDS of them in one blob is a split saved as id
# lists, or a CSV, TSV or text dump of the metadata under any name.
UUID = re.compile(rb"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)

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
        and not any(s in NON_IMAGE_BLOCKED for s in suffixes)
    )


def image_dir(path: str) -> str | None:
    """The IMAGE_BUDGETS directory an allowed image sits in, else None."""
    lower = path.lower()
    return next((d for d in IMAGE_BUDGETS if lower.startswith(d)), None)


def embedded_images(data: bytes) -> int:
    """Image markers followed by a payload of at least MIN_IMAGE_PAYLOAD base64 characters."""
    n_images = 0
    for marker in IMAGE_MARKERS.finditer(data):
        chunk = _PAYLOAD.match(data, marker.end())
        payload = _NOT_BASE64.sub(b"", chunk.group()) if chunk else b""
        if len(payload) >= MIN_IMAGE_PAYLOAD:
            n_images += 1
    return n_images


def embedded_problems(data: bytes) -> list[str]:
    """Images or dataset records hidden inside a text file, whatever its suffix."""
    problems = []
    n_images = embedded_images(data)
    if n_images:
        problems.append(f"{n_images} embedded images (notebook outputs, attachments or data URIs)")
    n_records = len(RECORD_KEYS.findall(data))
    if n_records > MAX_METADATA_RECORDS:
        problems.append(f"dataset metadata ({n_records} record keys)")
    n_ids = len(UUID.findall(data))
    if n_ids > MAX_METADATA_RECORDS:
        problems.append(f"dataset ids ({n_ids} UUIDs)")
    return problems


def notebook_problems(data: bytes) -> list[str]:
    """Notebooks are committed with outputs cleared and no attachments (.claude/rules/training.md):
    a text output can hold a metadata dump and an SVG plot carries no base64 payload, so any
    output or attachment is refused, not only the images the checks above recognise."""
    try:
        cells = json.loads(data).get("cells", [])
    except (ValueError, AttributeError):
        return []  # not a notebook in the nbformat 4 shape; the other checks still apply
    if not isinstance(cells, list):
        return []
    cells = [c for c in cells if isinstance(c, dict)]
    problems = []
    if n_out := sum(bool(c.get("outputs")) for c in cells):
        problems.append(f"notebook outputs not cleared ({n_out} cells)")
    if n_att := sum(bool(c.get("attachments")) for c in cells):
        problems.append(f"notebook attachments ({n_att} cells)")
    return problems


def path_problems(path: str) -> list[str]:
    lower = path.lower()
    pure = PurePosixPath(lower.rstrip(". ~"))  # split.csv~, split.csv. and "split.csv " too
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
        is_image = data[8:12] in HEIF_BRANDS
        problems.append("HEIC/AVIF image file" if is_image else "MP4/MOV video file")
    if is_allowed_image(path):
        problems = [p for p in problems if p not in ("JPEG file", "PNG file")]
    else:
        problems += embedded_problems(data)
    if path.lower().endswith(".ipynb"):
        problems += notebook_problems(data)
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
    findings += budget_problems(entries, contents)
    return findings


def budget_problems(entries: list[Entry], contents: dict[str, bytes]) -> list[tuple[Entry, str]]:
    """File-count and byte budgets per IMAGE_BUDGETS directory, per commit (commit "" is the
    index or the staged set, so --all sees the whole tree)."""
    groups: dict[tuple[str, str], dict[str, Entry]] = {}
    for entry in entries:
        directory = image_dir(entry.path) if is_allowed_image(entry.path) else None
        if directory is not None:
            groups.setdefault((entry.commit, directory), {})[entry.path] = entry
    findings: list[tuple[Entry, str]] = []
    for (commit, directory), by_path in sorted(groups.items()):
        max_files, max_bytes = IMAGE_BUDGETS[directory]
        where = Entry(directory, "", commit)
        if len(by_path) > max_files:
            findings.append((where, f"{len(by_path)} images under {directory} (cap {max_files})"))
        total = sum(len(contents[e.blob]) for e in by_path.values())
        if total > max_bytes:
            findings.append(
                (where, f"images under {directory} total {total:,} bytes (cap {max_bytes:,})")
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
