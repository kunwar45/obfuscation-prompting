"""Canonical artifact store: the public HF dataset repo for this project.

All run artifacts — generated datasets, results JSONs, plots, activation
captures, archived experiment snapshots — live in the HuggingFace dataset
repo below. Local `data/`, `results/`, `activations/`, etc. are working
copies only; anything worth keeping gets pushed to the Hub, and readers
fall back to the Hub when a local file is missing.

Folder names inside the repo mirror the repository root, so an
`activation_path` like `activations/<run>/<model>/<prompt_id>.npz` is the
same string locally and on the Hub.

CLI wrapper: `python -m scripts.hf_sync {pull,push} [folder ...]`.
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import EntryNotFoundError

HF_ARTIFACT_REPO = "kunwar45/obfuscation-prompting"
HF_REPO_TYPE = "dataset"

# Top-level artifact folders, identical locally and in the HF repo.
ARTIFACT_FOLDERS = ("data", "results", "activations", "saved_experiments", "vertex_downloads")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def repo_relative(path: str | Path) -> str | None:
    """Map a local artifact path to its path inside the HF repo, or None.

    Anchors on the first path component that names an artifact folder, so
    absolute paths written on other machines (e.g. `/app/activations/...`
    from a Vertex container) still resolve. Non-artifact paths return None.
    """
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part in ARTIFACT_FOLDERS:
            return "/".join(parts[i:])
    return None


def resolve_artifact(path: str | Path) -> Path:
    """Return a readable local file for `path`, fetching from the Hub if needed.

    A file that exists locally (as given, or relative to the repo root) wins;
    otherwise the file is downloaded from HF_ARTIFACT_REPO into the HF cache
    and that cache path is returned.
    """
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute() and (REPO_ROOT / p).exists():
        return REPO_ROOT / p
    rel = repo_relative(p)
    if rel is None:
        raise FileNotFoundError(
            f"{path} does not exist locally and is not under an artifact folder "
            f"{ARTIFACT_FOLDERS}, so it cannot be fetched from {HF_ARTIFACT_REPO}"
        )
    try:
        return Path(hf_hub_download(HF_ARTIFACT_REPO, rel, repo_type=HF_REPO_TYPE))
    except EntryNotFoundError as exc:
        raise FileNotFoundError(
            f"{path} not found locally or in {HF_ARTIFACT_REPO} (as {rel})"
        ) from exc


def pull(folders: list[str] | None = None, dest: str | Path = REPO_ROOT) -> Path:
    """Materialize artifact folders from the Hub into `dest` (default: repo root)."""
    folders = list(folders) if folders else list(ARTIFACT_FOLDERS)
    unknown = sorted(set(folders) - set(ARTIFACT_FOLDERS))
    if unknown:
        raise ValueError(f"Unknown artifact folder(s) {unknown}; expected {ARTIFACT_FOLDERS}")
    snapshot_download(
        HF_ARTIFACT_REPO,
        repo_type=HF_REPO_TYPE,
        local_dir=str(dest),
        allow_patterns=[f"{folder}/**" for folder in folders],
    )
    return Path(dest)


def push(folders: list[str], commit_message: str | None = None) -> None:
    """Upload local artifact folders to the Hub (requires a write token)."""
    unknown = sorted(set(folders) - set(ARTIFACT_FOLDERS))
    if unknown:
        raise ValueError(f"Unknown artifact folder(s) {unknown}; expected {ARTIFACT_FOLDERS}")
    api = HfApi()
    for folder in folders:
        local = REPO_ROOT / folder
        if not local.is_dir():
            raise FileNotFoundError(f"{local} does not exist — nothing to push")
        api.upload_folder(
            repo_id=HF_ARTIFACT_REPO,
            repo_type=HF_REPO_TYPE,
            folder_path=str(local),
            path_in_repo=folder,
            commit_message=commit_message or f"Update {folder}/",
        )
