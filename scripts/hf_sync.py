#!/usr/bin/env python3
"""Sync run artifacts with the canonical HF dataset repo.

Usage (from the repo root):
    python -m scripts.hf_sync pull                     # fetch every artifact folder
    python -m scripts.hf_sync pull activations results # fetch specific folders
    python -m scripts.hf_sync push results activations # publish local runs to the Hub
    python -m scripts.hf_sync push results -m "framing rerun, 50 scenarios"

`pull` writes into the matching folders at the repo root; `push` requires a
HF write token (`hf auth login`).
"""

from __future__ import annotations

import argparse

from src.storage.hf_artifacts import ARTIFACT_FOLDERS, HF_ARTIFACT_REPO, pull, push


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_pull = sub.add_parser("pull", help=f"Download artifact folders from {HF_ARTIFACT_REPO}")
    p_pull.add_argument(
        "folders",
        nargs="*",
        help=f"Folders to fetch (default: all of {', '.join(ARTIFACT_FOLDERS)})",
    )

    p_push = sub.add_parser("push", help=f"Upload local artifact folders to {HF_ARTIFACT_REPO}")
    p_push.add_argument("folders", nargs="+", choices=ARTIFACT_FOLDERS)
    p_push.add_argument("-m", "--message", default=None, help="Commit message on the Hub")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "pull":
        dest = pull(args.folders or None)
        print(f"PULLED {', '.join(args.folders or ARTIFACT_FOLDERS)} -> {dest}")
    else:
        push(args.folders, commit_message=args.message)
        print(f"PUSHED {', '.join(args.folders)} -> {HF_ARTIFACT_REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
