#!/usr/bin/env python3

import argparse
import getpass
import os
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VERTEX_JOBS_DIR = ROOT / "vertex_jobs"
RUNS_DIR = VERTEX_JOBS_DIR / "runs"
DEFAULT_BUCKET_PREFIX = "gs://genai-1010101/obfuscation-prompting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a per-run Vertex YAML config from a shared template."
    )
    parser.add_argument(
        "--template",
        default="experiment_l4.yaml",
        help="Template file under vertex_jobs/, e.g. experiment_l4.yaml or smoke_l4.yaml",
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Short run identifier used in the output filename.",
    )
    parser.add_argument(
        "--owner",
        default="",
        help="Short owner/team-member slug. Defaults to $VERTEX_RUN_OWNER or the current username.",
    )
    parser.add_argument(
        "--display-name",
        default="",
        help="Optional VERTEX_DISPLAY_NAME override. Defaults to <owner>-<run-name>.",
    )
    parser.add_argument(
        "--experiment-name",
        default="",
        help="Optional VERTEX_EXPERIMENT_NAME override.",
    )
    parser.add_argument(
        "--image-uri",
        default="",
        help="Optional container image URI override.",
    )
    parser.add_argument(
        "--results-gcs-uri",
        default=DEFAULT_BUCKET_PREFIX,
        help="GCS base prefix for uploaded artifacts.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional explicit output path. Defaults to vertex_jobs/runs/<timestamp>_<run-name>.yaml",
    )
    return parser.parse_args()


def replace_first(text: str, pattern: str, repl: str) -> str:
    return re.sub(pattern, repl, text, count=1, flags=re.MULTILINE)


def replace_env_value(text: str, env_name: str, new_value: str) -> str:
    pattern = (
        rf"(- name: {re.escape(env_name)}\n"
        rf"\s+value: )([^\n]+)"
    )
    replacement = rf"\1{new_value}"
    return re.sub(pattern, replacement, text, count=1)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def resolve_owner(explicit_owner: str) -> str:
    raw_owner = explicit_owner or os.environ.get("VERTEX_RUN_OWNER") or getpass.getuser()
    safe_owner = slugify(raw_owner)
    if not safe_owner:
        raise SystemExit(
            "Could not determine a safe run owner. Pass --owner or set VERTEX_RUN_OWNER."
        )
    return safe_owner


def main() -> int:
    args = parse_args()

    template_path = VERTEX_JOBS_DIR / args.template
    if not template_path.is_file():
        raise SystemExit(f"Template not found: {template_path}")

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    owner_slug = resolve_owner(args.owner)
    safe_run_name = slugify(args.run_name)
    if not safe_run_name:
        raise SystemExit("Run name must contain at least one alphanumeric character.")

    run_slug = f"{owner_slug}-{safe_run_name}"
    output_path = Path(args.output) if args.output else RUNS_DIR / f"{timestamp}_{run_slug}.yaml"

    text = template_path.read_text(encoding="utf-8")
    text = replace_env_value(text, "VERTEX_RESULTS_GCS_URI", args.results_gcs_uri)

    display_name = args.display_name or run_slug
    text = replace_env_value(text, "VERTEX_DISPLAY_NAME", display_name)
    if args.experiment_name:
        text = replace_env_value(text, "VERTEX_EXPERIMENT_NAME", args.experiment_name)
    if args.image_uri:
        text = replace_first(
            text,
            r"(^\s*imageUri:\s*)([^\n]+)",
            rf"\1{args.image_uri}",
        )
        text = replace_env_value(text, "VERTEX_IMAGE_URI", args.image_uri)

    header = (
        f"# Generated from vertex_jobs/{args.template}\n"
        f"# Owner: {owner_slug}\n"
        f"# Run name: {safe_run_name}\n"
        f"# Display name: {display_name}\n"
        f"# Generated at: {timestamp} UTC\n\n"
    )
    output_path.write_text(header + text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
