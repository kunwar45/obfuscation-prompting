#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from google.cloud import storage


DEFAULT_ARTIFACT_DIRS = ("results", "data", "activations")
METADATA_DIR = ".vertex_job"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run an experiment command inside Vertex AI and sync artifacts "
            "to a GCS run directory."
        )
    )
    parser.add_argument(
        "--results-gcs-uri",
        default=os.getenv("VERTEX_RESULTS_GCS_URI", ""),
        help="Base GCS URI, e.g. gs://bucket/obfuscation-prompting",
    )
    parser.add_argument(
        "--experiment-name",
        default=os.getenv("VERTEX_EXPERIMENT_NAME", ""),
        help="Logical experiment name used in the GCS path.",
    )
    parser.add_argument(
        "--display-name",
        default=os.getenv("VERTEX_DISPLAY_NAME", ""),
        help="Optional display name used in metadata and GCS path.",
    )
    parser.add_argument(
        "--image-uri",
        default=os.getenv("VERTEX_IMAGE_URI", ""),
        help="Optional image URI to record in metadata.",
    )
    parser.add_argument(
        "--artifact-dir",
        action="append",
        default=[],
        help=(
            "Artifact directory to upload. May be provided multiple times. "
            "Defaults to results, data, and activations."
        ),
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run, passed after '--'.",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("missing command; pass it after '--'")
    if not args.results_gcs_uri:
        parser.error(
            "--results-gcs-uri is required or set VERTEX_RESULTS_GCS_URI"
        )
    return args


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "gs" or not parsed.netloc:
        raise ValueError(f"expected gs://bucket/path, got: {uri}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return parsed.netloc, prefix


def make_run_slug(display_name: str, experiment_name: str, timestamp: str) -> str:
    logical_name = display_name or experiment_name or "vertex-run"
    safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in logical_name)
    short_id = uuid.uuid4().hex[:8]
    return f"{safe_name}/{timestamp}_{short_id}"


def collect_vertex_env() -> dict[str, str]:
    prefixes = ("AIP_", "CLOUD_ML_", "VERTEX_", "GOOGLE_CLOUD_")
    keys = sorted(k for k in os.environ if k.startswith(prefixes))
    return {k: os.environ[k] for k in keys}


def build_metadata(
    args: argparse.Namespace,
    command: list[str],
    run_started_at: datetime,
    run_slug: str,
) -> dict[str, Any]:
    bucket, prefix = parse_gcs_uri(args.results_gcs_uri)
    run_gcs_uri = f"gs://{bucket}/" + "/".join(
        p for p in (prefix, args.experiment_name, run_slug) if p
    )
    return {
        "run_id": run_slug.split("/")[-1],
        "run_slug": run_slug,
        "experiment_name": args.experiment_name,
        "display_name": args.display_name,
        "image_uri": args.image_uri,
        "hostname": socket.gethostname(),
        "started_at": run_started_at.isoformat(),
        "command": command,
        "command_shell": shlex.join(command),
        "artifact_dirs": args.artifact_dir or list(DEFAULT_ARTIFACT_DIRS),
        "results_gcs_base_uri": args.results_gcs_uri,
        "results_gcs_run_uri": run_gcs_uri,
        "vertex_env": collect_vertex_env(),
    }


def write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def upload_file(
    client: storage.Client,
    bucket_name: str,
    source_path: Path,
    destination_blob: str,
) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(str(source_path))


def upload_tree(
    client: storage.Client,
    bucket_name: str,
    source_dir: Path,
    destination_prefix: str,
) -> int:
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(source_dir).as_posix()
        blob_name = "/".join(part for part in (destination_prefix, rel) if part)
        upload_file(client, bucket_name, path, blob_name)
        count += 1
    return count


def sync_artifacts(
    results_gcs_uri: str,
    experiment_name: str,
    run_slug: str,
    artifact_dirs: list[str],
    metadata_path: Path,
) -> dict[str, Any]:
    client = storage.Client()
    bucket_name, base_prefix = parse_gcs_uri(results_gcs_uri)
    run_prefix = "/".join(
        p for p in (base_prefix, experiment_name, run_slug) if p
    )

    uploaded: dict[str, Any] = {"metadata": None, "artifacts": {}}

    metadata_blob = "/".join((run_prefix, "metadata.json"))
    upload_file(client, bucket_name, metadata_path, metadata_blob)
    uploaded["metadata"] = f"gs://{bucket_name}/{metadata_blob}"

    for artifact_dir in artifact_dirs:
        src = Path(artifact_dir)
        if not src.exists():
            uploaded["artifacts"][artifact_dir] = {
                "uploaded": False,
                "reason": "missing",
            }
            continue
        if not src.is_dir():
            uploaded["artifacts"][artifact_dir] = {
                "uploaded": False,
                "reason": "not_a_directory",
            }
            continue
        dest_prefix = "/".join((run_prefix, artifact_dir))
        file_count = upload_tree(client, bucket_name, src, dest_prefix)
        uploaded["artifacts"][artifact_dir] = {
            "uploaded": True,
            "file_count": file_count,
            "gcs_uri": f"gs://{bucket_name}/{dest_prefix}",
        }
    return uploaded


def run_blob_prefix(
    results_gcs_uri: str,
    experiment_name: str,
    run_slug: str,
) -> tuple[str, str]:
    bucket_name, base_prefix = parse_gcs_uri(results_gcs_uri)
    run_prefix = "/".join(
        p for p in (base_prefix, experiment_name, run_slug) if p
    )
    return bucket_name, run_prefix


def main() -> int:
    args = parse_args()
    artifact_dirs = args.artifact_dir or list(DEFAULT_ARTIFACT_DIRS)

    started_at = utc_now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    run_slug = make_run_slug(args.display_name, args.experiment_name, timestamp)
    metadata_path = Path(METADATA_DIR) / "metadata.json"
    metadata = build_metadata(args, args.command, started_at, run_slug)
    write_metadata(metadata_path, metadata)

    print(
        f"VERTEX_JOB_RUNNER_START run_uri={metadata['results_gcs_run_uri']}",
        flush=True,
    )
    print(f"VERTEX_JOB_RUNNER_COMMAND {metadata['command_shell']}", flush=True)

    proc = subprocess.run(args.command, check=False)
    exit_code = proc.returncode

    metadata["exit_code"] = exit_code
    metadata["finished_at"] = utc_now().isoformat()
    metadata["status"] = "succeeded" if exit_code == 0 else "failed"
    write_metadata(metadata_path, metadata)

    try:
        bucket_name, run_prefix = run_blob_prefix(
            args.results_gcs_uri,
            args.experiment_name,
            run_slug,
        )
        upload_summary = sync_artifacts(
            results_gcs_uri=args.results_gcs_uri,
            experiment_name=args.experiment_name,
            run_slug=run_slug,
            artifact_dirs=artifact_dirs,
            metadata_path=metadata_path,
        )
        metadata["upload_summary"] = upload_summary
        write_metadata(metadata_path, metadata)
        upload_file(
            storage.Client(),
            bucket_name,
            metadata_path,
            "/".join((run_prefix, "metadata.json")),
        )
        print(
            f"VERTEX_JOB_RUNNER_UPLOAD_COMPLETE {upload_summary['metadata']}",
            flush=True,
        )
    except Exception as exc:
        print(f"VERTEX_JOB_RUNNER_UPLOAD_FAILED {exc}", file=sys.stderr, flush=True)
        return exit_code or 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
