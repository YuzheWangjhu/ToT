from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .api_error_rerun import find_failed_run_specs, rerun_result_api_status
from .core import (
    API_STATUS_HARD_ERROR,
    API_STATUS_SOFT_ERROR,
    TIME_MODES,
    TRACK_IDS,
)


API_JOB_NAME_MARKERS = ("gemini", "gpt_audio")


def active_api_jobs() -> list[dict[str, str]]:
    completed = subprocess.run(
        ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i|%j|%T"],
        check=True,
        capture_output=True,
        text=True,
    )
    jobs = []
    for line in completed.stdout.splitlines():
        job_id, job_name, state = line.split("|", maxsplit=2)
        if any(marker in job_name.lower() for marker in API_JOB_NAME_MARKERS):
            jobs.append({"job_id": job_id, "job_name": job_name, "state": state})
    return jobs


def archive_unresolved_results(
    *,
    model_dir: Path,
    tracks: Sequence[str],
    time_mode: str,
    archive_timestamp: str,
    dry_run: bool,
) -> dict[str, object]:
    results_root = Path(model_dir) / ("results" if time_mode == "binned" else "results_continuous")
    specs = find_failed_run_specs(
        results_root,
        tracks,
        statuses=(API_STATUS_SOFT_ERROR, API_STATUS_HARD_ERROR),
    )
    archive_root = (
        results_root
        / "api_error_archive"
        / "retry_policy_v2"
        / archive_timestamp
    )
    manifest_entries = []
    for spec in specs:
        try:
            relative_path = spec.result_path.relative_to(results_root)
        except ValueError:
            relative_path = spec.result_path.relative_to(Path(model_dir))
        result = json.loads(spec.result_path.read_text(encoding="utf-8"))
        manifest_entries.append(
            {
                "relative_path": relative_path.as_posix(),
                "sha256": hashlib.sha256(spec.result_path.read_bytes()).hexdigest(),
                "api_status": rerun_result_api_status(result),
                "track_id": spec.track_id,
                "condition": spec.condition,
                "regime": spec.regime,
                "episode_id": spec.episode_id,
                "task_id": spec.task_id,
            }
        )
        if not dry_run:
            destination = archive_root / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"Archive destination already exists: {destination}")
            shutil.move(str(spec.result_path), str(destination))
    manifest = {
        "policy_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(Path(model_dir).resolve()),
        "results_root": str(results_root.resolve()),
        "num_archived": len(manifest_entries),
        "files": manifest_entries,
    }
    if not dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)
        manifest_path = archive_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Archive unresolved API-error results before retry-policy-v2 regeneration."
    )
    parser.add_argument("--model-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", default=list(TRACK_IDS), choices=TRACK_IDS)
    parser.add_argument("--time-mode", default="binned", choices=TIME_MODES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    jobs = active_api_jobs()
    if jobs:
        details = ", ".join(
            f"{job['job_id']}:{job['job_name']}:{job['state']}" for job in jobs
        )
        raise RuntimeError(f"Refusing API-result migration while API jobs are active: {details}")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    total = 0
    for model_dir in args.model_dirs:
        manifest = archive_unresolved_results(
            model_dir=model_dir,
            tracks=args.tracks,
            time_mode=args.time_mode,
            archive_timestamp=timestamp,
            dry_run=args.dry_run,
        )
        count = int(manifest["num_archived"])
        total += count
        print(f"{model_dir}: {count} unresolved API-error results", flush=True)
    action = "Would archive" if args.dry_run else "Archived"
    print(f"{action} {total} results under retry_policy_v2/{timestamp}.", flush=True)
    return 0
