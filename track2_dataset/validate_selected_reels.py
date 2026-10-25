#!/usr/bin/env python3
"""Validate selected Track 2 reel manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTED_REELS = SCRIPT_DIR / "track2_role_ipc_selected_reels_test_v1.csv"
DEFAULT_SELECTED_SCENES = SCRIPT_DIR / "track2_role_ipc_selected_reel_scenes_test_v1.csv"
DEFAULT_AUDITED_CANDIDATES = SCRIPT_DIR / "track2_role_ipc_reel_candidates_task_audit_v1.csv"
DEFAULT_GAP_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected Track 2 reel manifests.")
    parser.add_argument("--selected-reels", type=Path, default=DEFAULT_SELECTED_REELS)
    parser.add_argument("--selected-scenes", type=Path, default=DEFAULT_SELECTED_SCENES)
    parser.add_argument("--audited-candidates", type=Path, default=DEFAULT_AUDITED_CANDIDATES)
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def validate(args: argparse.Namespace) -> tuple[int, int]:
    require_file(args.selected_reels, "selected reels CSV")
    require_file(args.selected_scenes, "selected reel scenes CSV")
    require_file(args.audited_candidates, "audited candidates CSV")

    reels = pd.read_csv(args.selected_reels, keep_default_na=False)
    scenes = pd.read_csv(args.selected_scenes, keep_default_na=False)
    audited = pd.read_csv(args.audited_candidates, keep_default_na=False)

    require_columns(
        reels,
        {
            "reel_id",
            "candidate_reel_id",
            "source_split",
            "audit_reel_duration_pass",
            "audit_scenario_card_unique",
            "reel_task_ready",
            "selected_reel_audio_duration_sec",
        },
        "selected reels",
    )
    require_columns(
        scenes,
        {
            "reel_id",
            "candidate_reel_id",
            "source_split",
            "scene_rank",
            "scene_id",
            "scene_start_seconds",
            "scene_end_seconds",
            "gap_after_seconds",
        },
        "selected reel scenes",
    )
    require_columns(audited, {"candidate_reel_id"}, "audited candidates")

    if reels["reel_id"].duplicated().any():
        duplicates = sorted(reels.loc[reels["reel_id"].duplicated(), "reel_id"].unique())
        raise ValueError(f"Duplicate reel_id values: {duplicates[:10]}")
    if reels["candidate_reel_id"].duplicated().any():
        duplicates = sorted(
            reels.loc[reels["candidate_reel_id"].duplicated(), "candidate_reel_id"].unique()
        )
        raise ValueError(f"Duplicate selected candidate_reel_id values: {duplicates[:10]}")

    if set(reels["source_split"]) != {"test"}:
        raise ValueError(f"Selected reels must all be source_split=test: {sorted(set(reels['source_split']))}")
    for column in ("audit_reel_duration_pass", "audit_scenario_card_unique", "reel_task_ready"):
        if not bool_series(reels[column]).all():
            bad = reels.loc[~bool_series(reels[column]), "reel_id"].tolist()
            raise ValueError(f"Selected reels failed {column}: {bad[:10]}")

    audited_ids = set(audited["candidate_reel_id"])
    missing_candidate_ids = sorted(set(reels["candidate_reel_id"]) - audited_ids)
    if missing_candidate_ids:
        raise ValueError(f"Selected candidates missing from audited candidate CSV: {missing_candidate_ids[:10]}")

    scene_reel_ids = set(scenes["reel_id"])
    selected_reel_ids = set(reels["reel_id"])
    if scene_reel_ids != selected_reel_ids:
        raise ValueError(
            "Selected scene reel IDs do not match selected reels: "
            f"missing={sorted(selected_reel_ids - scene_reel_ids)[:10]}, "
            f"extra={sorted(scene_reel_ids - selected_reel_ids)[:10]}"
        )

    if scenes["scene_id"].duplicated().any():
        duplicates = sorted(scenes.loc[scenes["scene_id"].duplicated(), "scene_id"].unique())
        raise ValueError(f"Source scene_id reused across selected reels: {duplicates[:10]}")

    reels_by_id = reels.set_index("reel_id")
    for reel_id, group in scenes.groupby("reel_id", sort=False):
        group = group.sort_values("scene_rank")
        ranks = group["scene_rank"].astype(int).tolist()
        if ranks != [1, 2, 3, 4]:
            raise ValueError(f"{reel_id}: expected scene ranks [1,2,3,4], got {ranks}")
        starts = group["scene_start_seconds"].astype(float).tolist()
        ends = group["scene_end_seconds"].astype(float).tolist()
        gaps = group["gap_after_seconds"].astype(float).tolist()
        if abs(starts[0]) > 1e-6:
            raise ValueError(f"{reel_id}: first scene must start at 0.0, got {starts[0]}")
        for index in range(3):
            if abs(gaps[index] - args.gap_seconds) > 1e-6:
                raise ValueError(f"{reel_id}: rank {index + 1} gap must be {args.gap_seconds}, got {gaps[index]}")
            expected_next_start = ends[index] + gaps[index]
            if abs(starts[index + 1] - expected_next_start) > 1e-6:
                raise ValueError(
                    f"{reel_id}: rank {index + 2} start {starts[index + 1]} "
                    f"does not match previous end+gap {expected_next_start}"
                )
        if abs(gaps[3]) > 1e-6:
            raise ValueError(f"{reel_id}: final gap must be 0.0, got {gaps[3]}")
        expected_duration = float(reels_by_id.loc[reel_id, "selected_reel_audio_duration_sec"])
        if abs(ends[-1] - expected_duration) > 1e-6:
            raise ValueError(f"{reel_id}: final scene end {ends[-1]} != selected duration {expected_duration}")

    return len(reels), len(scenes)


def main() -> None:
    args = parse_args()
    reel_count, scene_count = validate(args)
    print("selected reel validation passed")
    print(f"selected_reels {reel_count}")
    print(f"selected_scenes {scene_count}")


if __name__ == "__main__":
    main()
