from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .adapters import API_ERROR_HARD, API_ERROR_SOFT, adapter_from_config, load_adapter_config
from .core import (
    API_STATUS_HARD_ERROR,
    API_STATUS_SOFT_ERROR,
    CONDITIONS,
    REGIMES,
    TIME_MODES,
    TRACK_IDS,
    _parse_config_from_manifest,
    default_dataset_root_for_time_mode,
    result_api_status,
    run_benchmark,
)


@dataclass(frozen=True)
class FailedRunSpec:
    track_id: str
    condition: str
    regime: str | None
    episode_id: str | None
    result_path: Path
    task_id: str | None = None
    benchmark_results_root: Path | None = None


def _contains_unrecovered_prompt_api_error(value: Any) -> bool:
    if isinstance(value, dict):
        is_invoked_prompt = (
            value.get("asked", True) is not False
            and bool(value.get("prompt"))
            and "raw_output" in value
        )
        if is_invoked_prompt and (value.get("parsed") or {}).get("reason") == "api_error":
            return True
        return any(_contains_unrecovered_prompt_api_error(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_unrecovered_prompt_api_error(child) for child in value)
    return False


def result_has_unrecovered_api_error(result: dict[str, Any]) -> bool:
    return rerun_result_api_status(result) in {API_STATUS_SOFT_ERROR, API_STATUS_HARD_ERROR}


def rerun_result_api_status(result: dict[str, Any]) -> str:
    if result.get("track2_result_schema_version") is None:
        return result_api_status(result)
    if result.get("run_status") != "api_error" and not result.get("unrecovered_api_errors"):
        return "complete"

    error_classes: list[str] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            error_class = value.get("error_class")
            if error_class in {API_ERROR_SOFT, API_ERROR_HARD}:
                error_classes.append(str(error_class))
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(result.get("unrecovered_api_errors") or result.get("api_attempts") or [])
    if API_ERROR_HARD in error_classes:
        return API_STATUS_HARD_ERROR
    return API_STATUS_SOFT_ERROR


def _track2_results_roots(results_root: Path) -> list[Path]:
    root = Path(results_root)
    candidates = [root / "track_2"]
    if root.name in {"results", "results_continuous"}:
        candidates.append(root.parent / "results_track2")
    elif root.name == "results_track2":
        candidates.append(root)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate.is_dir()))


def find_failed_run_specs(
    results_root: Path,
    tracks: Sequence[str],
    *,
    statuses: Sequence[str] = (API_STATUS_SOFT_ERROR,),
) -> list[FailedRunSpec]:
    selected_statuses = {str(status) for status in statuses}
    specs: list[FailedRunSpec] = []
    for track_id in tracks:
        if str(track_id) == "2":
            for track2_root in _track2_results_roots(Path(results_root)):
                for result_path in sorted(track2_root.glob("*/*/*/*.json")):
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    if result.get("track2_result_schema_version") is None:
                        continue
                    if rerun_result_api_status(result) not in selected_statuses:
                        continue
                    specs.append(
                        FailedRunSpec(
                            track_id="2",
                            condition=str(result["condition"]),
                            regime=None,
                            episode_id=None,
                            task_id=str(result["task_id"]),
                            result_path=result_path,
                            benchmark_results_root=track2_root,
                        )
                    )
            continue
        episode_results_root = Path(results_root) / f"track_{track_id}" / "episode_results"
        if not episode_results_root.is_dir():
            continue
        for result_path in sorted(episode_results_root.rglob("*.json")):
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result_api_status(result) not in selected_statuses:
                continue
            specs.append(
                FailedRunSpec(
                    track_id=str(track_id),
                    condition=str(result["condition"]),
                    regime=str(result["regime"]),
                    episode_id=str(result["episode_id"]),
                    result_path=result_path,
                )
            )
    track_order = {track_id: index for index, track_id in enumerate(TRACK_IDS)}
    condition_order = {condition: index for index, condition in enumerate(CONDITIONS)}
    regime_order = {regime: index for index, regime in enumerate(REGIMES)}
    return sorted(
        specs,
        key=lambda spec: (
            track_order[spec.track_id],
            condition_order[spec.condition],
            regime_order.get(spec.regime, -1),
            spec.episode_id or spec.task_id or "",
        ),
    )


def _track_root(root: Path, track_id: str) -> Path:
    candidate = Path(root) / f"track_{track_id}"
    if (candidate / "dataset" / "manifest.json").is_file():
        return candidate
    if (Path(root) / "dataset" / "manifest.json").is_file():
        return Path(root)
    raise RuntimeError(f"Missing dataset manifest for Track {track_id} under {root}.")


def _run_selected_specs(
    *,
    specs: Sequence[FailedRunSpec],
    dataset_root: Path,
    results_root: Path,
    adapter: Any,
    parallelism: int,
    time_mode: str,
) -> None:
    track2_grouped: dict[tuple[str, Path], list[str]] = defaultdict(list)
    grouped: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for spec in specs:
        if spec.track_id == "2":
            if spec.task_id is None or spec.benchmark_results_root is None:
                raise RuntimeError("Track 2 failed-run specification is incomplete.")
            track2_grouped[(spec.condition, spec.benchmark_results_root)].append(spec.task_id)
            continue
        if spec.regime is None or spec.episode_id is None:
            raise RuntimeError("Legacy failed-run specification is incomplete.")
        grouped[(spec.track_id, spec.condition, spec.regime)].append(spec.episode_id)

    if track2_grouped:
        from .track2 import run_track2_benchmark

        track2_dataset_root = (
            Path(__file__).resolve().parent.parent
            / "track2_dataset"
            / "generated_data"
            / "track2_role_ipc_test_v2"
        )
        for (condition, track2_results_root), task_ids in track2_grouped.items():
            print(
                f"rerun-api-errors track=2 condition={condition} tasks={len(task_ids)}",
                flush=True,
            )
            run_track2_benchmark(
                dataset_root=track2_dataset_root,
                results_root=track2_results_root,
                adapter=adapter,
                task_ids=task_ids,
                conditions=[condition],
                reprocess=True,
                parallelism=parallelism,
            )

    for (track_id, condition, regime), episode_ids in grouped.items():
        track_dataset_root = _track_root(dataset_root, track_id)
        track_results_root = Path(results_root) / f"track_{track_id}"
        manifest_path = track_dataset_root / "dataset" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = None if manifest.get("track_id") == "1c" else _parse_config_from_manifest(manifest_path)
        print(
            f"rerun-api-errors track={track_id} condition={condition} regime={regime} "
            f"instances={len(episode_ids)}",
            flush=True,
        )
        run_benchmark(
            dataset_root=track_dataset_root,
            results_root=track_results_root,
            adapter=adapter,
            config=config,
            episode_ids=episode_ids,
            conditions=[condition],
            regimes=[regime],
            reprocess=True,
            parallelism=parallelism,
            time_mode=time_mode,
        )


def _rebuild_track_outputs(
    *,
    tracks: Sequence[str],
    dataset_root: Path,
    results_root: Path,
    adapter: Any,
    time_mode: str,
) -> None:
    for track_id in tracks:
        if track_id == "2":
            from .track2 import rebuild_track2_outputs

            for track2_root in _track2_results_roots(Path(results_root)):
                rebuild_track2_outputs(track2_root)
            continue
        track_dataset_root = _track_root(dataset_root, track_id)
        track_results_root = Path(results_root) / f"track_{track_id}"
        manifest_path = track_dataset_root / "dataset" / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config = None if manifest.get("track_id") == "1c" else _parse_config_from_manifest(manifest_path)
        run_benchmark(
            dataset_root=track_dataset_root,
            results_root=track_results_root,
            adapter=adapter,
            config=config,
            reprocess=False,
            parallelism=1,
            time_mode=time_mode,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun only benchmark instances with unresolved soft API errors. "
            "Model/protocol failures and recovered transient errors are never selected."
        )
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--tracks", nargs="+", default=list(TRACK_IDS), choices=TRACK_IDS)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--time-mode", default="binned", choices=TIME_MODES)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.parallelism < 1:
        raise ValueError("--parallelism must be at least 1.")
    if args.max_instances is not None and args.max_instances < 1:
        raise ValueError("--max-instances must be at least 1.")

    repo_root = Path(__file__).resolve().parent.parent
    model_dir = args.model_dir.resolve()
    try:
        model_dir.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"--model-dir must be inside the repository: {model_dir}") from exc

    results_root = model_dir / ("results" if args.time_mode == "binned" else "results_continuous")
    dataset_root = repo_root / default_dataset_root_for_time_mode(args.time_mode)
    specs = find_failed_run_specs(results_root, args.tracks)
    if args.max_instances is not None:
        specs = specs[: args.max_instances]

    print(f"Selected {len(specs)} unresolved soft API-error instances.", flush=True)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for spec in specs:
        counts[(spec.track_id, spec.condition)] += 1
    for (track_id, condition), count in sorted(counts.items()):
        print(f"  track={track_id} condition={condition}: {count}", flush=True)

    if args.dry_run or not specs:
        return 0

    adapter = adapter_from_config(load_adapter_config(args.adapter_config))
    selected_tracks = list(dict.fromkeys(spec.track_id for spec in specs))
    try:
        _run_selected_specs(
            specs=specs,
            dataset_root=dataset_root,
            results_root=results_root,
            adapter=adapter,
            parallelism=args.parallelism,
            time_mode=args.time_mode,
        )
    finally:
        _rebuild_track_outputs(
            tracks=selected_tracks,
            dataset_root=dataset_root,
            results_root=results_root,
            adapter=adapter,
            time_mode=args.time_mode,
        )

    remaining = find_failed_run_specs(results_root, args.tracks)
    print(f"Remaining unresolved soft API-error instances: {len(remaining)}", flush=True)
    return 0
