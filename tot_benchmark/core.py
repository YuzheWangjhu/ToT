from __future__ import annotations

import argparse
import csv
import json
import concurrent.futures
import inspect
import math
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import soundfile as sf

from .adapters import (
    API_ERROR_HARD,
    API_ERROR_SOFT,
    APIAdapterError,
    AdapterSession,
    BaseAdapter,
    adapter_from_config,
    classify_api_error,
    load_adapter_config,
)
from .text_protocol import normalize_first_line
from .time_mode import TIME_MODE_BINNED, TIME_MODE_CONTINUOUS, TIME_MODES, validate_time_mode

DISCOVERY = "Discovery"
COUNT_KNOWN = "Count-known"
INVENTORY_KNOWN = "Inventory-known"
REGIMES = [DISCOVERY, COUNT_KNOWN, INVENTORY_KNOWN]
ACTIVE = "active"
FULL_CONTEXT = "full_context"
ACTIVE_SAME_SESSION_PERIODIC_PROBE = "active_same_session_periodic_probe"
DEFAULT_CONDITIONS = [ACTIVE, FULL_CONTEXT, ACTIVE_SAME_SESSION_PERIODIC_PROBE]
CONDITIONS = list(DEFAULT_CONDITIONS)
ACTION_SYSTEM_PROMPT = (
    "You are participating in an audio exploration benchmark. "
    "Follow the requested output grammar exactly. Return plain text only."
)
SEQUENCE_DELIMITER = " -> "
BEHAVIORAL_PROTOCOL_VERSION = 2
TRACK_1A = "1a"
TRACK_1B = "1b"
TRACK_1C = "1c"
TRACK_2 = "2"
TRACK_IDS = [TRACK_1A, TRACK_1B, TRACK_1C, TRACK_2]
MetricValue = float | None
TRACK_1B_MAX_OVERLAP = 2
TRACK_1B_MAX_PLACEMENT_ATTEMPTS = 2000
API_STATUS_COMPLETE = "complete"
API_STATUS_SOFT_ERROR = "soft_api_error"
API_STATUS_HARD_ERROR = "hard_api_error"
API_STATUSES = {API_STATUS_COMPLETE, API_STATUS_SOFT_ERROR, API_STATUS_HARD_ERROR}
API_HARD_MAX_ATTEMPTS = 3
API_SOFT_MAX_RETRIES = 1
API_RETRY_MAX_SLEEP_SECONDS = 600.0
TRACK_1B_MIN_BIN_COVERAGE_RATE = 0.30
TRACK_1B_MIN_PARTIAL_OVERLAP_RATE = 0.25
DEFAULT_DATASET_ROOTS = {
    TIME_MODE_BINNED: "generated_data/theory_of_time_v1",
    TIME_MODE_CONTINUOUS: "generated_data/theory_of_time_hard_v1",
}


def default_dataset_root_for_time_mode(time_mode: str) -> str:
    return DEFAULT_DATASET_ROOTS[validate_time_mode(time_mode)]


@dataclass(frozen=True)
class BenchmarkConfig:
    track_id: str = TRACK_1A
    time_mode: str = TIME_MODE_BINNED
    num_episodes: int = 10
    seed: int = 20260413
    num_bins: int = 10
    bin_duration_sec: float = 1.0
    episode_duration_sec: float = 10.0
    max_query_bins: int = 3
    max_steps: int = 8
    max_total_queried_bins: int = 20
    probe_every_n: int = 2
    min_events: int = 3
    max_events: int = 5
    sequence_delimiter: str = SEQUENCE_DELIMITER
    api_retry_sleep_sec: float = 60.0

    def to_manifest_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if validate_time_mode(self.time_mode) == TIME_MODE_BINNED:
            payload.pop("time_mode", None)
        return payload


@dataclass(frozen=True)
class AssetRecord:
    label: str
    category: str
    path: Path
    duration_sec: float
    samplerate: int
    channels: int


class ProgressReporter:
    def __init__(self, total: int, *, label: str, eta_total: int | None = None) -> None:
        self.total = max(int(total), 1)
        self.eta_total = self.total if eta_total is None else max(int(eta_total), 0)
        self.label = label
        self.current = 0
        self.eta_current = 0
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.width = 28
        self.start_time = time.time()

    @staticmethod
    def _format_seconds(value: float) -> str:
        total_seconds = max(0, int(round(value)))
        minutes, seconds = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def update(self, current: int, detail: str, *, count_for_eta: bool = True) -> None:
        self.current = max(0, min(int(current), self.total))
        if count_for_eta:
            self.eta_current = min(self.eta_current + 1, self.eta_total)
        elapsed = max(0.0, time.time() - self.start_time)
        if self.eta_current > 0:
            rate = elapsed / self.eta_current
            eta = rate * max(0, self.eta_total - self.eta_current)
        elif self.eta_total == 0:
            eta = 0.0
        else:
            eta = math.nan
        filled = int(self.width * self.current / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        percent = int(100 * self.current / self.total)
        line = (
            f"{self.label} [{bar}] {percent:3d}% "
            f"({self.current}/{self.total}) "
            f"elapsed={self._format_seconds(elapsed)} "
            f"eta={self._format_seconds(eta) if not math.isnan(eta) else '--:--'} "
            f"{detail}"
        )
        if self.is_tty:
            print(f"\r{line}", end="", flush=True)
            if self.current >= self.total:
                print(flush=True)
        else:
            print(line, flush=True)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _validate_track_id(track_id: str) -> str:
    normalized = str(track_id).strip().lower()
    _assert(normalized in TRACK_IDS, f"Unknown track_id: {track_id}")
    return normalized


def normalize_tracks(tracks: Sequence[str] | None) -> list[str]:
    if tracks is None:
        return [TRACK_1A]
    normalized = [_validate_track_id(track) for track in tracks]
    _assert(bool(normalized), "At least one track must be selected.")
    return normalized


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


def _track_root(base_root: Path, track_id: str, tracks: Sequence[str]) -> Path:
    track_specific_root = Path(base_root) / f"track_{track_id}"
    if track_id == TRACK_1C and track_specific_root.exists():
        return track_specific_root
    return Path(base_root) if len(tracks) == 1 else track_specific_root


def scan_source_assets(audio_root: Path, config: BenchmarkConfig) -> list[AssetRecord]:
    track_id = _validate_track_id(config.track_id)
    assets: list[AssetRecord] = []
    for path in sorted(audio_root.rglob("*.wav")):
        info = sf.info(str(path))
        if track_id == TRACK_1A and info.duration >= config.bin_duration_sec:
            continue
        if track_id == TRACK_1B and info.duration > config.episode_duration_sec:
            continue
        _assert(info.channels == 1, f"Expected mono audio, got {info.channels} channels for {path}")
        label = path.name.split("__voice-")[0]
        assets.append(
            AssetRecord(
                label=label,
                category=path.parent.name,
                path=path,
                duration_sec=float(info.duration),
                samplerate=int(info.samplerate),
                channels=int(info.channels),
            )
        )
    _assert(bool(assets), f"No source assets under {audio_root} satisfied track {track_id} constraints.")
    samplerates = {asset.samplerate for asset in assets}
    _assert(len(samplerates) == 1, f"Expected one shared samplerate, found {sorted(samplerates)}")
    labels = {asset.label for asset in assets}
    _assert(len(labels) >= config.max_events, "Not enough unique labels to build benchmark episodes.")
    return assets


def _read_mono_audio(path: Path, expected_sr: int) -> np.ndarray:
    audio, samplerate = sf.read(str(path), dtype="float32", always_2d=False)
    _assert(samplerate == expected_sr, f"Samplerate mismatch for {path}: {samplerate} != {expected_sr}")
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32)


def _write_audio(path: Path, audio: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate, subtype="PCM_16")


def _relative_to_root(path: Path, root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _track_prior_text(config: BenchmarkConfig) -> str:
    if _validate_track_id(config.track_id) == TRACK_1B:
        return " Events may span multiple bins, and at most two events may overlap in any bin."
    return ""


def _build_regime_variants(labels_present: Sequence[str], config: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    label_text = ", ".join(labels_present)
    track_prior = _track_prior_text(config)
    return {
        DISCOVERY: {
            "name": DISCOVERY,
            "known_event_count": None,
            "known_inventory": None,
            "prior_text": (
                "You know the full vocabulary, that each episode contains between 3 and 5 events, "
                "and that labels do not repeat. You do not know the exact event count or which labels are present."
                f"{track_prior}"
            ),
        },
        COUNT_KNOWN: {
            "name": COUNT_KNOWN,
            "known_event_count": len(labels_present),
            "known_inventory": None,
            "prior_text": (
                "You know the full vocabulary, that labels do not repeat, and that this episode contains exactly "
                f"{len(labels_present)} events. You do not know which labels are present."
                f"{track_prior}"
            ),
        },
        INVENTORY_KNOWN: {
            "name": INVENTORY_KNOWN,
            "known_event_count": len(labels_present),
            "known_inventory": list(labels_present),
            "prior_text": (
                "You know the full vocabulary, that labels do not repeat, that this episode contains exactly "
                f"{len(labels_present)} events, and that the present labels are: {label_text}. "
                "You do not know where they occur."
                f"{track_prior}"
            ),
        },
    }


def _sequence_answer(sequence: Sequence[str], config: BenchmarkConfig) -> str:
    return config.sequence_delimiter.join(sequence)


def _pairwise_answer(spans: dict[str, dict[str, int]], label_a: str, label_b: str) -> str:
    return "BEFORE" if spans[label_a]["start_bin"] < spans[label_b]["start_bin"] else "AFTER"


def _localization_answer(span: dict[str, int]) -> str:
    return f"START {span['start_bin']} END {span['end_bin']}"


def _gap_answer(left_span: dict[str, int], right_span: dict[str, int]) -> str:
    return "ADJACENT" if right_span["start_bin"] == left_span["end_bin"] + 1 else "GAP"


def _spans_overlap(left_span: dict[str, int], right_span: dict[str, int]) -> bool:
    return max(left_span["start_bin"], right_span["start_bin"]) <= min(
        left_span["end_bin"],
        right_span["end_bin"],
    )


def _span_bin_count(asset: AssetRecord, config: BenchmarkConfig) -> int:
    return max(1, int(math.ceil(asset.duration_sec / config.bin_duration_sec)))


def _event_time_interval(start_bin: int, asset: AssetRecord, config: BenchmarkConfig) -> tuple[float, float]:
    start_sec = start_bin * config.bin_duration_sec
    return start_sec, start_sec + asset.duration_sec


def _event_bin_coverage_sec(
    start_bin: int,
    asset: AssetRecord,
    bin_index: int,
    config: BenchmarkConfig,
) -> float:
    event_start_sec, event_end_sec = _event_time_interval(start_bin, asset, config)
    bin_start_sec = bin_index * config.bin_duration_sec
    bin_end_sec = bin_start_sec + config.bin_duration_sec
    return max(0.0, min(event_end_sec, bin_end_sec) - max(event_start_sec, bin_start_sec))


def _event_bins_meet_min_coverage(start_bin: int, asset: AssetRecord, config: BenchmarkConfig) -> bool:
    span_bins = _span_bin_count(asset, config)
    min_coverage_sec = TRACK_1B_MIN_BIN_COVERAGE_RATE * config.bin_duration_sec
    return all(
        _event_bin_coverage_sec(start_bin, asset, start_bin + offset, config) >= min_coverage_sec
        for offset in range(span_bins)
    )


def _interval_overlap_sec(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def _interval_contains(container: tuple[float, float], contained: tuple[float, float]) -> bool:
    return container[0] <= contained[0] and contained[1] <= container[1]


def _partial_overlap_meets_min_rate(
    left_start_bin: int,
    left_asset: AssetRecord,
    right_start_bin: int,
    right_asset: AssetRecord,
    config: BenchmarkConfig,
) -> bool:
    left_interval = _event_time_interval(left_start_bin, left_asset, config)
    right_interval = _event_time_interval(right_start_bin, right_asset, config)
    overlap_sec = _interval_overlap_sec(left_interval, right_interval)
    if overlap_sec <= 0.0:
        return True
    if _interval_contains(left_interval, right_interval) or _interval_contains(right_interval, left_interval):
        return True
    return (
        overlap_sec >= TRACK_1B_MIN_PARTIAL_OVERLAP_RATE * left_asset.duration_sec
        and overlap_sec >= TRACK_1B_MIN_PARTIAL_OVERLAP_RATE * right_asset.duration_sec
    )


def _all_partial_overlaps_meet_min_rate(
    events: Sequence[tuple[str, AssetRecord, int, int]],
    config: BenchmarkConfig,
) -> bool:
    for left_index, (_left_label, left_asset, left_start_bin, _left_end_bin) in enumerate(events):
        for _right_label, right_asset, right_start_bin, _right_end_bin in events[left_index + 1 :]:
            if not _partial_overlap_meets_min_rate(
                left_start_bin,
                left_asset,
                right_start_bin,
                right_asset,
                config,
            ):
                return False
    return True


def _normalize_audio_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return np.asarray(audio / peak, dtype=np.float32)
    return audio


def _event_has_multibin_span(spans: dict[str, dict[str, int]]) -> bool:
    return any(span["end_bin"] > span["start_bin"] for span in spans.values())


def _raster_has_overlap(raster: Sequence[Sequence[str]]) -> bool:
    return any(len(bin_labels) > 1 for bin_labels in raster)


def _raster_max_overlap_ok(raster: Sequence[Sequence[str]], max_overlap: int) -> bool:
    return all(len(bin_labels) <= max_overlap for bin_labels in raster)


def _non_overlapping_consecutive_pairs(
    sequence: Sequence[str],
    spans: dict[str, dict[str, int]],
) -> list[tuple[str, str]]:
    return [
        (left, right)
        for left, right in zip(sequence[:-1], sequence[1:])
        if not _spans_overlap(spans[left], spans[right])
    ]


def _build_episode_layout(
    assets_by_label: dict[str, list[AssetRecord]],
    labels: Sequence[str],
    rng: random.Random,
    config: BenchmarkConfig,
) -> tuple[list[tuple[str, AssetRecord, int, int]], list[list[str]], dict[str, dict[str, int]]]:
    track_id = _validate_track_id(config.track_id)
    if track_id == TRACK_1A:
        event_count = rng.randint(config.min_events, config.max_events)
        selected_labels = rng.sample(list(labels), event_count)
        event_bins = sorted(rng.sample(list(range(config.num_bins)), event_count))
        raster: list[list[str]] = [[] for _ in range(config.num_bins)]
        spans: dict[str, dict[str, int]] = {}
        events: list[tuple[str, AssetRecord, int, int]] = []
        for label, start_bin in sorted(zip(selected_labels, event_bins), key=lambda item: item[1]):
            asset = rng.choice(assets_by_label[label])
            raster[start_bin].append(label)
            spans[label] = {"start_bin": start_bin, "end_bin": start_bin}
            events.append((label, asset, start_bin, start_bin))
        return events, [sorted(bin_labels) for bin_labels in raster], spans

    for _attempt in range(TRACK_1B_MAX_PLACEMENT_ATTEMPTS):
        event_count = rng.randint(config.min_events, config.max_events)
        selected_labels = rng.sample(list(labels), event_count)
        selected_assets = {label: rng.choice(assets_by_label[label]) for label in selected_labels}
        spans: dict[str, dict[str, int]] = {}
        raster: list[list[str]] = [[] for _ in range(config.num_bins)]
        used_starts: set[int] = set()
        events: list[tuple[str, AssetRecord, int, int]] = []

        labels_by_span = sorted(
            selected_labels,
            key=lambda label: _span_bin_count(selected_assets[label], config),
            reverse=True,
        )
        placement_ok = True
        for label in labels_by_span:
            asset = selected_assets[label]
            span_bins = _span_bin_count(asset, config)
            if span_bins > config.num_bins:
                placement_ok = False
                break
            candidate_starts = list(range(config.num_bins - span_bins + 1))
            candidate_starts = [
                start_bin
                for start_bin in candidate_starts
                if _event_bins_meet_min_coverage(start_bin, asset, config)
            ]
            rng.shuffle(candidate_starts)
            placed = False
            for start_bin in candidate_starts:
                if start_bin in used_starts:
                    continue
                end_bin = start_bin + span_bins - 1
                if any(len(raster[bin_index]) >= TRACK_1B_MAX_OVERLAP for bin_index in range(start_bin, end_bin + 1)):
                    continue
                used_starts.add(start_bin)
                for bin_index in range(start_bin, end_bin + 1):
                    raster[bin_index].append(label)
                spans[label] = {"start_bin": start_bin, "end_bin": end_bin}
                events.append((label, asset, start_bin, end_bin))
                placed = True
                break
            if not placed:
                placement_ok = False
                break

        if not placement_ok:
            continue
        raster = [sorted(bin_labels) for bin_labels in raster]
        sequence = [
            label
            for label, _span in sorted(spans.items(), key=lambda item: (item[1]["start_bin"], item[0]))
        ]
        if not _event_has_multibin_span(spans):
            continue
        if not _raster_has_overlap(raster):
            continue
        if not _raster_max_overlap_ok(raster, TRACK_1B_MAX_OVERLAP):
            continue
        if not _all_partial_overlaps_meet_min_rate(events, config):
            continue
        if not _non_overlapping_consecutive_pairs(sequence, spans):
            continue
        events.sort(key=lambda item: (item[2], item[0]))
        return events, raster, spans

    raise RuntimeError(
        f"Unable to generate a valid track {track_id} episode after "
        f"{TRACK_1B_MAX_PLACEMENT_ATTEMPTS} attempts."
    )


def build_dataset(
    audio_root: Path,
    output_root: Path,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    config = config or BenchmarkConfig()
    track_id = _validate_track_id(config.track_id)
    if validate_time_mode(config.time_mode) == TIME_MODE_CONTINUOUS:
        from .continuous_tracks import build_continuous_dataset

        return build_continuous_dataset(audio_root=audio_root, output_root=output_root, config=config)
    root = Path(output_root)
    assets = scan_source_assets(Path(audio_root), config)
    assets_by_label: dict[str, list[AssetRecord]] = {}
    for asset in assets:
        assets_by_label.setdefault(asset.label, []).append(asset)

    rng = random.Random(config.seed)
    labels = sorted(assets_by_label)
    samplerate = assets[0].samplerate
    frames_per_bin = int(round(config.bin_duration_sec * samplerate))
    total_frames = int(round(config.episode_duration_sec * samplerate))
    dataset_dir = root / "dataset"
    episodes_dir = dataset_dir / "episodes"
    manifest_path = dataset_dir / "manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    progress = ProgressReporter(config.num_episodes, label="dataset-build")

    episodes: list[dict[str, Any]] = []
    vocabulary = sorted(labels)
    for episode_index in range(config.num_episodes):
        layout_events, raster, spans = _build_episode_layout(assets_by_label, labels, rng, config)
        event_count = len(layout_events)
        audio = np.zeros(total_frames, dtype=np.float32)
        bin_assignments: list[dict[str, Any]] = []
        asset_provenance: dict[str, dict[str, Any]] = {}

        for label, asset, start_bin, end_bin in layout_events:
            clip = _read_mono_audio(asset.path, samplerate)
            start_frame = start_bin * frames_per_bin
            end_frame = start_frame + clip.shape[0]
            _assert(end_frame <= total_frames, f"Event {label} exceeds episode duration in {track_id}.")
            audio[start_frame:end_frame] += clip
            bin_assignments.append({"label": label, "start_bin": start_bin, "end_bin": end_bin})
            asset_provenance[label] = {
                "path": _relative_to_root(asset.path, root),
                "category": asset.category,
                "duration_sec": round(asset.duration_sec, 6),
            }
        audio = _normalize_audio_peak(audio)

        sequence = [item["label"] for item in sorted(bin_assignments, key=lambda item: item["start_bin"])]
        pair_candidates = [
            {"label_a": label_a, "label_b": label_b}
            for index, label_a in enumerate(sequence)
            for label_b in sequence[index + 1 :]
        ]
        pair_queries = rng.sample(pair_candidates, 2)
        localization_label = rng.choice(sequence)
        consecutive_pairs = _non_overlapping_consecutive_pairs(sequence, spans)
        gap_relations = [
            {
                "left_label": left,
                "right_label": right,
                "relation": _gap_answer(spans[left], spans[right]),
            }
            for left, right in consecutive_pairs
        ]
        gap_query = rng.choice(gap_relations) if gap_relations else None

        episode_id = f"episode_{episode_index:03d}"
        episode_audio_path = episodes_dir / f"{episode_id}.wav"
        _write_audio(episode_audio_path, audio, samplerate)

        behavioral_queries = {
            "sequence": {
                "query_id": "sequence_1",
                "prompt": "Recover the left-to-right ordered event sequence.",
            },
            "pairwise_order": [
                {
                    "query_id": f"pairwise_{query_index + 1}",
                    "label_a": pair["label_a"],
                    "label_b": pair["label_b"],
                }
                for query_index, pair in enumerate(pair_queries)
            ],
            "localization": [
                {
                    "query_id": "localization_1",
                    "label": localization_label,
                }
            ],
            "gap_adjacency": (
                [
                    {
                        "query_id": "gap_1",
                        "left_label": gap_query["left_label"],
                        "right_label": gap_query["right_label"],
                    }
                ]
                if gap_query
                else []
            ),
        }
        behavioral_answers = {
            "sequence": {
                "query_id": "sequence_1",
                "answer": _sequence_answer(sequence, config),
            },
            "pairwise_order": {
                query["query_id"]: _pairwise_answer(spans, query["label_a"], query["label_b"])
                for query in behavioral_queries["pairwise_order"]
            },
            "localization": {
                "localization_1": _localization_answer(spans[localization_label]),
            },
            "gap_adjacency": {"gap_1": gap_query["relation"]} if gap_query else {},
        }

        episodes.append(
            {
                "episode_id": episode_id,
                "audio_path": _relative_to_root(episode_audio_path, root),
                "regime_variants": _build_regime_variants(sequence, config),
                "labels_present": sequence,
                "event_count": event_count,
                "event_bin_assignments": bin_assignments,
                "ground_truth_raster": raster,
                "ground_truth_sequence": sequence,
                "ground_truth_spans": spans,
                "ground_truth_gap_relations": gap_relations,
                "behavioral_queries": behavioral_queries,
                "behavioral_answers": behavioral_answers,
                "asset_provenance": asset_provenance,
                "build_seed": config.seed,
            }
        )
        progress.update(episode_index + 1, f"episode={episode_id} events={event_count}")

    manifest = {
        "benchmark_defaults": config.to_manifest_dict(),
        "vocabulary": vocabulary,
        "dataset_root": _relative_to_root(dataset_dir, root),
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def parse_action(raw_text: str) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text)
    if stripped == "STOP":
        return {"format_valid": True, "action": "STOP"}
    match = re.fullmatch(r"QUERY\s+(\d+)\s+(\d+)", stripped)
    if not match:
        return {"format_valid": False, "reason": "invalid_action_format"}
    return {
        "format_valid": True,
        "action": "QUERY",
        "start_bin": int(match.group(1)),
        "length_bin": int(match.group(2)),
    }


def validate_query(parsed: dict[str, Any], config: BenchmarkConfig, listened_bin_total: int) -> tuple[bool, str]:
    start_bin = parsed["start_bin"]
    length_bin = parsed["length_bin"]
    if length_bin < 1:
        return False, "length_bin_must_be_positive"
    if length_bin > config.max_query_bins:
        return False, "length_bin_exceeds_max_query_bins"
    if start_bin < 0 or start_bin >= config.num_bins:
        return False, "start_bin_out_of_range"
    if start_bin + length_bin > config.num_bins:
        return False, "query_range_out_of_range"
    if listened_bin_total + length_bin > config.max_total_queried_bins:
        return False, "query_exceeds_total_bin_budget"
    return True, ""


def _format_raster_bin(bin_state: list[str] | None) -> str:
    if bin_state is None:
        return "??"
    if not bin_state:
        return "00"
    return " ".join(bin_state)


def _vertical_raster_scaffold(num_bins: int) -> str:
    return "\n".join(f"{index}: ??" for index in range(num_bins))


def parse_raster(raw_text: str, vocabulary: Sequence[str], num_bins: int) -> dict[str, Any]:
    text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    lines = text.splitlines()
    if len(lines) < num_bins:
        return {"format_valid": False, "reason": "wrong_row_count", "bins": None, "tokens": None}
    lines = lines[:num_bins]

    allowed = set(vocabulary)
    bins: list[list[str] | None] = []
    seen_indices: set[int] = set()
    for expected_index, line in enumerate(lines):
        match = re.fullmatch(r"\s*(\d+)\s*:\s*(.*?)\s*", line)
        if not match:
            return {"format_valid": False, "reason": "invalid_raster_row", "bins": None, "tokens": None}

        bin_index = int(match.group(1))
        if bin_index in seen_indices:
            return {"format_valid": False, "reason": "duplicate_bin_index", "bins": None, "tokens": None}
        seen_indices.add(bin_index)
        if bin_index != expected_index:
            return {"format_valid": False, "reason": "bin_index_out_of_order", "bins": None, "tokens": None}

        contents = match.group(2).strip()
        if not contents:
            return {"format_valid": False, "reason": "empty_raster_bin", "bins": None, "tokens": None}
        tokens = contents.split()
        special_tokens = {"00", "??"} & set(tokens)
        if special_tokens:
            if len(tokens) != 1:
                return {"format_valid": False, "reason": "mixed_special_raster_token", "bins": None, "tokens": None}
            bins.append(None if tokens[0] == "??" else [])
            continue
        if any(token not in allowed for token in tokens):
            return {"format_valid": False, "reason": "invalid_raster_token", "bins": None, "tokens": None}
        if len(set(tokens)) != len(tokens):
            return {"format_valid": False, "reason": "duplicate_raster_label", "bins": None, "tokens": None}
        bins.append(sorted(tokens))

    return {
        "format_valid": True,
        "bins": bins,
        "tokens": [_format_raster_bin(bin_state) for bin_state in bins],
    }


def parse_sequence_answer(raw_text: str, vocabulary: Sequence[str]) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text)
    tokens = [token.strip() for token in stripped.split(SEQUENCE_DELIMITER)]
    if any(not token for token in tokens):
        return {"format_valid": False, "reason": "empty_sequence_token"}
    allowed = set(vocabulary)
    if any(token not in allowed for token in tokens):
        return {"format_valid": False, "reason": "unknown_sequence_token"}
    return {"format_valid": True, "tokens": tokens}


def parse_pairwise_answer(raw_text: str, allow_unexplored: bool = False) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).upper()
    allowed = {"BEFORE", "AFTER"}
    if allow_unexplored:
        allowed.add("UNEXPLORED")
    if stripped not in allowed:
        return {"format_valid": False, "reason": "invalid_pairwise_answer"}
    return {"format_valid": True, "answer": stripped}


def parse_localization_answer(raw_text: str, num_bins: int, allow_unexplored: bool = False) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).upper()
    if allow_unexplored and stripped == "UNEXPLORED":
        return {"format_valid": True, "answer": "UNEXPLORED"}
    match = re.fullmatch(r"START\s+(\d+)\s+END\s+(\d+)", stripped)
    if not match:
        return {"format_valid": False, "reason": "invalid_localization_answer"}
    start_bin = int(match.group(1))
    end_bin = int(match.group(2))
    if start_bin < 0 or end_bin < 0 or start_bin >= num_bins or end_bin >= num_bins or end_bin < start_bin:
        return {"format_valid": False, "reason": "localization_out_of_range"}
    return {"format_valid": True, "start_bin": start_bin, "end_bin": end_bin}


def parse_gap_answer(raw_text: str, allow_unexplored: bool = False) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).upper()
    allowed = {"ADJACENT", "GAP"}
    if allow_unexplored:
        allowed.add("UNEXPLORED")
    if stripped not in allowed:
        return {"format_valid": False, "reason": "invalid_gap_answer"}
    return {"format_valid": True, "answer": stripped}


def _spans_from_raster(bins: Sequence[Sequence[str] | None]) -> dict[str, dict[str, int]]:
    label_bins: dict[str, list[int]] = {}
    for bin_index, bin_state in enumerate(bins):
        for label in bin_state or []:
            label_bins.setdefault(label, []).append(bin_index)
    return {
        label: {"start_bin": min(indices), "end_bin": max(indices)}
        for label, indices in label_bins.items()
    }


def _sequence_from_spans(spans: dict[str, dict[str, int]]) -> list[str]:
    return [
        label
        for label, _ in sorted(spans.items(), key=lambda item: (item[1]["start_bin"], item[0]))
    ]


def score_raster(
    predicted_bins: Sequence[Sequence[str] | None],
    ground_truth_bins: Sequence[Sequence[str]],
    observed_bins: set[int],
) -> dict[str, float]:
    scores: list[float] = []
    for index, (predicted, gold) in enumerate(zip(predicted_bins, ground_truth_bins)):
        if predicted is not None and set(predicted) == set(gold):
            scores.append(1.0)
        elif predicted is None and index not in observed_bins:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return {"bin_score_mean": float(sum(scores) / len(scores))}


def score_belief_metrics(
    final_raster: dict[str, Any] | None,
    episode: dict[str, Any],
    observed_bins: set[int],
) -> dict[str, float]:
    metrics = {
        "belief_bin_score_mean": 0.0,
        "belief_sequence_exact": 0.0,
        "belief_pairwise_accuracy": 0.0,
        "belief_localization_exact": 0.0,
        "belief_localization_mean_iou": 0.0,
        "belief_gap_accuracy": 0.0,
    }
    if not final_raster or not final_raster.get("format_valid"):
        return metrics

    bins = final_raster["bins"]
    metrics["belief_bin_score_mean"] = score_raster(bins, episode["ground_truth_raster"], observed_bins)[
        "bin_score_mean"
    ]
    predicted_spans = _spans_from_raster(bins)
    predicted_sequence = _sequence_from_spans(predicted_spans)
    metrics["belief_sequence_exact"] = float(predicted_sequence == episode["ground_truth_sequence"])

    pair_scores: list[float] = []
    true_spans = episode["ground_truth_spans"]
    sequence = episode["ground_truth_sequence"]
    for index, label_a in enumerate(sequence):
        for label_b in sequence[index + 1 :]:
            if label_a not in predicted_spans or label_b not in predicted_spans:
                pair_scores.append(0.0)
            else:
                pred_answer = _pairwise_answer(predicted_spans, label_a, label_b)
                gold_answer = _pairwise_answer(true_spans, label_a, label_b)
                pair_scores.append(float(pred_answer == gold_answer))
    if pair_scores:
        metrics["belief_pairwise_accuracy"] = float(sum(pair_scores) / len(pair_scores))

    exact_scores: list[float] = []
    ious: list[float] = []
    for label, gold_span in true_spans.items():
        pred_span = predicted_spans.get(label)
        if pred_span is None:
            exact_scores.append(0.0)
            ious.append(0.0)
            continue
        exact_scores.append(float(pred_span == gold_span))
        pred_bins = set(range(pred_span["start_bin"], pred_span["end_bin"] + 1))
        gold_bins = set(range(gold_span["start_bin"], gold_span["end_bin"] + 1))
        intersection = len(pred_bins & gold_bins)
        union = len(pred_bins | gold_bins)
        ious.append(float(intersection / union) if union else 0.0)
    if exact_scores:
        metrics["belief_localization_exact"] = float(sum(exact_scores) / len(exact_scores))
        metrics["belief_localization_mean_iou"] = float(sum(ious) / len(ious))

    gap_scores: list[float] = []
    for relation in episode["ground_truth_gap_relations"]:
        left = relation["left_label"]
        right = relation["right_label"]
        if left not in predicted_spans or right not in predicted_spans:
            gap_scores.append(0.0)
            continue
        pred_relation = _gap_answer(predicted_spans[left], predicted_spans[right])
        gap_scores.append(float(pred_relation == relation["relation"]))
    if gap_scores:
        metrics["belief_gap_accuracy"] = float(sum(gap_scores) / len(gap_scores))
    return metrics


def _event_is_explored(episode: dict[str, Any], label: str, observed_bins: set[int]) -> bool:
    span = episode["ground_truth_spans"][label]
    return any(bin_index in observed_bins for bin_index in range(span["start_bin"], span["end_bin"] + 1))


def _explored_events_in_order(episode: dict[str, Any], observed_bins: set[int]) -> list[str]:
    return [
        label for label in episode["ground_truth_sequence"] if _event_is_explored(episode, label, observed_bins)
    ]


def _empty_task_output(reason: str = "not_evaluable") -> dict[str, Any]:
    return {
        "asked": False,
        "prompt": "",
        "raw_output": "",
        "parsed": {"format_valid": False, "reason": reason},
        "repair_attempts": [],
    }


def _build_behavioral_plan(episode: dict[str, Any], observed_bins: set[int]) -> dict[str, Any]:
    sequence = list(episode["ground_truth_sequence"])
    explored_events = _explored_events_in_order(episode, observed_bins)
    explored_set = set(explored_events)
    unexplored_events = [label for label in sequence if label not in explored_set]
    ordered_pairs = [
        (label_a, label_b)
        for index, label_a in enumerate(sequence)
        for label_b in sequence[index + 1 :]
    ]
    consecutive_pairs = _non_overlapping_consecutive_pairs(sequence, episode["ground_truth_spans"])

    core_sequence = None
    if explored_events:
        core_sequence = {
            "query_id": "core_sequence_1",
            "answer": SEQUENCE_DELIMITER.join(explored_events),
            "labels": list(explored_events),
        }

    core_pairwise = []
    if len(explored_events) >= 2:
        label_a = explored_events[0]
        label_b = explored_events[-1]
        core_pairwise.append(
            {
                "query_id": "core_pairwise_1",
                "label_a": label_a,
                "label_b": label_b,
                "answer": _pairwise_answer(episode["ground_truth_spans"], label_a, label_b),
            }
        )

    core_localization = []
    if explored_events:
        label = explored_events[0]
        core_localization.append(
            {
                "query_id": "core_localization_1",
                "label": label,
                "answer": _localization_answer(episode["ground_truth_spans"][label]),
            }
        )

    core_gap = []
    for left_label, right_label in consecutive_pairs:
        if left_label in explored_set and right_label in explored_set:
            core_gap.append(
                {
                    "query_id": "core_gap_1",
                    "left_label": left_label,
                    "right_label": right_label,
                    "answer": _gap_answer(
                        episode["ground_truth_spans"][left_label],
                        episode["ground_truth_spans"][right_label],
                    ),
                }
            )
            break

    uncertainty_pairwise = []
    for label_a, label_b in ordered_pairs:
        if label_a not in explored_set or label_b not in explored_set:
            uncertainty_pairwise.append(
                {
                    "query_id": "uncertainty_pairwise_1",
                    "label_a": label_a,
                    "label_b": label_b,
                    "answer": "UNEXPLORED",
                }
            )
            break

    uncertainty_localization = []
    if unexplored_events:
        uncertainty_localization.append(
            {
                "query_id": "uncertainty_localization_1",
                "label": unexplored_events[0],
                "answer": "UNEXPLORED",
            }
        )

    uncertainty_gap = []
    for left_label, right_label in consecutive_pairs:
        if left_label not in explored_set or right_label not in explored_set:
            uncertainty_gap.append(
                {
                    "query_id": "uncertainty_gap_1",
                    "left_label": left_label,
                    "right_label": right_label,
                    "answer": "UNEXPLORED",
                }
            )
            break

    coverage_metrics = {
        "event_coverage_rate": float(len(explored_events) / len(sequence)) if sequence else 0.0,
        "event_coverage_count": len(explored_events),
        "pairwise_core_evaluable": float(bool(core_pairwise)),
        "localization_core_evaluable": float(bool(core_localization)),
        "gap_core_evaluable": float(bool(core_gap)),
    }

    return {
        "explored_events": explored_events,
        "unexplored_events": unexplored_events,
        "coverage_metrics": coverage_metrics,
        "core": {
            "sequence": core_sequence,
            "pairwise_order": core_pairwise,
            "localization": core_localization,
            "gap_adjacency": core_gap,
        },
        "uncertainty": {
            "pairwise_order": uncertainty_pairwise,
            "localization": uncertainty_localization,
            "gap_adjacency": uncertainty_gap,
        },
    }


def _score_behavioral_outputs(outputs: dict[str, Any], plan: dict[str, Any], prefix: str) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        f"{prefix}_behavior_sequence_exact": None,
        f"{prefix}_behavior_pairwise_accuracy": None,
        f"{prefix}_behavior_localization_exact": None,
        f"{prefix}_behavior_localization_iou": None,
        f"{prefix}_behavior_gap_accuracy": None,
    }

    sequence_query = plan.get("sequence")
    sequence_output = outputs.get("sequence")
    if sequence_query:
        if sequence_output and sequence_output.get("parsed", {}).get("format_valid"):
            expected_tokens = [token.strip() for token in str(sequence_query["answer"]).split(SEQUENCE_DELIMITER)]
            metrics[f"{prefix}_behavior_sequence_exact"] = float(sequence_output["parsed"]["tokens"] == expected_tokens)
        else:
            metrics[f"{prefix}_behavior_sequence_exact"] = 0.0

    pair_queries = plan.get("pairwise_order", [])
    pair_outputs = outputs.get("pairwise_order", {})
    if pair_queries:
        pair_scores = []
        for query in pair_queries:
            parsed = pair_outputs.get(query["query_id"], {}).get("parsed", {})
            pair_scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_pairwise_accuracy"] = float(sum(pair_scores) / len(pair_scores))

    localization_queries = plan.get("localization", [])
    localization_outputs = outputs.get("localization", {})
    if localization_queries:
        exact_scores: list[float] = []
        iou_scores: list[float] = []
        for query in localization_queries:
            parsed = localization_outputs.get(query["query_id"], {}).get("parsed", {})
            if not parsed.get("format_valid"):
                exact_scores.append(0.0)
                iou_scores.append(0.0)
                continue
            if query["answer"] == "UNEXPLORED":
                score = float(parsed.get("answer") == "UNEXPLORED")
                exact_scores.append(score)
                iou_scores.append(score)
                continue
            gold_answer = parse_localization_answer(query["answer"], 10)
            pred_span = {"start_bin": parsed["start_bin"], "end_bin": parsed["end_bin"]}
            gold_span = {"start_bin": gold_answer["start_bin"], "end_bin": gold_answer["end_bin"]}
            exact_scores.append(float(pred_span == gold_span))
            pred_bins = set(range(pred_span["start_bin"], pred_span["end_bin"] + 1))
            gold_bins = set(range(gold_span["start_bin"], gold_span["end_bin"] + 1))
            intersection = len(pred_bins & gold_bins)
            union = len(pred_bins | gold_bins)
            iou_scores.append(float(intersection / union) if union else 0.0)
        metrics[f"{prefix}_behavior_localization_exact"] = float(sum(exact_scores) / len(exact_scores))
        metrics[f"{prefix}_behavior_localization_iou"] = float(sum(iou_scores) / len(iou_scores))

    gap_queries = plan.get("gap_adjacency", [])
    gap_outputs = outputs.get("gap_adjacency", {})
    if gap_queries:
        gap_scores = []
        for query in gap_queries:
            parsed = gap_outputs.get(query["query_id"], {}).get("parsed", {})
            gap_scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_gap_accuracy"] = float(sum(gap_scores) / len(gap_scores))

    return metrics


def _score_uncertainty_metrics(outputs: dict[str, Any], plan: dict[str, Any]) -> dict[str, MetricValue]:
    metrics: dict[str, MetricValue] = {
        "uncertainty_pairwise_accuracy": None,
        "uncertainty_localization_accuracy": None,
        "uncertainty_gap_accuracy": None,
        "uncertainty_pairwise_count": float(len(plan.get("pairwise_order", []))),
        "uncertainty_localization_count": float(len(plan.get("localization", []))),
        "uncertainty_gap_count": float(len(plan.get("gap_adjacency", []))),
    }

    if plan.get("pairwise_order"):
        scores = []
        for query in plan["pairwise_order"]:
            parsed = outputs.get("pairwise_order", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics["uncertainty_pairwise_accuracy"] = float(sum(scores) / len(scores))

    if plan.get("localization"):
        scores = []
        for query in plan["localization"]:
            parsed = outputs.get("localization", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics["uncertainty_localization_accuracy"] = float(sum(scores) / len(scores))

    if plan.get("gap_adjacency"):
        scores = []
        for query in plan["gap_adjacency"]:
            parsed = outputs.get("gap_adjacency", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics["uncertainty_gap_accuracy"] = float(sum(scores) / len(scores))

    return metrics


class EpisodeEnvironment:
    def __init__(self, root: Path, episode: dict[str, Any], config: BenchmarkConfig, run_dir: Path) -> None:
        self.root = root
        self.episode = episode
        self.config = config
        self.run_dir = run_dir
        self.audio_path = root / episode["audio_path"]
        self.audio, self.samplerate = sf.read(str(self.audio_path), dtype="float32", always_2d=False)
        if isinstance(self.audio, np.ndarray) and self.audio.ndim == 2:
            self.audio = self.audio.mean(axis=1)
        self.frames_per_bin = int(round(config.bin_duration_sec * self.samplerate))
        self.action_history: list[dict[str, Any]] = []
        self.repair_log: list[dict[str, Any]] = []
        self.query_audio_paths: list[Path] = []
        self.observed_bins: set[int] = set()
        self.listened_bin_total = 0
        self.stop_reason = ""

    def remaining_steps(self) -> int:
        return self.config.max_steps - len(self.action_history)

    def remaining_bin_budget(self) -> int:
        return self.config.max_total_queried_bins - self.listened_bin_total

    def clip_for_query(self, step_index: int, start_bin: int, length_bin: int) -> tuple[Path, dict[str, Any]]:
        start_frame = start_bin * self.frames_per_bin
        end_frame = (start_bin + length_bin) * self.frames_per_bin
        clip = np.asarray(self.audio[start_frame:end_frame], dtype=np.float32)
        clip_path = self.run_dir / "query_audio" / f"{self.episode['episode_id']}_step_{step_index:02d}_{start_bin}_{length_bin}.wav"
        _write_audio(clip_path, clip, self.samplerate)
        self.query_audio_paths.append(clip_path)
        observed = set(range(start_bin, start_bin + length_bin))
        self.observed_bins |= observed
        self.listened_bin_total += length_bin
        metadata = {
            "step_index": step_index,
            "start_bin": start_bin,
            "length_bin": length_bin,
            "end_bin_exclusive": start_bin + length_bin,
            "duration_sec": round(length_bin * self.config.bin_duration_sec, 6),
            "audio_path": str(clip_path),
        }
        self.action_history.append(metadata)
        return clip_path, metadata


def _episode_context_header(episode: dict[str, Any], regime_payload: dict[str, Any], config: BenchmarkConfig, vocabulary: Sequence[str]) -> str:
    return (
        f"Vocabulary: {', '.join(vocabulary)}\n"
        f"Timeline bins: 0 to {config.num_bins - 1}, one second per bin.\n"
        f"Budget: max_steps={config.max_steps}, max_total_queried_bins={config.max_total_queried_bins}, "
        f"max_query_bins={config.max_query_bins}.\n"
        f"{regime_payload['prior_text']}"
    )


def build_action_initial_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> str:
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "You do not have access to the full clip.\n"
        "Return exactly one line:\n"
        "- QUERY <start_bin> <length_bin>\n"
        "- STOP\n"
        "Choose your first action now."
    )


def build_action_followup_prompt(metadata: dict[str, Any], env: EpisodeEnvironment) -> str:
    return (
        f"You inspected bins {metadata['start_bin']} to {metadata['end_bin_exclusive'] - 1}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining queried-bin budget: {env.remaining_bin_budget()}.\n"
        "Return exactly one line:\n"
        "- QUERY <start_bin> <length_bin>\n"
        "- STOP"
    )


def build_action_repair_prompt(reason: str, env: EpisodeEnvironment) -> str:
    return (
        f"Your previous action was invalid: {reason}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining queried-bin budget: {env.remaining_bin_budget()}.\n"
        "Return exactly one valid line now:\n"
        "- QUERY <start_bin> <length_bin>\n"
        "- STOP"
    )


def _query_history_text(action_history: Sequence[dict[str, Any]]) -> str:
    if not action_history:
        return "No intervals have been queried yet."
    lines = [
        f"step {item['step_index']}: bins {item['start_bin']} to {item['end_bin_exclusive'] - 1}"
        for item in action_history
    ]
    return "\n".join(lines)


def _observed_bins_from_history(action_history: Sequence[dict[str, Any]]) -> set[int]:
    return {
        bin_index
        for item in action_history
        for bin_index in range(item["start_bin"], item["end_bin_exclusive"])
    }


def _observed_bins_text(action_history: Sequence[dict[str, Any]], config: BenchmarkConfig) -> str:
    observed_bins = _observed_bins_from_history(action_history)
    if not observed_bins:
        return "No bins have been observed yet."
    return ", ".join(str(index) for index in range(config.num_bins) if index in observed_bins)


def build_probe_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
) -> str:
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "You are revealing your current timeline belief for diagnostic scoring.\n"
        "The attached audio crops are the intervals you have already queried, in chronological order.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Return exactly {config.num_bins} lines, one per bin, with no extra text.\n"
        "Each row must be '<bin_index>: <contents>'.\n"
        "Use only vocabulary labels, 00 for empty/silence, or ?? for unresolved.\n"
        "Multiple labels in the same row mean overlap.\n"
        f"Fill this scaffold:\n{_vertical_raster_scaffold(config.num_bins)}"
    )


def build_same_session_probe_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    probe_kind: str,
) -> str:
    checkpoint_text = "final terminal" if probe_kind == "final" else "periodic checkpoint"
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"This is a {checkpoint_text} belief probe in the same exploration session.\n"
        "Report your current global timeline belief based only on the exploration so far.\n"
        "Use only evidence obtained through queried intervals. Do not assume unqueried intervals are known.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Observed bins: {_observed_bins_text(action_history, config)}\n"
        f"Return exactly {config.num_bins} lines, one per bin, with no extra text.\n"
        "Each row must be '<bin_index>: <contents>'.\n"
        "Use only vocabulary labels, 00 for empty/silence, or ?? for unresolved.\n"
        "Multiple labels in the same row mean overlap.\n"
        f"Fill this scaffold:\n{_vertical_raster_scaffold(config.num_bins)}"
    )


def build_full_context_probe_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> str:
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "You have access to the full 10-second audio clip.\n"
        f"Return exactly {config.num_bins} lines, one per bin, with no extra text.\n"
        "Each row must be '<bin_index>: <contents>'.\n"
        "Use only vocabulary labels, 00 for empty/silence, or ?? for unresolved.\n"
        "Multiple labels in the same row mean overlap.\n"
        f"Fill this scaffold:\n{_vertical_raster_scaffold(config.num_bins)}"
    )


def build_sequence_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    full_context: bool,
    explored_only: bool = False,
) -> str:
    evidence = "the full clip attached below" if full_context else "the attached queried crops"
    task_text = (
        "Return the left-to-right ordered event sequence for only the events that are directly established by the evidence. "
        "Do not guess undiscovered events."
        if explored_only
        else "Return the left-to-right ordered event sequence."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"{task_text} Use exactly this delimiter: {config.sequence_delimiter}\n"
        "Return only one line and nothing else."
    )


def build_pairwise_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    full_context: bool,
    allow_unexplored: bool = False,
) -> str:
    evidence = "the full clip attached below" if full_context else "the attached queried crops"
    answer_text = (
        "Return exactly BEFORE, AFTER, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly BEFORE or AFTER."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Does {query['label_a']} occur BEFORE or AFTER {query['label_b']}?\n"
        f"{answer_text}"
    )


def build_localization_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    full_context: bool,
    allow_unexplored: bool = False,
) -> str:
    evidence = "the full clip attached below" if full_context else "the attached queried crops"
    answer_text = (
        "Return exactly START <i> END <j>, or UNEXPLORED if the evidence does not directly establish this event."
        if allow_unexplored
        else "Return exactly START <i> END <j>."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Give the start and end bins for {query['label']}.\n"
        f"{answer_text}"
    )


def build_gap_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    full_context: bool,
    allow_unexplored: bool = False,
) -> str:
    evidence = "the full clip attached below" if full_context else "the attached queried crops"
    answer_text = (
        "Return exactly ADJACENT, GAP, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly ADJACENT or GAP."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Are {query['left_label']} and {query['right_label']} ADJACENT, or is there a GAP?\n"
        f"{answer_text}"
    )


def build_same_session_sequence_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
) -> str:
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "Use the exploration so far in this same session and the final raster belief you just reported.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Observed bins: {_observed_bins_text(action_history, config)}\n"
        "Return the left-to-right ordered event sequence for only the events that are directly established by the evidence. "
        "Do not guess undiscovered events. "
        f"Use exactly this delimiter: {config.sequence_delimiter}\n"
        "Return only one line and nothing else."
    )


def build_same_session_pairwise_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    allow_unexplored: bool = False,
) -> str:
    answer_text = (
        "Return exactly BEFORE, AFTER, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly BEFORE or AFTER."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "Use the exploration so far in this same session and the final raster belief you just reported.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Observed bins: {_observed_bins_text(action_history, config)}\n"
        f"Does {query['label_a']} occur BEFORE or AFTER {query['label_b']}?\n"
        f"{answer_text}"
    )


def build_same_session_localization_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    allow_unexplored: bool = False,
) -> str:
    answer_text = (
        "Return exactly START <i> END <j>, or UNEXPLORED if the evidence does not directly establish this event."
        if allow_unexplored
        else "Return exactly START <i> END <j>."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "Use the exploration so far in this same session and the final raster belief you just reported.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Observed bins: {_observed_bins_text(action_history, config)}\n"
        f"Give the start and end bins for {query['label']}.\n"
        f"{answer_text}"
    )


def build_same_session_gap_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    query: dict[str, Any],
    allow_unexplored: bool = False,
) -> str:
    answer_text = (
        "Return exactly ADJACENT, GAP, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly ADJACENT or GAP."
    )
    return (
        f"{_episode_context_header(episode, regime_payload, config, vocabulary)}\n"
        "Use the exploration so far in this same session and the final raster belief you just reported.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Observed bins: {_observed_bins_text(action_history, config)}\n"
        f"Are {query['left_label']} and {query['right_label']} ADJACENT, or is there a GAP?\n"
        f"{answer_text}"
    )


def build_repair_prompt(kind: str, reason: str) -> str:
    return (
        f"Your previous {kind} output was invalid: {reason}.\n"
        "Return exactly one valid answer now with no explanation."
    )


def build_probe_repair_prompt(reason: str, config: BenchmarkConfig) -> str:
    return (
        f"Your previous probe output was invalid: {reason}.\n"
        f"Return exactly {config.num_bins} lines, one per bin, with no extra text.\n"
        "Each row must be '<bin_index>: <contents>'.\n"
        "Use only vocabulary labels, 00 for empty/silence, or ?? for unresolved.\n"
        "Multiple labels in the same row mean overlap.\n"
        f"Fill this scaffold:\n{_vertical_raster_scaffold(config.num_bins)}"
    )


def _api_failure_parse(reason: str) -> dict[str, Any]:
    return {"format_valid": False, "reason": reason, "api_error": True}


def _adapter_send(
    adapter: BaseAdapter,
    session: Any,
    prompt_text: str,
    audio_paths: Sequence[Path] | None,
    *,
    preserve_multiline: bool = False,
) -> str:
    kwargs: dict[str, Any] = {"audio_paths": audio_paths}
    try:
        parameters = inspect.signature(adapter.send).parameters
    except (TypeError, ValueError):
        parameters = {}
    if "preserve_multiline" in parameters or any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        kwargs["preserve_multiline"] = bool(preserve_multiline)
    return adapter.send(session, prompt_text, **kwargs)


def _close_adapter_session(adapter: BaseAdapter, session: Any) -> None:
    close_session = getattr(adapter, "close_session", None)
    if isinstance(session, AdapterSession) and close_session is not None:
        close_session(session)


def _send_with_api_retry(
    adapter: BaseAdapter,
    session: Any,
    prompt_text: str,
    audio_paths: Sequence[Path] | None,
    sleep_seconds: float,
    *,
    preserve_multiline: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    hard_attempts = 0
    soft_retries = 0
    attempt_index = 0
    while True:
        attempt_index += 1
        try:
            raw = _adapter_send(
                adapter,
                session,
                prompt_text,
                audio_paths,
                preserve_multiline=preserve_multiline,
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "success",
                    "outcome": "success",
                    "reason": "api_success",
                    "error_class": None,
                    "status_code": None,
                    "message": "",
                    "retry_delay_seconds": 0.0,
                }
            )
            return raw, attempts
        except APIAdapterError as exc:
            attempt: dict[str, Any] = {
                "attempt": attempt_index,
                "status": "error",
                "outcome": "error",
                "reason": "api_error",
                "error": str(exc),
                "message": str(exc),
                "error_class": exc.error_class,
                "status_code": exc.status_code,
                "provider_retry_delay_seconds": exc.retry_after_seconds,
                "retry_delay_seconds": 0.0,
            }
            attempts.append(attempt)
            if exc.error_class == API_ERROR_HARD:
                hard_attempts += 1
                if hard_attempts >= API_HARD_MAX_ATTEMPTS:
                    return None, attempts
                continue
            if soft_retries >= API_SOFT_MAX_RETRIES:
                return None, attempts
            soft_retries += 1
            delay = (
                max(0.0, float(exc.retry_after_seconds))
                if exc.retry_after_seconds is not None
                else max(0.0, float(sleep_seconds))
            )
            bounded_delay = min(delay, API_RETRY_MAX_SLEEP_SECONDS)
            attempt["retry_delay_seconds"] = bounded_delay
            time.sleep(bounded_delay)


def _send_with_one_repair(
    adapter: BaseAdapter,
    session: Any,
    prompt_text: str,
    parse_fn,
    audio_paths: Sequence[Path] | None,
    repair_kind: str,
    *,
    api_retry_sleep_sec: float = 60.0,
    repair_prompt_fn=None,
    preserve_multiline: bool = False,
) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    repair_attempts: list[dict[str, Any]] = []
    api_attempts: list[dict[str, Any]] = []
    raw, initial_api_attempts = _send_with_api_retry(
        adapter,
        session,
        prompt_text,
        audio_paths,
        api_retry_sleep_sec,
        preserve_multiline=preserve_multiline,
    )
    api_attempts.extend(initial_api_attempts)
    if raw is None:
        return "", _api_failure_parse("api_error"), repair_attempts, api_attempts
    parsed = parse_fn(raw)
    if parsed.get("format_valid"):
        return raw, parsed, repair_attempts, api_attempts
    repair_attempts.append({"attempt": 1, "raw_output": raw, "reason": parsed.get("reason", "invalid")})
    repair_prompt = (
        repair_prompt_fn(parsed.get("reason", "invalid"))
        if repair_prompt_fn is not None
        else build_repair_prompt(repair_kind, parsed.get("reason", "invalid"))
    )
    repair_raw, repair_api_attempts = _send_with_api_retry(
        adapter,
        session,
        repair_prompt,
        [],
        api_retry_sleep_sec,
        preserve_multiline=preserve_multiline,
    )
    api_attempts.extend(repair_api_attempts)
    if repair_raw is None:
        return "", _api_failure_parse("api_error"), repair_attempts, api_attempts
    repaired = parse_fn(repair_raw)
    repair_attempts.append(
        {"attempt": 2, "raw_output": repair_raw, "reason": repaired.get("reason", ""), "repair": True}
    )
    return repair_raw, repaired, repair_attempts, api_attempts


def _run_probe_sidecar(
    adapter: BaseAdapter,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
) -> dict[str, Any]:
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    try:
        prompt = build_probe_prompt(episode, regime_payload, config, vocabulary, action_history)
        raw, parsed, repairs, api_attempts = _send_with_one_repair(
            adapter,
            session,
            prompt,
            lambda text: parse_raster(text, vocabulary, config.num_bins),
            audio_paths,
            "probe",
            api_retry_sleep_sec=config.api_retry_sleep_sec,
            repair_prompt_fn=lambda reason: build_probe_repair_prompt(reason, config),
            preserve_multiline=True,
        )
        return {
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
            "repair_attempts": repairs,
            "api_attempts": api_attempts,
        }
    finally:
        _close_adapter_session(adapter, session)


def _run_full_context_probe(
    adapter: BaseAdapter,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    audio_path: Path,
) -> dict[str, Any]:
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    try:
        prompt = build_full_context_probe_prompt(episode, regime_payload, config, vocabulary)
        raw, parsed, repairs, api_attempts = _send_with_one_repair(
            adapter,
            session,
            prompt,
            lambda text: parse_raster(text, vocabulary, config.num_bins),
            [audio_path],
            "probe",
            api_retry_sleep_sec=config.api_retry_sleep_sec,
            repair_prompt_fn=lambda reason: build_probe_repair_prompt(reason, config),
            preserve_multiline=True,
        )
        return {
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
            "repair_attempts": repairs,
            "api_attempts": api_attempts,
        }
    finally:
        _close_adapter_session(adapter, session)


def _run_behavioral_prompts(
    adapter: BaseAdapter,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
    full_context: bool,
) -> dict[str, Any]:
    observed_bins = set(range(config.num_bins)) if full_context else {
        bin_index
        for item in action_history
        for bin_index in range(item["start_bin"], item["end_bin_exclusive"])
    }
    plan = _build_behavioral_plan(episode, observed_bins)

    sequence_session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    core_sequence_query = plan["core"]["sequence"]
    if core_sequence_query is None:
        core_sequence_output = _empty_task_output()
    else:
        sequence_prompt = build_sequence_prompt(
            episode, regime_payload, config, vocabulary, action_history, full_context, explored_only=True
        )
        try:
            sequence_raw, sequence_parsed, sequence_repairs, sequence_api_attempts = _send_with_one_repair(
                adapter,
                sequence_session,
                sequence_prompt,
                lambda text: parse_sequence_answer(text, vocabulary),
                audio_paths,
                "sequence",
                api_retry_sleep_sec=config.api_retry_sleep_sec,
            )
        finally:
            _close_adapter_session(adapter, sequence_session)
        core_sequence_output = {
            "asked": True,
            "prompt": sequence_prompt,
            "raw_output": sequence_raw,
            "parsed": sequence_parsed,
            "repair_attempts": sequence_repairs,
            "api_attempts": sequence_api_attempts,
        }

    def _run_query_map(
        queries: Sequence[dict[str, Any]],
        prompt_builder,
        parse_fn,
        repair_kind: str,
    ) -> dict[str, Any]:
        outputs: dict[str, Any] = {}
        for query in queries:
            session = adapter.create_session(ACTION_SYSTEM_PROMPT)
            prompt = prompt_builder(query)
            try:
                raw, parsed, repairs, api_attempts = _send_with_one_repair(
                    adapter,
                    session,
                    prompt,
                    parse_fn,
                    audio_paths,
                    repair_kind,
                    api_retry_sleep_sec=config.api_retry_sleep_sec,
                )
            finally:
                _close_adapter_session(adapter, session)
            outputs[query["query_id"]] = {
                "asked": True,
                "prompt": prompt,
                "raw_output": raw,
                "parsed": parsed,
                "repair_attempts": repairs,
                "api_attempts": api_attempts,
            }
        return outputs

    core_pairwise_outputs = _run_query_map(
        plan["core"]["pairwise_order"],
        lambda query: build_pairwise_prompt(
            episode, regime_payload, config, vocabulary, action_history, query, full_context
        ),
        lambda text: parse_pairwise_answer(text),
        "pairwise",
    )
    core_localization_outputs = _run_query_map(
        plan["core"]["localization"],
        lambda query: build_localization_prompt(
            episode, regime_payload, config, vocabulary, action_history, query, full_context
        ),
        lambda text: parse_localization_answer(text, config.num_bins),
        "localization",
    )
    core_gap_outputs = _run_query_map(
        plan["core"]["gap_adjacency"],
        lambda query: build_gap_prompt(
            episode, regime_payload, config, vocabulary, action_history, query, full_context
        ),
        lambda text: parse_gap_answer(text),
        "gap",
    )

    uncertainty_pairwise_outputs = _run_query_map(
        plan["uncertainty"]["pairwise_order"],
        lambda query: build_pairwise_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            full_context,
            allow_unexplored=True,
        ),
        lambda text: parse_pairwise_answer(text, allow_unexplored=True),
        "pairwise",
    )
    uncertainty_localization_outputs = _run_query_map(
        plan["uncertainty"]["localization"],
        lambda query: build_localization_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            full_context,
            allow_unexplored=True,
        ),
        lambda text: parse_localization_answer(text, config.num_bins, allow_unexplored=True),
        "localization",
    )
    uncertainty_gap_outputs = _run_query_map(
        plan["uncertainty"]["gap_adjacency"],
        lambda query: build_gap_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            full_context,
            allow_unexplored=True,
        ),
        lambda text: parse_gap_answer(text, allow_unexplored=True),
        "gap",
    )

    core_outputs = {
        "sequence": core_sequence_output,
        "pairwise_order": core_pairwise_outputs,
        "localization": core_localization_outputs,
        "gap_adjacency": core_gap_outputs,
    }
    uncertainty_outputs = {
        "pairwise_order": uncertainty_pairwise_outputs,
        "localization": uncertainty_localization_outputs,
        "gap_adjacency": uncertainty_gap_outputs,
    }
    core_metrics = _score_behavioral_outputs(core_outputs, plan["core"], "core")
    uncertainty_metrics = _score_uncertainty_metrics(uncertainty_outputs, plan["uncertainty"])

    return {
        "behavioral_protocol_version": BEHAVIORAL_PROTOCOL_VERSION,
        "coverage_metrics": plan["coverage_metrics"],
        "core_behavioral_plan": plan["core"],
        "uncertainty_plan": plan["uncertainty"],
        "core_behavioral_outputs": core_outputs,
        "core_behavioral_metrics": core_metrics,
        "uncertainty_outputs": uncertainty_outputs,
        "uncertainty_metrics": uncertainty_metrics,
        "behavioral_outputs": core_outputs,
        "behavioral_metrics": {
            "behavior_sequence_exact": core_metrics["core_behavior_sequence_exact"],
            "behavior_pairwise_accuracy": core_metrics["core_behavior_pairwise_accuracy"],
            "behavior_localization_exact": core_metrics["core_behavior_localization_exact"],
            "behavior_localization_iou": core_metrics["core_behavior_localization_iou"],
            "behavior_gap_accuracy": core_metrics["core_behavior_gap_accuracy"],
        },
    }


def _prefix_metric_keys(metrics: dict[str, MetricValue], prefix: str) -> dict[str, MetricValue]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}


def _run_same_session_probe(
    adapter: BaseAdapter,
    session: Any,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
    probe_kind: str,
) -> dict[str, Any]:
    prompt = build_same_session_probe_prompt(
        episode,
        regime_payload,
        config,
        vocabulary,
        action_history,
        probe_kind,
    )
    raw, parsed, repairs, api_attempts = _send_with_one_repair(
        adapter,
        session,
        prompt,
        lambda text: parse_raster(text, vocabulary, config.num_bins),
        audio_paths,
        "probe",
        api_retry_sleep_sec=config.api_retry_sleep_sec,
        repair_prompt_fn=lambda reason: build_probe_repair_prompt(reason, config),
        preserve_multiline=True,
    )
    observed_bins = _observed_bins_from_history(action_history)
    return {
        "probe_kind": probe_kind,
        "checkpoint_step": len(action_history),
        "observed_bins": sorted(observed_bins),
        "listened_bin_total": sum(int(item["length_bin"]) for item in action_history),
        "prompt": prompt,
        "raw_output": raw,
        "parsed": parsed,
        "repair_attempts": repairs,
        "api_attempts": api_attempts,
        "belief_metrics": score_belief_metrics(parsed, episode, observed_bins),
    }


def _run_same_session_behavioral_prompts(
    adapter: BaseAdapter,
    session: Any,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    observed_bins = _observed_bins_from_history(action_history)
    plan = _build_behavioral_plan(episode, observed_bins)
    core_outputs = {
        "sequence": _empty_task_output(),
        "pairwise_order": {},
        "localization": {},
        "gap_adjacency": {},
    }
    uncertainty_outputs = {
        "pairwise_order": {},
        "localization": {},
        "gap_adjacency": {},
    }

    def _run_one(prompt: str, parse_fn, repair_kind: str) -> dict[str, Any]:
        raw, parsed, repairs, api_attempts = _send_with_one_repair(
            adapter,
            session,
            prompt,
            parse_fn,
            [],
            repair_kind,
            api_retry_sleep_sec=config.api_retry_sleep_sec,
        )
        return {
            "asked": True,
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
            "repair_attempts": repairs,
            "api_attempts": api_attempts,
        }

    for query in plan["core"]["localization"]:
        prompt = build_same_session_localization_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
        )
        core_outputs["localization"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_localization_answer(text, config.num_bins),
            "localization",
        )

    for query in plan["core"]["pairwise_order"]:
        prompt = build_same_session_pairwise_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
        )
        core_outputs["pairwise_order"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_pairwise_answer(text),
            "pairwise",
        )

    for query in plan["core"]["gap_adjacency"]:
        prompt = build_same_session_gap_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
        )
        core_outputs["gap_adjacency"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_gap_answer(text),
            "gap",
        )

    if plan["core"]["sequence"] is not None:
        prompt = build_same_session_sequence_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
        )
        core_outputs["sequence"] = _run_one(
            prompt,
            lambda text: parse_sequence_answer(text, vocabulary),
            "sequence",
        )

    for query in plan["uncertainty"]["localization"]:
        prompt = build_same_session_localization_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            allow_unexplored=True,
        )
        uncertainty_outputs["localization"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_localization_answer(text, config.num_bins, allow_unexplored=True),
            "localization",
        )

    for query in plan["uncertainty"]["pairwise_order"]:
        prompt = build_same_session_pairwise_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            allow_unexplored=True,
        )
        uncertainty_outputs["pairwise_order"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_pairwise_answer(text, allow_unexplored=True),
            "pairwise",
        )

    for query in plan["uncertainty"]["gap_adjacency"]:
        prompt = build_same_session_gap_prompt(
            episode,
            regime_payload,
            config,
            vocabulary,
            action_history,
            query,
            allow_unexplored=True,
        )
        uncertainty_outputs["gap_adjacency"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_gap_answer(text, allow_unexplored=True),
            "gap",
        )

    core_metrics = _score_behavioral_outputs(core_outputs, plan["core"], "same_session_core")
    uncertainty_metrics = _prefix_metric_keys(
        _score_uncertainty_metrics(uncertainty_outputs, plan["uncertainty"]),
        "same_session_",
    )
    behavioral_metrics = {
        "same_session_behavior_sequence_exact": core_metrics["same_session_core_behavior_sequence_exact"],
        "same_session_behavior_pairwise_accuracy": core_metrics["same_session_core_behavior_pairwise_accuracy"],
        "same_session_behavior_localization_exact": core_metrics["same_session_core_behavior_localization_exact"],
        "same_session_behavior_localization_iou": core_metrics["same_session_core_behavior_localization_iou"],
        "same_session_behavior_gap_accuracy": core_metrics["same_session_core_behavior_gap_accuracy"],
    }

    return {
        "behavioral_protocol_version": BEHAVIORAL_PROTOCOL_VERSION,
        "coverage_metrics": plan["coverage_metrics"],
        "same_session_core_behavioral_plan": plan["core"],
        "same_session_uncertainty_plan": plan["uncertainty"],
        "same_session_core_behavioral_outputs": core_outputs,
        "same_session_core_behavioral_metrics": core_metrics,
        "same_session_uncertainty_outputs": uncertainty_outputs,
        "same_session_uncertainty_metrics": uncertainty_metrics,
        "same_session_behavioral_outputs": core_outputs,
        "same_session_behavioral_metrics": behavioral_metrics,
    }


def _format_metrics(
    action_attempts: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    core_behavioral: dict[str, Any],
    uncertainty_behavioral: dict[str, Any],
) -> dict[str, MetricValue]:
    action_format = [float(item["final_parse"].get("format_valid", False)) for item in action_attempts]
    action_semantic = [float(item.get("semantic_valid", False)) for item in action_attempts]
    probe_valid = [float(item["parsed"].get("format_valid", False)) for item in probes]
    repair_attempts = 0
    repair_successes = 0
    for item in action_attempts:
        if item.get("repair_attempted"):
            repair_attempts += 1
            repair_successes += int(item["final_parse"].get("format_valid", False) and item.get("semantic_valid", False))
    for probe in probes:
        if probe["repair_attempts"]:
            repair_attempts += 1
            repair_successes += int(probe["parsed"].get("format_valid", False))
    task_groups = [
        core_behavioral["sequence"],
        *core_behavioral["pairwise_order"].values(),
        *core_behavioral["localization"].values(),
        *core_behavioral["gap_adjacency"].values(),
        *uncertainty_behavioral["pairwise_order"].values(),
        *uncertainty_behavioral["localization"].values(),
        *uncertainty_behavioral["gap_adjacency"].values(),
    ]
    for task_group in task_groups:
        if not task_group.get("asked", True):
            continue
        if task_group["repair_attempts"]:
            repair_attempts += 1
            repair_successes += int(task_group["parsed"].get("format_valid", False))

    return {
        "action_format_valid_rate": float(sum(action_format) / len(action_format)) if action_format else None,
        "action_semantic_valid_rate": float(sum(action_semantic) / len(action_semantic)) if action_semantic else None,
        "probe_valid_rate": float(sum(probe_valid) / len(probe_valid)) if probe_valid else None,
        "repair_success_rate": float(repair_successes / repair_attempts) if repair_attempts else None,
    }


def _build_episode_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "episode_id": result["episode_id"],
        "condition": result["condition"],
        "regime": result["regime"],
        "model_id": result["model_id"],
        "api_status": result_api_status(result),
        "stop_reason": result["stop_reason"],
        "behavioral_protocol_version": result.get("behavioral_protocol_version", 1),
    }
    if "listened_bin_total" in result:
        row["listened_bin_total"] = result["listened_bin_total"]
    if "observed_bins" in result:
        row["observed_bin_count"] = len(result["observed_bins"])
    if "listened_seconds_total" in result:
        row["listened_seconds_total"] = result["listened_seconds_total"]
    if "query_intervals" in result:
        row["observed_seconds_unique"] = result.get("exploration_metrics", {}).get("observed_seconds_unique")
    for key in ["track_id", "track_display_id", "time_mode", "conversation_id", "variant_id", "category", "category_label"]:
        if key in result:
            row[key] = result[key]
    for key in METRIC_GROUP_KEYS:
        if key in result:
            row.update(result[key])
    return row


def _episode_result_path(result: dict[str, Any], results_root: Path) -> Path:
    return (
        results_root
        / "episode_results"
        / result["condition"]
        / result["regime"]
        / f"{result['episode_id']}.json"
    )


def _selected_episode_result_path(
    results_root: Path,
    condition: str,
    regime: str,
    episode_id: str,
) -> Path:
    return results_root / "episode_results" / condition / regime / f"{episode_id}.json"


def _api_error_class_from_attempts(attempts: Any) -> str:
    if not isinstance(attempts, list):
        return API_ERROR_SOFT
    for attempt in reversed(attempts):
        if not isinstance(attempt, dict) or attempt.get("reason") != "api_error":
            continue
        error_class = attempt.get("error_class")
        if error_class in {API_ERROR_SOFT, API_ERROR_HARD}:
            return str(error_class)
        message = str(attempt.get("retry_error") or attempt.get("error") or "")
        return classify_api_error(message)
    return API_ERROR_SOFT


def _unresolved_prompt_api_error_classes(value: Any) -> list[str]:
    classes: list[str] = []
    if isinstance(value, dict):
        is_unresolved_prompt = (
            value.get("asked", True) is not False
            and bool(value.get("prompt"))
            and (value.get("parsed") or {}).get("reason") == "api_error"
        )
        if is_unresolved_prompt:
            classes.append(
                _api_error_class_from_attempts(
                    value.get("api_attempts") or value.get("repair_attempts")
                )
            )
            return classes
        for child in value.values():
            classes.extend(_unresolved_prompt_api_error_classes(child))
    elif isinstance(value, list):
        for child in value:
            classes.extend(_unresolved_prompt_api_error_classes(child))
    return classes


def result_api_status(result: dict[str, Any]) -> str:
    explicit_status = result.get("api_status")
    if explicit_status in API_STATUSES:
        return str(explicit_status)
    unresolved_classes = _unresolved_prompt_api_error_classes(result)
    if str(result.get("stop_reason", "")).endswith("api_error"):
        action_attempts = result.get("repair_attempts")
        if isinstance(action_attempts, list) and action_attempts:
            final_attempt = action_attempts[-1]
            if isinstance(final_attempt, dict):
                unresolved_classes.append(
                    _api_error_class_from_attempts(
                        final_attempt.get("api_attempts")
                        or final_attempt.get("api_failures")
                    )
                )
        if not unresolved_classes:
            unresolved_classes.append(API_ERROR_SOFT)
    if API_ERROR_HARD in unresolved_classes:
        return API_STATUS_HARD_ERROR
    if API_ERROR_SOFT in unresolved_classes:
        return API_STATUS_SOFT_ERROR
    return API_STATUS_COMPLETE


def _write_episode_result(result: dict[str, Any], results_root: Path) -> None:
    result["api_status"] = result_api_status(result)
    episode_path = _episode_result_path(result, results_root)
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    episode_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


def _write_flat_csv(results: Sequence[dict[str, Any]], path: Path) -> None:
    rows = [
        _build_episode_row(result)
        for result in results
        if result_api_status(result) == API_STATUS_COMPLETE
    ]
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


METRIC_GROUP_KEYS = [
    "format_metrics",
    "master_metrics",
    "belief_metrics",
    "behavioral_metrics",
    "core_behavioral_metrics",
    "uncertainty_metrics",
    "coverage_metrics",
    "same_session_behavioral_metrics",
    "same_session_core_behavioral_metrics",
    "same_session_uncertainty_metrics",
    "track1c_metrics",
    "exploration_metrics",
]


def _bounded_score(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    return min(1.0, max(0.0, float(value)))


def _weighted_score(components: Sequence[tuple[float | None, float]]) -> float | None:
    available = [(float(value), float(weight)) for value, weight in components if value is not None]
    weight_total = sum(weight for _value, weight in available)
    if weight_total <= 0.0:
        return None
    return float(sum(value * weight for value, weight in available) / weight_total)


def _numeric_count(value: Any) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    return max(0.0, float(value))


def _count_weighted_score(components: Sequence[tuple[float | None, float | None]]) -> float | None:
    available = [
        (float(value), float(count))
        for value, count in components
        if value is not None and count is not None and float(count) > 0.0
    ]
    count_total = sum(count for _value, count in available)
    if count_total <= 0.0:
        return None
    return float(sum(value * count for value, count in available) / count_total)


def _generic_master_prefix(track_id: str, condition: str) -> str:
    normalized_track = _validate_track_id(track_id)
    _assert(normalized_track in {TRACK_1A, TRACK_1B}, f"Unsupported generic master-score track: {track_id}")
    if condition == FULL_CONTEXT:
        suffix = "full_context"
    elif condition == ACTIVE:
        suffix = "active"
    elif condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        suffix = "active_same_session"
    else:
        raise RuntimeError(f"Unsupported condition for generic master score: {condition}")
    return f"track{normalized_track}_{suffix}"


def _generic_core_answer_score(core_metrics: dict[str, MetricValue], prefix: str) -> float:
    localization_exact = core_metrics.get(f"{prefix}_behavior_localization_exact")
    if localization_exact is None:
        localization_exact = core_metrics.get(f"{prefix}_behavior_localization_exact_250ms")
    score = _weighted_score(
        [
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_sequence_exact")), 0.20),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_pairwise_accuracy")), 0.20),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_localization_iou")), 0.25),
            (_bounded_score(localization_exact), 0.15),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_gap_accuracy")), 0.20),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_overlap_accuracy")), 0.10),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_overlap_iou")), 0.10),
            (_bounded_score(core_metrics.get(f"{prefix}_behavior_max_concurrency_accuracy")), 0.05),
        ]
    )
    return 0.0 if score is None else score


def _generic_uncertainty_score(uncertainty_metrics: dict[str, MetricValue], prefix: str) -> float | None:
    return _count_weighted_score(
        [
            (
                _bounded_score(uncertainty_metrics.get(f"{prefix}uncertainty_pairwise_accuracy")),
                _numeric_count(uncertainty_metrics.get(f"{prefix}uncertainty_pairwise_count")),
            ),
            (
                _bounded_score(uncertainty_metrics.get(f"{prefix}uncertainty_localization_accuracy")),
                _numeric_count(uncertainty_metrics.get(f"{prefix}uncertainty_localization_count")),
            ),
            (
                _bounded_score(uncertainty_metrics.get(f"{prefix}uncertainty_gap_accuracy")),
                _numeric_count(uncertainty_metrics.get(f"{prefix}uncertainty_gap_count")),
            ),
            (
                _bounded_score(uncertainty_metrics.get(f"{prefix}uncertainty_overlap_accuracy")),
                _numeric_count(uncertainty_metrics.get(f"{prefix}uncertainty_overlap_count")),
            ),
        ]
    )


def _generic_answer_score(
    *,
    belief_metrics: dict[str, MetricValue],
    core_metrics: dict[str, MetricValue],
    uncertainty_metrics: dict[str, MetricValue],
    core_prefix: str,
    uncertainty_prefix: str,
) -> float:
    core_score = _generic_core_answer_score(core_metrics, core_prefix)
    belief_structure_score = _weighted_score(
        [
            (_bounded_score(belief_metrics.get("belief_sequence_exact")), 0.40),
            (_bounded_score(belief_metrics.get("belief_pairwise_accuracy")), 0.35),
            (_bounded_score(belief_metrics.get("belief_gap_accuracy")), 0.25),
            (_bounded_score(belief_metrics.get("belief_overlap_pair_f1")), 0.20),
            (_bounded_score(belief_metrics.get("belief_overlap_interval_iou")), 0.15),
            (_bounded_score(belief_metrics.get("max_concurrency_accuracy")), 0.10),
        ]
    )
    answer_score = _weighted_score(
        [
            (core_score, 0.50),
            (belief_structure_score, 0.20),
            (_generic_uncertainty_score(uncertainty_metrics, uncertainty_prefix), 0.10),
            (_bounded_score(belief_metrics.get("belief_bin_score_mean", belief_metrics.get("belief_time_score_mean"))), 0.15),
            (_bounded_score(belief_metrics.get("belief_localization_mean_iou")), 0.05),
        ]
    )
    return 0.0 if answer_score is None else answer_score


def score_generic_master_metrics(
    *,
    track_id: str,
    condition: str,
    stop_reason: str,
    listened_bin_total: int | float,
    format_metrics: dict[str, MetricValue],
    exploration_metrics: dict[str, MetricValue],
    belief_metrics: dict[str, MetricValue],
    core_metrics: dict[str, MetricValue],
    uncertainty_metrics: dict[str, MetricValue],
) -> dict[str, float]:
    prefix = _generic_master_prefix(track_id, condition)
    if condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        core_prefix = "same_session_core"
        uncertainty_prefix = "same_session_"
    else:
        core_prefix = "core"
        uncertainty_prefix = ""

    answer_score = _generic_answer_score(
        belief_metrics=belief_metrics,
        core_metrics=core_metrics,
        uncertainty_metrics=uncertainty_metrics,
        core_prefix=core_prefix,
        uncertainty_prefix=uncertainty_prefix,
    )
    if condition == FULL_CONTEXT:
        return {
            f"{prefix}_answer_score": answer_score,
            f"{prefix}_master_score": 100.0 * answer_score,
        }

    action_format = _bounded_score(format_metrics.get("action_format_valid_rate")) or 0.0
    action_semantic = _bounded_score(format_metrics.get("action_semantic_valid_rate")) or 0.0
    non_invalid_episode = 0.0 if str(stop_reason).startswith("invalid_action:") else 1.0
    protocol_score = non_invalid_episode * math.sqrt(action_format * action_semantic)

    exploration_core = _weighted_score(
        [
            (_bounded_score(exploration_metrics.get("event_coverage_rate")), 0.70),
            (
                _bounded_score(
                    exploration_metrics.get(
                        "event_bin_coverage_rate",
                        exploration_metrics.get("event_time_coverage_rate"),
                    )
                ),
                0.30,
            ),
        ]
    )
    exploration_core = 0.0 if exploration_core is None else exploration_core
    repeated_bin_total = max(
        0.0,
        float(exploration_metrics.get("repeated_bin_total", exploration_metrics.get("repeated_seconds_total", 0.0)) or 0.0),
    )
    listened_total = max(0.0, float(listened_bin_total))
    nonredundant_query_rate = 0.0 if listened_total <= 0.0 else 1.0 - min(1.0, repeated_bin_total / listened_total)
    exploration_score = exploration_core * nonredundant_query_rate

    master_score = 100.0 * (protocol_score ** 0.25) * (exploration_score ** 0.35) * (answer_score ** 0.40)
    return {
        f"{prefix}_protocol_score": protocol_score,
        f"{prefix}_exploration_score": exploration_score,
        f"{prefix}_answer_score": answer_score,
        f"{prefix}_master_score": master_score,
    }


def _ensure_generic_master_metrics(
    result: dict[str, Any],
    episode: dict[str, Any],
    config: BenchmarkConfig,
) -> dict[str, Any]:
    track_id = _validate_track_id(config.track_id)
    if track_id not in {TRACK_1A, TRACK_1B}:
        return result
    prefix = _generic_master_prefix(track_id, str(result["condition"]))
    master_metrics = dict(result.get("master_metrics") or {})
    if f"{prefix}_master_score" in master_metrics:
        return result

    enriched = dict(result)
    enriched["track_id"] = track_id
    action_history = list(enriched.get("action_history") or [])
    observed_bins = {int(index) for index in enriched.get("observed_bins") or []}
    exploration_metrics = dict(enriched.get("exploration_metrics") or {})
    if enriched["condition"] != FULL_CONTEXT and not exploration_metrics:
        exploration_metrics = _score_exploration_metrics(
            episode,
            config,
            list(enriched.get("repair_attempts") or []),
            action_history,
            observed_bins,
        )
        enriched["exploration_metrics"] = exploration_metrics

    if enriched["condition"] == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        core_metrics = dict(enriched.get("same_session_core_behavioral_metrics") or {})
        uncertainty_metrics = dict(enriched.get("same_session_uncertainty_metrics") or {})
    else:
        core_metrics = dict(enriched.get("core_behavioral_metrics") or {})
        uncertainty_metrics = dict(enriched.get("uncertainty_metrics") or {})

    master_metrics.update(
        score_generic_master_metrics(
            track_id=track_id,
            condition=str(enriched["condition"]),
            stop_reason=str(enriched.get("stop_reason", "")),
            listened_bin_total=float(enriched.get("listened_bin_total", 0.0)),
            format_metrics=dict(enriched.get("format_metrics") or {}),
            exploration_metrics=exploration_metrics,
            belief_metrics=dict(enriched.get("belief_metrics") or {}),
            core_metrics=core_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )
    )
    enriched["master_metrics"] = master_metrics
    return enriched


def _numeric_metric_values(result: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for group_key in METRIC_GROUP_KEYS:
        group = result.get(group_key, {})
        if not isinstance(group, dict):
            continue
        for key, value in group.items():
            if isinstance(value, bool):
                values[key] = float(value)
            elif isinstance(value, (int, float)):
                values[key] = float(value)
    return values


def _write_summary_json(results: Sequence[dict[str, Any]], path: Path) -> None:
    api_status_counts = {
        status: sum(1 for result in results if result_api_status(result) == status)
        for status in sorted(API_STATUSES)
    }
    complete_results = [
        result for result in results if result_api_status(result) == API_STATUS_COMPLETE
    ]
    metric_keys = sorted({key for result in complete_results for key in _numeric_metric_values(result)})
    aggregates: dict[str, Any] = {
        "num_rows": len(complete_results),
        "num_results_total": len(results),
        "api_status_counts": api_status_counts,
        "metrics": {},
    }
    for key in metric_keys:
        values = [
            metrics[key]
            for result in complete_results
            for metrics in [_numeric_metric_values(result)]
            if key in metrics
        ]
        aggregates["metrics"][key] = float(sum(values) / len(values)) if values else 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(aggregates, indent=2, sort_keys=True), encoding="utf-8")


def _score_exploration_metrics(
    episode: dict[str, Any],
    config: BenchmarkConfig,
    action_attempts: Sequence[dict[str, Any]],
    action_history: Sequence[dict[str, Any]],
    observed_bins: set[int],
) -> dict[str, float]:
    event_bins = {index for index, labels in enumerate(episode["ground_truth_raster"]) if labels}
    silent_bins = set(range(config.num_bins)) - event_bins
    queried_bin_total = sum(int(item["length_bin"]) for item in action_history)
    seen_bins: set[int] = set()
    redundant_query_count = 0
    partial_redundant_query_count = 0
    for item in action_history:
        query_bins = set(range(item["start_bin"], item["end_bin_exclusive"]))
        overlap = len(query_bins & seen_bins)
        if overlap == len(query_bins) and query_bins:
            redundant_query_count += 1
        elif overlap > 0:
            partial_redundant_query_count += 1
        seen_bins |= query_bins

    event_bin_coverage_count = len(observed_bins & event_bins)
    silent_bin_coverage_count = len(observed_bins & silent_bins)
    invalid_action_count = sum(float(not item.get("semantic_valid", False)) for item in action_attempts)
    repair_attempted_action_count = sum(float(bool(item.get("repair_attempted"))) for item in action_attempts)
    return {
        "query_count": float(len(action_history)),
        "action_attempt_count": float(len(action_attempts)),
        "invalid_action_count": float(invalid_action_count),
        "repair_attempted_action_count": float(repair_attempted_action_count),
        "budget_usage_rate": float(queried_bin_total / config.max_total_queried_bins) if config.max_total_queried_bins else 0.0,
        "step_usage_rate": float(len(action_history) / config.max_steps) if config.max_steps else 0.0,
        "event_bin_coverage_count": float(event_bin_coverage_count),
        "event_bin_coverage_rate": float(event_bin_coverage_count / len(event_bins)) if event_bins else 0.0,
        "silent_bin_coverage_count": float(silent_bin_coverage_count),
        "silent_bin_coverage_rate": float(silent_bin_coverage_count / len(silent_bins)) if silent_bins else 0.0,
        "repeated_bin_total": float(max(0, queried_bin_total - len(observed_bins))),
        "redundant_query_count": float(redundant_query_count),
        "partial_redundant_query_count": float(partial_redundant_query_count),
    }


def _run_active_condition(
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    run_dir: Path,
) -> dict[str, Any]:
    regime_payload = episode["regime_variants"][regime]
    env = EpisodeEnvironment(root, episode, config, run_dir)
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    prompt = build_action_initial_prompt(episode, regime_payload, config, vocabulary)
    action_attempts: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    raw_action, raw_action_api_attempts = _send_with_api_retry(
        adapter,
        session,
        prompt,
        [],
        config.api_retry_sleep_sec,
    )
    while True:
        if raw_action is None:
            final_parse = _api_failure_parse("api_error")
            action_attempts.append(
                {
                    "initial_raw_output": "",
                    "final_raw_output": "",
                    "final_parse": final_parse,
                    "semantic_valid": False,
                    "repair_attempted": False,
                    "reason": "api_error",
                    "api_attempts": raw_action_api_attempts,
                }
            )
            env.stop_reason = "invalid_action:api_error"
            break
        parsed = parse_action(raw_action)
        repair_attempted = False
        api_attempts = list(raw_action_api_attempts)
        if parsed.get("format_valid") and parsed.get("action") == "QUERY":
            semantic_valid, reason = validate_query(parsed, config, env.listened_bin_total)
        else:
            semantic_valid = bool(parsed.get("format_valid"))
            reason = parsed.get("reason", "")
        if not parsed.get("format_valid") or (parsed.get("action") == "QUERY" and not semantic_valid):
            repair_attempted = True
            repair_reason = reason or parsed.get("reason", "invalid_action")
            repaired_raw, repair_api_attempts = _send_with_api_retry(
                adapter,
                session,
                build_action_repair_prompt(repair_reason, env),
                [],
                config.api_retry_sleep_sec,
            )
            api_attempts.extend(repair_api_attempts)
            if repaired_raw is None:
                repaired = _api_failure_parse("api_error")
                semantic_valid = False
                reason = "api_error"
                final_raw = ""
                final_parse = repaired
            else:
                repaired = parse_action(repaired_raw)
                if repaired.get("format_valid") and repaired.get("action") == "QUERY":
                    semantic_valid, reason = validate_query(repaired, config, env.listened_bin_total)
                else:
                    semantic_valid = bool(repaired.get("format_valid"))
                    reason = repaired.get("reason", "")
                final_raw = repaired_raw
                final_parse = repaired
        else:
            final_raw = raw_action
            final_parse = parsed

        attempt_record = {
            "initial_raw_output": raw_action,
            "final_raw_output": final_raw,
            "final_parse": final_parse,
            "semantic_valid": bool(final_parse.get("format_valid")) and (
                final_parse.get("action") != "QUERY" or semantic_valid
            ),
            "repair_attempted": repair_attempted,
            "reason": reason,
        }
        attempt_record["api_attempts"] = api_attempts
        action_attempts.append(attempt_record)

        if not attempt_record["semantic_valid"]:
            env.stop_reason = f"invalid_action:{reason or final_parse.get('reason', 'invalid')}"
            break
        if final_parse["action"] == "STOP":
            env.stop_reason = "model_stop"
            break
        clip_path, metadata = env.clip_for_query(
            len(env.action_history) + 1,
            final_parse["start_bin"],
            final_parse["length_bin"],
        )
        if len(env.action_history) % config.probe_every_n == 0:
            probe_result = _run_probe_sidecar(
                adapter,
                episode,
                regime_payload,
                config,
                vocabulary,
                env.action_history,
                env.query_audio_paths,
            )
            probe_result["checkpoint_step"] = len(env.action_history)
            probes.append(probe_result)
            env.repair_log.extend(probe_result["repair_attempts"])
        if len(env.action_history) >= config.max_steps:
            env.stop_reason = "max_steps_reached"
            break
        if env.listened_bin_total >= config.max_total_queried_bins:
            env.stop_reason = "max_total_queried_bins_reached"
            break
        raw_action, raw_action_api_attempts = _send_with_api_retry(
            adapter,
            session,
            build_action_followup_prompt(metadata, env),
            [clip_path],
            config.api_retry_sleep_sec,
        )

    final_probe = _run_probe_sidecar(
        adapter,
        episode,
        regime_payload,
        config,
        vocabulary,
        env.action_history,
        env.query_audio_paths,
    )
    final_probe["checkpoint_step"] = len(env.action_history)
    probes.append(final_probe)

    behavioral_outputs = _run_behavioral_prompts(
        adapter,
        episode,
        regime_payload,
        config,
        vocabulary,
        env.action_history,
        env.query_audio_paths,
        False,
    )
    belief_metrics = score_belief_metrics(final_probe["parsed"], episode, env.observed_bins)
    format_metrics = _format_metrics(
        action_attempts,
        probes,
        behavioral_outputs["core_behavioral_outputs"],
        behavioral_outputs["uncertainty_outputs"],
    )
    exploration_metrics = _score_exploration_metrics(
        episode,
        config,
        action_attempts,
        env.action_history,
        env.observed_bins,
    )
    master_metrics = score_generic_master_metrics(
        track_id=config.track_id,
        condition=ACTIVE,
        stop_reason=env.stop_reason,
        listened_bin_total=env.listened_bin_total,
        format_metrics=format_metrics,
        exploration_metrics=exploration_metrics,
        belief_metrics=belief_metrics,
        core_metrics=behavioral_outputs["core_behavioral_metrics"],
        uncertainty_metrics=behavioral_outputs["uncertainty_metrics"],
    )
    _close_adapter_session(adapter, session)

    return {
        "episode_id": episode["episode_id"],
        "track_id": config.track_id,
        "condition": "active",
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": env.action_history,
        "repair_attempts": action_attempts,
        "repair_log": env.repair_log,
        "stop_reason": env.stop_reason,
        "observed_bins": sorted(env.observed_bins),
        "listened_bin_total": env.listened_bin_total,
        "probe_outputs": probes,
        "final_raster": final_probe,
        "behavioral_protocol_version": behavioral_outputs["behavioral_protocol_version"],
        "coverage_metrics": behavioral_outputs["coverage_metrics"],
        "core_behavioral_outputs": behavioral_outputs["core_behavioral_outputs"],
        "core_behavioral_metrics": behavioral_outputs["core_behavioral_metrics"],
        "uncertainty_outputs": behavioral_outputs["uncertainty_outputs"],
        "uncertainty_metrics": behavioral_outputs["uncertainty_metrics"],
        "behavioral_outputs": behavioral_outputs["behavioral_outputs"],
        "belief_metrics": belief_metrics,
        "behavioral_metrics": behavioral_outputs["behavioral_metrics"],
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
        "master_metrics": master_metrics,
    }


def _run_active_same_session_periodic_probe_condition(
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    run_dir: Path,
) -> dict[str, Any]:
    regime_payload = episode["regime_variants"][regime]
    env = EpisodeEnvironment(root, episode, config, run_dir)
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    prompt = build_action_initial_prompt(episode, regime_payload, config, vocabulary)
    action_attempts: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    raw_action, raw_action_api_attempts = _send_with_api_retry(
        adapter,
        session,
        prompt,
        [],
        config.api_retry_sleep_sec,
    )
    final_probe_audio_paths: list[Path] = []

    while True:
        if raw_action is None:
            final_parse = _api_failure_parse("api_error")
            action_attempts.append(
                {
                    "initial_raw_output": "",
                    "final_raw_output": "",
                    "final_parse": final_parse,
                    "semantic_valid": False,
                    "repair_attempted": False,
                    "reason": "api_error",
                    "api_attempts": raw_action_api_attempts,
                }
            )
            env.stop_reason = "invalid_action:api_error"
            break

        parsed = parse_action(raw_action)
        repair_attempted = False
        api_attempts = list(raw_action_api_attempts)
        if parsed.get("format_valid") and parsed.get("action") == "QUERY":
            semantic_valid, reason = validate_query(parsed, config, env.listened_bin_total)
        else:
            semantic_valid = bool(parsed.get("format_valid"))
            reason = parsed.get("reason", "")
        if not parsed.get("format_valid") or (parsed.get("action") == "QUERY" and not semantic_valid):
            repair_attempted = True
            repair_reason = reason or parsed.get("reason", "invalid_action")
            repaired_raw, repair_api_attempts = _send_with_api_retry(
                adapter,
                session,
                build_action_repair_prompt(repair_reason, env),
                [],
                config.api_retry_sleep_sec,
            )
            api_attempts.extend(repair_api_attempts)
            if repaired_raw is None:
                repaired = _api_failure_parse("api_error")
                semantic_valid = False
                reason = "api_error"
                final_raw = ""
                final_parse = repaired
            else:
                repaired = parse_action(repaired_raw)
                if repaired.get("format_valid") and repaired.get("action") == "QUERY":
                    semantic_valid, reason = validate_query(repaired, config, env.listened_bin_total)
                else:
                    semantic_valid = bool(repaired.get("format_valid"))
                    reason = repaired.get("reason", "")
                final_raw = repaired_raw
                final_parse = repaired
        else:
            final_raw = raw_action
            final_parse = parsed

        attempt_record = {
            "initial_raw_output": raw_action,
            "final_raw_output": final_raw,
            "final_parse": final_parse,
            "semantic_valid": bool(final_parse.get("format_valid")) and (
                final_parse.get("action") != "QUERY" or semantic_valid
            ),
            "repair_attempted": repair_attempted,
            "reason": reason,
        }
        attempt_record["api_attempts"] = api_attempts
        action_attempts.append(attempt_record)

        if not attempt_record["semantic_valid"]:
            env.stop_reason = f"invalid_action:{reason or final_parse.get('reason', 'invalid')}"
            break
        if final_parse["action"] == "STOP":
            env.stop_reason = "model_stop"
            break

        clip_path, metadata = env.clip_for_query(
            len(env.action_history) + 1,
            final_parse["start_bin"],
            final_parse["length_bin"],
        )
        consumed_latest_audio = False
        if len(env.action_history) % config.probe_every_n == 0:
            probe_result = _run_same_session_probe(
                adapter,
                session,
                episode,
                regime_payload,
                config,
                vocabulary,
                env.action_history,
                [clip_path],
                "periodic",
            )
            probes.append(probe_result)
            env.repair_log.extend(probe_result["repair_attempts"])
            consumed_latest_audio = True

        if len(env.action_history) >= config.max_steps:
            env.stop_reason = "max_steps_reached"
            final_probe_audio_paths = [] if consumed_latest_audio else [clip_path]
            break
        if env.listened_bin_total >= config.max_total_queried_bins:
            env.stop_reason = "max_total_queried_bins_reached"
            final_probe_audio_paths = [] if consumed_latest_audio else [clip_path]
            break

        raw_action, raw_action_api_attempts = _send_with_api_retry(
            adapter,
            session,
            build_action_followup_prompt(metadata, env),
            [] if consumed_latest_audio else [clip_path],
            config.api_retry_sleep_sec,
        )

    final_probe = _run_same_session_probe(
        adapter,
        session,
        episode,
        regime_payload,
        config,
        vocabulary,
        env.action_history,
        final_probe_audio_paths,
        "final",
    )
    probes.append(final_probe)
    env.repair_log.extend(final_probe["repair_attempts"])

    behavioral_outputs = _run_same_session_behavioral_prompts(
        adapter,
        session,
        episode,
        regime_payload,
        config,
        vocabulary,
        env.action_history,
    )
    format_metrics = _format_metrics(
        action_attempts,
        probes,
        behavioral_outputs["same_session_core_behavioral_outputs"],
        behavioral_outputs["same_session_uncertainty_outputs"],
    )
    exploration_metrics = _score_exploration_metrics(
        episode,
        config,
        action_attempts,
        env.action_history,
        env.observed_bins,
    )
    master_metrics = score_generic_master_metrics(
        track_id=config.track_id,
        condition=ACTIVE_SAME_SESSION_PERIODIC_PROBE,
        stop_reason=env.stop_reason,
        listened_bin_total=env.listened_bin_total,
        format_metrics=format_metrics,
        exploration_metrics=exploration_metrics,
        belief_metrics=final_probe["belief_metrics"],
        core_metrics=behavioral_outputs["same_session_core_behavioral_metrics"],
        uncertainty_metrics=behavioral_outputs["same_session_uncertainty_metrics"],
    )
    _close_adapter_session(adapter, session)

    return {
        "episode_id": episode["episode_id"],
        "track_id": config.track_id,
        "condition": ACTIVE_SAME_SESSION_PERIODIC_PROBE,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": env.action_history,
        "repair_attempts": action_attempts,
        "repair_log": env.repair_log,
        "stop_reason": env.stop_reason,
        "observed_bins": sorted(env.observed_bins),
        "listened_bin_total": env.listened_bin_total,
        "probe_outputs": probes,
        "final_raster": final_probe,
        "behavioral_protocol_version": behavioral_outputs["behavioral_protocol_version"],
        "coverage_metrics": behavioral_outputs["coverage_metrics"],
        "same_session_core_behavioral_plan": behavioral_outputs["same_session_core_behavioral_plan"],
        "same_session_uncertainty_plan": behavioral_outputs["same_session_uncertainty_plan"],
        "same_session_core_behavioral_outputs": behavioral_outputs["same_session_core_behavioral_outputs"],
        "same_session_core_behavioral_metrics": behavioral_outputs["same_session_core_behavioral_metrics"],
        "same_session_uncertainty_outputs": behavioral_outputs["same_session_uncertainty_outputs"],
        "same_session_uncertainty_metrics": behavioral_outputs["same_session_uncertainty_metrics"],
        "same_session_behavioral_outputs": behavioral_outputs["same_session_behavioral_outputs"],
        "same_session_behavioral_metrics": behavioral_outputs["same_session_behavioral_metrics"],
        "belief_metrics": final_probe["belief_metrics"],
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
        "master_metrics": master_metrics,
    }


def _run_full_context_condition(
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> dict[str, Any]:
    regime_payload = episode["regime_variants"][regime]
    audio_path = root / episode["audio_path"]
    final_probe = _run_full_context_probe(adapter, episode, regime_payload, config, vocabulary, audio_path)
    behavioral_outputs = _run_behavioral_prompts(
        adapter,
        episode,
        regime_payload,
        config,
        vocabulary,
        [],
        [audio_path],
        True,
    )
    observed_bins = set(range(config.num_bins))
    belief_metrics = score_belief_metrics(final_probe["parsed"], episode, observed_bins)
    format_metrics = _format_metrics(
        [],
        [final_probe],
        behavioral_outputs["core_behavioral_outputs"],
        behavioral_outputs["uncertainty_outputs"],
    )
    master_metrics = score_generic_master_metrics(
        track_id=config.track_id,
        condition=FULL_CONTEXT,
        stop_reason=FULL_CONTEXT,
        listened_bin_total=config.num_bins,
        format_metrics=format_metrics,
        exploration_metrics={},
        belief_metrics=belief_metrics,
        core_metrics=behavioral_outputs["core_behavioral_metrics"],
        uncertainty_metrics=behavioral_outputs["uncertainty_metrics"],
    )
    return {
        "episode_id": episode["episode_id"],
        "track_id": config.track_id,
        "condition": "full_context",
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": [],
        "repair_attempts": [],
        "repair_log": final_probe["repair_attempts"],
        "stop_reason": "full_context",
        "observed_bins": list(range(config.num_bins)),
        "listened_bin_total": config.num_bins,
        "probe_outputs": [final_probe],
        "final_raster": final_probe,
        "behavioral_protocol_version": behavioral_outputs["behavioral_protocol_version"],
        "coverage_metrics": behavioral_outputs["coverage_metrics"],
        "core_behavioral_outputs": behavioral_outputs["core_behavioral_outputs"],
        "core_behavioral_metrics": behavioral_outputs["core_behavioral_metrics"],
        "uncertainty_outputs": behavioral_outputs["uncertainty_outputs"],
        "uncertainty_metrics": behavioral_outputs["uncertainty_metrics"],
        "behavioral_outputs": behavioral_outputs["behavioral_outputs"],
        "belief_metrics": belief_metrics,
        "behavioral_metrics": behavioral_outputs["behavioral_metrics"],
        "format_metrics": format_metrics,
        "master_metrics": master_metrics,
    }


def run_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    adapter: BaseAdapter,
    config: BenchmarkConfig | None = None,
    episode_ids: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    reprocess: bool = False,
    parallelism: int = 1,
    time_mode: str | None = None,
) -> list[dict[str, Any]]:
    config = config or BenchmarkConfig()
    requested_time_mode = validate_time_mode(time_mode if time_mode is not None else config.time_mode)
    config = replace(config, time_mode=requested_time_mode)
    root = Path(dataset_root)
    manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("track_id") == TRACK_1C and "benchmark_defaults" not in manifest:
        from .track1c import run_track1c_benchmark

        return run_track1c_benchmark(
            dataset_root=root,
            results_root=results_root,
            adapter=adapter,
            episode_ids=episode_ids,
            conditions=conditions,
            regimes=regimes,
            reprocess=reprocess,
            parallelism=parallelism,
            time_mode=requested_time_mode,
        )
    manifest_time_mode = validate_time_mode(
        manifest.get("time_mode")
        or manifest.get("benchmark_defaults", {}).get("time_mode", TIME_MODE_BINNED)
    )
    _assert(
        manifest_time_mode == requested_time_mode,
        f"Manifest time_mode {manifest_time_mode} does not match requested time_mode {requested_time_mode}.",
    )
    if requested_time_mode == TIME_MODE_CONTINUOUS:
        from .continuous_tracks import run_continuous_benchmark

        return run_continuous_benchmark(
            dataset_root=root,
            results_root=results_root,
            adapter=adapter,
            config=config,
            episode_ids=episode_ids,
            conditions=conditions,
            regimes=regimes,
            reprocess=reprocess,
            parallelism=parallelism,
        )
    vocabulary = manifest["vocabulary"]
    manifest_episodes = manifest["episodes"]
    manifest_episode_ids = [str(episode["episode_id"]) for episode in manifest_episodes]

    if episode_ids is None:
        selected_episodes = list(manifest_episodes)
    else:
        requested_episode_ids = [str(episode_id) for episode_id in episode_ids]
        unknown_episode_ids = sorted(set(requested_episode_ids) - set(manifest_episode_ids))
        _assert(not unknown_episode_ids, f"Unknown episode_ids: {unknown_episode_ids}")
        requested_set = set(requested_episode_ids)
        selected_episodes = [episode for episode in manifest_episodes if episode["episode_id"] in requested_set]

    selected_conditions = list(DEFAULT_CONDITIONS if conditions is None else [str(condition) for condition in conditions])
    unknown_conditions = sorted(set(selected_conditions) - set(CONDITIONS))
    _assert(not unknown_conditions, f"Unknown conditions: {unknown_conditions}")

    selected_regimes = list(REGIMES if regimes is None else [str(regime) for regime in regimes])
    unknown_regimes = sorted(set(selected_regimes) - set(REGIMES))
    _assert(not unknown_regimes, f"Unknown regimes: {unknown_regimes}")

    run_specs: list[tuple[str, str, dict[str, Any]]] = []
    for episode in selected_episodes:
        for regime in selected_regimes:
            for condition in selected_conditions:
                run_specs.append((condition, regime, episode))

    total_runs = len(run_specs)
    _assert(total_runs > 0, "Selected benchmark run is empty.")
    cached_spec_keys = {
        (condition, regime, episode["episode_id"])
        for condition, regime, episode in run_specs
        if (not reprocess)
        and _selected_episode_result_path(
            results_root,
            condition,
            regime,
            episode["episode_id"],
        ).is_file()
    }
    progress = ProgressReporter(total_runs, label="benchmark-run", eta_total=total_runs - len(cached_spec_keys))

    def _run_one_spec(spec: tuple[str, str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        condition, regime, episode = spec
        spec_key = (condition, regime, episode["episode_id"])
        existing_path = _selected_episode_result_path(
            results_root,
            condition,
            regime,
            episode["episode_id"],
        )
        if spec_key in cached_spec_keys:
            cached_result = json.loads(existing_path.read_text(encoding="utf-8"))
            return _ensure_generic_master_metrics(cached_result, episode, config), True

        if condition == ACTIVE:
            run_dir = results_root / "runtime" / ACTIVE / regime / episode["episode_id"]
            result = _run_active_condition(
                root,
                episode,
                regime,
                adapter,
                config,
                vocabulary,
                run_dir,
            )
        elif condition == FULL_CONTEXT:
            result = _run_full_context_condition(
                root,
                episode,
                regime,
                adapter,
                config,
                vocabulary,
            )
        elif condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
            run_dir = results_root / "runtime" / ACTIVE_SAME_SESSION_PERIODIC_PROBE / regime / episode["episode_id"]
            result = _run_active_same_session_periodic_probe_condition(
                root,
                episode,
                regime,
                adapter,
                config,
                vocabulary,
                run_dir,
            )
        else:
            raise RuntimeError(f"Unsupported condition: {condition}")
        _write_episode_result(result, results_root)
        return result, False

    max_workers = max(1, int(parallelism))
    results_by_index: list[dict[str, Any] | None] = [None] * total_runs
    if max_workers == 1 or total_runs == 1:
        for index, spec in enumerate(run_specs):
            result, from_cache = _run_one_spec(spec)
            results_by_index[index] = result
            progress.update(
                index + 1,
                f"condition={spec[0]} regime={spec[1]} episode={spec[2]['episode_id']}",
                count_for_eta=not from_cache,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, total_runs)) as executor:
            future_to_index = {executor.submit(_run_one_spec, spec): index for index, spec in enumerate(run_specs)}
            completed_runs = 0
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                spec = run_specs[index]
                result, from_cache = future.result()
                results_by_index[index] = result
                completed_runs += 1
                progress.update(
                    completed_runs,
                    f"condition={spec[0]} regime={spec[1]} episode={spec[2]['episode_id']}",
                    count_for_eta=not from_cache,
                )

    results = [result for result in results_by_index if result is not None]

    _write_flat_csv(results, results_root / "episode_metrics.csv")
    _write_summary_json(results, results_root / "summary.json")
    return results


def build_datasets_for_tracks(
    *,
    audio_root: Path,
    output_root: Path,
    tracks: Sequence[str] | None = None,
    config: BenchmarkConfig | None = None,
    time_mode: str | None = None,
) -> dict[str, dict[str, Any]]:
    selected_tracks = normalize_tracks(tracks)
    base_config = config or BenchmarkConfig()
    requested_time_mode = validate_time_mode(time_mode if time_mode is not None else base_config.time_mode)
    base_config = replace(base_config, time_mode=requested_time_mode)
    manifests: dict[str, dict[str, Any]] = {}
    for track_id in selected_tracks:
        if track_id == TRACK_2:
            raise RuntimeError("Track 2 dataset generation is handled by track2_dataset/build_track2_dataset_v2.py.")
        if track_id == TRACK_1C:
            raise RuntimeError("Track 1c dataset generation is handled by track1c/build_dataset.py.")
        track_config = replace(base_config, track_id=track_id)
        manifests[track_id] = build_dataset(
            audio_root=audio_root,
            output_root=_track_root(output_root, track_id, selected_tracks),
            config=track_config,
        )
    return manifests


def run_benchmark_for_tracks(
    *,
    dataset_root: Path,
    results_root: Path,
    adapter: BaseAdapter,
    tracks: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    reprocess: bool = False,
    parallelism: int = 1,
    time_mode: str = TIME_MODE_BINNED,
    track2_dataset_root: Path | None = None,
    track2_task_ids: Sequence[str] | None = None,
    track2_reel_ids: Sequence[str] | None = None,
    track2_families: Sequence[str] | None = None,
    track2_variants: Sequence[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    selected_tracks = normalize_tracks(tracks)
    requested_time_mode = validate_time_mode(time_mode)
    selected_conditions = list(DEFAULT_CONDITIONS if conditions is None else conditions)
    supported_conditions = getattr(adapter, "supported_conditions", None)
    if supported_conditions is not None and any(track != TRACK_2 for track in selected_tracks):
        unsupported_conditions = sorted(set(selected_conditions) - set(supported_conditions))
        _assert(
            not unsupported_conditions,
            f"{type(adapter).__name__} does not support conditions: {unsupported_conditions}",
        )
    results: dict[str, list[dict[str, Any]]] = {}
    for track_id in selected_tracks:
        if track_id == TRACK_2:
            from .track2 import run_track2_benchmark

            resolved_track2_root = Path(track2_dataset_root) if track2_dataset_root is not None else (
                Path(__file__).resolve().parent.parent
                / "track2_dataset"
                / "generated_data"
                / "track2_role_ipc_test_v2"
            )
            track2_results_root = _track_root(results_root, TRACK_2, selected_tracks)
            results[track_id] = run_track2_benchmark(
                dataset_root=resolved_track2_root,
                results_root=track2_results_root,
                adapter=adapter,
                task_ids=track2_task_ids,
                reel_ids=track2_reel_ids,
                families=track2_families,
                variants=track2_variants,
                conditions=conditions,
                reprocess=reprocess,
                parallelism=parallelism,
            )
            continue
        track_dataset_root = _track_root(dataset_root, track_id, selected_tracks)
        if not (track_dataset_root / "dataset" / "manifest.json").is_file():
            fallback_root = Path(dataset_root) / f"track_{track_id}"
            if (fallback_root / "dataset" / "manifest.json").is_file():
                track_dataset_root = fallback_root
        track_results_root = _track_root(results_root, track_id, selected_tracks)
        manifest = json.loads((track_dataset_root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("track_id") == TRACK_1C and "benchmark_defaults" not in manifest:
            from .track1c import run_track1c_benchmark

            results[track_id] = run_track1c_benchmark(
                dataset_root=track_dataset_root,
                results_root=track_results_root,
                adapter=adapter,
                conditions=conditions,
                reprocess=reprocess,
                parallelism=parallelism,
                time_mode=requested_time_mode,
            )
            continue
        config = _parse_config_from_manifest(track_dataset_root / "dataset" / "manifest.json")
        _assert(
            _validate_track_id(config.track_id) == track_id,
            f"Manifest track_id {config.track_id} does not match requested track {track_id}.",
        )
        _assert(
            validate_time_mode(config.time_mode) == requested_time_mode,
            f"Manifest time_mode {config.time_mode} does not match requested time_mode {requested_time_mode}.",
        )
        results[track_id] = run_benchmark(
            dataset_root=track_dataset_root,
            results_root=track_results_root,
            adapter=adapter,
            config=config,
            conditions=conditions,
            reprocess=reprocess,
            parallelism=parallelism,
            time_mode=requested_time_mode,
        )
    return results


def _parse_config_from_manifest(manifest_path: Path) -> BenchmarkConfig:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("track_id") == TRACK_1C and "benchmark_defaults" not in manifest:
        return BenchmarkConfig(track_id=TRACK_1C)
    defaults = manifest["benchmark_defaults"]
    return BenchmarkConfig(**defaults)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Theory of Time benchmark utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build-dataset", help="Build the evaluation dataset.")
    build_parser.add_argument("--audio-root", default="audios")
    build_parser.add_argument("--output-root", default=None)
    build_parser.add_argument("--seed", type=int, default=BenchmarkConfig.seed)
    build_parser.add_argument("--tracks", nargs="+", default=[TRACK_1A], choices=TRACK_IDS)
    build_parser.add_argument("--time-mode", default=TIME_MODE_BINNED, choices=TIME_MODES)

    run_parser = subparsers.add_parser("run", help="Run the benchmark for one adapter.")
    run_parser.add_argument("--dataset-root", default=None)
    run_parser.add_argument("--track2-dataset-root", default=None)
    run_parser.add_argument("--results-root", required=True)
    run_parser.add_argument("--adapter-config", required=True)
    run_parser.add_argument("--tracks", nargs="+", default=[TRACK_1A], choices=TRACK_IDS)
    run_parser.add_argument("--conditions", nargs="+", default=None, choices=CONDITIONS)
    run_parser.add_argument("--parallelism", type=int, default=1)
    run_parser.add_argument("--reprocess", type=parse_bool_arg, default=False)
    run_parser.add_argument("--time-mode", default=TIME_MODE_BINNED, choices=TIME_MODES)
    run_parser.add_argument("--track2-task-ids", nargs="+", default=None)
    run_parser.add_argument("--track2-reel-ids", nargs="+", default=None)
    run_parser.add_argument("--track2-families", nargs="+", default=None)
    run_parser.add_argument("--track2-variants", nargs="+", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "build-dataset":
        time_mode = validate_time_mode(args.time_mode)
        config = BenchmarkConfig(seed=int(args.seed), time_mode=time_mode)
        build_datasets_for_tracks(
            audio_root=Path(args.audio_root),
            output_root=Path(args.output_root or default_dataset_root_for_time_mode(time_mode)),
            tracks=args.tracks,
            config=config,
            time_mode=time_mode,
        )
        return 0
    if args.command == "run":
        time_mode = validate_time_mode(args.time_mode)
        adapter_config = load_adapter_config(Path(args.adapter_config))
        adapter = adapter_from_config(adapter_config)
        run_benchmark_for_tracks(
            dataset_root=Path(args.dataset_root or default_dataset_root_for_time_mode(time_mode)),
            results_root=Path(args.results_root),
            adapter=adapter,
            tracks=args.tracks,
            conditions=args.conditions,
            reprocess=bool(args.reprocess),
            parallelism=int(args.parallelism),
            time_mode=time_mode,
            track2_dataset_root=Path(args.track2_dataset_root) if args.track2_dataset_root else None,
            track2_task_ids=args.track2_task_ids,
            track2_reel_ids=args.track2_reel_ids,
            track2_families=args.track2_families,
            track2_variants=args.track2_variants,
        )
        return 0
    raise RuntimeError(f"Unsupported command: {args.command}")
