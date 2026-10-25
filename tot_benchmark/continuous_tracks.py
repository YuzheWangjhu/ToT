from __future__ import annotations

import concurrent.futures
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .adapters import BaseAdapter
from .core import (
    ACTION_SYSTEM_PROMPT,
    ACTIVE,
    ACTIVE_SAME_SESSION_PERIODIC_PROBE,
    CONDITIONS,
    COUNT_KNOWN,
    DEFAULT_CONDITIONS,
    DISCOVERY,
    FULL_CONTEXT,
    INVENTORY_KNOWN,
    REGIMES,
    SEQUENCE_DELIMITER,
    TRACK_1A,
    TRACK_1B,
    TRACK_1B_MAX_OVERLAP,
    TRACK_IDS,
    AssetRecord,
    BenchmarkConfig,
    ProgressReporter,
    _api_failure_parse,
    _assert,
    _close_adapter_session,
    _normalize_audio_peak,
    _read_mono_audio,
    _relative_to_root,
    _selected_episode_result_path,
    _send_with_api_retry,
    _send_with_one_repair,
    _validate_track_id,
    _write_audio,
    _write_episode_result,
    _write_flat_csv,
    _write_summary_json,
    parse_gap_answer,
    parse_pairwise_answer,
    parse_sequence_answer,
    scan_source_assets,
    score_generic_master_metrics,
)
from .text_protocol import normalize_first_line
from .time_mode import (
    CONTINUOUS_EXACT_THRESHOLD_SEC,
    TIME_MODE_CONTINUOUS,
    continuous_redundancy_counts,
    format_seconds,
    gap_from_spans,
    interval_contained,
    interval_iou,
    interval_overlap,
    interval_set_intersection_duration,
    interval_set_unique_duration,
    localization_exact_within_threshold,
    max_concurrency_from_segments,
    mean_label_iou,
    merge_intervals,
    parse_continuous_action,
    parse_continuous_localization_answer,
    parse_continuous_segments,
    relation_from_spans,
    segment_text,
    segment_time_accuracy,
    segments_from_event_spans,
    spans_from_segments,
    validate_continuous_query,
)

CONTINUOUS_PROTOCOL_VERSION = 3
CONTINUOUS_PLACEMENT_ATTEMPTS = 4000
MIN_TRACK1A_GAP_SECONDS = 0.20


def _continuous_regime_variants(labels_present: Sequence[str], config: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    label_text = ", ".join(labels_present)
    track_prior = " Events may have duration and at most two events may overlap." if config.track_id == TRACK_1B else ""
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


def _sample_noninteger_start(rng: random.Random, max_start: float) -> float:
    max_centisecond = int(math.floor(max_start * 100.0))
    _assert(max_centisecond > 0, "No valid centisecond placement range.")
    for _ in range(100):
        centiseconds = rng.randint(0, max_centisecond)
        value = centiseconds / 100.0
        if abs(value - round(value)) > 1e-9:
            return value
    return min(max_start, 0.37)


def _max_concurrency(intervals: Sequence[tuple[float, float]]) -> int:
    points = sorted({point for interval in intervals for point in interval})
    maximum = 0
    for start, end in zip(points[:-1], points[1:]):
        midpoint = (start + end) / 2.0
        count = sum(float(left) <= midpoint < float(right) for left, right in intervals)
        maximum = max(maximum, count)
    return maximum


def _has_non_overlapping_consecutive_pair(sequence: Sequence[str], spans: dict[str, dict[str, float]]) -> bool:
    for left, right in zip(sequence[:-1], sequence[1:]):
        if interval_overlap(
            (spans[left]["start_time_seconds"], spans[left]["end_time_seconds"]),
            (spans[right]["start_time_seconds"], spans[right]["end_time_seconds"]),
        ) <= 0.0:
            return True
    return False


def _continuous_gap_relations(sequence: Sequence[str], spans: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for left, right in zip(sequence[:-1], sequence[1:]):
        left_span = spans[left]
        right_span = spans[right]
        if interval_overlap(
            (left_span["start_time_seconds"], left_span["end_time_seconds"]),
            (right_span["start_time_seconds"], right_span["end_time_seconds"]),
        ) > 0.0:
            continue
        relations.append(
            {
                "left_label": left,
                "right_label": right,
                "relation": gap_from_spans(left_span, right_span),
            }
        )
    return relations


def _continuous_overlap_relations(sequence: Sequence[str], spans: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []
    for index, left in enumerate(sequence):
        for right in sequence[index + 1 :]:
            left_interval = (spans[left]["start_time_seconds"], spans[left]["end_time_seconds"])
            right_interval = (spans[right]["start_time_seconds"], spans[right]["end_time_seconds"])
            overlap_start = max(left_interval[0], right_interval[0])
            overlap_end = min(left_interval[1], right_interval[1])
            if overlap_end <= overlap_start:
                continue
            relations.append(
                {
                    "label_a": left,
                    "label_b": right,
                    "start_time_seconds": round(overlap_start, 2),
                    "end_time_seconds": round(overlap_end, 2),
                }
            )
    return relations


def _place_track1a_continuous(
    assets_by_label: dict[str, list[AssetRecord]],
    labels: Sequence[str],
    rng: random.Random,
    config: BenchmarkConfig,
) -> list[tuple[str, AssetRecord, float, float]]:
    for _attempt in range(CONTINUOUS_PLACEMENT_ATTEMPTS):
        event_count = rng.randint(config.min_events, config.max_events)
        selected_labels = rng.sample(list(labels), event_count)
        events: list[tuple[str, AssetRecord, float, float]] = []
        for label in selected_labels:
            asset = rng.choice(assets_by_label[label])
            start = _sample_noninteger_start(rng, config.episode_duration_sec - asset.duration_sec)
            end = round(start + asset.duration_sec, 2)
            start = round(start, 2)
            events.append((label, asset, start, end))
        events.sort(key=lambda item: item[2])
        ok = True
        for left, right in zip(events[:-1], events[1:]):
            if left[3] + MIN_TRACK1A_GAP_SECONDS > right[2]:
                ok = False
                break
        if ok:
            return events
    raise RuntimeError("Unable to place continuous Track 1a episode.")


def _place_track1b_continuous(
    assets_by_label: dict[str, list[AssetRecord]],
    labels: Sequence[str],
    rng: random.Random,
    config: BenchmarkConfig,
) -> list[tuple[str, AssetRecord, float, float]]:
    for _attempt in range(CONTINUOUS_PLACEMENT_ATTEMPTS):
        event_count = rng.randint(config.min_events, config.max_events)
        selected_labels = rng.sample(list(labels), event_count)
        events: list[tuple[str, AssetRecord, float, float]] = []
        for label in selected_labels:
            asset = rng.choice(assets_by_label[label])
            if asset.duration_sec > config.episode_duration_sec:
                break
            start = _sample_noninteger_start(rng, config.episode_duration_sec - asset.duration_sec)
            start = round(start, 2)
            end = round(start + asset.duration_sec, 2)
            events.append((label, asset, start, end))
        if len(events) != event_count:
            continue
        intervals = [(start, end) for _label, _asset, start, end in events]
        if _max_concurrency(intervals) > TRACK_1B_MAX_OVERLAP:
            continue
        spans = {
            label: {"start_time_seconds": start, "end_time_seconds": end}
            for label, _asset, start, end in events
        }
        sequence = [
            label
            for label, _span in sorted(spans.items(), key=lambda item: (item[1]["start_time_seconds"], item[0]))
        ]
        if not _continuous_overlap_relations(sequence, spans):
            continue
        if not _has_non_overlapping_consecutive_pair(sequence, spans):
            continue
        events.sort(key=lambda item: (item[2], item[0]))
        return events
    raise RuntimeError("Unable to place continuous Track 1b episode.")


def build_continuous_dataset(
    audio_root: Path,
    output_root: Path,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    track_id = _validate_track_id(config.track_id)
    _assert(track_id in {TRACK_1A, TRACK_1B}, "Continuous dataset builder supports Tracks 1a and 1b.")
    assets = scan_source_assets(Path(audio_root), config)
    assets_by_label: dict[str, list[AssetRecord]] = {}
    for asset in assets:
        assets_by_label.setdefault(asset.label, []).append(asset)
    labels = sorted(assets_by_label)
    samplerate = assets[0].samplerate
    total_frames = int(round(config.episode_duration_sec * samplerate))
    root = Path(output_root)
    dataset_dir = root / "dataset"
    episodes_dir = dataset_dir / "episodes"
    manifest_path = dataset_dir / "manifest.json"
    progress = ProgressReporter(config.num_episodes, label="continuous-dataset-build")
    rng = random.Random(config.seed)
    vocabulary = sorted(labels)
    episodes: list[dict[str, Any]] = []

    for episode_index in range(config.num_episodes):
        if track_id == TRACK_1A:
            layout_events = _place_track1a_continuous(assets_by_label, labels, rng, config)
        else:
            layout_events = _place_track1b_continuous(assets_by_label, labels, rng, config)
        audio = np.zeros(total_frames, dtype=np.float32)
        event_assignments: list[dict[str, Any]] = []
        asset_provenance: dict[str, dict[str, Any]] = {}
        spans: dict[str, dict[str, float]] = {}

        for label, asset, start_sec, end_sec in layout_events:
            clip = _read_mono_audio(asset.path, samplerate)
            start_frame = int(round(start_sec * samplerate))
            end_frame = start_frame + clip.shape[0]
            _assert(end_frame <= total_frames, f"Event {label} exceeds episode duration in {track_id}.")
            audio[start_frame:end_frame] += clip
            end_time = round(start_sec + clip.shape[0] / float(samplerate), 2)
            event_assignments.append(
                {
                    "label": label,
                    "start_time_seconds": round(start_sec, 2),
                    "end_time_seconds": end_time,
                }
            )
            spans[label] = {"start_time_seconds": round(start_sec, 2), "end_time_seconds": end_time}
            asset_provenance[label] = {
                "path": _relative_to_root(asset.path, root),
                "category": asset.category,
                "duration_sec": round(asset.duration_sec, 6),
            }
        audio = _normalize_audio_peak(audio)
        sequence = [
            item["label"]
            for item in sorted(event_assignments, key=lambda item: (item["start_time_seconds"], item["label"]))
        ]
        gap_relations = _continuous_gap_relations(sequence, spans)
        overlap_relations = _continuous_overlap_relations(sequence, spans)
        gold_segments = segments_from_event_spans(spans, audio_duration_seconds=config.episode_duration_sec)

        episode_id = f"episode_{episode_index:03d}"
        episode_audio_path = episodes_dir / f"{episode_id}.wav"
        _write_audio(episode_audio_path, audio, samplerate)
        episodes.append(
            {
                "episode_id": episode_id,
                "track_id": track_id,
                "time_mode": TIME_MODE_CONTINUOUS,
                "track_display_id": f"{track_id}-H",
                "audio_path": _relative_to_root(episode_audio_path, root),
                "audio_duration_seconds": round(config.episode_duration_sec, 6),
                "regime_variants": _continuous_regime_variants(sequence, config),
                "labels_present": sequence,
                "event_count": len(sequence),
                "event_time_assignments": event_assignments,
                "ground_truth_sequence": sequence,
                "ground_truth_event_spans": spans,
                "ground_truth_segments": gold_segments,
                "ground_truth_gap_relations": gap_relations,
                "ground_truth_overlap_relations": overlap_relations,
                "ground_truth_max_concurrency": max_concurrency_from_segments(gold_segments),
                "asset_provenance": asset_provenance,
                "build_seed": config.seed,
            }
        )
        progress.update(episode_index + 1, f"episode={episode_id} events={len(sequence)}")

    manifest = {
        "benchmark_defaults": config.to_manifest_dict(),
        "track_id": track_id,
        "time_mode": TIME_MODE_CONTINUOUS,
        "track_display_id": f"{track_id}-H",
        "vocabulary": vocabulary,
        "dataset_root": _relative_to_root(dataset_dir, root),
        "episodes": episodes,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


class ContinuousEpisodeEnvironment:
    def __init__(self, root: Path, episode: dict[str, Any], config: BenchmarkConfig, run_dir: Path) -> None:
        self.root = Path(root)
        self.episode = episode
        self.config = config
        self.run_dir = Path(run_dir)
        self.audio_path = self.root / episode["audio_path"]
        self.audio, self.samplerate = sf.read(str(self.audio_path), dtype="float32", always_2d=False)
        if isinstance(self.audio, np.ndarray) and self.audio.ndim == 2:
            self.audio = self.audio.mean(axis=1)
        self.action_history: list[dict[str, Any]] = []
        self.repair_log: list[dict[str, Any]] = []
        self.query_audio_paths: list[Path] = []
        self.query_intervals: list[tuple[float, float]] = []
        self.listened_seconds_total = 0.0
        self.stop_reason = ""

    @property
    def audio_duration_seconds(self) -> float:
        return float(self.episode.get("audio_duration_seconds", self.config.episode_duration_sec))

    @property
    def max_single_query_seconds(self) -> float:
        return float(self.config.max_query_bins) * float(self.config.bin_duration_sec)

    @property
    def total_listening_budget_seconds(self) -> float:
        return float(self.config.max_total_queried_bins) * float(self.config.bin_duration_sec)

    def remaining_steps(self) -> int:
        return self.config.max_steps - len(self.action_history)

    def remaining_seconds_budget(self) -> float:
        return self.total_listening_budget_seconds - self.listened_seconds_total

    def clip_for_query(self, step_index: int, start_sec: float, end_sec: float) -> tuple[Path, dict[str, Any]]:
        start_frame = int(round(float(start_sec) * self.samplerate))
        end_frame = int(round(float(end_sec) * self.samplerate))
        clip = np.asarray(self.audio[start_frame:end_frame], dtype=np.float32)
        start_cs = int(round(float(start_sec) * 100.0))
        end_cs = int(round(float(end_sec) * 100.0))
        clip_path = (
            self.run_dir
            / "query_audio"
            / f"{self.episode['episode_id']}_step_{step_index:02d}_{start_cs}_{end_cs}.wav"
        )
        _write_audio(clip_path, clip, self.samplerate)
        self.query_audio_paths.append(clip_path)
        duration = float(end_sec) - float(start_sec)
        self.listened_seconds_total += duration
        self.query_intervals.append((float(start_sec), float(end_sec)))
        metadata = {
            "step_index": step_index,
            "start_time_seconds": round(float(start_sec), 2),
            "end_time_seconds": round(float(end_sec), 2),
            "duration_seconds": round(duration, 6),
            "audio_path": str(clip_path),
        }
        self.action_history.append(metadata)
        return clip_path, metadata


def _continuous_context_header(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> str:
    max_single = config.max_query_bins * config.bin_duration_sec
    total_budget = config.max_total_queried_bins * config.bin_duration_sec
    duration = float(episode.get("audio_duration_seconds", config.episode_duration_sec))
    return (
        f"Vocabulary: {', '.join(vocabulary)}\n"
        f"Time mode: continuous hard mode ({config.track_id}-H).\n"
        f"Audio duration: {format_seconds(duration)} seconds.\n"
        f"Budget: max_steps={config.max_steps}, total_listening_budget_seconds={format_seconds(total_budget)}, "
        f"max_single_query_seconds={format_seconds(max_single)}.\n"
        f"{regime_payload['prior_text']}"
    )


def build_continuous_action_initial_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> str:
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        "You do not have access to the full clip.\n"
        "Return exactly one line:\n"
        "- QUERY <start_seconds> <end_seconds>\n"
        "- STOP\n"
        "Timestamps must use exactly two decimals, for example QUERY 2.00 5.00.\n"
        "Choose your first action now."
    )


def build_continuous_action_followup_prompt(metadata: dict[str, Any], env: ContinuousEpisodeEnvironment) -> str:
    return (
        f"You inspected seconds {format_seconds(metadata['start_time_seconds'])} to "
        f"{format_seconds(metadata['end_time_seconds'])}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining listening budget: {format_seconds(env.remaining_seconds_budget())} seconds.\n"
        "Return exactly one line:\n"
        "- QUERY <start_seconds> <end_seconds>\n"
        "- STOP\n"
        "Timestamps must use exactly two decimals."
    )


def build_continuous_action_repair_prompt(reason: str, env: ContinuousEpisodeEnvironment) -> str:
    return (
        f"Your previous action was invalid: {reason}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining listening budget: {format_seconds(env.remaining_seconds_budget())} seconds.\n"
        "Return exactly one valid line now:\n"
        "- QUERY <start_seconds> <end_seconds>\n"
        "- STOP\n"
        "Timestamps must use exactly two decimals."
    )


def _continuous_query_history_text(action_history: Sequence[dict[str, Any]]) -> str:
    if not action_history:
        return "No intervals have been queried yet."
    return "\n".join(
        f"step {item['step_index']}: seconds {format_seconds(item['start_time_seconds'])} "
        f"to {format_seconds(item['end_time_seconds'])}"
        for item in action_history
    )


def _continuous_probe_prompt(
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    *,
    full_context: bool,
    same_session: bool,
    probe_kind: str = "",
) -> str:
    if full_context:
        evidence_text = "You have access to the full audio clip."
    elif same_session:
        evidence_text = "Report your current global timeline belief based only on the exploration so far."
    else:
        evidence_text = "The attached audio crops are the intervals you have already queried, in chronological order."
    overlap_text = " Multiple labels in one segment mean overlap." if config.track_id == TRACK_1B else ""
    checkpoint = f"This is a {probe_kind} belief probe.\n" if probe_kind else ""
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"{checkpoint}"
        f"{evidence_text}\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        "Return a continuous timeline belief with one segment per line and no extra text.\n"
        "Each row must be: START <seconds> END <seconds> LABEL <label(s)>.\n"
        "All timestamps must use exactly two decimals.\n"
        "Use only vocabulary labels, 00 for silence, or ?? for unexplored."
        f"{overlap_text}\n"
        f"Cover the full interval from START 0.00 to END {format_seconds(episode.get('audio_duration_seconds', config.episode_duration_sec))}."
    )


def _continuous_probe_repair_prompt(reason: str, episode: dict[str, Any], config: BenchmarkConfig) -> str:
    return (
        f"Your previous probe output was invalid: {reason}.\n"
        "Return exactly one valid continuous timeline with no explanation.\n"
        "Each row must be: START <seconds> END <seconds> LABEL <label(s)>.\n"
        "All timestamps must use exactly two decimals.\n"
        f"Cover the full interval from START 0.00 to END {format_seconds(episode.get('audio_duration_seconds', config.episode_duration_sec))}."
    )


def _parse_segments_for_episode(raw_text: str, episode: dict[str, Any], config: BenchmarkConfig, vocabulary: Sequence[str]) -> dict[str, Any]:
    return parse_continuous_segments(
        raw_text,
        allowed_labels=vocabulary,
        audio_duration_seconds=float(episode.get("audio_duration_seconds", config.episode_duration_sec)),
        allow_multi_label=config.track_id == TRACK_1B,
    )


def _run_continuous_probe(
    adapter: BaseAdapter,
    session: Any,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
    *,
    full_context: bool,
    same_session: bool,
    probe_kind: str = "",
) -> dict[str, Any]:
    prompt = _continuous_probe_prompt(
        episode,
        regime_payload,
        config,
        vocabulary,
        action_history,
        full_context=full_context,
        same_session=same_session,
        probe_kind=probe_kind,
    )
    raw, parsed, repairs, api_attempts = _send_with_one_repair(
        adapter,
        session,
        prompt,
        lambda text: _parse_segments_for_episode(text, episode, config, vocabulary),
        audio_paths,
        "continuous_probe",
        api_retry_sleep_sec=config.api_retry_sleep_sec,
        repair_prompt_fn=lambda reason: _continuous_probe_repair_prompt(reason, episode, config),
        preserve_multiline=True,
    )
    query_intervals = _query_intervals_from_history(action_history)
    return {
        "probe_kind": probe_kind,
        "checkpoint_step": len(action_history),
        "query_intervals": [
            {"start_time_seconds": start, "end_time_seconds": end}
            for start, end in query_intervals
        ],
        "listened_seconds_total": sum(float(item["duration_seconds"]) for item in action_history),
        "prompt": prompt,
        "raw_output": raw,
        "parsed": parsed,
        "repair_attempts": repairs,
        "api_attempts": api_attempts,
        "belief_metrics": score_continuous_belief_metrics(parsed, episode, query_intervals),
    }


def _query_intervals_from_history(action_history: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(item["start_time_seconds"]), float(item["end_time_seconds"]))
        for item in action_history
    ]


def _continuous_event_spans(episode: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        label: {
            "start_time_seconds": float(span["start_time_seconds"]),
            "end_time_seconds": float(span["end_time_seconds"]),
        }
        for label, span in episode["ground_truth_event_spans"].items()
    }


def _event_is_observed(span: dict[str, float], query_intervals: Sequence[tuple[float, float]]) -> bool:
    return interval_contained(
        query_intervals,
        (float(span["start_time_seconds"]), float(span["end_time_seconds"])),
    )


def _build_continuous_behavioral_plan(
    episode: dict[str, Any],
    query_intervals: Sequence[tuple[float, float]],
    *,
    full_context: bool,
) -> dict[str, Any]:
    sequence = list(episode["ground_truth_sequence"])
    spans = _continuous_event_spans(episode)
    if full_context:
        observed_set = set(sequence)
    else:
        observed_set = {label for label in sequence if _event_is_observed(spans[label], query_intervals)}
    explored_events = [label for label in sequence if label in observed_set]
    unexplored_events = [label for label in sequence if label not in observed_set]
    ordered_pairs = [
        (label_a, label_b)
        for index, label_a in enumerate(sequence)
        for label_b in sequence[index + 1 :]
    ]
    consecutive_pairs = [
        (relation["left_label"], relation["right_label"])
        for relation in episode.get("ground_truth_gap_relations", [])
    ]

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
                "answer": relation_from_spans(spans[label_a], spans[label_b]),
            }
        )

    core_localization = []
    if explored_events:
        label = explored_events[0]
        span = spans[label]
        core_localization.append(
            {
                "query_id": "core_localization_1",
                "label": label,
                "answer": f"START {format_seconds(span['start_time_seconds'])} END {format_seconds(span['end_time_seconds'])}",
                "gold_span": dict(span),
            }
        )

    core_gap = []
    for left_label, right_label in consecutive_pairs:
        if left_label in observed_set and right_label in observed_set:
            core_gap.append(
                {
                    "query_id": "core_gap_1",
                    "left_label": left_label,
                    "right_label": right_label,
                    "answer": gap_from_spans(spans[left_label], spans[right_label]),
                }
            )
            break

    uncertainty_pairwise = []
    for label_a, label_b in ordered_pairs:
        if label_a not in observed_set or label_b not in observed_set:
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
        if left_label not in observed_set or right_label not in observed_set:
            uncertainty_gap.append(
                {
                    "query_id": "uncertainty_gap_1",
                    "left_label": left_label,
                    "right_label": right_label,
                    "answer": "UNEXPLORED",
                }
            )
            break

    core_overlap: list[dict[str, Any]] = []
    core_overlap_localization: list[dict[str, Any]] = []
    core_max_concurrency: list[dict[str, Any]] = []
    uncertainty_overlap: list[dict[str, Any]] = []
    if episode.get("track_id") == TRACK_1B:
        overlap_relations = episode.get("ground_truth_overlap_relations", [])
        for label_a, label_b in ordered_pairs:
            if label_a in observed_set and label_b in observed_set:
                relation = next(
                    (
                        item
                        for item in overlap_relations
                        if {item["label_a"], item["label_b"]} == {label_a, label_b}
                    ),
                    None,
                )
                core_overlap.append(
                    {
                        "query_id": "core_overlap_1",
                        "label_a": label_a,
                        "label_b": label_b,
                        "answer": "YES" if relation else "NO",
                    }
                )
                if relation:
                    core_overlap_localization.append(
                        {
                            "query_id": "core_overlap_interval_1",
                            "label_a": label_a,
                            "label_b": label_b,
                            "answer": (
                                f"START {format_seconds(relation['start_time_seconds'])} "
                                f"END {format_seconds(relation['end_time_seconds'])}"
                            ),
                            "gold_span": {
                                "start_time_seconds": float(relation["start_time_seconds"]),
                                "end_time_seconds": float(relation["end_time_seconds"]),
                            },
                        }
                    )
                break
            uncertainty_overlap.append(
                {
                    "query_id": "uncertainty_overlap_1",
                    "label_a": label_a,
                    "label_b": label_b,
                    "answer": "UNEXPLORED",
                }
            )
            break
        if len(observed_set) == len(sequence):
            core_max_concurrency.append(
                {
                    "query_id": "core_max_concurrency_1",
                    "answer": str(int(episode.get("ground_truth_max_concurrency", 0))),
                }
            )

    coverage_metrics = {
        "event_coverage_rate": float(len(explored_events) / len(sequence)) if sequence else 0.0,
        "event_coverage_count": float(len(explored_events)),
        "pairwise_core_evaluable": float(bool(core_pairwise)),
        "localization_core_evaluable": float(bool(core_localization)),
        "gap_core_evaluable": float(bool(core_gap)),
        "overlap_core_evaluable": float(bool(core_overlap)) if episode.get("track_id") == TRACK_1B else None,
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
            "overlap": core_overlap,
            "overlap_localization": core_overlap_localization,
            "max_concurrency": core_max_concurrency,
        },
        "uncertainty": {
            "pairwise_order": uncertainty_pairwise,
            "localization": uncertainty_localization,
            "gap_adjacency": uncertainty_gap,
            "overlap": uncertainty_overlap,
        },
    }


def _parse_yes_no_unexplored(raw_text: str, *, allow_unexplored: bool = False) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).strip().upper()
    allowed = {"YES", "NO"}
    if allow_unexplored:
        allowed.add("UNEXPLORED")
    if stripped not in allowed:
        return {"format_valid": False, "reason": "invalid_yes_no_answer"}
    return {"format_valid": True, "answer": stripped}


def _parse_max_concurrency(raw_text: str) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).strip()
    if not re.fullmatch(r"\d+", stripped):
        return {"format_valid": False, "reason": "invalid_max_concurrency"}
    return {"format_valid": True, "answer": stripped}


def _continuous_sequence_prompt(episode, regime_payload, config, vocabulary, action_history, *, full_context, same_session, explored_only):
    evidence = "the full clip attached below" if full_context else "the queried clips"
    if same_session:
        evidence = "the exploration so far in this same session"
    task_text = (
        "Return the left-to-right ordered event sequence for only events directly established by the evidence. "
        "Do not guess undiscovered events."
        if explored_only
        else "Return the left-to-right ordered event sequence."
    )
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"{task_text} Use exactly this delimiter: {SEQUENCE_DELIMITER}\n"
        "Return only one line and nothing else."
    )


def _continuous_pairwise_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, full_context, same_session, allow_unexplored):
    evidence = "the full clip attached below" if full_context else "the queried clips"
    if same_session:
        evidence = "the exploration so far in this same session"
    answer_text = (
        "Return exactly BEFORE, AFTER, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly BEFORE or AFTER."
    )
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"Does {query['label_a']} occur BEFORE or AFTER {query['label_b']}?\n"
        f"{answer_text}"
    )


def _continuous_localization_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, full_context, same_session, allow_unexplored):
    evidence = "the full clip attached below" if full_context else "the queried clips"
    if same_session:
        evidence = "the exploration so far in this same session"
    answer_text = (
        "Return exactly START <seconds> END <seconds>, or UNEXPLORED if the evidence does not directly establish this event."
        if allow_unexplored
        else "Return exactly START <seconds> END <seconds>."
    )
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"Give the start and end timestamps for {query['label']}.\n"
        "Timestamps must use exactly two decimals.\n"
        f"{answer_text}"
    )


def _continuous_gap_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, full_context, same_session, allow_unexplored):
    evidence = "the full clip attached below" if full_context else "the queried clips"
    if same_session:
        evidence = "the exploration so far in this same session"
    answer_text = (
        "Return exactly ADJACENT, GAP, or UNEXPLORED. If the evidence does not directly establish one or both events, return UNEXPLORED."
        if allow_unexplored
        else "Return exactly ADJACENT or GAP."
    )
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"Are {query['left_label']} and {query['right_label']} ADJACENT, or is there a GAP?\n"
        f"{answer_text}"
    )


def _continuous_overlap_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, same_session, allow_unexplored):
    evidence = "the exploration so far in this same session" if same_session else "the queried clips"
    answer_text = "Return exactly YES, NO, or UNEXPLORED." if allow_unexplored else "Return exactly YES or NO."
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"Do {query['label_a']} and {query['label_b']} overlap in time?\n"
        f"{answer_text}"
    )


def _continuous_overlap_interval_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, same_session):
    evidence = "the exploration so far in this same session" if same_session else "the queried clips"
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"When does the overlap between {query['label_a']} and {query['label_b']} occur?\n"
        "Return exactly START <seconds> END <seconds>. Timestamps must use exactly two decimals."
    )


def _continuous_max_concurrency_prompt(episode, regime_payload, config, vocabulary, action_history, query, *, same_session):
    evidence = "the exploration so far in this same session" if same_session else "the queried clips"
    return (
        f"{_continuous_context_header(episode, regime_payload, config, vocabulary)}\n"
        f"Use {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        "What is the maximum number of simultaneous events directly established by the evidence?\n"
        "Return exactly one integer."
    )


def _empty_continuous_task(reason: str = "not_evaluable") -> dict[str, Any]:
    return {
        "asked": False,
        "prompt": "",
        "raw_output": "",
        "parsed": {"format_valid": False, "reason": reason},
        "repair_attempts": [],
    }


def _score_continuous_behavioral_outputs(outputs: dict[str, Any], plan: dict[str, Any], prefix: str) -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        f"{prefix}_behavior_sequence_exact": None,
        f"{prefix}_behavior_pairwise_accuracy": None,
        f"{prefix}_behavior_localization_exact_250ms": None,
        f"{prefix}_behavior_localization_iou": None,
        f"{prefix}_behavior_gap_accuracy": None,
        f"{prefix}_behavior_overlap_accuracy": None,
        f"{prefix}_behavior_overlap_iou": None,
        f"{prefix}_behavior_max_concurrency_accuracy": None,
    }
    if plan.get("sequence"):
        parsed = outputs.get("sequence", {}).get("parsed", {})
        expected = [token.strip() for token in str(plan["sequence"]["answer"]).split(SEQUENCE_DELIMITER)]
        metrics[f"{prefix}_behavior_sequence_exact"] = float(parsed.get("format_valid") and parsed.get("tokens") == expected)
    if plan.get("pairwise_order"):
        scores = []
        for query in plan["pairwise_order"]:
            parsed = outputs.get("pairwise_order", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_pairwise_accuracy"] = float(sum(scores) / len(scores))
    if plan.get("localization"):
        exact_scores = []
        ious = []
        for query in plan["localization"]:
            parsed = outputs.get("localization", {}).get(query["query_id"], {}).get("parsed", {})
            if not parsed.get("format_valid"):
                exact_scores.append(0.0)
                ious.append(0.0)
                continue
            gold = query["gold_span"]
            pred_interval = (parsed["start_time_seconds"], parsed["end_time_seconds"])
            gold_interval = (gold["start_time_seconds"], gold["end_time_seconds"])
            exact_scores.append(localization_exact_within_threshold(pred_interval, gold_interval))
            ious.append(interval_iou(pred_interval, gold_interval))
        metrics[f"{prefix}_behavior_localization_exact_250ms"] = float(sum(exact_scores) / len(exact_scores))
        metrics[f"{prefix}_behavior_localization_iou"] = float(sum(ious) / len(ious))
    if plan.get("gap_adjacency"):
        scores = []
        for query in plan["gap_adjacency"]:
            parsed = outputs.get("gap_adjacency", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_gap_accuracy"] = float(sum(scores) / len(scores))
    if plan.get("overlap"):
        scores = []
        for query in plan["overlap"]:
            parsed = outputs.get("overlap", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_overlap_accuracy"] = float(sum(scores) / len(scores))
    if plan.get("overlap_localization"):
        ious = []
        for query in plan["overlap_localization"]:
            parsed = outputs.get("overlap_localization", {}).get(query["query_id"], {}).get("parsed", {})
            if not parsed.get("format_valid"):
                ious.append(0.0)
                continue
            gold = query["gold_span"]
            ious.append(
                interval_iou(
                    (parsed["start_time_seconds"], parsed["end_time_seconds"]),
                    (gold["start_time_seconds"], gold["end_time_seconds"]),
                )
            )
        metrics[f"{prefix}_behavior_overlap_iou"] = float(sum(ious) / len(ious))
    if plan.get("max_concurrency"):
        scores = []
        for query in plan["max_concurrency"]:
            parsed = outputs.get("max_concurrency", {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}_behavior_max_concurrency_accuracy"] = float(sum(scores) / len(scores))
    return metrics


def _score_continuous_uncertainty(outputs: dict[str, Any], plan: dict[str, Any], prefix: str = "") -> dict[str, float | None]:
    metrics: dict[str, float | None] = {
        f"{prefix}uncertainty_pairwise_accuracy": None,
        f"{prefix}uncertainty_localization_accuracy": None,
        f"{prefix}uncertainty_gap_accuracy": None,
        f"{prefix}uncertainty_overlap_accuracy": None,
        f"{prefix}uncertainty_pairwise_count": float(len(plan.get("pairwise_order", []))),
        f"{prefix}uncertainty_localization_count": float(len(plan.get("localization", []))),
        f"{prefix}uncertainty_gap_count": float(len(plan.get("gap_adjacency", []))),
        f"{prefix}uncertainty_overlap_count": float(len(plan.get("overlap", []))),
    }
    for key, output_key in [
        ("pairwise", "pairwise_order"),
        ("localization", "localization"),
        ("gap", "gap_adjacency"),
        ("overlap", "overlap"),
    ]:
        queries = plan.get(output_key, [])
        if not queries:
            continue
        scores = []
        for query in queries:
            parsed = outputs.get(output_key, {}).get(query["query_id"], {}).get("parsed", {})
            scores.append(float(parsed.get("format_valid") and parsed.get("answer") == query["answer"]))
        metrics[f"{prefix}uncertainty_{key}_accuracy"] = float(sum(scores) / len(scores))
    return metrics


def _run_continuous_behavioral_prompts(
    adapter: BaseAdapter,
    session: Any | None,
    episode: dict[str, Any],
    regime_payload: dict[str, Any],
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
    *,
    full_context: bool,
    same_session: bool,
) -> dict[str, Any]:
    query_intervals = _query_intervals_from_history(action_history)
    plan = _build_continuous_behavioral_plan(episode, query_intervals, full_context=full_context)
    audio_for_questions = list(audio_paths)

    def _session() -> Any:
        return session if same_session and session is not None else adapter.create_session(ACTION_SYSTEM_PROMPT)

    def _run_one(prompt: str, parse_fn, repair_kind: str, *, preserve_multiline: bool = False) -> dict[str, Any]:
        prompt_session = _session()
        try:
            raw, parsed, repairs, api_attempts = _send_with_one_repair(
                adapter,
                prompt_session,
                prompt,
                parse_fn,
                [] if same_session else audio_for_questions,
                repair_kind,
                api_retry_sleep_sec=config.api_retry_sleep_sec,
                preserve_multiline=preserve_multiline,
            )
        finally:
            if not same_session:
                _close_adapter_session(adapter, prompt_session)
        return {
            "asked": True,
            "prompt": prompt,
            "raw_output": raw,
            "parsed": parsed,
            "repair_attempts": repairs,
            "api_attempts": api_attempts,
        }

    core_outputs = {
        "sequence": _empty_continuous_task(),
        "pairwise_order": {},
        "localization": {},
        "gap_adjacency": {},
        "overlap": {},
        "overlap_localization": {},
        "max_concurrency": {},
    }
    uncertainty_outputs = {"pairwise_order": {}, "localization": {}, "gap_adjacency": {}, "overlap": {}}
    if plan["core"]["sequence"] is not None:
        prompt = _continuous_sequence_prompt(
            episode, regime_payload, config, vocabulary, action_history,
            full_context=full_context, same_session=same_session, explored_only=True,
        )
        core_outputs["sequence"] = _run_one(prompt, lambda text: parse_sequence_answer(text, vocabulary), "sequence")
    for query in plan["core"]["pairwise_order"]:
        prompt = _continuous_pairwise_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=False,
        )
        core_outputs["pairwise_order"][query["query_id"]] = _run_one(prompt, lambda text: parse_pairwise_answer(text), "pairwise")
    for query in plan["core"]["localization"]:
        prompt = _continuous_localization_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=False,
        )
        core_outputs["localization"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_continuous_localization_answer(
                text,
                audio_duration_seconds=float(episode["audio_duration_seconds"]),
            ),
            "localization",
        )
    for query in plan["core"]["gap_adjacency"]:
        prompt = _continuous_gap_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=False,
        )
        core_outputs["gap_adjacency"][query["query_id"]] = _run_one(prompt, lambda text: parse_gap_answer(text), "gap")
    for query in plan["core"]["overlap"]:
        prompt = _continuous_overlap_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            same_session=same_session, allow_unexplored=False,
        )
        core_outputs["overlap"][query["query_id"]] = _run_one(prompt, lambda text: _parse_yes_no_unexplored(text), "overlap")
    for query in plan["core"]["overlap_localization"]:
        prompt = _continuous_overlap_interval_prompt(
            episode, regime_payload, config, vocabulary, action_history, query, same_session=same_session
        )
        core_outputs["overlap_localization"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_continuous_localization_answer(
                text,
                audio_duration_seconds=float(episode["audio_duration_seconds"]),
            ),
            "overlap_localization",
        )
    for query in plan["core"]["max_concurrency"]:
        prompt = _continuous_max_concurrency_prompt(
            episode, regime_payload, config, vocabulary, action_history, query, same_session=same_session
        )
        core_outputs["max_concurrency"][query["query_id"]] = _run_one(prompt, _parse_max_concurrency, "max_concurrency")

    for query in plan["uncertainty"]["pairwise_order"]:
        prompt = _continuous_pairwise_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=True,
        )
        uncertainty_outputs["pairwise_order"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_pairwise_answer(text, allow_unexplored=True),
            "pairwise",
        )
    for query in plan["uncertainty"]["localization"]:
        prompt = _continuous_localization_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=True,
        )
        uncertainty_outputs["localization"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_continuous_localization_answer(
                text,
                audio_duration_seconds=float(episode["audio_duration_seconds"]),
                allow_unexplored=True,
            ),
            "localization",
        )
    for query in plan["uncertainty"]["gap_adjacency"]:
        prompt = _continuous_gap_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            full_context=full_context, same_session=same_session, allow_unexplored=True,
        )
        uncertainty_outputs["gap_adjacency"][query["query_id"]] = _run_one(
            prompt,
            lambda text: parse_gap_answer(text, allow_unexplored=True),
            "gap",
        )
    for query in plan["uncertainty"]["overlap"]:
        prompt = _continuous_overlap_prompt(
            episode, regime_payload, config, vocabulary, action_history, query,
            same_session=same_session, allow_unexplored=True,
        )
        uncertainty_outputs["overlap"][query["query_id"]] = _run_one(
            prompt,
            lambda text: _parse_yes_no_unexplored(text, allow_unexplored=True),
            "overlap",
        )

    prefix = "same_session_core" if same_session else "core"
    uncertainty_prefix = "same_session_" if same_session else ""
    core_metrics = _score_continuous_behavioral_outputs(core_outputs, plan["core"], prefix)
    uncertainty_metrics = _score_continuous_uncertainty(uncertainty_outputs, plan["uncertainty"], uncertainty_prefix)
    behavioral_metrics = {
        f"{'same_session_' if same_session else ''}behavior_sequence_exact": core_metrics[f"{prefix}_behavior_sequence_exact"],
        f"{'same_session_' if same_session else ''}behavior_pairwise_accuracy": core_metrics[f"{prefix}_behavior_pairwise_accuracy"],
        f"{'same_session_' if same_session else ''}behavior_localization_exact_250ms": core_metrics[f"{prefix}_behavior_localization_exact_250ms"],
        f"{'same_session_' if same_session else ''}behavior_localization_iou": core_metrics[f"{prefix}_behavior_localization_iou"],
        f"{'same_session_' if same_session else ''}behavior_gap_accuracy": core_metrics[f"{prefix}_behavior_gap_accuracy"],
    }
    result = {
        "behavioral_protocol_version": CONTINUOUS_PROTOCOL_VERSION,
        "coverage_metrics": plan["coverage_metrics"],
        "core_behavioral_outputs": core_outputs,
        "core_behavioral_metrics": core_metrics,
        "uncertainty_outputs": uncertainty_outputs,
        "uncertainty_metrics": uncertainty_metrics,
        "behavioral_outputs": core_outputs,
        "behavioral_metrics": behavioral_metrics,
    }
    if same_session:
        return {
            "behavioral_protocol_version": CONTINUOUS_PROTOCOL_VERSION,
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
    return result


def score_continuous_belief_metrics(
    final_segments: dict[str, Any] | None,
    episode: dict[str, Any],
    query_intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    metrics = {
        "belief_time_score_mean": 0.0,
        "belief_sequence_exact": 0.0,
        "belief_pairwise_accuracy": 0.0,
        "belief_localization_exact_250ms": 0.0,
        "belief_localization_mean_iou": 0.0,
        "belief_gap_accuracy": 0.0,
        "belief_overlap_pair_f1": 0.0,
        "belief_overlap_interval_iou": 0.0,
        "max_concurrency_accuracy": 0.0,
    }
    if not final_segments or not final_segments.get("format_valid"):
        return metrics
    predicted_segments = final_segments["segments"]
    expected_segments = episode["ground_truth_segments"]
    metrics["belief_time_score_mean"] = segment_time_accuracy(
        predicted_segments,
        expected_segments,
        audio_duration_seconds=float(episode["audio_duration_seconds"]),
        observed_intervals=query_intervals,
    )
    predicted_spans = spans_from_segments(predicted_segments)
    gold_spans = _continuous_event_spans(episode)
    predicted_sequence = [
        label
        for label, _ in sorted(predicted_spans.items(), key=lambda item: (item[1]["start_time_seconds"], item[0]))
    ]
    gold_sequence = list(episode["ground_truth_sequence"])
    metrics["belief_sequence_exact"] = float(predicted_sequence == gold_sequence)
    pair_scores = []
    for index, label_a in enumerate(gold_sequence):
        for label_b in gold_sequence[index + 1 :]:
            if label_a not in predicted_spans or label_b not in predicted_spans:
                pair_scores.append(0.0)
                continue
            pair_scores.append(
                float(
                    relation_from_spans(predicted_spans[label_a], predicted_spans[label_b])
                    == relation_from_spans(gold_spans[label_a], gold_spans[label_b])
                )
            )
    if pair_scores:
        metrics["belief_pairwise_accuracy"] = float(sum(pair_scores) / len(pair_scores))
    exact_scores = []
    ious = []
    for label, gold_span in gold_spans.items():
        pred_span = predicted_spans.get(label)
        if pred_span is None:
            exact_scores.append(0.0)
            ious.append(0.0)
            continue
        pred_interval = (pred_span["start_time_seconds"], pred_span["end_time_seconds"])
        gold_interval = (gold_span["start_time_seconds"], gold_span["end_time_seconds"])
        exact_scores.append(localization_exact_within_threshold(pred_interval, gold_interval))
        ious.append(interval_iou(pred_interval, gold_interval))
    if exact_scores:
        metrics["belief_localization_exact_250ms"] = float(sum(exact_scores) / len(exact_scores))
        metrics["belief_localization_mean_iou"] = float(sum(ious) / len(ious))
    gap_scores = []
    for relation in episode.get("ground_truth_gap_relations", []):
        left = relation["left_label"]
        right = relation["right_label"]
        if left not in predicted_spans or right not in predicted_spans:
            gap_scores.append(0.0)
            continue
        gap_scores.append(float(gap_from_spans(predicted_spans[left], predicted_spans[right]) == relation["relation"]))
    if gap_scores:
        metrics["belief_gap_accuracy"] = float(sum(gap_scores) / len(gap_scores))
    if episode.get("track_id") == TRACK_1B:
        gold_pairs = {
            tuple(sorted((item["label_a"], item["label_b"])))
            for item in episode.get("ground_truth_overlap_relations", [])
        }
        predicted_pairs = set()
        for segment in predicted_segments:
            labels = segment.get("labels") or []
            if len(labels) >= 2:
                for left_index, left in enumerate(labels):
                    for right in labels[left_index + 1 :]:
                        predicted_pairs.add(tuple(sorted((left, right))))
        true_positive = len(gold_pairs & predicted_pairs)
        precision = true_positive / len(predicted_pairs) if predicted_pairs else 0.0
        recall = true_positive / len(gold_pairs) if gold_pairs else 0.0
        metrics["belief_overlap_pair_f1"] = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall > 0.0
            else 0.0
        )
        overlap_ious = []
        for relation in episode.get("ground_truth_overlap_relations", []):
            labels = {relation["label_a"], relation["label_b"]}
            pred_intervals = [
                (float(segment["start_time_seconds"]), float(segment["end_time_seconds"]))
                for segment in predicted_segments
                if labels <= set(segment.get("labels") or [])
            ]
            gold_interval = (float(relation["start_time_seconds"]), float(relation["end_time_seconds"]))
            pred_duration = interval_set_unique_duration(pred_intervals)
            if pred_duration <= 0.0:
                overlap_ious.append(0.0)
                continue
            intersection = interval_set_intersection_duration(pred_intervals, [gold_interval])
            union = pred_duration + (gold_interval[1] - gold_interval[0]) - intersection
            overlap_ious.append(float(intersection / union) if union > 0.0 else 0.0)
        if overlap_ious:
            metrics["belief_overlap_interval_iou"] = float(sum(overlap_ious) / len(overlap_ious))
        metrics["max_concurrency_accuracy"] = float(
            max_concurrency_from_segments(predicted_segments) == int(episode.get("ground_truth_max_concurrency", 0))
        )
    return metrics


def _score_continuous_exploration_metrics(
    episode: dict[str, Any],
    config: BenchmarkConfig,
    action_attempts: Sequence[dict[str, Any]],
    action_history: Sequence[dict[str, Any]],
) -> dict[str, float | None]:
    query_intervals = _query_intervals_from_history(action_history)
    listened, unique, redundant_query_count, partial_redundant_query_count = continuous_redundancy_counts(query_intervals)
    repeated = max(0.0, listened - unique)
    event_spans = _continuous_event_spans(episode)
    event_intervals = [
        (span["start_time_seconds"], span["end_time_seconds"])
        for span in event_spans.values()
    ]
    event_union = merge_intervals(event_intervals)
    full_duration = float(episode["audio_duration_seconds"])
    silent_intervals: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in event_union:
        if start > cursor:
            silent_intervals.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < full_duration:
        silent_intervals.append((cursor, full_duration))
    event_time_covered = interval_set_intersection_duration(query_intervals, event_union)
    silent_time_covered = interval_set_intersection_duration(query_intervals, silent_intervals)
    observed_events = [
        label
        for label, span in event_spans.items()
        if _event_is_observed(span, query_intervals)
    ]
    total_budget = config.max_total_queried_bins * config.bin_duration_sec
    return {
        "query_count": float(len(action_history)),
        "action_attempt_count": float(len(action_attempts)),
        "invalid_action_count": float(sum(float(not item.get("semantic_valid", False)) for item in action_attempts)),
        "repair_attempted_action_count": float(sum(float(bool(item.get("repair_attempted"))) for item in action_attempts)),
        "budget_usage_rate": float(listened / total_budget) if total_budget else 0.0,
        "step_usage_rate": float(len(action_history) / config.max_steps) if config.max_steps else 0.0,
        "listened_seconds_total": float(listened),
        "observed_seconds_unique": float(unique),
        "repeated_seconds_total": float(repeated),
        "event_coverage_rate": float(len(observed_events) / len(event_spans)) if event_spans else 0.0,
        "event_coverage_count": float(len(observed_events)),
        "event_time_coverage_seconds": float(event_time_covered),
        "event_time_coverage_rate": float(event_time_covered / interval_set_unique_duration(event_union)) if event_union else 0.0,
        "silent_time_coverage_seconds": float(silent_time_covered),
        "silent_time_coverage_rate": float(silent_time_covered / interval_set_unique_duration(silent_intervals)) if silent_intervals else 0.0,
        "redundant_query_count": float(redundant_query_count),
        "partial_redundant_query_count": float(partial_redundant_query_count),
    }


def _continuous_format_metrics(
    action_attempts: Sequence[dict[str, Any]],
    probes: Sequence[dict[str, Any]],
    output_groups: Sequence[Any],
) -> dict[str, float | None]:
    action_format = [float(item["final_parse"].get("format_valid", False)) for item in action_attempts]
    action_semantic = [float(item.get("semantic_valid", False)) for item in action_attempts]
    probe_valid = [float(item["parsed"].get("format_valid", False)) for item in probes]
    repair_attempts = 0
    repair_successes = 0
    for item in action_attempts:
        if item.get("repair_attempted"):
            repair_attempts += 1
            repair_successes += int(item["final_parse"].get("format_valid", False) and item.get("semantic_valid", False))
    stack = list(probes) + list(output_groups)
    while stack:
        item = stack.pop()
        if isinstance(item, dict) and "repair_attempts" in item and "parsed" in item:
            if item.get("asked", True) and item["repair_attempts"]:
                repair_attempts += 1
                repair_successes += int(item["parsed"].get("format_valid", False))
            continue
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return {
        "action_format_valid_rate": float(sum(action_format) / len(action_format)) if action_format else None,
        "action_semantic_valid_rate": float(sum(action_semantic) / len(action_semantic)) if action_semantic else None,
        "probe_valid_rate": float(sum(probe_valid) / len(probe_valid)) if probe_valid else None,
        "repair_success_rate": float(repair_successes / repair_attempts) if repair_attempts else None,
    }


def _continuous_validate(parsed: dict[str, Any], env: ContinuousEpisodeEnvironment) -> tuple[bool, str]:
    return validate_continuous_query(
        parsed,
        audio_duration_seconds=env.audio_duration_seconds,
        max_single_query_seconds=env.max_single_query_seconds,
        total_listening_budget_seconds=env.total_listening_budget_seconds,
        listened_seconds_total=env.listened_seconds_total,
    )


def _run_continuous_action_condition(
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
    run_dir: Path,
    *,
    same_session: bool,
) -> dict[str, Any]:
    regime_payload = episode["regime_variants"][regime]
    env = ContinuousEpisodeEnvironment(root, episode, config, run_dir)
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    prompt = build_continuous_action_initial_prompt(episode, regime_payload, config, vocabulary)
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
        parsed = parse_continuous_action(raw_action)
        repair_attempted = False
        api_attempts = list(raw_action_api_attempts)
        if parsed.get("format_valid") and parsed.get("action") == "QUERY":
            semantic_valid, reason = _continuous_validate(parsed, env)
        else:
            semantic_valid = bool(parsed.get("format_valid"))
            reason = parsed.get("reason", "")
        if not parsed.get("format_valid") or (parsed.get("action") == "QUERY" and not semantic_valid):
            repair_attempted = True
            repair_reason = reason or parsed.get("reason", "invalid_action")
            repaired_raw, repair_api_attempts = _send_with_api_retry(
                adapter,
                session,
                build_continuous_action_repair_prompt(repair_reason, env),
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
                repaired = parse_continuous_action(repaired_raw)
                if repaired.get("format_valid") and repaired.get("action") == "QUERY":
                    semantic_valid, reason = _continuous_validate(repaired, env)
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
            final_parse["start_time_seconds"],
            final_parse["end_time_seconds"],
        )
        consumed_latest_audio = False
        if len(env.action_history) % config.probe_every_n == 0:
            probe_session = session if same_session else adapter.create_session(ACTION_SYSTEM_PROMPT)
            try:
                probe_result = _run_continuous_probe(
                    adapter,
                    probe_session,
                    episode,
                    regime_payload,
                    config,
                    vocabulary,
                    env.action_history,
                    [clip_path] if same_session else env.query_audio_paths,
                    full_context=False,
                    same_session=same_session,
                    probe_kind="periodic" if same_session else "",
                )
            finally:
                if not same_session:
                    _close_adapter_session(adapter, probe_session)
            probes.append(probe_result)
            env.repair_log.extend(probe_result["repair_attempts"])
            consumed_latest_audio = same_session
        if len(env.action_history) >= config.max_steps:
            env.stop_reason = "max_steps_reached"
            final_probe_audio_paths = [] if consumed_latest_audio else [clip_path]
            break
        if env.listened_seconds_total >= env.total_listening_budget_seconds - 1e-9:
            env.stop_reason = "max_total_queried_seconds_reached"
            final_probe_audio_paths = [] if consumed_latest_audio else [clip_path]
            break
        raw_action, raw_action_api_attempts = _send_with_api_retry(
            adapter,
            session,
            build_continuous_action_followup_prompt(metadata, env),
            [] if same_session and consumed_latest_audio else [clip_path],
            config.api_retry_sleep_sec,
        )

    final_probe_session = session if same_session else adapter.create_session(ACTION_SYSTEM_PROMPT)
    try:
        final_probe = _run_continuous_probe(
            adapter,
            final_probe_session,
            episode,
            regime_payload,
            config,
            vocabulary,
            env.action_history,
            final_probe_audio_paths if same_session else env.query_audio_paths,
            full_context=False,
            same_session=same_session,
            probe_kind="final" if same_session else "",
        )
    finally:
        if not same_session:
            _close_adapter_session(adapter, final_probe_session)
    probes.append(final_probe)
    env.repair_log.extend(final_probe["repair_attempts"])

    behavioral_outputs = _run_continuous_behavioral_prompts(
        adapter,
        session if same_session else None,
        episode,
        regime_payload,
        config,
        vocabulary,
        env.action_history,
        env.query_audio_paths,
        full_context=False,
        same_session=same_session,
    )
    format_metrics = _continuous_format_metrics(
        action_attempts,
        probes,
        [
            behavioral_outputs.get("core_behavioral_outputs", {}),
            behavioral_outputs.get("uncertainty_outputs", {}),
            behavioral_outputs.get("same_session_core_behavioral_outputs", {}),
            behavioral_outputs.get("same_session_uncertainty_outputs", {}),
        ],
    )
    exploration_metrics = _score_continuous_exploration_metrics(episode, config, action_attempts, env.action_history)
    master_metrics = score_generic_master_metrics(
        track_id=config.track_id,
        condition=ACTIVE_SAME_SESSION_PERIODIC_PROBE if same_session else ACTIVE,
        stop_reason=env.stop_reason,
        listened_bin_total=env.listened_seconds_total,
        format_metrics=format_metrics,
        exploration_metrics=exploration_metrics,
        belief_metrics=final_probe["belief_metrics"],
        core_metrics=behavioral_outputs.get("same_session_core_behavioral_metrics", behavioral_outputs.get("core_behavioral_metrics", {})),
        uncertainty_metrics=behavioral_outputs.get("same_session_uncertainty_metrics", behavioral_outputs.get("uncertainty_metrics", {})),
    )
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"],
        "track_id": config.track_id,
        "track_display_id": f"{config.track_id}-H",
        "time_mode": TIME_MODE_CONTINUOUS,
        "condition": ACTIVE_SAME_SESSION_PERIODIC_PROBE if same_session else ACTIVE,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": env.action_history,
        "repair_attempts": action_attempts,
        "repair_log": env.repair_log,
        "stop_reason": env.stop_reason,
        "query_intervals": [
            {"start_time_seconds": start, "end_time_seconds": end}
            for start, end in env.query_intervals
        ],
        "listened_seconds_total": round(env.listened_seconds_total, 6),
        "probe_outputs": probes,
        "final_segments": final_probe,
        "behavioral_protocol_version": behavioral_outputs["behavioral_protocol_version"],
        "coverage_metrics": behavioral_outputs["coverage_metrics"],
        "belief_metrics": final_probe["belief_metrics"],
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
        "master_metrics": master_metrics,
    }
    if same_session:
        result.update(
            {
                "same_session_core_behavioral_plan": behavioral_outputs["same_session_core_behavioral_plan"],
                "same_session_uncertainty_plan": behavioral_outputs["same_session_uncertainty_plan"],
                "same_session_core_behavioral_outputs": behavioral_outputs["same_session_core_behavioral_outputs"],
                "same_session_core_behavioral_metrics": behavioral_outputs["same_session_core_behavioral_metrics"],
                "same_session_uncertainty_outputs": behavioral_outputs["same_session_uncertainty_outputs"],
                "same_session_uncertainty_metrics": behavioral_outputs["same_session_uncertainty_metrics"],
                "same_session_behavioral_outputs": behavioral_outputs["same_session_behavioral_outputs"],
                "same_session_behavioral_metrics": behavioral_outputs["same_session_behavioral_metrics"],
            }
        )
    else:
        result.update(
            {
                "core_behavioral_outputs": behavioral_outputs["core_behavioral_outputs"],
                "core_behavioral_metrics": behavioral_outputs["core_behavioral_metrics"],
                "uncertainty_outputs": behavioral_outputs["uncertainty_outputs"],
                "uncertainty_metrics": behavioral_outputs["uncertainty_metrics"],
                "behavioral_outputs": behavioral_outputs["behavioral_outputs"],
                "behavioral_metrics": behavioral_outputs["behavioral_metrics"],
            }
        )
    _close_adapter_session(adapter, session)
    return result


def _run_continuous_full_context_condition(
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    vocabulary: Sequence[str],
) -> dict[str, Any]:
    regime_payload = episode["regime_variants"][regime]
    audio_path = root / episode["audio_path"]
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    try:
        final_probe = _run_continuous_probe(
            adapter,
            session,
            episode,
            regime_payload,
            config,
            vocabulary,
            [],
            [audio_path],
            full_context=True,
            same_session=False,
        )
    finally:
        _close_adapter_session(adapter, session)
    full_interval = [(0.0, float(episode["audio_duration_seconds"]))]
    final_probe["belief_metrics"] = score_continuous_belief_metrics(final_probe["parsed"], episode, full_interval)
    behavioral_outputs = _run_continuous_behavioral_prompts(
        adapter,
        None,
        episode,
        regime_payload,
        config,
        vocabulary,
        [],
        [audio_path],
        full_context=True,
        same_session=False,
    )
    format_metrics = _continuous_format_metrics(
        [],
        [final_probe],
        [behavioral_outputs["core_behavioral_outputs"], behavioral_outputs["uncertainty_outputs"]],
    )
    master_metrics = score_generic_master_metrics(
        track_id=config.track_id,
        condition=FULL_CONTEXT,
        stop_reason=FULL_CONTEXT,
        listened_bin_total=float(episode["audio_duration_seconds"]),
        format_metrics=format_metrics,
        exploration_metrics={},
        belief_metrics=final_probe["belief_metrics"],
        core_metrics=behavioral_outputs["core_behavioral_metrics"],
        uncertainty_metrics=behavioral_outputs["uncertainty_metrics"],
    )
    return {
        "episode_id": episode["episode_id"],
        "track_id": config.track_id,
        "track_display_id": f"{config.track_id}-H",
        "time_mode": TIME_MODE_CONTINUOUS,
        "condition": FULL_CONTEXT,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": [],
        "repair_attempts": [],
        "repair_log": final_probe["repair_attempts"],
        "stop_reason": FULL_CONTEXT,
        "query_intervals": [{"start_time_seconds": 0.0, "end_time_seconds": float(episode["audio_duration_seconds"])}],
        "listened_seconds_total": float(episode["audio_duration_seconds"]),
        "probe_outputs": [final_probe],
        "final_segments": final_probe,
        "behavioral_protocol_version": behavioral_outputs["behavioral_protocol_version"],
        "coverage_metrics": behavioral_outputs["coverage_metrics"],
        "core_behavioral_outputs": behavioral_outputs["core_behavioral_outputs"],
        "core_behavioral_metrics": behavioral_outputs["core_behavioral_metrics"],
        "uncertainty_outputs": behavioral_outputs["uncertainty_outputs"],
        "uncertainty_metrics": behavioral_outputs["uncertainty_metrics"],
        "behavioral_outputs": behavioral_outputs["behavioral_outputs"],
        "belief_metrics": final_probe["belief_metrics"],
        "behavioral_metrics": behavioral_outputs["behavioral_metrics"],
        "format_metrics": format_metrics,
        "master_metrics": master_metrics,
    }


def ensure_continuous_master_metrics(result: dict[str, Any], episode: dict[str, Any], config: BenchmarkConfig) -> dict[str, Any]:
    if result.get("time_mode") != TIME_MODE_CONTINUOUS:
        return result
    prefix_suffix = {
        ACTIVE: "active",
        FULL_CONTEXT: "full_context",
        ACTIVE_SAME_SESSION_PERIODIC_PROBE: "active_same_session",
    }[str(result["condition"])]
    master_key = f"track{config.track_id}_{prefix_suffix}_master_score"
    if master_key in dict(result.get("master_metrics") or {}):
        return result
    enriched = dict(result)
    if enriched["condition"] == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        core_metrics = dict(enriched.get("same_session_core_behavioral_metrics") or {})
        uncertainty_metrics = dict(enriched.get("same_session_uncertainty_metrics") or {})
    else:
        core_metrics = dict(enriched.get("core_behavioral_metrics") or {})
        uncertainty_metrics = dict(enriched.get("uncertainty_metrics") or {})
    master_metrics = dict(enriched.get("master_metrics") or {})
    master_metrics.update(
        score_generic_master_metrics(
            track_id=config.track_id,
            condition=str(enriched["condition"]),
            stop_reason=str(enriched.get("stop_reason", "")),
            listened_bin_total=float(enriched.get("listened_seconds_total", 0.0)),
            format_metrics=dict(enriched.get("format_metrics") or {}),
            exploration_metrics=dict(enriched.get("exploration_metrics") or {}),
            belief_metrics=dict(enriched.get("belief_metrics") or {}),
            core_metrics=core_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )
    )
    enriched["master_metrics"] = master_metrics
    return enriched


def run_continuous_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    adapter: BaseAdapter,
    config: BenchmarkConfig,
    episode_ids: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    reprocess: bool = False,
    parallelism: int = 1,
) -> list[dict[str, Any]]:
    root = Path(dataset_root)
    manifest = json.loads((root / "dataset" / "manifest.json").read_text(encoding="utf-8"))
    _assert(manifest.get("time_mode") == TIME_MODE_CONTINUOUS, "Expected continuous-mode manifest.")
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

    run_specs = [
        (condition, regime, episode)
        for episode in selected_episodes
        for regime in selected_regimes
        for condition in selected_conditions
    ]
    _assert(bool(run_specs), "Selected continuous benchmark run is empty.")
    cached_spec_keys = {
        (condition, regime, episode["episode_id"])
        for condition, regime, episode in run_specs
        if (not reprocess)
        and _selected_episode_result_path(
            Path(results_root),
            condition,
            regime,
            episode["episode_id"],
        ).is_file()
    }
    progress = ProgressReporter(len(run_specs), label="continuous-benchmark-run", eta_total=len(run_specs) - len(cached_spec_keys))

    def _run_one_spec(spec: tuple[str, str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        condition, regime, episode = spec
        spec_key = (condition, regime, episode["episode_id"])
        existing_path = _selected_episode_result_path(Path(results_root), condition, regime, episode["episode_id"])
        if spec_key in cached_spec_keys:
            cached = json.loads(existing_path.read_text(encoding="utf-8"))
            return ensure_continuous_master_metrics(cached, episode, config), True
        if condition == ACTIVE:
            result = _run_continuous_action_condition(
                root,
                episode,
                regime,
                adapter,
                config,
                vocabulary,
                Path(results_root) / "runtime" / ACTIVE / regime / episode["episode_id"],
                same_session=False,
            )
        elif condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
            result = _run_continuous_action_condition(
                root,
                episode,
                regime,
                adapter,
                config,
                vocabulary,
                Path(results_root) / "runtime" / ACTIVE_SAME_SESSION_PERIODIC_PROBE / regime / episode["episode_id"],
                same_session=True,
            )
        elif condition == FULL_CONTEXT:
            result = _run_continuous_full_context_condition(root, episode, regime, adapter, config, vocabulary)
        else:
            raise RuntimeError(f"Unsupported condition: {condition}")
        _write_episode_result(result, Path(results_root))
        return result, False

    results_by_index: list[dict[str, Any] | None] = [None] * len(run_specs)
    max_workers = max(1, int(parallelism))
    if max_workers == 1 or len(run_specs) == 1:
        for index, spec in enumerate(run_specs):
            result, from_cache = _run_one_spec(spec)
            results_by_index[index] = result
            progress.update(
                index + 1,
                f"condition={spec[0]} regime={spec[1]} episode={spec[2]['episode_id']}",
                count_for_eta=not from_cache,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(run_specs))) as executor:
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
    _write_flat_csv(results, Path(results_root) / "episode_metrics.csv")
    _write_summary_json(results, Path(results_root) / "summary.json")
    return results


__all__ = [
    "ContinuousEpisodeEnvironment",
    "build_continuous_dataset",
    "build_continuous_action_initial_prompt",
    "parse_continuous_segments",
    "run_continuous_benchmark",
    "score_continuous_belief_metrics",
]
