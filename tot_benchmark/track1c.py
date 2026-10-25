from __future__ import annotations

import concurrent.futures
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from .adapters import BaseAdapter
from .core import (
    ACTION_SYSTEM_PROMPT,
    ACTIVE,
    ACTIVE_SAME_SESSION_PERIODIC_PROBE,
    COUNT_KNOWN,
    DEFAULT_CONDITIONS,
    DISCOVERY,
    FULL_CONTEXT,
    INVENTORY_KNOWN,
    REGIMES,
    BenchmarkConfig,
    ProgressReporter,
    _api_failure_parse,
    _assert,
    _close_adapter_session,
    _selected_episode_result_path,
    _send_with_api_retry,
    _send_with_one_repair,
    _write_episode_result,
    _write_flat_csv,
    _write_summary_json,
    build_action_repair_prompt,
    build_repair_prompt,
    parse_action,
    validate_query,
)
from .text_protocol import normalize_first_line
from .time_mode import (
    CONTINUOUS_EXACT_THRESHOLD_SEC,
    TIME_MODE_BINNED,
    TIME_MODE_CONTINUOUS,
    continuous_redundancy_counts,
    format_seconds,
    interval_contained,
    interval_set_intersection_duration,
    interval_set_unique_duration,
    label_iou_from_segments,
    mean_label_iou,
    merge_intervals,
    parse_continuous_action,
    parse_continuous_localization_answer,
    parse_continuous_segments,
    segment_exact_match_within_threshold,
    segment_time_accuracy,
    validate_continuous_query,
    validate_time_mode,
)


TRACK1C_PROTOCOL_VERSION = 1
TRACK1C_SEMANTIC_GROUPS = [
    ("generic", "generic_questions", "generic_answer_rules"),
    ("category", "category_specific_questions", "category_specific_answer_rules"),
    ("content", "conversation_content_specific_questions", "conversation_content_specific_answer_rules"),
]
TRACK1C_TIMING_GROUP = "timing"
UNKNOWN = "UNKNOWN"


def load_track1c_manifest(dataset_root: Path) -> dict[str, Any]:
    manifest_path = Path(dataset_root) / "dataset" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _assert(manifest.get("track_id") == "1c", f"Expected Track 1c manifest: {manifest_path}")
    return manifest


def _track1c_config_for_episode(episode: dict[str, Any], base_config: BenchmarkConfig | None = None) -> BenchmarkConfig:
    num_bins = max(1, int(math.ceil(float(episode["audio_duration_seconds"]))))
    config = base_config or BenchmarkConfig()
    return replace(
        config,
        track_id="1c",
        num_bins=num_bins,
        bin_duration_sec=1.0,
        episode_duration_sec=float(episode["audio_duration_seconds"]),
        max_query_bins=3,
        max_total_queried_bins=int(math.ceil(2.0 * num_bins)),
        max_steps=int(math.ceil(0.8 * num_bins)),
    )


def _category_label(raw_episode: dict[str, Any]) -> str:
    source_row = raw_episode.get("source_row", {})
    return str(source_row.get("category_label") or source_row.get("category") or "").strip()


def normalize_track1c_episode(raw_episode: dict[str, Any]) -> dict[str, Any]:
    episode = dict(raw_episode)
    conversation_id = str(episode["conversation_id"])
    variant_id = str(episode["variant_id"])
    source_row = episode.get("source_row", {})
    config = _track1c_config_for_episode(episode)
    episode.update(
        {
            "episode_id": f"{conversation_id}__{variant_id}",
            "audio_path": str(episode["audio_file_path"]),
            "num_bins": config.num_bins,
            "max_query_bins": config.max_query_bins,
            "max_steps": config.max_steps,
            "max_total_queried_bins": config.max_total_queried_bins,
            "track_id": "1c",
            "category": str(source_row.get("category", "")).strip(),
            "category_label": _category_label(episode),
        }
    )
    return episode


def normalize_track1c_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [normalize_track1c_episode(episode) for episode in manifest["episodes"]]


def _track1c_regime_payload(episode: dict[str, Any], regime: str) -> dict[str, Any]:
    sentence_count = len(episode["sentences"])
    category_label = episode.get("category_label") or episode.get("category") or "unknown category"
    payloads = {
        DISCOVERY: (
            "You know this is a short synthetic two-speaker conversation. "
            "You do not know the sentence count, category, or transcript content."
        ),
        COUNT_KNOWN: (
            "You know this is a short synthetic two-speaker conversation with exactly "
            f"{sentence_count} spoken sentences. You do not know the category or transcript content."
        ),
        INVENTORY_KNOWN: (
            "You know this is a short synthetic two-speaker conversation with exactly "
            f"{sentence_count} spoken sentences. The category focus is: {category_label}. "
            "You do not know the transcript content."
        ),
    }
    _assert(regime in payloads, f"Unknown Track 1c regime: {regime}")
    return {"name": regime, "prior_text": payloads[regime]}


def _track1c_context_header(episode: dict[str, Any], regime: str) -> str:
    config = _track1c_config_for_episode(episode)
    regime_payload = _track1c_regime_payload(episode, regime)
    return (
        "You are participating in Track 1c of an audio exploration benchmark.\n"
        "The audio is a two-speaker conversation. Speaker A is the first speaker in the full conversation; "
        "Speaker B is the other speaker.\n"
        f"Timeline bins: 0 to {config.num_bins - 1}, one second per bin. "
        "The final bin may be shorter than one second.\n"
        f"Budget: max_steps={config.max_steps}, max_total_queried_bins={config.max_total_queried_bins}, "
        f"max_query_bins={config.max_query_bins}.\n"
        f"{regime_payload['prior_text']}"
    )


def _track1c_continuous_context_header(episode: dict[str, Any], regime: str) -> str:
    config = _track1c_config_for_episode(episode, BenchmarkConfig(time_mode=TIME_MODE_CONTINUOUS))
    regime_payload = _track1c_regime_payload(episode, regime)
    max_single = config.max_query_bins * config.bin_duration_sec
    total_budget = config.max_total_queried_bins * config.bin_duration_sec
    return (
        "You are participating in Track 1c-H of an audio exploration benchmark.\n"
        "The audio is a two-speaker conversation. Speaker A is the first speaker in the full conversation; "
        "Speaker B is the other speaker.\n"
        f"Time mode: continuous hard mode. Audio duration: {format_seconds(float(episode['audio_duration_seconds']))} seconds.\n"
        f"Budget: max_steps={config.max_steps}, total_listening_budget_seconds={format_seconds(total_budget)}, "
        f"max_single_query_seconds={format_seconds(max_single)}.\n"
        f"{regime_payload['prior_text']}"
    )


def build_track1c_action_initial_prompt(episode: dict[str, Any], regime: str) -> str:
    return (
        f"{_track1c_context_header(episode, regime)}\n"
        "You do not have access to the full clip.\n"
        "Return exactly one line:\n"
        "- QUERY <start_bin> <length_bin>\n"
        "- STOP\n"
        "Choose your first action now."
    )


def build_track1c_continuous_action_initial_prompt(episode: dict[str, Any], regime: str) -> str:
    return (
        f"{_track1c_continuous_context_header(episode, regime)}\n"
        "You do not have access to the full clip.\n"
        "Return exactly one line:\n"
        "- QUERY <start_seconds> <end_seconds>\n"
        "- STOP\n"
        "Timestamps must use exactly two decimals, for example QUERY 2.00 5.00.\n"
        "Choose your first action now."
    )


def build_track1c_action_followup_prompt(metadata: dict[str, Any], env: "Track1cEnvironment") -> str:
    return (
        f"You inspected bins {metadata['start_bin']} to {metadata['end_bin_exclusive'] - 1}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining queried-bin budget: {env.remaining_bin_budget()}.\n"
        "Return exactly one line:\n"
        "- QUERY <start_bin> <length_bin>\n"
        "- STOP"
    )


def build_track1c_continuous_action_followup_prompt(metadata: dict[str, Any], env: "Track1cContinuousEnvironment") -> str:
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


def build_track1c_continuous_action_repair_prompt(reason: str, env: "Track1cContinuousEnvironment") -> str:
    return (
        f"Your previous action was invalid: {reason}.\n"
        f"Remaining step budget: {env.remaining_steps()}.\n"
        f"Remaining listening budget: {format_seconds(env.remaining_seconds_budget())} seconds.\n"
        "Return exactly one valid line now:\n"
        "- QUERY <start_seconds> <end_seconds>\n"
        "- STOP\n"
        "Timestamps must use exactly two decimals."
    )


def _query_history_text(action_history: Sequence[dict[str, Any]]) -> str:
    if not action_history:
        return "No intervals have been queried yet."
    return "\n".join(
        f"step {item['step_index']}: bins {item['start_bin']} to {item['end_bin_exclusive'] - 1}"
        for item in action_history
    )


def _continuous_query_history_text(action_history: Sequence[dict[str, Any]]) -> str:
    if not action_history:
        return "No intervals have been queried yet."
    return "\n".join(
        f"step {item['step_index']}: seconds {format_seconds(item['start_time_seconds'])} "
        f"to {format_seconds(item['end_time_seconds'])}"
        for item in action_history
    )


def _observed_bins_from_history(action_history: Sequence[dict[str, Any]]) -> set[int]:
    return {
        bin_index
        for item in action_history
        for bin_index in range(int(item["start_bin"]), int(item["end_bin_exclusive"]))
    }


def build_track1c_question_prompt(
    episode: dict[str, Any],
    regime: str,
    question: dict[str, Any],
    action_history: Sequence[dict[str, Any]],
    *,
    full_context: bool,
    checkpoint_sentence_id: str,
) -> str:
    evidence = "the full conversation audio attached below" if full_context else "the attached queried audio clips"
    return (
        f"{_track1c_context_header(episode, regime)}\n"
        f"Use only {evidence}.\n"
        f"Query history:\n{_query_history_text(action_history)}\n"
        f"Checkpoint: {checkpoint_sentence_id}.\n"
        "Answer according to the conversation evidence available at this checkpoint.\n"
        f"{question['prompt']}\n"
        "Return exactly one line and no explanation."
    )


def build_track1c_continuous_question_prompt(
    episode: dict[str, Any],
    regime: str,
    question: dict[str, Any],
    action_history: Sequence[dict[str, Any]],
    *,
    full_context: bool,
    checkpoint_sentence_id: str,
) -> str:
    evidence = "the full conversation audio attached below" if full_context else "the attached queried audio clips"
    if str(question.get("question_type", "")) == "speaker_activity_raster":
        task_text = (
            "Return the speaker/silence timeline for the audio heard so far.\n"
            "Use A, B, AB, 00, and ??. A = first speaker in the full conversation. "
            "B = the other speaker. AB = overlapping speech. 00 = silence. ?? = unexplored.\n"
            "Return one segment per line: START <seconds> END <seconds> LABEL <A|B|AB|00|??>.\n"
            "All timestamps must use exactly two decimals. Cover the full audio duration."
        )
    elif str(question.get("question_type", "")) == "semantic_evidence_localization":
        task_text = (
            f"{question['prompt']}\n"
            "Return exactly START <seconds> END <seconds>, or UNKNOWN. Timestamps must use exactly two decimals."
        )
    else:
        task_text = f"{question['prompt']}\nReturn exactly one line and no explanation."
    return (
        f"{_track1c_continuous_context_header(episode, regime)}\n"
        f"Use only {evidence}.\n"
        f"Query history:\n{_continuous_query_history_text(action_history)}\n"
        f"Checkpoint: {checkpoint_sentence_id}.\n"
        "Answer according to the conversation evidence available at this checkpoint.\n"
        f"{task_text}"
    )


def _write_audio(path: Path, audio: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate, subtype="PCM_16")


class Track1cEnvironment:
    def __init__(self, root: Path, episode: dict[str, Any], run_dir: Path) -> None:
        self.root = Path(root)
        self.dataset_dir = self.root / "dataset"
        self.episode = episode
        self.config = _track1c_config_for_episode(episode)
        self.run_dir = Path(run_dir)
        self.audio_path = self.dataset_dir / episode["audio_path"]
        self.audio, self.samplerate = sf.read(str(self.audio_path), dtype="float32", always_2d=False)
        if isinstance(self.audio, np.ndarray) and self.audio.ndim == 2:
            self.audio = self.audio.mean(axis=1)
        self.action_history: list[dict[str, Any]] = []
        self.query_audio_paths: list[Path] = []
        self.observed_bins: set[int] = set()
        self.listened_bin_total = 0
        self.stop_reason = ""

    def remaining_steps(self) -> int:
        return self.config.max_steps - len(self.action_history)

    def remaining_bin_budget(self) -> int:
        return self.config.max_total_queried_bins - self.listened_bin_total

    def clip_for_query(self, step_index: int, start_bin: int, length_bin: int) -> tuple[Path, dict[str, Any]]:
        start_sec = float(start_bin)
        end_sec = min(float(start_bin + length_bin), float(self.episode["audio_duration_seconds"]))
        start_frame = int(round(start_sec * self.samplerate))
        end_frame = int(round(end_sec * self.samplerate))
        clip = np.asarray(self.audio[start_frame:end_frame], dtype=np.float32)
        clip_path = (
            self.run_dir
            / "query_audio"
            / f"{self.episode['episode_id']}_step_{step_index:02d}_{start_bin}_{length_bin}.wav"
        )
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
            "start_time_seconds": round(start_sec, 6),
            "end_time_seconds": round(end_sec, 6),
            "duration_sec": round(max(0.0, end_sec - start_sec), 6),
            "audio_path": str(clip_path),
        }
        self.action_history.append(metadata)
        return clip_path, metadata


class Track1cContinuousEnvironment:
    def __init__(self, root: Path, episode: dict[str, Any], run_dir: Path) -> None:
        self.root = Path(root)
        self.dataset_dir = self.root / "dataset"
        self.episode = episode
        self.config = _track1c_config_for_episode(episode, BenchmarkConfig(time_mode=TIME_MODE_CONTINUOUS))
        self.run_dir = Path(run_dir)
        self.audio_path = self.dataset_dir / episode["audio_path"]
        self.audio, self.samplerate = sf.read(str(self.audio_path), dtype="float32", always_2d=False)
        if isinstance(self.audio, np.ndarray) and self.audio.ndim == 2:
            self.audio = self.audio.mean(axis=1)
        self.action_history: list[dict[str, Any]] = []
        self.query_audio_paths: list[Path] = []
        self.query_intervals: list[tuple[float, float]] = []
        self.listened_seconds_total = 0.0
        self.stop_reason = ""

    @property
    def audio_duration_seconds(self) -> float:
        return float(self.episode["audio_duration_seconds"])

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
        self.query_intervals.append((float(start_sec), float(end_sec)))
        duration = float(end_sec) - float(start_sec)
        self.listened_seconds_total += duration
        metadata = {
            "step_index": step_index,
            "start_time_seconds": round(float(start_sec), 2),
            "end_time_seconds": round(float(end_sec), 2),
            "duration_seconds": round(duration, 6),
            "audio_path": str(clip_path),
        }
        self.action_history.append(metadata)
        return clip_path, metadata


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    sorted_intervals = sorted((float(start), float(end)) for start, end in intervals)
    merged = [sorted_intervals[0]]
    for start, end in sorted_intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def observed_sentence_ids(
    episode: dict[str, Any],
    query_intervals: Sequence[tuple[float, float]],
) -> list[str]:
    merged = _merge_intervals(query_intervals)
    observed: list[str] = []
    epsilon = 1e-6
    for sentence in episode["sentences"]:
        start = float(sentence["sentence_start_time"])
        end = float(sentence["sentence_end_time"])
        if any(interval_start <= start + epsilon and interval_end + epsilon >= end for interval_start, interval_end in merged):
            observed.append(str(sentence["sentence_id"]))
    return observed


def _query_intervals_from_history(action_history: Sequence[dict[str, Any]]) -> list[tuple[float, float]]:
    return [
        (float(item["start_time_seconds"]), float(item["end_time_seconds"]))
        for item in action_history
    ]


def _sentence_order_index(episode: dict[str, Any]) -> dict[str, int]:
    return {str(sentence["sentence_id"]): index for index, sentence in enumerate(episode["sentences"])}


def newly_crossed_checkpoints(
    episode: dict[str, Any],
    previous_observed_sentence_ids: Sequence[str],
    current_observed_sentence_ids: Sequence[str],
) -> list[str]:
    previous = set(previous_observed_sentence_ids)
    current = set(current_observed_sentence_ids)
    order = _sentence_order_index(episode)
    crossed = sorted(current - previous, key=lambda sentence_id: order[sentence_id])
    return crossed


def _question_maps(episode: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    questions: dict[str, dict[str, Any]] = {}
    rules: dict[str, dict[str, Any]] = {}
    groups: dict[str, str] = {}
    metadata = episode["parsed_metadata"]
    for group_name, question_key, rule_key in TRACK1C_SEMANTIC_GROUPS:
        for question in metadata.get(question_key, []):
            question_id = str(question["question_id"])
            questions[question_id] = dict(question)
            groups[question_id] = group_name
        for rule in metadata.get(rule_key, []):
            rules[str(rule["question_id"])] = dict(rule)
    for question in metadata.get("timing_questions", []):
        question_id = str(question["question_id"])
        questions[question_id] = dict(question)
        groups[question_id] = TRACK1C_TIMING_GROUP
    return questions, rules, groups


def semantic_questions_for_checkpoint(
    episode: dict[str, Any],
    checkpoint_sentence_id: str,
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    questions, rules, groups = _question_maps(episode)
    items: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for question_id, rule in rules.items():
        if checkpoint_sentence_id in rule.get("ask_after", []):
            items.append((groups[question_id], questions[question_id], rule))
    order = {question_id: index for index, question_id in enumerate(questions)}
    items.sort(key=lambda item: order[str(item[1]["question_id"])])
    return items


def all_semantic_questions_for_end(episode: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    questions, rules, groups = _question_maps(episode)
    items = [(groups[question_id], questions[question_id], rule) for question_id, rule in rules.items()]
    order = {question_id: index for index, question_id in enumerate(questions)}
    items.sort(key=lambda item: order[str(item[1]["question_id"])])
    return items


def timing_questions(episode: dict[str, Any]) -> list[dict[str, Any]]:
    return [dict(question) for question in episode["parsed_metadata"].get("timing_questions", [])]


def expected_generic_answer(episode: dict[str, Any], question: dict[str, Any], observed_ids: set[str]) -> str:
    first_sentence_id = str(episode["sentences"][0]["sentence_id"])
    if first_sentence_id not in observed_ids:
        return UNKNOWN
    observed_sentences = [
        sentence for sentence in episode["sentences"] if str(sentence["sentence_id"]) in observed_ids
    ]
    speakers = [str(sentence["speaker"]) for sentence in observed_sentences]
    collapsed: list[str] = []
    for speaker in speakers:
        if not collapsed or collapsed[-1] != speaker:
            collapsed.append(speaker)
    question_type = str(question.get("question_type", ""))
    if question_type == "speaker_handoff_path":
        return ">".join(collapsed) if collapsed else UNKNOWN
    if question_type == "speaker_handoff_count":
        return str(max(0, len(collapsed) - 1))
    if question_type == "final_heard_speaker":
        return speakers[-1] if speakers else UNKNOWN
    raise RuntimeError(f"Unsupported generic Track 1c question_type: {question_type}")


def expected_rule_answer(rule: dict[str, Any], observed_ids: set[str], episode: dict[str, Any] | None = None) -> str:
    default_answer: str | None = None
    for evidence_rule in rule.get("evidence_rules", []):
        if "default" in evidence_rule:
            default_answer = str(evidence_rule["default"])
            continue
        required = {str(sentence_id) for sentence_id in evidence_rule.get("if_observed_all", [])}
        if required and required <= observed_ids:
            return str(evidence_rule["answer"])
    if default_answer is not None:
        return default_answer
    prefix_trace = rule.get("prefix_answer_trace")
    if isinstance(prefix_trace, dict) and episode is not None:
        latest_sentence_id: str | None = None
        for sentence in episode["sentences"]:
            sentence_id = str(sentence["sentence_id"])
            if sentence_id in observed_ids and sentence_id in prefix_trace:
                latest_sentence_id = sentence_id
        if latest_sentence_id is not None:
            return str(prefix_trace[latest_sentence_id])
        if "START" in prefix_trace:
            return str(prefix_trace["START"])
    raise RuntimeError(f"Track 1c rule has no satisfied evidence rule, default, or prefix trace: {rule.get('question_id')}")


def expected_semantic_answer(
    episode: dict[str, Any],
    group_name: str,
    question: dict[str, Any],
    rule: dict[str, Any],
    observed_sentence_ids: Sequence[str],
    *,
    full_context: bool,
) -> str:
    if full_context:
        return str(rule["final_answer_full_evidence"])
    observed = {str(sentence_id) for sentence_id in observed_sentence_ids}
    if group_name == "generic":
        return expected_generic_answer(episode, question, observed)
    return expected_rule_answer(rule, observed, episode)


def parse_track1c_semantic_answer(raw_text: str, question: dict[str, Any]) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).strip()
    if not stripped:
        return {"format_valid": False, "reason": "empty_answer"}
    if "allowed_answers" in question:
        allowed = {str(answer) for answer in question["allowed_answers"]}
        if stripped not in allowed:
            return {"format_valid": False, "reason": "answer_not_allowed", "answer": stripped}
        return {"format_valid": True, "answer": stripped}

    question_type = str(question.get("question_type", ""))
    upper = stripped.upper()
    if question_type == "speaker_handoff_path":
        if upper == UNKNOWN or re.fullmatch(r"[AB](>[AB])*", stripped):
            return {"format_valid": True, "answer": upper if upper == UNKNOWN else stripped}
        return {"format_valid": False, "reason": "invalid_speaker_handoff_path", "answer": stripped}
    if question_type == "speaker_handoff_count":
        if upper == UNKNOWN or re.fullmatch(r"\d+", stripped):
            return {"format_valid": True, "answer": upper if upper == UNKNOWN else stripped}
        return {"format_valid": False, "reason": "invalid_speaker_handoff_count", "answer": stripped}
    if question_type == "procedure_order_supported":
        if upper == UNKNOWN or re.fullmatch(r"[A-Z0-9_]+(>[A-Z0-9_]+)*", stripped):
            return {"format_valid": True, "answer": upper if upper == UNKNOWN else stripped}
        return {"format_valid": False, "reason": "invalid_procedure_order", "answer": stripped}
    return {"format_valid": False, "reason": f"unsupported_question_type:{question_type}", "answer": stripped}


def parse_track1c_timing_localization(raw_text: str, num_bins: int) -> dict[str, Any]:
    stripped = normalize_first_line(raw_text).upper()
    if stripped == UNKNOWN:
        return {"format_valid": True, "answer": UNKNOWN}
    match = re.fullmatch(r"START\s+(\d+)\s+END\s+(\d+)", stripped)
    if not match:
        return {"format_valid": False, "reason": "invalid_timing_localization"}
    start_bin = int(match.group(1))
    end_bin = int(match.group(2))
    if start_bin < 0 or end_bin < start_bin or end_bin >= num_bins:
        return {"format_valid": False, "reason": "timing_localization_out_of_range"}
    return {"format_valid": True, "start_bin": start_bin, "end_bin": end_bin}


def parse_track1c_speaker_raster(raw_text: str, num_bins: int) -> dict[str, Any]:
    text = normalize_first_line(raw_text)
    tokens = [token.strip().upper() for token in text.split(",")]
    if len(tokens) != num_bins:
        return {"format_valid": False, "reason": "wrong_raster_token_count", "tokens": tokens}
    allowed = {"A", "B", "AB", "00", "??"}
    if any(token not in allowed for token in tokens):
        return {"format_valid": False, "reason": "invalid_raster_token", "tokens": tokens}
    return {"format_valid": True, "tokens": tokens}


def parse_track1c_speaker_segments(raw_text: str, audio_duration_seconds: float) -> dict[str, Any]:
    parsed = parse_continuous_segments(
        raw_text,
        allowed_labels=["A", "B", "AB"],
        audio_duration_seconds=float(audio_duration_seconds),
        allow_multi_label=False,
    )
    if not parsed.get("format_valid"):
        return parsed
    for segment in parsed["segments"]:
        labels = segment.get("labels")
        if labels == ["AB"]:
            continue
        if labels is not None and len(labels) > 1:
            return {"format_valid": False, "reason": "speaker_segment_must_have_single_label", "segments": None}
    return parsed


def _target_evidence_sentence_ids(episode: dict[str, Any]) -> list[str]:
    raw = str(episode.get("source_row", {}).get("target_evidence_sentences", "")).strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def expected_timing_localization_answer(
    episode: dict[str, Any],
    observed_sentence_ids: Sequence[str],
) -> str:
    observed = set(observed_sentence_ids)
    target_ids = [sentence_id for sentence_id in _target_evidence_sentence_ids(episode) if sentence_id in observed]
    if not target_ids:
        return UNKNOWN
    sentences_by_id = {str(sentence["sentence_id"]): sentence for sentence in episode["sentences"]}
    min_start = min(float(sentences_by_id[sentence_id]["sentence_start_time"]) for sentence_id in target_ids)
    max_end = max(float(sentences_by_id[sentence_id]["sentence_end_time"]) for sentence_id in target_ids)
    start_bin = int(math.floor(min_start))
    end_bin = min(int(math.ceil(max_end) - 1), int(episode["num_bins"]) - 1)
    return f"START {start_bin} END {end_bin}"


def expected_continuous_timing_localization_answer(
    episode: dict[str, Any],
    observed_sentence_ids: Sequence[str],
) -> str:
    observed = set(observed_sentence_ids)
    target_ids = [sentence_id for sentence_id in _target_evidence_sentence_ids(episode) if sentence_id in observed]
    if not target_ids:
        return UNKNOWN
    sentences_by_id = {str(sentence["sentence_id"]): sentence for sentence in episode["sentences"]}
    min_start = min(float(sentences_by_id[sentence_id]["sentence_start_time"]) for sentence_id in target_ids)
    max_end = max(float(sentences_by_id[sentence_id]["sentence_end_time"]) for sentence_id in target_ids)
    return f"START {format_seconds(min_start)} END {format_seconds(max_end)}"


def expected_speaker_raster_tokens(episode: dict[str, Any], observed_bins: set[int]) -> list[str]:
    tokens: list[str] = []
    num_bins = int(episode["num_bins"])
    for bin_index in range(num_bins):
        if bin_index not in observed_bins:
            tokens.append("??")
            continue
        bin_start = float(bin_index)
        bin_end = min(float(bin_index + 1), float(episode["audio_duration_seconds"]))
        speakers = set()
        for sentence in episode["sentences"]:
            start = float(sentence["sentence_start_time"])
            end = float(sentence["sentence_end_time"])
            if max(bin_start, start) < min(bin_end, end):
                speakers.add(str(sentence["speaker"]))
        if speakers == {"A", "B"}:
            tokens.append("AB")
        elif speakers == {"A"}:
            tokens.append("A")
        elif speakers == {"B"}:
            tokens.append("B")
        else:
            tokens.append("00")
    return tokens


def expected_speaker_segments(
    episode: dict[str, Any],
    query_intervals: Sequence[tuple[float, float]],
    *,
    full_context: bool,
) -> list[dict[str, Any]]:
    duration = float(episode["audio_duration_seconds"])
    observed_intervals = [(0.0, duration)] if full_context else list(query_intervals)
    breakpoints = {0.0, round(duration, 2)}
    for start, end in observed_intervals:
        breakpoints.add(round(float(start), 2))
        breakpoints.add(round(float(end), 2))
    for sentence in episode["sentences"]:
        breakpoints.add(round(float(sentence["sentence_start_time"]), 2))
        breakpoints.add(round(float(sentence["sentence_end_time"]), 2))
    for pause in episode.get("pauses", []):
        breakpoints.add(round(float(pause["pause_start_time"]), 2))
        breakpoints.add(round(float(pause["pause_end_time"]), 2))
    points = sorted(point for point in breakpoints if 0.0 <= point <= round(duration, 2))
    segments: list[dict[str, Any]] = []
    for start, end in zip(points[:-1], points[1:]):
        if end <= start:
            continue
        if not interval_contained(observed_intervals, (start, end), tolerance=0.01):
            labels: list[str] | None = None
        else:
            speakers = sorted(
                {
                    str(sentence["speaker"])
                    for sentence in episode["sentences"]
                    if max(start, float(sentence["sentence_start_time"])) < min(end, float(sentence["sentence_end_time"]))
                }
            )
            if speakers == ["A", "B"]:
                labels = ["AB"]
            elif speakers == ["A"]:
                labels = ["A"]
            elif speakers == ["B"]:
                labels = ["B"]
            else:
                labels = []
        if segments and segments[-1]["labels"] == labels and abs(float(segments[-1]["end_time_seconds"]) - start) <= 0.01:
            segments[-1]["end_time_seconds"] = end
        else:
            segments.append({"start_time_seconds": start, "end_time_seconds": end, "labels": labels})
    return segments


def _speaker_segment_text(segments: Sequence[dict[str, Any]]) -> str:
    lines = []
    for segment in segments:
        labels = segment.get("labels")
        if labels is None:
            label_text = "??"
        elif not labels:
            label_text = "00"
        else:
            label_text = " ".join(labels)
        lines.append(
            f"START {format_seconds(segment['start_time_seconds'])} "
            f"END {format_seconds(segment['end_time_seconds'])} LABEL {label_text}"
        )
    return "\n".join(lines)


def _score_timing_localization(parsed: dict[str, Any], expected_answer: str) -> tuple[float, float]:
    if expected_answer == UNKNOWN:
        score = float(parsed.get("format_valid") and parsed.get("answer") == UNKNOWN)
        return score, score
    if not parsed.get("format_valid") or parsed.get("answer") == UNKNOWN:
        return 0.0, 0.0
    match = re.fullmatch(r"START\s+(\d+)\s+END\s+(\d+)", expected_answer)
    if not match:
        raise RuntimeError(f"Invalid expected timing answer: {expected_answer}")
    expected = {"start_bin": int(match.group(1)), "end_bin": int(match.group(2))}
    pred_bins = set(range(parsed["start_bin"], parsed["end_bin"] + 1))
    gold_bins = set(range(expected["start_bin"], expected["end_bin"] + 1))
    intersection = len(pred_bins & gold_bins)
    union = len(pred_bins | gold_bins)
    return float(pred_bins == gold_bins), float(intersection / union) if union else 0.0


def _score_continuous_timing_localization(parsed: dict[str, Any], expected_answer: str) -> tuple[float, float, dict[str, Any]]:
    if expected_answer == UNKNOWN:
        score = float(parsed.get("format_valid") and parsed.get("answer") == UNKNOWN)
        return score, score, {"expected_answer": expected_answer}
    if not parsed.get("format_valid") or parsed.get("answer") == UNKNOWN:
        return 0.0, 0.0, {"expected_answer": expected_answer}
    expected = parse_continuous_localization_answer(
        expected_answer,
        audio_duration_seconds=float("inf"),
    )
    if not expected.get("format_valid"):
        raise RuntimeError(f"Invalid expected timing answer: {expected_answer}")
    pred_interval = (float(parsed["start_time_seconds"]), float(parsed["end_time_seconds"]))
    gold_interval = (float(expected["start_time_seconds"]), float(expected["end_time_seconds"]))
    intersection = max(0.0, min(pred_interval[1], gold_interval[1]) - max(pred_interval[0], gold_interval[0]))
    union = max(pred_interval[1], gold_interval[1]) - min(pred_interval[0], gold_interval[0])
    exact = float(
        abs(pred_interval[0] - gold_interval[0]) <= CONTINUOUS_EXACT_THRESHOLD_SEC
        and abs(pred_interval[1] - gold_interval[1]) <= CONTINUOUS_EXACT_THRESHOLD_SEC
    )
    return exact, float(intersection / union) if union else 0.0, {
        "predicted_start_time_seconds": pred_interval[0],
        "predicted_end_time_seconds": pred_interval[1],
        "gold_start_time_seconds": gold_interval[0],
        "gold_end_time_seconds": gold_interval[1],
    }


def _score_speaker_raster(parsed: dict[str, Any], expected_tokens: Sequence[str]) -> tuple[float, float]:
    if not parsed.get("format_valid"):
        return 0.0, 0.0
    predicted = list(parsed["tokens"])
    expected = list(expected_tokens)
    token_score = sum(float(left == right) for left, right in zip(predicted, expected)) / len(expected)
    return float(predicted == expected), float(token_score)


def _score_speaker_segments(
    parsed: dict[str, Any],
    expected_segments_value: Sequence[dict[str, Any]],
    *,
    episode: dict[str, Any],
) -> dict[str, float]:
    if not parsed.get("format_valid"):
        return {
            "speaker_segment_time_accuracy": 0.0,
            "speaker_segment_exact_250ms": 0.0,
            "speaker_segment_iou": 0.0,
            "speech_silence_iou": 0.0,
            "unexplored_time_accuracy": 0.0,
            "false_speech_on_silence_rate": 1.0,
            "false_certainty_on_unexplored_time": 1.0,
        }
    predicted = parsed["segments"]
    expected = list(expected_segments_value)
    duration = float(episode["audio_duration_seconds"])
    time_accuracy = segment_time_accuracy(
        predicted,
        expected,
        audio_duration_seconds=duration,
        observed_intervals=None,
        credit_unexplored=0.0,
    )
    speaker_iou = mean_label_iou(predicted, expected, ["A", "B", "AB"])
    predicted_speech = _segment_intervals_by_predicate(predicted, lambda labels: labels in (["A"], ["B"], ["AB"]))
    expected_speech = _segment_intervals_by_predicate(expected, lambda labels: labels in (["A"], ["B"], ["AB"]))
    predicted_silence = _segment_intervals_by_predicate(predicted, lambda labels: labels == [])
    expected_silence = _segment_intervals_by_predicate(expected, lambda labels: labels == [])
    speech_silence_iou = _mean_interval_class_iou(
        [
            (predicted_speech, expected_speech),
            (predicted_silence, expected_silence),
        ]
    )
    breakpoints = {0.0, round(duration, 2)}
    for segment in predicted + expected:
        breakpoints.add(round(float(segment["start_time_seconds"]), 2))
        breakpoints.add(round(float(segment["end_time_seconds"]), 2))
    points = sorted(point for point in breakpoints if 0.0 <= point <= round(duration, 2))
    unexplored_total = 0.0
    unexplored_correct = 0.0
    silence_total = 0.0
    false_speech = 0.0
    false_certainty_total = 0.0
    false_certainty = 0.0
    for start, end in zip(points[:-1], points[1:]):
        segment_duration = max(0.0, end - start)
        if segment_duration <= 0.0:
            continue
        pred = _segment_label_at(predicted, start, end)
        exp = _segment_label_at(expected, start, end)
        if exp is None:
            unexplored_total += segment_duration
            false_certainty_total += segment_duration
            if pred is None:
                unexplored_correct += segment_duration
            else:
                false_certainty += segment_duration
        if exp == []:
            silence_total += segment_duration
            if pred in (["A"], ["B"], ["AB"]):
                false_speech += segment_duration
    return {
        "speaker_segment_time_accuracy": float(time_accuracy),
        "speaker_segment_exact_250ms": segment_exact_match_within_threshold(predicted, expected),
        "speaker_segment_iou": float(speaker_iou),
        "speech_silence_iou": float(speech_silence_iou),
        "unexplored_time_accuracy": float(unexplored_correct / unexplored_total) if unexplored_total else 1.0,
        "false_speech_on_silence_rate": float(false_speech / silence_total) if silence_total else 0.0,
        "false_certainty_on_unexplored_time": float(false_certainty / false_certainty_total) if false_certainty_total else 0.0,
    }


def _segment_intervals_by_predicate(
    segments: Sequence[dict[str, Any]],
    predicate,
) -> list[tuple[float, float]]:
    return [
        (float(segment["start_time_seconds"]), float(segment["end_time_seconds"]))
        for segment in segments
        if predicate(None if segment.get("labels") is None else list(segment.get("labels") or []))
    ]


def _mean_interval_class_iou(classes: Sequence[tuple[Sequence[tuple[float, float]], Sequence[tuple[float, float]]]]) -> float:
    values = []
    for predicted, expected in classes:
        pred_duration = interval_set_unique_duration(predicted)
        expected_duration = interval_set_unique_duration(expected)
        if pred_duration <= 0.0 and expected_duration <= 0.0:
            continue
        intersection = interval_set_intersection_duration(predicted, expected)
        union = pred_duration + expected_duration - intersection
        values.append(float(intersection / union) if union > 0.0 else 0.0)
    return float(sum(values) / len(values)) if values else 0.0


def _segment_label_at(segments: Sequence[dict[str, Any]], start: float, end: float) -> list[str] | None:
    midpoint = (float(start) + float(end)) / 2.0
    for segment in segments:
        if float(segment["start_time_seconds"]) - 1e-9 <= midpoint < float(segment["end_time_seconds"]) + 1e-9:
            labels = segment.get("labels")
            return None if labels is None else list(labels)
    return []


def _send_track1c_question(
    adapter: BaseAdapter,
    session: Any,
    prompt: str,
    parse_fn,
    audio_paths: Sequence[Path],
    repair_kind: str,
    api_retry_sleep_sec: float,
) -> dict[str, Any]:
    raw, parsed, repairs, api_attempts = _send_with_one_repair(
        adapter,
        session,
        prompt,
        parse_fn,
        audio_paths,
        repair_kind,
        api_retry_sleep_sec=api_retry_sleep_sec,
        repair_prompt_fn=lambda reason: build_repair_prompt(repair_kind, reason),
    )
    return {
        "raw_output": raw,
        "parsed": parsed,
        "repair_attempts": repairs,
        "api_attempts": api_attempts,
    }


def _run_checkpoint_questions(
    *,
    adapter: BaseAdapter,
    episode: dict[str, Any],
    regime: str,
    checkpoint_sentence_id: str,
    observed_sentence_ids_value: Sequence[str],
    observed_bins: set[int],
    action_history: Sequence[dict[str, Any]],
    audio_paths: Sequence[Path],
    full_context: bool,
    same_session: bool,
    session: Any | None,
    consume_audio_once: bool,
    api_retry_sleep_sec: float,
    time_mode: str = TIME_MODE_BINNED,
) -> tuple[list[dict[str, Any]], bool]:
    outputs: list[dict[str, Any]] = []
    audio_consumed = False
    continuous_mode = validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS

    semantic_items = (
        all_semantic_questions_for_end(episode)
        if checkpoint_sentence_id == "END"
        else semantic_questions_for_checkpoint(episode, checkpoint_sentence_id)
    )
    question_items: list[tuple[str, dict[str, Any], dict[str, Any] | None]] = [
        (group_name, question, rule) for group_name, question, rule in semantic_items
    ]
    question_items.extend((TRACK1C_TIMING_GROUP, question, None) for question in timing_questions(episode))

    for group_name, question, rule in question_items:
        question_session = session if same_session and session is not None else adapter.create_session(ACTION_SYSTEM_PROMPT)
        if continuous_mode:
            prompt = build_track1c_continuous_question_prompt(
                episode,
                regime,
                question,
                action_history,
                full_context=full_context,
                checkpoint_sentence_id=checkpoint_sentence_id,
            )
        else:
            prompt = build_track1c_question_prompt(
                episode,
                regime,
                question,
                action_history,
                full_context=full_context,
                checkpoint_sentence_id=checkpoint_sentence_id,
            )
        current_audio_paths = list(audio_paths)
        if consume_audio_once and audio_consumed:
            current_audio_paths = []
        if current_audio_paths:
            audio_consumed = True

        question_type = str(question.get("question_type", ""))
        if group_name == TRACK1C_TIMING_GROUP and question_type == "semantic_evidence_localization":
            if continuous_mode:
                expected_answer = expected_continuous_timing_localization_answer(episode, observed_sentence_ids_value)
                result = _send_track1c_question(
                    adapter,
                    question_session,
                    prompt,
                    lambda text: parse_continuous_localization_answer(
                        text,
                        audio_duration_seconds=float(episode["audio_duration_seconds"]),
                        unknown_token=UNKNOWN,
                    ),
                    current_audio_paths,
                    "timing_localization",
                    api_retry_sleep_sec,
                )
                exact, iou, raw_span_details = _score_continuous_timing_localization(result["parsed"], expected_answer)
                metric_details = {
                    "timing_localization_exact_250ms": exact,
                    "timing_localization_iou": iou,
                    "timing_localization_span_details": raw_span_details,
                }
            else:
                expected_answer = expected_timing_localization_answer(episode, observed_sentence_ids_value)
                result = _send_track1c_question(
                    adapter,
                    question_session,
                    prompt,
                    lambda text: parse_track1c_timing_localization(text, int(episode["num_bins"])),
                    current_audio_paths,
                    "timing_localization",
                    api_retry_sleep_sec,
                )
                exact, iou = _score_timing_localization(result["parsed"], expected_answer)
                metric_details = {"timing_localization_exact": exact, "timing_localization_iou": iou}
            score = exact
        elif group_name == TRACK1C_TIMING_GROUP and question_type == "speaker_activity_raster":
            if continuous_mode:
                query_intervals = _query_intervals_from_history(action_history)
                expected_segments_value = expected_speaker_segments(
                    episode,
                    query_intervals,
                    full_context=full_context,
                )
                expected_answer = _speaker_segment_text(expected_segments_value)
                result = _send_track1c_question(
                    adapter,
                    question_session,
                    prompt,
                    lambda text: parse_track1c_speaker_segments(text, float(episode["audio_duration_seconds"])),
                    current_audio_paths,
                    "speaker_segment",
                    api_retry_sleep_sec,
                )
                segment_scores = _score_speaker_segments(result["parsed"], expected_segments_value, episode=episode)
                score = segment_scores["speaker_segment_exact_250ms"]
                metric_details = {
                    "speaker_segment_time_accuracy": segment_scores["speaker_segment_time_accuracy"],
                    "speaker_segment_exact_250ms": segment_scores["speaker_segment_exact_250ms"],
                    "speaker_segment_iou": segment_scores["speaker_segment_iou"],
                    "speech_silence_iou": segment_scores["speech_silence_iou"],
                    "unexplored_time_accuracy": segment_scores["unexplored_time_accuracy"],
                    "false_speech_on_silence_rate": segment_scores["false_speech_on_silence_rate"],
                    "false_certainty_on_unexplored_time": segment_scores["false_certainty_on_unexplored_time"],
                }
            else:
                expected_tokens = expected_speaker_raster_tokens(episode, observed_bins)
                expected_answer = ",".join(expected_tokens)
                result = _send_track1c_question(
                    adapter,
                    question_session,
                    prompt,
                    lambda text: parse_track1c_speaker_raster(text, int(episode["num_bins"])),
                    current_audio_paths,
                    "speaker_raster",
                    api_retry_sleep_sec,
                )
                exact, token_accuracy = _score_speaker_raster(result["parsed"], expected_tokens)
                score = exact
                metric_details = {
                    "speaker_raster_exact": exact,
                    "speaker_raster_token_accuracy": token_accuracy,
                }
        else:
            assert rule is not None
            expected_answer = expected_semantic_answer(
                episode,
                group_name,
                question,
                rule,
                observed_sentence_ids_value,
                full_context=full_context,
            )
            result = _send_track1c_question(
                adapter,
                question_session,
                prompt,
                lambda text, question=question: parse_track1c_semantic_answer(text, question),
                current_audio_paths,
                "semantic_answer",
                api_retry_sleep_sec,
            )
            parsed = result["parsed"]
            score = float(parsed.get("format_valid") and parsed.get("answer") == expected_answer)
            metric_details = {}

        outputs.append(
            {
                "checkpoint_sentence_id": checkpoint_sentence_id,
                "checkpoint_step": len(action_history),
                "observed_sentence_ids": list(observed_sentence_ids_value),
                "observed_bins": sorted(observed_bins),
                "question_group": group_name,
                "question_id": question["question_id"],
                "question_type": question.get("question_type", ""),
                "prompt": prompt,
                "expected_answer": expected_answer,
                "raw_output": result["raw_output"],
                "parsed": result["parsed"],
                "score": score,
                "repair_attempts": result["repair_attempts"],
                "api_attempts": result["api_attempts"],
                "metric_details": metric_details,
            }
        )
        if not same_session:
            _close_adapter_session(adapter, question_session)
    return outputs, audio_consumed


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def score_track1c_checkpoint_outputs(
    outputs: Sequence[dict[str, Any]],
    *,
    time_mode: str = TIME_MODE_BINNED,
) -> dict[str, float | None]:
    continuous_mode = validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS
    semantic = [item for item in outputs if item["question_group"] in {"generic", "category", "content"}]
    generic = [item for item in outputs if item["question_group"] == "generic"]
    category = [item for item in outputs if item["question_group"] == "category"]
    content = [item for item in outputs if item["question_group"] == "content"]
    timing_loc = [
        item for item in outputs if item.get("question_type") == "semantic_evidence_localization"
    ]
    raster = [item for item in outputs if item.get("question_type") == "speaker_activity_raster"]
    unknown = [item for item in outputs if str(item["expected_answer"]) == UNKNOWN]
    non_unknown = [item for item in outputs if str(item["expected_answer"]) != UNKNOWN]

    update_scores: list[float] = []
    stale_scores: list[float] = []
    previous_expected_by_question: dict[str, str] = {}
    for item in outputs:
        question_id = str(item["question_id"])
        expected_answer = str(item["expected_answer"])
        parsed_answer = str(item.get("parsed", {}).get("answer", ""))
        previous_expected = previous_expected_by_question.get(question_id)
        if previous_expected is not None and previous_expected != expected_answer and expected_answer != UNKNOWN:
            update_scores.append(float(item["score"]))
            stale_scores.append(float(parsed_answer == previous_expected))
        previous_expected_by_question[question_id] = expected_answer

    metrics = {
        "track1c_semantic_accuracy": _mean_or_none([float(item["score"]) for item in semantic]),
        "track1c_generic_accuracy": _mean_or_none([float(item["score"]) for item in generic]),
        "track1c_category_accuracy": _mean_or_none([float(item["score"]) for item in category]),
        "track1c_content_accuracy": _mean_or_none([float(item["score"]) for item in content]),
        "track1c_timing_localization_iou": _mean_or_none(
            [float(item["metric_details"]["timing_localization_iou"]) for item in timing_loc]
        ),
        "track1c_unknown_accuracy": _mean_or_none([float(item["score"]) for item in unknown]),
        "track1c_non_unknown_accuracy": _mean_or_none([float(item["score"]) for item in non_unknown]),
        "track1c_update_success_rate": _mean_or_none(update_scores),
        "track1c_stale_answer_rate": _mean_or_none(stale_scores),
    }
    if continuous_mode:
        metrics.update(
            {
                "track1c_timing_localization_exact_250ms": _mean_or_none(
                    [float(item["metric_details"]["timing_localization_exact_250ms"]) for item in timing_loc]
                ),
                "track1c_speaker_segment_time_accuracy": _mean_or_none(
                    [float(item["metric_details"]["speaker_segment_time_accuracy"]) for item in raster]
                ),
                "track1c_speaker_segment_exact_250ms": _mean_or_none(
                    [float(item["metric_details"]["speaker_segment_exact_250ms"]) for item in raster]
                ),
                "track1c_speaker_segment_iou": _mean_or_none(
                    [float(item["metric_details"]["speaker_segment_iou"]) for item in raster]
                ),
                "track1c_speech_silence_iou": _mean_or_none(
                    [float(item["metric_details"]["speech_silence_iou"]) for item in raster]
                ),
                "track1c_unexplored_time_accuracy": _mean_or_none(
                    [float(item["metric_details"]["unexplored_time_accuracy"]) for item in raster]
                ),
                "track1c_false_speech_on_silence_rate": _mean_or_none(
                    [float(item["metric_details"]["false_speech_on_silence_rate"]) for item in raster]
                ),
                "track1c_false_certainty_on_unexplored_time": _mean_or_none(
                    [float(item["metric_details"]["false_certainty_on_unexplored_time"]) for item in raster]
                ),
            }
        )
    else:
        metrics.update(
            {
                "track1c_timing_localization_exact": _mean_or_none(
                    [float(item["metric_details"]["timing_localization_exact"]) for item in timing_loc]
                ),
                "track1c_speaker_raster_token_accuracy": _mean_or_none(
                    [float(item["metric_details"]["speaker_raster_token_accuracy"]) for item in raster]
                ),
                "track1c_speaker_raster_exact": _mean_or_none(
                    [float(item["metric_details"]["speaker_raster_exact"]) for item in raster]
                ),
            }
        )
    return metrics


def _score_track1c_format_metrics(
    action_attempts: Sequence[dict[str, Any]],
    checkpoint_outputs: Sequence[dict[str, Any]],
) -> dict[str, float | None]:
    action_format = [float(item["final_parse"].get("format_valid", False)) for item in action_attempts]
    action_semantic = [float(item.get("semantic_valid", False)) for item in action_attempts]
    output_valid = [float(item["parsed"].get("format_valid", False)) for item in checkpoint_outputs]
    repair_attempts = 0
    repair_successes = 0
    for item in action_attempts:
        if item.get("repair_attempted"):
            repair_attempts += 1
            repair_successes += int(item["final_parse"].get("format_valid", False) and item.get("semantic_valid", False))
    for item in checkpoint_outputs:
        if item["repair_attempts"]:
            repair_attempts += 1
            repair_successes += int(item["parsed"].get("format_valid", False))
    return {
        "action_format_valid_rate": _mean_or_none(action_format),
        "action_semantic_valid_rate": _mean_or_none(action_semantic),
        "probe_valid_rate": _mean_or_none(output_valid),
        "repair_success_rate": float(repair_successes / repair_attempts) if repair_attempts else None,
    }


def _sentence_acquisition_order(episode: dict[str, Any], action_history: Sequence[dict[str, Any]]) -> list[str]:
    observed_so_far: list[str] = []
    acquired: list[str] = []
    for step_index in range(1, len(action_history) + 1):
        current = observed_sentence_ids(episode, _query_intervals_from_history(action_history[:step_index]))
        for sentence_id in newly_crossed_checkpoints(episode, observed_so_far, current):
            acquired.append(sentence_id)
        observed_so_far = current
    return acquired


def score_track1c_exploration_metrics(
    episode: dict[str, Any],
    action_attempts: Sequence[dict[str, Any]],
    action_history: Sequence[dict[str, Any]],
    observed_sentence_ids_value: Sequence[str],
    observed_bins: set[int],
) -> dict[str, float | None]:
    sentence_ids = [str(sentence["sentence_id"]) for sentence in episode["sentences"]]
    observed_set = set(observed_sentence_ids_value)
    target_ids = set(_target_evidence_sentence_ids(episode))
    acquired_order = _sentence_acquisition_order(episode, action_history)
    canonical_index = {sentence_id: index for index, sentence_id in enumerate(sentence_ids)}
    inversions = 0
    for left_index, left_sentence_id in enumerate(acquired_order):
        for right_sentence_id in acquired_order[left_index + 1 :]:
            if canonical_index[left_sentence_id] > canonical_index[right_sentence_id]:
                inversions += 1

    prefix_len = 0
    for sentence_id in sentence_ids:
        if sentence_id in observed_set:
            prefix_len += 1
        else:
            break
    prefix_observed = set(sentence_ids[:prefix_len])
    queried_bin_total = sum(int(item["length_bin"]) for item in action_history)
    seen_bins: set[int] = set()
    redundant_query_count = 0
    partial_redundant_query_count = 0
    for item in action_history:
        query_bins = set(range(int(item["start_bin"]), int(item["end_bin_exclusive"])))
        overlap = len(query_bins & seen_bins)
        if overlap == len(query_bins) and query_bins:
            redundant_query_count += 1
        elif overlap > 0:
            partial_redundant_query_count += 1
        seen_bins |= query_bins

    all_target_step: float | None = None
    all_target_bins: float | None = None
    if target_ids:
        for step_index in range(1, len(action_history) + 1):
            current = set(observed_sentence_ids(episode, _query_intervals_from_history(action_history[:step_index])))
            if target_ids <= current:
                all_target_step = float(step_index)
                all_target_bins = float(sum(int(item["length_bin"]) for item in action_history[:step_index]))
                break

    config = _track1c_config_for_episode(episode)
    skipped_prior = 0
    for sentence_id in observed_set:
        index = canonical_index[sentence_id]
        skipped_prior += sum(1 for prior_id in sentence_ids[:index] if prior_id not in observed_set)

    return {
        "query_count": float(len(action_history)),
        "action_attempt_count": float(len(action_attempts)),
        "invalid_action_count": float(sum(float(not item.get("semantic_valid", False)) for item in action_attempts)),
        "repair_attempted_action_count": float(sum(float(bool(item.get("repair_attempted"))) for item in action_attempts)),
        "budget_usage_rate": float(queried_bin_total / config.max_total_queried_bins) if config.max_total_queried_bins else 0.0,
        "step_usage_rate": float(len(action_history) / config.max_steps) if config.max_steps else 0.0,
        "repeated_bin_total": float(max(0, queried_bin_total - len(observed_bins))),
        "redundant_query_count": float(redundant_query_count),
        "partial_redundant_query_count": float(partial_redundant_query_count),
        "sentence_coverage_rate": float(len(observed_set) / len(sentence_ids)) if sentence_ids else 0.0,
        "target_evidence_sentence_recall": float(len(observed_set & target_ids) / len(target_ids)) if target_ids else None,
        "checkpoint_crossed_count": float(len(observed_set)),
        "all_target_evidence_found_step": all_target_step,
        "all_target_evidence_found_queried_bins": all_target_bins,
        "prefix_contiguous_observed_rate": float(len(prefix_observed) / len(observed_set)) if observed_set else 1.0,
        "out_of_order_sentence_acquisition_count": float(skipped_prior),
        "sentence_acquisition_inversion_count": float(inversions),
    }


def score_track1c_continuous_exploration_metrics(
    episode: dict[str, Any],
    action_attempts: Sequence[dict[str, Any]],
    action_history: Sequence[dict[str, Any]],
    observed_sentence_ids_value: Sequence[str],
) -> dict[str, float | None]:
    sentence_ids = [str(sentence["sentence_id"]) for sentence in episode["sentences"]]
    observed_set = set(observed_sentence_ids_value)
    target_ids = set(_target_evidence_sentence_ids(episode))
    query_intervals = _query_intervals_from_history(action_history)
    listened, unique, redundant_query_count, partial_redundant_query_count = continuous_redundancy_counts(query_intervals)
    acquired_order = _sentence_acquisition_order(episode, action_history)
    canonical_index = {sentence_id: index for index, sentence_id in enumerate(sentence_ids)}
    inversions = 0
    for left_index, left_sentence_id in enumerate(acquired_order):
        for right_sentence_id in acquired_order[left_index + 1 :]:
            if canonical_index[left_sentence_id] > canonical_index[right_sentence_id]:
                inversions += 1
    prefix_len = 0
    for sentence_id in sentence_ids:
        if sentence_id in observed_set:
            prefix_len += 1
        else:
            break
    prefix_observed = set(sentence_ids[:prefix_len])
    all_target_step: float | None = None
    all_target_seconds: float | None = None
    if target_ids:
        for step_index in range(1, len(action_history) + 1):
            current = set(observed_sentence_ids(episode, _query_intervals_from_history(action_history[:step_index])))
            if target_ids <= current:
                all_target_step = float(step_index)
                all_target_seconds = float(sum(float(item["duration_seconds"]) for item in action_history[:step_index]))
                break
    skipped_prior = 0
    for sentence_id in observed_set:
        index = canonical_index[sentence_id]
        skipped_prior += sum(1 for prior_id in sentence_ids[:index] if prior_id not in observed_set)
    config = _track1c_config_for_episode(episode, BenchmarkConfig(time_mode=TIME_MODE_CONTINUOUS))
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
        "repeated_seconds_total": float(max(0.0, listened - unique)),
        "redundant_query_count": float(redundant_query_count),
        "partial_redundant_query_count": float(partial_redundant_query_count),
        "sentence_coverage_rate": float(len(observed_set) / len(sentence_ids)) if sentence_ids else 0.0,
        "target_evidence_sentence_recall": float(len(observed_set & target_ids) / len(target_ids)) if target_ids else None,
        "checkpoint_crossed_count": float(len(observed_set)),
        "all_target_evidence_found_step": all_target_step,
        "all_target_evidence_found_listened_seconds": all_target_seconds,
        "prefix_contiguous_observed_rate": float(len(prefix_observed) / len(observed_set)) if observed_set else 1.0,
        "out_of_order_sentence_acquisition_count": float(skipped_prior),
        "sentence_acquisition_inversion_count": float(inversions),
    }



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


def _track1c_answer_score(
    track1c_metrics: dict[str, float | None],
    *,
    time_mode: str = TIME_MODE_BINNED,
) -> float:
    semantic_balanced = _weighted_score(
        [
            (_bounded_score(track1c_metrics.get("track1c_generic_accuracy")), 1.0),
            (_bounded_score(track1c_metrics.get("track1c_category_accuracy")), 1.0),
            (_bounded_score(track1c_metrics.get("track1c_content_accuracy")), 1.0),
        ]
    )
    if validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS:
        speaker_segment_score = _weighted_score(
            [
                (_bounded_score(track1c_metrics.get("track1c_speaker_segment_time_accuracy")), 0.50),
                (_bounded_score(track1c_metrics.get("track1c_speaker_segment_iou")), 0.30),
                (_bounded_score(track1c_metrics.get("track1c_unexplored_time_accuracy")), 0.20),
            ]
        )
        answer_score = _weighted_score(
            [
                (semantic_balanced, 0.45),
                (_bounded_score(track1c_metrics.get("track1c_non_unknown_accuracy")), 0.17),
                (_bounded_score(track1c_metrics.get("track1c_unknown_accuracy")), 0.10),
                (_bounded_score(track1c_metrics.get("track1c_update_success_rate")), 0.13),
                (_bounded_score(track1c_metrics.get("track1c_timing_localization_iou")), 0.05),
                (speaker_segment_score, 0.10),
            ]
        )
        return 0.0 if answer_score is None else answer_score
    answer_score = _weighted_score(
        [
            (semantic_balanced, 0.50),
            (_bounded_score(track1c_metrics.get("track1c_non_unknown_accuracy")), 0.20),
            (_bounded_score(track1c_metrics.get("track1c_unknown_accuracy")), 0.10),
            (_bounded_score(track1c_metrics.get("track1c_update_success_rate")), 0.15),
            (_bounded_score(track1c_metrics.get("track1c_timing_localization_iou")), 0.05),
        ]
    )
    return 0.0 if answer_score is None else answer_score


def score_track1c_master_metrics(
    *,
    condition: str,
    stop_reason: str,
    listened_bin_total: int | float,
    track1c_metrics: dict[str, float | None],
    format_metrics: dict[str, float | None],
    exploration_metrics: dict[str, float | None],
    time_mode: str = TIME_MODE_BINNED,
) -> dict[str, float]:
    continuous_mode = validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS
    answer_score = _track1c_answer_score(track1c_metrics, time_mode=time_mode)
    if condition == FULL_CONTEXT:
        return {
            "track1c_full_context_answer_score": answer_score,
            "track1c_full_context_master_score": 100.0 * answer_score,
        }

    if condition == ACTIVE:
        prefix = "track1c_active"
    elif condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        prefix = "track1c_active_same_session"
    else:
        raise RuntimeError(f"Unsupported Track 1c condition for master score: {condition}")

    action_format = _bounded_score(format_metrics.get("action_format_valid_rate")) or 0.0
    action_semantic = _bounded_score(format_metrics.get("action_semantic_valid_rate")) or 0.0
    non_invalid_episode = 0.0 if str(stop_reason).startswith("invalid_action:") else 1.0
    protocol_score = non_invalid_episode * math.sqrt(action_format * action_semantic)

    exploration_core = _weighted_score(
        [
            (_bounded_score(exploration_metrics.get("target_evidence_sentence_recall")), 0.70),
            (_bounded_score(exploration_metrics.get("sentence_coverage_rate")), 0.30),
        ]
    )
    exploration_core = 0.0 if exploration_core is None else exploration_core
    repeated_bin_total = max(
        0.0,
        float(exploration_metrics.get("repeated_bin_total", exploration_metrics.get("repeated_seconds_total", 0.0)) or 0.0),
    )
    listened_total = max(0.0, float(listened_bin_total))
    nonredundant_query_rate = 0.0 if listened_total <= 0.0 else 1.0 - min(1.0, repeated_bin_total / listened_total)
    exploration_quality = _weighted_score(
        [
            (_bounded_score(exploration_metrics.get("prefix_contiguous_observed_rate")), 0.50),
            (nonredundant_query_rate, 0.50),
        ]
    )
    exploration_quality = 0.0 if exploration_quality is None else exploration_quality
    exploration_score = exploration_core * exploration_quality

    master_score = 100.0 * (protocol_score ** 0.25) * (exploration_score ** 0.35) * (answer_score ** 0.40)
    return {
        f"{prefix}_protocol_score": protocol_score,
        f"{prefix}_exploration_score": exploration_score,
        f"{prefix}_answer_score": answer_score,
        f"{prefix}_master_score": master_score,
    }


def _track1c_master_key_for_condition(condition: str) -> str:
    if condition == FULL_CONTEXT:
        return "track1c_full_context_master_score"
    if condition == ACTIVE:
        return "track1c_active_master_score"
    if condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
        return "track1c_active_same_session_master_score"
    raise RuntimeError(f"Unsupported Track 1c condition for master score: {condition}")


def ensure_track1c_master_metrics(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("track_id") != "1c":
        return result
    condition = str(result["condition"])
    track1c_metrics = dict(result.get("track1c_metrics") or {})
    if _track1c_master_key_for_condition(condition) in track1c_metrics:
        return result
    enriched = dict(result)
    track1c_metrics.update(
        score_track1c_master_metrics(
            condition=condition,
            stop_reason=str(result.get("stop_reason", "")),
            listened_bin_total=float(result.get("listened_bin_total", 0.0)),
            track1c_metrics=track1c_metrics,
            format_metrics=dict(result.get("format_metrics") or {}),
            exploration_metrics=dict(result.get("exploration_metrics") or {}),
            time_mode=str(result.get("time_mode", TIME_MODE_BINNED)),
        )
    )
    enriched["track1c_metrics"] = track1c_metrics
    return enriched

def _run_track1c_action_loop(
    *,
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    run_dir: Path,
    same_session: bool,
    time_mode: str = TIME_MODE_BINNED,
) -> dict[str, Any]:
    if validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS:
        return _run_track1c_continuous_action_loop(
            root=root,
            episode=episode,
            regime=regime,
            adapter=adapter,
            run_dir=run_dir,
            same_session=same_session,
        )
    env = Track1cEnvironment(root, episode, run_dir)
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    prompt = build_track1c_action_initial_prompt(episode, regime)
    action_attempts: list[dict[str, Any]] = []
    checkpoint_outputs: list[dict[str, Any]] = []
    observed_sentence_ids_value: list[str] = []
    raw_action, raw_action_api_attempts = _send_with_api_retry(
        adapter,
        session,
        prompt,
        [],
        env.config.api_retry_sleep_sec,
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
            semantic_valid, reason = validate_query(parsed, env.config, env.listened_bin_total)
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
                env.config.api_retry_sleep_sec,
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
                    semantic_valid, reason = validate_query(repaired, env.config, env.listened_bin_total)
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

        previous_observed = list(observed_sentence_ids_value)
        clip_path, metadata = env.clip_for_query(
            len(env.action_history) + 1,
            final_parse["start_bin"],
            final_parse["length_bin"],
        )
        observed_sentence_ids_value = observed_sentence_ids(episode, _query_intervals_from_history(env.action_history))
        crossed = newly_crossed_checkpoints(episode, previous_observed, observed_sentence_ids_value)
        consumed_latest_audio = False
        for checkpoint_sentence_id in crossed:
            outputs, audio_consumed = _run_checkpoint_questions(
                adapter=adapter,
                episode=episode,
                regime=regime,
                checkpoint_sentence_id=checkpoint_sentence_id,
                observed_sentence_ids_value=observed_sentence_ids_value,
                observed_bins=env.observed_bins,
                action_history=env.action_history,
                audio_paths=[] if same_session and consumed_latest_audio else ([clip_path] if same_session else list(env.query_audio_paths)),
                full_context=False,
                same_session=same_session,
                session=session,
                consume_audio_once=same_session,
                api_retry_sleep_sec=env.config.api_retry_sleep_sec,
            )
            checkpoint_outputs.extend(outputs)
            consumed_latest_audio = consumed_latest_audio or audio_consumed

        if len(env.action_history) >= env.config.max_steps:
            env.stop_reason = "max_steps_reached"
            break
        if env.listened_bin_total >= env.config.max_total_queried_bins:
            env.stop_reason = "max_total_queried_bins_reached"
            break

        raw_action, raw_action_api_attempts = _send_with_api_retry(
            adapter,
            session,
            build_track1c_action_followup_prompt(metadata, env),
            [] if same_session and consumed_latest_audio else [clip_path],
            env.config.api_retry_sleep_sec,
        )

    track1c_metrics = score_track1c_checkpoint_outputs(checkpoint_outputs)
    format_metrics = _score_track1c_format_metrics(action_attempts, checkpoint_outputs)
    exploration_metrics = score_track1c_exploration_metrics(
        episode,
        action_attempts,
        env.action_history,
        observed_sentence_ids_value,
        env.observed_bins,
    )
    condition = ACTIVE_SAME_SESSION_PERIODIC_PROBE if same_session else ACTIVE
    track1c_metrics = {
        **track1c_metrics,
        **score_track1c_master_metrics(
            condition=condition,
            stop_reason=env.stop_reason,
            listened_bin_total=env.listened_bin_total,
            track1c_metrics=track1c_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
        ),
    }
    result: dict[str, Any] = {
        "episode_id": episode["episode_id"],
        "track_id": "1c",
        "conversation_id": episode["conversation_id"],
        "variant_id": episode["variant_id"],
        "category": episode.get("category", ""),
        "category_label": episode.get("category_label", ""),
        "condition": condition,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": env.action_history,
        "repair_attempts": action_attempts,
        "stop_reason": env.stop_reason,
        "observed_bins": sorted(env.observed_bins),
        "observed_sentence_ids": observed_sentence_ids_value,
        "listened_bin_total": env.listened_bin_total,
        "checkpoint_outputs": checkpoint_outputs,
        "behavioral_protocol_version": TRACK1C_PROTOCOL_VERSION,
        "track1c_metrics": track1c_metrics,
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
    }
    _close_adapter_session(adapter, session)
    return result


def _validate_track1c_continuous_action(parsed: dict[str, Any], env: Track1cContinuousEnvironment) -> tuple[bool, str]:
    return validate_continuous_query(
        parsed,
        audio_duration_seconds=env.audio_duration_seconds,
        max_single_query_seconds=env.max_single_query_seconds,
        total_listening_budget_seconds=env.total_listening_budget_seconds,
        listened_seconds_total=env.listened_seconds_total,
    )


def _run_track1c_continuous_action_loop(
    *,
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    run_dir: Path,
    same_session: bool,
) -> dict[str, Any]:
    env = Track1cContinuousEnvironment(root, episode, run_dir)
    session = adapter.create_session(ACTION_SYSTEM_PROMPT)
    prompt = build_track1c_continuous_action_initial_prompt(episode, regime)
    action_attempts: list[dict[str, Any]] = []
    checkpoint_outputs: list[dict[str, Any]] = []
    observed_sentence_ids_value: list[str] = []
    raw_action, raw_action_api_attempts = _send_with_api_retry(
        adapter,
        session,
        prompt,
        [],
        env.config.api_retry_sleep_sec,
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

        parsed = parse_continuous_action(raw_action)
        repair_attempted = False
        api_attempts = list(raw_action_api_attempts)
        if parsed.get("format_valid") and parsed.get("action") == "QUERY":
            semantic_valid, reason = _validate_track1c_continuous_action(parsed, env)
        else:
            semantic_valid = bool(parsed.get("format_valid"))
            reason = parsed.get("reason", "")
        if not parsed.get("format_valid") or (parsed.get("action") == "QUERY" and not semantic_valid):
            repair_attempted = True
            repair_reason = reason or parsed.get("reason", "invalid_action")
            repaired_raw, repair_api_attempts = _send_with_api_retry(
                adapter,
                session,
                build_track1c_continuous_action_repair_prompt(repair_reason, env),
                [],
                env.config.api_retry_sleep_sec,
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
                    semantic_valid, reason = _validate_track1c_continuous_action(repaired, env)
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

        previous_observed = list(observed_sentence_ids_value)
        clip_path, metadata = env.clip_for_query(
            len(env.action_history) + 1,
            final_parse["start_time_seconds"],
            final_parse["end_time_seconds"],
        )
        observed_sentence_ids_value = observed_sentence_ids(episode, _query_intervals_from_history(env.action_history))
        crossed = newly_crossed_checkpoints(episode, previous_observed, observed_sentence_ids_value)
        consumed_latest_audio = False
        for checkpoint_sentence_id in crossed:
            outputs, audio_consumed = _run_checkpoint_questions(
                adapter=adapter,
                episode=episode,
                regime=regime,
                checkpoint_sentence_id=checkpoint_sentence_id,
                observed_sentence_ids_value=observed_sentence_ids_value,
                observed_bins=set(),
                action_history=env.action_history,
                audio_paths=[] if same_session and consumed_latest_audio else ([clip_path] if same_session else list(env.query_audio_paths)),
                full_context=False,
                same_session=same_session,
                session=session,
                consume_audio_once=same_session,
                api_retry_sleep_sec=env.config.api_retry_sleep_sec,
                time_mode=TIME_MODE_CONTINUOUS,
            )
            checkpoint_outputs.extend(outputs)
            consumed_latest_audio = consumed_latest_audio or audio_consumed

        if len(env.action_history) >= env.config.max_steps:
            env.stop_reason = "max_steps_reached"
            break
        if env.listened_seconds_total >= env.total_listening_budget_seconds - 1e-9:
            env.stop_reason = "max_total_queried_seconds_reached"
            break

        raw_action, raw_action_api_attempts = _send_with_api_retry(
            adapter,
            session,
            build_track1c_continuous_action_followup_prompt(metadata, env),
            [] if same_session and consumed_latest_audio else [clip_path],
            env.config.api_retry_sleep_sec,
        )

    outputs, _audio_consumed = _run_checkpoint_questions(
        adapter=adapter,
        episode=episode,
        regime=regime,
        checkpoint_sentence_id="END",
        observed_sentence_ids_value=observed_sentence_ids_value,
        observed_bins=set(),
        action_history=env.action_history,
        audio_paths=[] if same_session else list(env.query_audio_paths),
        full_context=False,
        same_session=same_session,
        session=session if same_session else None,
        consume_audio_once=same_session,
        api_retry_sleep_sec=env.config.api_retry_sleep_sec,
        time_mode=TIME_MODE_CONTINUOUS,
    )
    checkpoint_outputs.extend(outputs)

    track1c_metrics = score_track1c_checkpoint_outputs(checkpoint_outputs, time_mode=TIME_MODE_CONTINUOUS)
    format_metrics = _score_track1c_format_metrics(action_attempts, checkpoint_outputs)
    exploration_metrics = score_track1c_continuous_exploration_metrics(
        episode,
        action_attempts,
        env.action_history,
        observed_sentence_ids_value,
    )
    condition = ACTIVE_SAME_SESSION_PERIODIC_PROBE if same_session else ACTIVE
    track1c_metrics = {
        **track1c_metrics,
        **score_track1c_master_metrics(
            condition=condition,
            stop_reason=env.stop_reason,
            listened_bin_total=env.listened_seconds_total,
            track1c_metrics=track1c_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
            time_mode=TIME_MODE_CONTINUOUS,
        ),
    }
    result = {
        "episode_id": episode["episode_id"],
        "track_id": "1c",
        "track_display_id": "1c-H",
        "time_mode": TIME_MODE_CONTINUOUS,
        "conversation_id": episode["conversation_id"],
        "variant_id": episode["variant_id"],
        "category": episode.get("category", ""),
        "category_label": episode.get("category_label", ""),
        "condition": condition,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": env.action_history,
        "repair_attempts": action_attempts,
        "stop_reason": env.stop_reason,
        "query_intervals": [
            {"start_time_seconds": start, "end_time_seconds": end}
            for start, end in env.query_intervals
        ],
        "observed_sentence_ids": observed_sentence_ids_value,
        "listened_seconds_total": round(env.listened_seconds_total, 6),
        "checkpoint_outputs": checkpoint_outputs,
        "behavioral_protocol_version": TRACK1C_PROTOCOL_VERSION,
        "track1c_metrics": track1c_metrics,
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
    }
    _close_adapter_session(adapter, session)
    return result


def _run_track1c_full_context_condition(
    *,
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
    time_mode: str = TIME_MODE_BINNED,
) -> dict[str, Any]:
    if validate_time_mode(time_mode) == TIME_MODE_CONTINUOUS:
        return _run_track1c_continuous_full_context_condition(
            root=root,
            episode=episode,
            regime=regime,
            adapter=adapter,
        )
    audio_path = Path(root) / "dataset" / episode["audio_path"]
    observed_sentence_ids_value = [str(sentence["sentence_id"]) for sentence in episode["sentences"]]
    observed_bins = set(range(int(episode["num_bins"])))
    outputs, _audio_consumed = _run_checkpoint_questions(
        adapter=adapter,
        episode=episode,
        regime=regime,
        checkpoint_sentence_id="END",
        observed_sentence_ids_value=observed_sentence_ids_value,
        observed_bins=observed_bins,
        action_history=[],
        audio_paths=[audio_path],
        full_context=True,
        same_session=False,
        session=None,
        consume_audio_once=False,
        api_retry_sleep_sec=BenchmarkConfig().api_retry_sleep_sec,
        time_mode=TIME_MODE_BINNED,
    )
    track1c_metrics = score_track1c_checkpoint_outputs(outputs)
    format_metrics = _score_track1c_format_metrics([], outputs)
    exploration_metrics = score_track1c_exploration_metrics(
        episode,
        [],
        [],
        observed_sentence_ids_value,
        observed_bins,
    )
    track1c_metrics = {
        **track1c_metrics,
        **score_track1c_master_metrics(
            condition=FULL_CONTEXT,
            stop_reason=FULL_CONTEXT,
            listened_bin_total=int(episode["num_bins"]),
            track1c_metrics=track1c_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
            time_mode=TIME_MODE_BINNED,
        ),
    }
    return {
        "episode_id": episode["episode_id"],
        "track_id": "1c",
        "conversation_id": episode["conversation_id"],
        "variant_id": episode["variant_id"],
        "category": episode.get("category", ""),
        "category_label": episode.get("category_label", ""),
        "condition": FULL_CONTEXT,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": [],
        "repair_attempts": [],
        "stop_reason": FULL_CONTEXT,
        "observed_bins": sorted(observed_bins),
        "observed_sentence_ids": observed_sentence_ids_value,
        "listened_bin_total": int(episode["num_bins"]),
        "checkpoint_outputs": outputs,
        "behavioral_protocol_version": TRACK1C_PROTOCOL_VERSION,
        "track1c_metrics": track1c_metrics,
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
    }


def _run_track1c_continuous_full_context_condition(
    *,
    root: Path,
    episode: dict[str, Any],
    regime: str,
    adapter: BaseAdapter,
) -> dict[str, Any]:
    audio_path = Path(root) / "dataset" / episode["audio_path"]
    observed_sentence_ids_value = [str(sentence["sentence_id"]) for sentence in episode["sentences"]]
    full_interval = [(0.0, float(episode["audio_duration_seconds"]))]
    outputs, _audio_consumed = _run_checkpoint_questions(
        adapter=adapter,
        episode=episode,
        regime=regime,
        checkpoint_sentence_id="END",
        observed_sentence_ids_value=observed_sentence_ids_value,
        observed_bins=set(),
        action_history=[],
        audio_paths=[audio_path],
        full_context=True,
        same_session=False,
        session=None,
        consume_audio_once=False,
        api_retry_sleep_sec=BenchmarkConfig().api_retry_sleep_sec,
        time_mode=TIME_MODE_CONTINUOUS,
    )
    track1c_metrics = score_track1c_checkpoint_outputs(outputs, time_mode=TIME_MODE_CONTINUOUS)
    format_metrics = _score_track1c_format_metrics([], outputs)
    exploration_metrics = score_track1c_continuous_exploration_metrics(
        episode,
        [],
        [{"step_index": 1, "start_time_seconds": 0.0, "end_time_seconds": float(episode["audio_duration_seconds"]), "duration_seconds": float(episode["audio_duration_seconds"])}],
        observed_sentence_ids_value,
    )
    track1c_metrics = {
        **track1c_metrics,
        **score_track1c_master_metrics(
            condition=FULL_CONTEXT,
            stop_reason=FULL_CONTEXT,
            listened_bin_total=float(episode["audio_duration_seconds"]),
            track1c_metrics=track1c_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
            time_mode=TIME_MODE_CONTINUOUS,
        ),
    }
    return {
        "episode_id": episode["episode_id"],
        "track_id": "1c",
        "track_display_id": "1c-H",
        "time_mode": TIME_MODE_CONTINUOUS,
        "conversation_id": episode["conversation_id"],
        "variant_id": episode["variant_id"],
        "category": episode.get("category", ""),
        "category_label": episode.get("category_label", ""),
        "condition": FULL_CONTEXT,
        "model_id": adapter.model_id,
        "regime": regime,
        "action_history": [],
        "repair_attempts": [],
        "stop_reason": FULL_CONTEXT,
        "query_intervals": [{"start_time_seconds": 0.0, "end_time_seconds": float(episode["audio_duration_seconds"])}],
        "observed_sentence_ids": observed_sentence_ids_value,
        "listened_seconds_total": float(episode["audio_duration_seconds"]),
        "checkpoint_outputs": outputs,
        "behavioral_protocol_version": TRACK1C_PROTOCOL_VERSION,
        "track1c_metrics": track1c_metrics,
        "format_metrics": format_metrics,
        "exploration_metrics": exploration_metrics,
    }


def run_track1c_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    adapter: BaseAdapter,
    episode_ids: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    regimes: Sequence[str] | None = None,
    reprocess: bool = False,
    parallelism: int = 1,
    time_mode: str = TIME_MODE_BINNED,
) -> list[dict[str, Any]]:
    requested_time_mode = validate_time_mode(time_mode)
    root = Path(dataset_root)
    manifest = load_track1c_manifest(root)
    episodes = normalize_track1c_manifest(manifest)
    manifest_episode_ids = [str(episode["episode_id"]) for episode in episodes]

    if episode_ids is None:
        selected_episodes = list(episodes)
    else:
        requested_episode_ids = [str(episode_id) for episode_id in episode_ids]
        unknown_episode_ids = sorted(set(requested_episode_ids) - set(manifest_episode_ids))
        _assert(not unknown_episode_ids, f"Unknown episode_ids: {unknown_episode_ids}")
        requested_set = set(requested_episode_ids)
        selected_episodes = [episode for episode in episodes if episode["episode_id"] in requested_set]

    selected_conditions = list(DEFAULT_CONDITIONS if conditions is None else [str(condition) for condition in conditions])
    unknown_conditions = sorted(set(selected_conditions) - set(DEFAULT_CONDITIONS))
    _assert(not unknown_conditions, f"Unknown conditions: {unknown_conditions}")

    selected_regimes = list(REGIMES if regimes is None else [str(regime) for regime in regimes])
    unknown_regimes = sorted(set(selected_regimes) - set(REGIMES))
    _assert(not unknown_regimes, f"Unknown regimes: {unknown_regimes}")

    run_specs: list[tuple[str, str, dict[str, Any]]] = []
    for episode in selected_episodes:
        for regime in selected_regimes:
            for condition in selected_conditions:
                run_specs.append((condition, regime, episode))

    _assert(bool(run_specs), "Selected Track 1c benchmark run is empty.")
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
    progress = ProgressReporter(
        len(run_specs),
        label="track1c-run",
        eta_total=len(run_specs) - len(cached_spec_keys),
    )

    def _run_one_spec(spec: tuple[str, str, dict[str, Any]]) -> tuple[dict[str, Any], bool]:
        condition, regime, episode = spec
        spec_key = (condition, regime, episode["episode_id"])
        existing_path = _selected_episode_result_path(
            Path(results_root),
            condition,
            regime,
            episode["episode_id"],
        )
        if spec_key in cached_spec_keys:
            return ensure_track1c_master_metrics(json.loads(existing_path.read_text(encoding="utf-8"))), True
        if condition == FULL_CONTEXT:
            result = _run_track1c_full_context_condition(
                root=root,
                episode=episode,
                regime=regime,
                adapter=adapter,
                time_mode=requested_time_mode,
            )
        elif condition == ACTIVE:
            result = _run_track1c_action_loop(
                root=root,
                episode=episode,
                regime=regime,
                adapter=adapter,
                run_dir=Path(results_root) / "runtime" / ACTIVE / regime / episode["episode_id"],
                same_session=False,
                time_mode=requested_time_mode,
            )
        elif condition == ACTIVE_SAME_SESSION_PERIODIC_PROBE:
            result = _run_track1c_action_loop(
                root=root,
                episode=episode,
                regime=regime,
                adapter=adapter,
                run_dir=Path(results_root) / "runtime" / ACTIVE_SAME_SESSION_PERIODIC_PROBE / regime / episode["episode_id"],
                same_session=True,
                time_mode=requested_time_mode,
            )
        else:
            raise RuntimeError(f"Unsupported Track 1c condition: {condition}")
        _write_episode_result(result, Path(results_root))
        return result, False

    max_workers = max(1, int(parallelism))
    results_by_index: list[dict[str, Any] | None] = [None] * len(run_specs)
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
