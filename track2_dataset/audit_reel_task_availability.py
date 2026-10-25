#!/usr/bin/env python3
"""Audit Track 2 reel candidates for deterministic task availability."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REEL_INPUT = SCRIPT_DIR / "track2_role_ipc_reel_candidates_v1.csv"
DEFAULT_SCENE_INPUT = SCRIPT_DIR / "track2_role_ipc_scene_candidates_scene_filter_v1.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "track2_role_ipc_reel_candidates_task_audit_v1.csv"

DEFAULT_GAP_SECONDS = 0.5
DEFAULT_MIN_REEL_DURATION_SEC = 600.0
DEFAULT_MAX_REEL_DURATION_SEC = 1200.0
DEFAULT_EXACT_MIN_WORDS = 8
DEFAULT_EXACT_MAX_WORDS = 22
DEFAULT_EXACT_MIN_DURATION_SEC = 2.0
DEFAULT_EXACT_MAX_DURATION_SEC = 12.0
DEFAULT_BOUNDARY_MARGIN_SEC = 5.0
DEFAULT_MAX_TURN_OVERLAP_RATIO = 0.20
DEFAULT_REPEATED_MIN_SEPARATION_SEC = 10.0
DEFAULT_STRUCTURAL_TARGET_MIN_WORDS = 10
DEFAULT_STRUCTURAL_TARGET_MIN_DURATION_SEC = 5.0

TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
STOP_OR_FILLER = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "just",
    "kind",
    "like",
    "mm",
    "of",
    "oh",
    "okay",
    "on",
    "or",
    "so",
    "sort",
    "that",
    "the",
    "this",
    "to",
    "uh",
    "um",
    "well",
    "yeah",
    "yes",
    "you",
}


@dataclass(frozen=True)
class Turn:
    scene_id: str
    scene_rank: int
    speaker: str
    start: float
    end: float
    text: str
    tokens: tuple[str, ...]
    overlap_ratio: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    @property
    def normalized_text(self) -> str:
        return " ".join(self.tokens)


@dataclass(frozen=True)
class Word:
    scene_id: str
    scene_rank: int
    speaker: str
    start: float
    end: float
    token: str


@dataclass(frozen=True)
class SceneTranscript:
    scene_id: str
    local_turns: tuple[Turn, ...]
    local_words: tuple[Word, ...]
    local_vad: tuple[tuple[str, float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add deterministic task-availability audit fields to Track 2 reel candidates."
    )
    parser.add_argument("--reel-input", type=Path, default=DEFAULT_REEL_INPUT)
    parser.add_argument("--scene-input", type=Path, default=DEFAULT_SCENE_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Separately acquired Seamless Interaction datasets directory.")
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    parser.add_argument("--min-reel-duration-sec", type=float, default=DEFAULT_MIN_REEL_DURATION_SEC)
    parser.add_argument("--max-reel-duration-sec", type=float, default=DEFAULT_MAX_REEL_DURATION_SEC)
    parser.add_argument("--exact-min-words", type=int, default=DEFAULT_EXACT_MIN_WORDS)
    parser.add_argument("--exact-max-words", type=int, default=DEFAULT_EXACT_MAX_WORDS)
    parser.add_argument("--exact-min-duration-sec", type=float, default=DEFAULT_EXACT_MIN_DURATION_SEC)
    parser.add_argument("--exact-max-duration-sec", type=float, default=DEFAULT_EXACT_MAX_DURATION_SEC)
    parser.add_argument("--boundary-margin-sec", type=float, default=DEFAULT_BOUNDARY_MARGIN_SEC)
    parser.add_argument("--max-turn-overlap-ratio", type=float, default=DEFAULT_MAX_TURN_OVERLAP_RATIO)
    parser.add_argument(
        "--repeated-min-separation-sec", type=float, default=DEFAULT_REPEATED_MIN_SEPARATION_SEC
    )
    parser.add_argument(
        "--structural-target-min-words", type=int, default=DEFAULT_STRUCTURAL_TARGET_MIN_WORDS
    )
    parser.add_argument(
        "--structural-target-min-duration-sec",
        type=float,
        default=DEFAULT_STRUCTURAL_TARGET_MIN_DURATION_SEC,
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    require_file(path, "JSON file")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def tokens_from_text(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def normalized_word_token(text: str) -> str:
    tokens = tokens_from_text(text)
    return tokens[0] if len(tokens) == 1 else ""


def interval_overlap(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    for other_start, other_end in intervals:
        total += max(0.0, min(end, other_end) - max(start, other_start))
    return total


def valid_timestamp(start: Any, end: Any, scene_id: str, source: str) -> tuple[float, float]:
    if start is None or end is None:
        raise ValueError(f"{scene_id}: missing {source} timestamp")
    start_float = float(start)
    end_float = float(end)
    if math.isnan(start_float) or math.isnan(end_float):
        raise ValueError(f"{scene_id}: NaN {source} timestamp")
    if start_float < 0 or end_float < start_float:
        raise ValueError(f"{scene_id}: invalid {source} timestamp {start_float}-{end_float}")
    return start_float, end_float


def load_scene_transcript(scene: pd.Series, dataset_root: Path) -> SceneTranscript:
    scene_id = str(scene["scene_id"])
    raw_segments: list[dict[str, Any]] = []
    vad_by_speaker: dict[str, list[tuple[float, float]]] = {"participant1": [], "participant2": []}
    words: list[Word] = []

    for speaker in ("participant1", "participant2"):
        data = load_json(dataset_root / str(scene[f"{speaker}_json_relpath"]))
        transcript = data.get("metadata:transcript")
        vad = data.get("metadata:vad")
        if not isinstance(transcript, list):
            raise ValueError(f"{scene_id}: {speaker} metadata:transcript is not a list")
        if not isinstance(vad, list):
            raise ValueError(f"{scene_id}: {speaker} metadata:vad is not a list")

        for vad_index, vad_segment in enumerate(vad):
            start, end = valid_timestamp(
                vad_segment.get("start"), vad_segment.get("end"), scene_id, f"{speaker} vad {vad_index}"
            )
            if end > start:
                vad_by_speaker[speaker].append((start, end))

        for segment_index, segment in enumerate(transcript):
            if not isinstance(segment, dict):
                raise ValueError(f"{scene_id}: {speaker} transcript segment is not an object")
            start, end = valid_timestamp(
                segment.get("start"),
                segment.get("end"),
                scene_id,
                f"{speaker} transcript segment {segment_index}",
            )
            text = str(segment.get("transcript", ""))
            segment_tokens = tokens_from_text(text)
            if segment_tokens and end > start:
                raw_segments.append(
                    {
                        "speaker": speaker,
                        "start": start,
                        "end": end,
                        "text": text,
                        "tokens": tuple(segment_tokens),
                    }
                )

            segment_words = segment.get("words", [])
            if not isinstance(segment_words, list):
                raise ValueError(f"{scene_id}: {speaker} transcript words is not a list")
            for word_index, word in enumerate(segment_words):
                if not isinstance(word, dict):
                    raise ValueError(f"{scene_id}: {speaker} word is not an object")
                token = normalized_word_token(str(word.get("word", "")))
                if not token:
                    continue
                if word.get("start") is None or word.get("end") is None:
                    continue
                word_start, word_end = valid_timestamp(
                    word.get("start"), word.get("end"), scene_id, f"{speaker} word {word_index}"
                )
                if word_end > word_start:
                    words.append(Word(scene_id, 0, speaker, word_start, word_end, token))

    turns = build_turns(scene_id, raw_segments, vad_by_speaker)
    vad_rows = tuple(
        (speaker, start, end)
        for speaker, intervals in vad_by_speaker.items()
        for start, end in intervals
    )
    return SceneTranscript(scene_id, tuple(turns), tuple(sorted(words, key=lambda w: (w.start, w.end))), vad_rows)


def build_turns(
    scene_id: str,
    segments: list[dict[str, Any]],
    vad_by_speaker: dict[str, list[tuple[float, float]]],
) -> list[Turn]:
    turns: list[dict[str, Any]] = []
    for segment in sorted(segments, key=lambda item: (item["start"], item["end"])):
        if (
            turns
            and turns[-1]["speaker"] == segment["speaker"]
            and segment["start"] - turns[-1]["end"] <= 0.75
        ):
            turns[-1]["end"] = max(turns[-1]["end"], segment["end"])
            turns[-1]["text"] = f"{turns[-1]['text']} {segment['text']}".strip()
            turns[-1]["tokens"] = tuple(list(turns[-1]["tokens"]) + list(segment["tokens"]))
        else:
            turns.append(dict(segment))

    result: list[Turn] = []
    for turn in turns:
        other_speaker = "participant2" if turn["speaker"] == "participant1" else "participant1"
        duration = turn["end"] - turn["start"]
        overlap = interval_overlap(turn["start"], turn["end"], vad_by_speaker[other_speaker])
        overlap_ratio = overlap / duration if duration > 0 else 1.0
        result.append(
            Turn(
                scene_id=scene_id,
                scene_rank=0,
                speaker=turn["speaker"],
                start=turn["start"],
                end=turn["end"],
                text=turn["text"],
                tokens=tuple(turn["tokens"]),
                overlap_ratio=overlap_ratio,
            )
        )
    return result


def globalize_scene(
    scene_transcript: SceneTranscript,
    scene_rank: int,
    offset: float,
) -> tuple[list[Turn], list[Word], list[tuple[str, float, float]]]:
    turns = [
        Turn(
            scene_id=turn.scene_id,
            scene_rank=scene_rank,
            speaker=turn.speaker,
            start=turn.start + offset,
            end=turn.end + offset,
            text=turn.text,
            tokens=turn.tokens,
            overlap_ratio=turn.overlap_ratio,
        )
        for turn in scene_transcript.local_turns
    ]
    words = [
        Word(
            scene_id=word.scene_id,
            scene_rank=scene_rank,
            speaker=word.speaker,
            start=word.start + offset,
            end=word.end + offset,
            token=word.token,
        )
        for word in scene_transcript.local_words
    ]
    vad = [
        (speaker, start + offset, end + offset)
        for speaker, start, end in scene_transcript.local_vad
    ]
    return turns, words, vad


@lru_cache(maxsize=200_000)
def is_content_phrase(tokens: tuple[str, ...]) -> bool:
    if not any(any(ch.isalpha() for ch in token) for token in tokens):
        return False
    content_count = sum(token not in STOP_OR_FILLER and any(ch.isalpha() for ch in token) for token in tokens)
    return content_count >= max(1, math.ceil(len(tokens) / 2))


def turn_near_boundary(turn: Turn, boundaries: list[tuple[float, float]], margin: float) -> bool:
    scene_start, scene_end = boundaries[turn.scene_rank - 1]
    return turn.start - scene_start < margin or scene_end - turn.end < margin


def exact_quote_candidates(
    turns: list[Turn],
    boundaries: list[tuple[float, float]],
    args: argparse.Namespace,
) -> list[Turn]:
    normalized_counts = Counter(turn.normalized_text for turn in turns)
    candidates = []
    for turn in turns:
        if not (args.exact_min_words <= turn.word_count <= args.exact_max_words):
            continue
        if not (args.exact_min_duration_sec <= turn.duration <= args.exact_max_duration_sec):
            continue
        if normalized_counts[turn.normalized_text] != 1:
            continue
        if turn.overlap_ratio > args.max_turn_overlap_ratio:
            continue
        if turn_near_boundary(turn, boundaries, args.boundary_margin_sec):
            continue
        if not is_content_phrase(turn.tokens):
            continue
        candidates.append(turn)
    return candidates


def partial_quote_candidate_count(exact_candidates: list[Turn]) -> int:
    count = 0
    for turn in exact_candidates:
        if turn.word_count < 10:
            continue
        middle = turn.tokens[2:-2]
        maskable = [token for token in middle if token not in STOP_OR_FILLER and any(ch.isalpha() for ch in token)]
        if len(maskable) >= 2:
            count += 1
    return count


def repeated_phrase_candidates(words: list[Word], min_separation_sec: float) -> dict[tuple[str, ...], list[Word]]:
    candidates: dict[tuple[str, ...], list[Word]] = {}
    by_phrase: dict[tuple[str, ...], list[Word]] = defaultdict(list)
    words = sorted(words, key=lambda word: (word.start, word.end))
    for n in range(2, 6):
        for index in range(0, len(words) - n + 1):
            window = words[index : index + n]
            if len({word.scene_rank for word in window}) != 1:
                continue
            if len({word.speaker for word in window}) != 1:
                continue
            phrase = tuple(word.token for word in window)
            by_phrase[phrase].append(window[0])

    for phrase, starts in by_phrase.items():
        if not 2 <= len(starts) <= 5:
            continue
        if not is_content_phrase(phrase):
            continue
        unique_starts = sorted(starts, key=lambda word: word.start)
        if any(
            unique_starts[index + 1].start - unique_starts[index].start < min_separation_sec
            for index in range(len(unique_starts) - 1)
        ):
            continue
        candidates[phrase] = unique_starts
    return candidates


def scene_phrase_occurrences(
    scene_transcript: SceneTranscript,
    min_words: int,
    max_words: int,
) -> dict[tuple[str, ...], tuple[float, ...]]:
    by_phrase: dict[tuple[str, ...], list[float]] = defaultdict(list)
    words = sorted(scene_transcript.local_words, key=lambda word: (word.start, word.end))
    for n in range(min_words, max_words + 1):
        for index in range(0, len(words) - n + 1):
            window = words[index : index + n]
            if len({word.speaker for word in window}) != 1:
                continue
            phrase = tuple(word.token for word in window)
            if not is_content_phrase(phrase):
                continue
            by_phrase[phrase].append(window[0].start)
    return {phrase: tuple(starts) for phrase, starts in by_phrase.items()}


def repeated_phrase_candidate_count_from_scenes(
    scene_ids: list[str],
    offsets: list[float],
    scene_phrase_cache: dict[str, dict[str, dict[tuple[str, ...], tuple[float, ...]]]],
    phrase_band: str,
    min_separation_sec: float,
) -> int:
    phrase_maps = [scene_phrase_cache[scene_id][phrase_band] for scene_id in scene_ids]
    phrase_sets = [set(phrase_map) for phrase_map in phrase_maps]
    candidate_phrases: set[tuple[str, ...]] = set()
    for phrase_map in phrase_maps:
        candidate_phrases.update(
            phrase for phrase, starts in phrase_map.items() if len(starts) >= 2
        )
    for left_index in range(len(phrase_sets)):
        for right_index in range(left_index + 1, len(phrase_sets)):
            candidate_phrases.update(phrase_sets[left_index] & phrase_sets[right_index])

    count = 0
    for phrase in candidate_phrases:
        starts: list[float] = []
        for phrase_map, offset in zip(phrase_maps, offsets):
            starts.extend(start + offset for start in phrase_map.get(phrase, ()))
        if not 2 <= len(starts) <= 5:
            continue
        starts = sorted(starts)
        if any(starts[index + 1] - starts[index] < min_separation_sec for index in range(len(starts) - 1)):
            continue
        count += 1
    return count


def structural_relation_candidate_count(exact_candidates: list[Turn], all_turns: list[Turn], args: argparse.Namespace) -> int:
    count = 0
    sorted_turns = sorted(all_turns, key=lambda turn: (turn.start, turn.end))
    for anchor in exact_candidates:
        for candidate in sorted_turns:
            if candidate.start <= anchor.end:
                continue
            if candidate.speaker == anchor.speaker:
                continue
            if candidate.scene_rank != anchor.scene_rank:
                break
            if candidate.overlap_ratio > args.max_turn_overlap_ratio:
                continue
            if (
                candidate.word_count >= args.structural_target_min_words
                or candidate.duration >= args.structural_target_min_duration_sec
            ):
                count += 1
            break
    return count


def build_scene_offsets(reel: pd.Series, gap_seconds: float) -> tuple[list[str], list[float], list[tuple[float, float]]]:
    scene_ids = [str(reel[f"scene{index}_scene_id"]) for index in range(1, 5)]
    durations = [float(reel[f"scene{index}_duration_sec"]) for index in range(1, 5)]
    offsets: list[float] = []
    boundaries: list[tuple[float, float]] = []
    cursor = 0.0
    for duration in durations:
        offsets.append(cursor)
        boundaries.append((cursor, cursor + duration))
        cursor += duration + gap_seconds
    return scene_ids, offsets, boundaries


def audit_reel(
    reel: pd.Series,
    scene_cache: dict[str, SceneTranscript],
    scene_phrase_cache: dict[str, dict[str, dict[tuple[str, ...], tuple[float, ...]]]],
    absent_quote_texts_by_split: dict[str, set[str]],
    absent_quote_pool_sizes_by_split: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    scene_ids, offsets, boundaries = build_scene_offsets(reel, args.gap_seconds)
    turns: list[Turn] = []
    vad_rows: list[tuple[str, float, float]] = []
    timestamped_word_count = 0
    for rank, (scene_id, offset) in enumerate(zip(scene_ids, offsets), start=1):
        scene_turns, scene_words, scene_vad = globalize_scene(scene_cache[scene_id], rank, offset)
        turns.extend(scene_turns)
        vad_rows.extend(scene_vad)
        timestamped_word_count += len(scene_words)

    turns = sorted(turns, key=lambda turn: (turn.start, turn.end))
    exact_candidates = exact_quote_candidates(turns, boundaries, args)
    repeated_2_5_count = repeated_phrase_candidate_count_from_scenes(
        scene_ids, offsets, scene_phrase_cache, "2_5", args.repeated_min_separation_sec
    )
    repeated_5_8_count = repeated_phrase_candidate_count_from_scenes(
        scene_ids, offsets, scene_phrase_cache, "5_8", args.repeated_min_separation_sec
    )
    repeated_6_8_count = repeated_phrase_candidate_count_from_scenes(
        scene_ids, offsets, scene_phrase_cache, "6_8", args.repeated_min_separation_sec
    )
    reel_texts = {turn.normalized_text for turn in turns}
    split = str(reel["source_split"])
    split_absent_pool = absent_quote_texts_by_split[split]
    absent_source_count = absent_quote_pool_sizes_by_split[split] - sum(
        1 for text in reel_texts if text in split_absent_pool
    )
    structural_count = structural_relation_candidate_count(exact_candidates, turns, args)
    duration_pass = args.min_reel_duration_sec <= float(reel["reel_duration_sec"]) <= args.max_reel_duration_sec
    scenario_card_unique = not (
        str(reel["has_duplicate_prompt_hash"]).lower() == "true"
        or str(reel["has_duplicate_prompt_id"]).lower() == "true"
        or str(reel["has_duplicate_prompt_text"]).lower() == "true"
    )
    partial_count = partial_quote_candidate_count(exact_candidates)

    return {
        "audit_gap_seconds": args.gap_seconds,
        "audit_reel_duration_pass": duration_pass,
        "audit_scenario_card_unique": scenario_card_unique,
        "audit_turn_count": len(turns),
        "audit_timestamped_word_count": timestamped_word_count,
        "exact_quote_candidate_count": len(exact_candidates),
        "partial_quote_candidate_count": partial_count,
        "repeated_phrase_candidate_count": repeated_2_5_count,
        "repeated_phrase_2_5_candidate_count": repeated_2_5_count,
        "repeated_phrase_5_8_candidate_count": repeated_5_8_count,
        "repeated_phrase_6_8_candidate_count": repeated_6_8_count,
        "absent_quote_source_candidate_count": absent_source_count,
        "absent_quote_source_available": absent_source_count > 0,
        "structural_relation_candidate_count": structural_count,
        "has_exact_quote_candidate": len(exact_candidates) > 0,
        "has_partial_quote_candidate": partial_count > 0,
        "has_repeated_phrase_candidate": repeated_2_5_count > 0,
        "has_repeated_phrase_2_5_candidate": repeated_2_5_count > 0,
        "has_repeated_phrase_5_8_candidate": repeated_5_8_count > 0,
        "has_repeated_phrase_6_8_candidate": repeated_6_8_count > 0,
        "has_structural_relation_candidate": structural_count > 0,
        "reel_task_ready": (
            duration_pass
            and scenario_card_unique
            and len(exact_candidates) >= 2
            and partial_count >= 1
            and absent_source_count > 0
            and structural_count >= 1
        ),
        "reel_task_ready_with_repeated": (
            duration_pass
            and scenario_card_unique
            and len(exact_candidates) >= 2
            and repeated_2_5_count >= 1
            and absent_source_count > 0
            and structural_count >= 1
        ),
    }


def exact_absent_pool_by_split(
    scene_cache: dict[str, SceneTranscript],
    scene_df: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, set[str]]:
    pool: dict[str, set[str]] = defaultdict(set)
    placeholder_boundaries = [(0.0, 10_000.0)]
    for scene in scene_df.itertuples(index=False):
        scene_id = str(scene.scene_id)
        turns = [
            Turn(
                scene_id=turn.scene_id,
                scene_rank=1,
                speaker=turn.speaker,
                start=turn.start,
                end=turn.end,
                text=turn.text,
                tokens=turn.tokens,
                overlap_ratio=turn.overlap_ratio,
            )
            for turn in scene_cache[scene_id].local_turns
        ]
        for turn in exact_quote_candidates(turns, placeholder_boundaries, args):
            pool[str(scene.source_split)].add(turn.normalized_text)
    return pool


def main() -> None:
    args = parse_args()
    require_file(args.reel_input, "reel candidate CSV")
    require_file(args.scene_input, "filtered scene CSV")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {args.dataset_root}")

    reel_df = pd.read_csv(args.reel_input, keep_default_na=False)
    scene_df = pd.read_csv(args.scene_input, keep_default_na=False)
    if reel_df["candidate_reel_id"].duplicated().any():
        duplicates = sorted(
            reel_df.loc[reel_df["candidate_reel_id"].duplicated(), "candidate_reel_id"].unique()
        )
        raise ValueError(f"Duplicate candidate_reel_id values: {duplicates[:10]}")
    if scene_df["scene_id"].duplicated().any():
        duplicates = sorted(scene_df.loc[scene_df["scene_id"].duplicated(), "scene_id"].unique())
        raise ValueError(f"Duplicate scene_id values: {duplicates[:10]}")

    scenes_needed = {
        str(scene_id)
        for index in range(1, 5)
        for scene_id in reel_df[f"scene{index}_scene_id"].tolist()
    }
    scene_lookup = scene_df.set_index("scene_id", drop=False)
    missing_scenes = sorted(scene_id for scene_id in scenes_needed if scene_id not in scene_lookup.index)
    if missing_scenes:
        raise ValueError(f"Reel candidates reference missing scene IDs: {missing_scenes[:10]}")

    scene_cache = {
        scene_id: load_scene_transcript(scene_lookup.loc[scene_id], args.dataset_root)
        for scene_id in sorted(scenes_needed)
    }
    scene_phrase_cache = {
        scene_id: {
            "2_5": scene_phrase_occurrences(scene_transcript, 2, 5),
            "5_8": scene_phrase_occurrences(scene_transcript, 5, 8),
            "6_8": scene_phrase_occurrences(scene_transcript, 6, 8),
        }
        for scene_id, scene_transcript in scene_cache.items()
    }
    absent_pool = exact_absent_pool_by_split(scene_cache, scene_df[scene_df["scene_id"].isin(scenes_needed)], args)
    absent_pool_sizes = {split: len(pool) for split, pool in absent_pool.items()}
    audit_rows = [
        audit_reel(row, scene_cache, scene_phrase_cache, absent_pool, absent_pool_sizes, args)
        for _index, row in reel_df.iterrows()
    ]
    output = pd.concat([reel_df.reset_index(drop=True), pd.DataFrame(audit_rows)], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"wrote {args.output}")
    print(f"rows {len(output)}")
    print("task-ready by split")
    print(output.groupby("source_split")["reel_task_ready"].sum().sort_index().to_string())
    print("task-ready-with-repeated by split")
    print(output.groupby("source_split")["reel_task_ready_with_repeated"].sum().sort_index().to_string())


if __name__ == "__main__":
    main()
