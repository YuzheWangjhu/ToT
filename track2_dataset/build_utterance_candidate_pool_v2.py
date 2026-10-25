#!/usr/bin/env python3
"""Build Track 2 v2 utterance candidate pools.

The v2 design requires four real utterance candidates per scene, giving sixteen
real candidates per four-scene reel. The default thresholds are the accepted
middle ground for the current selected reels: preserve the original minimum
word/duration and overlap filters, allow utterances up to 30 words, and exclude
only the first/last two seconds of each scene.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

import audit_reel_task_availability as audit


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTED_REELS = SCRIPT_DIR / "track2_role_ipc_selected_reels_test_v1.csv"
DEFAULT_SELECTED_SCENES = SCRIPT_DIR / "track2_role_ipc_selected_reel_scenes_test_v1.csv"
DEFAULT_SCENE_METADATA = SCRIPT_DIR / "track2_role_ipc_scene_candidates_scene_filter_v1.csv"
DEFAULT_REAL_OUTPUT = SCRIPT_DIR / "track2_role_ipc_utterance_candidate_pool_v2.csv"
DEFAULT_ABSENT_OUTPUT = SCRIPT_DIR / "track2_role_ipc_absent_utterance_candidate_pool_v2.csv"
DEFAULT_AVAILABILITY_OUTPUT = SCRIPT_DIR / "track2_role_ipc_utterance_candidate_pool_v2_availability.csv"
DEFAULT_GAP_SECONDS = 0.5
DEFAULT_EXACT_MIN_WORDS = audit.DEFAULT_EXACT_MIN_WORDS
DEFAULT_EXACT_MAX_WORDS = 30
DEFAULT_EXACT_MIN_DURATION_SEC = audit.DEFAULT_EXACT_MIN_DURATION_SEC
DEFAULT_EXACT_MAX_DURATION_SEC = audit.DEFAULT_EXACT_MAX_DURATION_SEC
DEFAULT_BOUNDARY_MARGIN_SEC = 2.0
DEFAULT_MAX_TURN_OVERLAP_RATIO = audit.DEFAULT_MAX_TURN_OVERLAP_RATIO
REAL_CANDIDATES_PER_SCENE = 4
REAL_CANDIDATES_PER_REEL = 16
REAL_TARGETS_PER_QUESTION_SET = 7
QUESTION_SETS_PER_REEL = 2
RESERVE_CANDIDATES_PER_REEL = 2


@dataclass(frozen=True)
class Candidate:
    reel_id: str
    scene_rank: int
    scene_id: str
    quality_rank_in_scene: int
    turn: audit.Turn
    partial_text: str
    masked_token_indices: tuple[int, ...]

    @property
    def partial_maskable(self) -> bool:
        return bool(self.partial_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Track 2 v2 utterance candidate pools.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Separately acquired Seamless Interaction datasets directory.")
    parser.add_argument("--selected-reels", type=Path, default=DEFAULT_SELECTED_REELS)
    parser.add_argument("--selected-scenes", type=Path, default=DEFAULT_SELECTED_SCENES)
    parser.add_argument("--scene-metadata", type=Path, default=DEFAULT_SCENE_METADATA)
    parser.add_argument("--real-output", type=Path, default=DEFAULT_REAL_OUTPUT)
    parser.add_argument("--absent-output", type=Path, default=DEFAULT_ABSENT_OUTPUT)
    parser.add_argument("--availability-output", type=Path, default=DEFAULT_AVAILABILITY_OUTPUT)
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    parser.add_argument("--exact-min-words", type=int, default=DEFAULT_EXACT_MIN_WORDS)
    parser.add_argument("--exact-max-words", type=int, default=DEFAULT_EXACT_MAX_WORDS)
    parser.add_argument("--exact-min-duration-sec", type=float, default=DEFAULT_EXACT_MIN_DURATION_SEC)
    parser.add_argument("--exact-max-duration-sec", type=float, default=DEFAULT_EXACT_MAX_DURATION_SEC)
    parser.add_argument("--boundary-margin-sec", type=float, default=DEFAULT_BOUNDARY_MARGIN_SEC)
    parser.add_argument("--max-turn-overlap-ratio", type=float, default=DEFAULT_MAX_TURN_OVERLAP_RATIO)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def audit_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        exact_min_words=args.exact_min_words,
        exact_max_words=args.exact_max_words,
        exact_min_duration_sec=args.exact_min_duration_sec,
        exact_max_duration_sec=args.exact_max_duration_sec,
        boundary_margin_sec=args.boundary_margin_sec,
        max_turn_overlap_ratio=args.max_turn_overlap_ratio,
    )


def speaker_label(participant: str) -> str:
    if participant == "participant1":
        return "A"
    if participant == "participant2":
        return "B"
    raise ValueError(f"Unexpected speaker: {participant}")


def partial_prompt(turn: audit.Turn) -> tuple[str, tuple[int, ...]]:
    tokens = list(turn.tokens)
    middle_indices = [
        index
        for index, token in enumerate(tokens)
        if 2 <= index < len(tokens) - 2
        and token not in audit.STOP_OR_FILLER
        and any(ch.isalpha() for ch in token)
    ]
    if len(tokens) < 10 or len(middle_indices) < 2:
        return "", ()
    mask_count = max(2, round(len(middle_indices) * 0.33))
    mask_indices = tuple(middle_indices[:mask_count])
    masked = ["[MASK]" if index in mask_indices else token for index, token in enumerate(tokens)]
    return " ".join(masked), mask_indices


def candidate_sort_key(turn: audit.Turn) -> tuple[Any, ...]:
    return (-turn.word_count, -turn.duration, turn.start, turn.normalized_text)


def reel_scene_rows(selected_scenes: pd.DataFrame, reel_id: str) -> pd.DataFrame:
    rows = selected_scenes[selected_scenes["reel_id"] == reel_id].copy()
    rows["scene_rank_int"] = rows["scene_rank"].astype(int)
    rows = rows.sort_values("scene_rank_int")
    if len(rows) != 4:
        raise ValueError(f"{reel_id}: expected 4 selected scenes, got {len(rows)}")
    if rows["scene_rank_int"].tolist() != [1, 2, 3, 4]:
        raise ValueError(f"{reel_id}: scene ranks are not 1-4")
    return rows


def reel_turns_and_boundaries(
    reel_id: str,
    selected_scenes: pd.DataFrame,
    scene_cache: dict[str, audit.SceneTranscript],
) -> tuple[list[audit.Turn], list[tuple[float, float]]]:
    turns: list[audit.Turn] = []
    boundaries: list[tuple[float, float]] = []
    for scene in reel_scene_rows(selected_scenes, reel_id).itertuples(index=False):
        scene_id = str(scene.scene_id)
        scene_rank = int(scene.scene_rank)
        scene_start = float(scene.scene_start_seconds)
        scene_end = float(scene.scene_end_seconds)
        scene_turns, _scene_words, _scene_vad = audit.globalize_scene(scene_cache[scene_id], scene_rank, scene_start)
        turns.extend(scene_turns)
        boundaries.append((scene_start, scene_end))
    return sorted(turns, key=lambda turn: (turn.start, turn.end, turn.speaker)), boundaries


def exact_candidates_by_scene(
    reel_id: str,
    selected_scenes: pd.DataFrame,
    scene_cache: dict[str, audit.SceneTranscript],
    args: SimpleNamespace,
) -> dict[str, list[audit.Turn]]:
    turns, boundaries = reel_turns_and_boundaries(reel_id, selected_scenes, scene_cache)
    exact_candidates = audit.exact_quote_candidates(turns, boundaries, args)
    by_scene: dict[str, list[audit.Turn]] = {str(row.scene_id): [] for row in reel_scene_rows(selected_scenes, reel_id).itertuples(index=False)}
    for turn in exact_candidates:
        by_scene[turn.scene_id].append(turn)
    return {scene_id: sorted(turns_, key=candidate_sort_key) for scene_id, turns_ in by_scene.items()}


def build_real_candidates_for_reel(
    reel_id: str,
    selected_scenes: pd.DataFrame,
    scene_cache: dict[str, audit.SceneTranscript],
    args: SimpleNamespace,
) -> tuple[list[Candidate], list[dict[str, Any]]]:
    by_scene = exact_candidates_by_scene(reel_id, selected_scenes, scene_cache, args)
    availability_rows: list[dict[str, Any]] = []
    selected_candidates: list[Candidate] = []
    for scene in reel_scene_rows(selected_scenes, reel_id).itertuples(index=False):
        scene_id = str(scene.scene_id)
        scene_candidates = by_scene[scene_id]
        partial_count = sum(1 for turn in scene_candidates if partial_prompt(turn)[0])
        availability_rows.append(
            {
                "reel_id": reel_id,
                "scene_rank": int(scene.scene_rank),
                "scene_id": scene_id,
                "v2_exact_candidate_count": len(scene_candidates),
                "v2_partial_maskable_candidate_count": partial_count,
                "v2_has_four_exact_candidates": len(scene_candidates) >= REAL_CANDIDATES_PER_SCENE,
            }
        )
        for rank, turn in enumerate(scene_candidates[:REAL_CANDIDATES_PER_SCENE], start=1):
            partial_text, masked_indices = partial_prompt(turn)
            selected_candidates.append(
                Candidate(
                    reel_id=reel_id,
                    scene_rank=int(scene.scene_rank),
                    scene_id=scene_id,
                    quality_rank_in_scene=rank,
                    turn=turn,
                    partial_text=partial_text,
                    masked_token_indices=masked_indices,
                )
            )
    return selected_candidates, availability_rows


def assignment_sort_key(candidate: Candidate) -> tuple[Any, ...]:
    return (candidate.quality_rank_in_scene, -candidate.turn.word_count, candidate.turn.start, candidate.turn.normalized_text)


def assign_question_sets(candidates: list[Candidate]) -> dict[Candidate, tuple[str, str, int]]:
    if len(candidates) != REAL_CANDIDATES_PER_REEL:
        raise ValueError(f"Expected {REAL_CANDIDATES_PER_REEL} candidates, got {len(candidates)}")

    roles = [
        (1, "partial_utterance", 1),
        (1, "partial_utterance", 2),
        (2, "partial_utterance", 1),
        (2, "partial_utterance", 2),
        (1, "structural_real", 1),
        (1, "structural_real", 2),
        (1, "structural_real", 3),
        (2, "structural_real", 1),
        (2, "structural_real", 2),
        (2, "structural_real", 3),
        (1, "exact_utterance", 1),
        (1, "exact_utterance", 2),
        (2, "exact_utterance", 1),
        (2, "exact_utterance", 2),
    ]
    used: set[Candidate] = set()
    assignment: dict[Candidate, tuple[str, str, int]] = {}
    structural_scenes: dict[int, set[int]] = {1: set(), 2: set()}
    set_scene_counts: dict[int, dict[int, int]] = {1: {}, 2: {}}

    for question_set, role, role_slot in roles:
        available = [candidate for candidate in candidates if candidate not in used]
        if role == "partial_utterance":
            available = [candidate for candidate in available if candidate.partial_maskable]
        if role == "structural_real":
            available = [candidate for candidate in available if candidate.scene_rank not in structural_scenes[question_set]]
        if role in {"structural_real", "exact_utterance"}:
            available = sorted(
                available,
                key=lambda candidate: (
                    candidate.partial_maskable,
                    set_scene_counts[question_set].get(candidate.scene_rank, 0),
                    assignment_sort_key(candidate),
                ),
            )
        else:
            available = sorted(
                available,
                key=lambda candidate: (
                    set_scene_counts[question_set].get(candidate.scene_rank, 0),
                    assignment_sort_key(candidate),
                ),
            )
        if not available:
            raise ValueError(f"Could not assign role {role} {role_slot} for question set {question_set}")
        candidate = available[0]
        used.add(candidate)
        assignment[candidate] = (str(question_set), role, role_slot)
        if role == "structural_real":
            structural_scenes[question_set].add(candidate.scene_rank)
        set_scene_counts[question_set][candidate.scene_rank] = set_scene_counts[question_set].get(candidate.scene_rank, 0) + 1

    reserves = sorted([candidate for candidate in candidates if candidate not in used], key=assignment_sort_key)
    if len(reserves) != RESERVE_CANDIDATES_PER_REEL:
        raise ValueError(f"Expected {RESERVE_CANDIDATES_PER_REEL} reserve candidates, got {len(reserves)}")
    for index, candidate in enumerate(reserves, start=1):
        assignment[candidate] = ("reserve", "reserve", index)
    return assignment


def candidate_to_row(candidate: Candidate, assignment: tuple[str, str, int]) -> dict[str, Any]:
    question_set, role, role_index = assignment
    candidate_id = f"{candidate.reel_id}_S{candidate.scene_rank}_U{candidate.quality_rank_in_scene:02d}"
    return {
        "reel_id": candidate.reel_id,
        "scene_rank": candidate.scene_rank,
        "scene_id": candidate.scene_id,
        "utterance_candidate_id": candidate_id,
        "quality_rank_in_scene": candidate.quality_rank_in_scene,
        "assigned_question_set": question_set,
        "assigned_role": role,
        "assigned_role_index": role_index,
        "speaker": speaker_label(candidate.turn.speaker),
        "start_seconds": round(candidate.turn.start, 3),
        "end_seconds": round(candidate.turn.end, 3),
        "duration_seconds": round(candidate.turn.duration, 3),
        "word_count": candidate.turn.word_count,
        "text": candidate.turn.text.strip(),
        "normalized_text": candidate.turn.normalized_text,
        "partial_maskable": candidate.partial_maskable,
        "partial_text": candidate.partial_text,
        "masked_token_indices": json.dumps(list(candidate.masked_token_indices)),
        "overlap_ratio": round(candidate.turn.overlap_ratio, 6),
    }


def local_exact_source_candidates(
    selected_scenes: pd.DataFrame,
    scene_cache: dict[str, audit.SceneTranscript],
    args: SimpleNamespace,
) -> list[dict[str, Any]]:
    rows = []
    for scene in selected_scenes.itertuples(index=False):
        scene_id = str(scene.scene_id)
        turns = [
            audit.Turn(
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
        boundaries = [(0.0, float(scene.source_scene_duration_sec))]
        for turn in audit.exact_quote_candidates(turns, boundaries, args):
            rows.append(
                {
                    "source_reel_id": str(scene.reel_id),
                    "source_scene_rank": int(scene.scene_rank),
                    "source_scene_id": scene_id,
                    "source_session_id": str(scene.session_id),
                    "source_dyad_key": str(scene.dyad_key),
                    "speaker": speaker_label(turn.speaker),
                    "local_start_seconds": round(turn.start, 3),
                    "local_end_seconds": round(turn.end, 3),
                    "duration_seconds": round(turn.duration, 3),
                    "word_count": turn.word_count,
                    "text": turn.text.strip(),
                    "normalized_text": turn.normalized_text,
                }
            )
    return sorted(rows, key=lambda row: (-row["word_count"], -row["duration_seconds"], row["source_reel_id"], row["source_scene_rank"], row["local_start_seconds"]))


def fake_priority(target_reel: pd.Series, source: dict[str, Any]) -> tuple[int, Any, ...]:
    same_session = str(source["source_session_id"]) == str(target_reel["session_id"])
    same_dyad = str(source["source_dyad_key"]) == str(target_reel["dyad_key"])
    same_split = str(target_reel["source_split"]) == "test"
    if same_session:
        priority = 0
    elif same_dyad:
        priority = 1
    elif same_split:
        priority = 2
    else:
        priority = 3
    return (priority, -int(source["word_count"]), source["source_reel_id"], source["source_scene_rank"], source["local_start_seconds"])


def build_absent_rows(
    selected_reels: pd.DataFrame,
    selected_scenes: pd.DataFrame,
    scene_cache: dict[str, audit.SceneTranscript],
    real_reel_turn_texts: dict[str, set[str]],
    args: SimpleNamespace,
) -> list[dict[str, Any]]:
    source_candidates = local_exact_source_candidates(selected_scenes, scene_cache, args)
    rows = []
    for reel in selected_reels.sort_values("reel_id").itertuples(index=False):
        reel_series = pd.Series(reel._asdict())
        reel_id = str(reel.reel_id)
        used_normalized: set[str] = set()
        eligible = [
            source
            for source in source_candidates
            if source["source_reel_id"] != reel_id
            and source["normalized_text"] not in real_reel_turn_texts[reel_id]
        ]
        eligible = sorted(eligible, key=lambda source: fake_priority(reel_series, source))
        for question_set in (1, 2):
            chosen = next((source for source in eligible if source["normalized_text"] not in used_normalized), None)
            if chosen is None:
                raise ValueError(f"{reel_id}: no absent utterance candidate for question set {question_set}")
            used_normalized.add(chosen["normalized_text"])
            relation = "same_session_outside_reel"
            if chosen["source_session_id"] != str(reel.session_id):
                relation = "same_dyad_outside_reel" if chosen["source_dyad_key"] == str(reel.dyad_key) else "other_selected_reel"
            rows.append(
                {
                    "reel_id": reel_id,
                    "assigned_question_set": question_set,
                    "absent_candidate_id": f"{reel_id}_SET{question_set}_FAKE",
                    "source_relation": relation,
                    **chosen,
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    require_file(args.selected_reels, "selected reels CSV")
    require_file(args.selected_scenes, "selected scenes CSV")
    require_file(args.scene_metadata, "scene metadata CSV")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {args.dataset_root}")

    selected_reels = pd.read_csv(args.selected_reels, keep_default_na=False)
    selected_scenes = pd.read_csv(args.selected_scenes, keep_default_na=False)
    scene_metadata = pd.read_csv(args.scene_metadata, keep_default_na=False)
    if selected_scenes["scene_id"].duplicated().any():
        duplicates = sorted(selected_scenes.loc[selected_scenes["scene_id"].duplicated(), "scene_id"].unique())
        raise ValueError(f"Duplicate selected scene_id values: {duplicates[:10]}")
    scene_lookup = scene_metadata.set_index("scene_id", drop=False)
    selected_scene_ids = sorted(selected_scenes["scene_id"].unique())
    missing = sorted(scene_id for scene_id in selected_scene_ids if scene_id not in scene_lookup.index)
    if missing:
        raise ValueError(f"Selected scenes missing from scene metadata: {missing[:10]}")

    scene_cache = {
        scene_id: audit.load_scene_transcript(scene_lookup.loc[scene_id], args.dataset_root)
        for scene_id in selected_scene_ids
    }
    args_exact = audit_args(args)

    all_candidates: list[Candidate] = []
    availability_rows: list[dict[str, Any]] = []
    real_reel_turn_texts: dict[str, set[str]] = {}
    for reel_id in sorted(selected_reels["reel_id"].astype(str)):
        turns, _boundaries = reel_turns_and_boundaries(reel_id, selected_scenes, scene_cache)
        real_reel_turn_texts[reel_id] = {turn.normalized_text for turn in turns}
        reel_candidates, reel_availability = build_real_candidates_for_reel(reel_id, selected_scenes, scene_cache, args_exact)
        all_candidates.extend(reel_candidates)
        availability_rows.extend(reel_availability)

    availability = pd.DataFrame(availability_rows)
    args.availability_output.parent.mkdir(parents=True, exist_ok=True)
    availability.to_csv(args.availability_output, index=False)

    bad_rows = availability[~availability["v2_has_four_exact_candidates"]]
    if not bad_rows.empty:
        details = bad_rows[["reel_id", "scene_rank", "scene_id", "v2_exact_candidate_count"]].to_dict("records")
        raise ValueError(
            "Track 2 v2 candidate pool is not viable: some scenes have fewer than "
            f"{REAL_CANDIDATES_PER_SCENE} exact-eligible utterances. "
            f"Wrote diagnostics to {args.availability_output}. Blocking scenes: {details}"
        )

    if len(all_candidates) != len(selected_reels) * REAL_CANDIDATES_PER_REEL:
        raise ValueError(f"Expected {len(selected_reels) * REAL_CANDIDATES_PER_REEL} real candidates, got {len(all_candidates)}")

    real_rows = []
    for reel_id in sorted(selected_reels["reel_id"].astype(str)):
        reel_candidates = [candidate for candidate in all_candidates if candidate.reel_id == reel_id]
        assignment = assign_question_sets(reel_candidates)
        real_rows.extend(candidate_to_row(candidate, assignment[candidate]) for candidate in reel_candidates)

    real_output = pd.DataFrame(real_rows).sort_values(
        ["reel_id", "scene_rank", "quality_rank_in_scene"]
    )
    partial_counts = real_output[real_output["assigned_role"] == "partial_utterance"].groupby("reel_id").size()
    if not (partial_counts == 4).all():
        raise ValueError(f"Expected exactly four assigned partial utterances per reel, got {partial_counts.to_dict()}")
    args.real_output.parent.mkdir(parents=True, exist_ok=True)
    real_output.to_csv(args.real_output, index=False)

    absent_rows = build_absent_rows(selected_reels, selected_scenes, scene_cache, real_reel_turn_texts, args_exact)
    absent_output = pd.DataFrame(absent_rows).sort_values(["reel_id", "assigned_question_set"])
    absent_output.to_csv(args.absent_output, index=False)

    print(f"wrote {args.availability_output}")
    print(f"wrote {args.real_output}")
    print(f"wrote {args.absent_output}")
    print(f"real_candidates {len(real_output)}")
    print(f"absent_candidates {len(absent_output)}")


if __name__ == "__main__":
    main()
