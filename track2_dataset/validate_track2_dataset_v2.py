#!/usr/bin/env python3
"""Validate the audited Track 2 v2 dataset release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import audit_reel_task_availability as audit


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATASET_DIR = SCRIPT_DIR / "generated_data" / "track2_role_ipc_test_v2" / "dataset"
BUILDER = SCRIPT_DIR / "build_track2_dataset_v2.py"
V1_DIR = SCRIPT_DIR / "generated_data" / "track2_role_ipc_test_v1" / "dataset"
EXPECTED_COUNTS = {"2A.1": 110, "2A.2": 176, "2A.3": 44, "2B.1": 44, "2B.2": 44}
EXPECTED_PER_REEL = {"2A.1": 5, "2A.2": 8, "2A.3": 2, "2B.1": 2, "2B.2": 2}
EXPECTED_T2R_0009_CARD = (
    "Two people are discussing their experiences with recovery, sharing personal stories about "
    "challenges they have faced and what gives them hope. They compare perspectives on resilience "
    "and offer encouragement and opinions while trying to connect over similar struggles."
)
EXPECTED_EXCLUSIONS = 18
OVERLAP_EXCLUSIONS = {
    "T2R_0001_S4_U02", "T2R_0001_S4_U03", "T2R_0002_S3_U04",
    "T2R_0007_S4_U01", "T2R_0007_S4_U02", "T2R_0008_S2_U01",
    "T2R_0008_S4_U02", "T2R_0009_S3_U02", "T2R_0010_S2_U04",
    "T2R_0010_S4_U01", "T2R_0012_S4_U04", "T2R_0014_S2_U04",
    "T2R_0017_S3_U01", "T2R_0018_S4_U02",
}
AMBIGUOUS_EXCLUSIONS = {
    "T2R_0012_S2_U01", "T2R_0013_S3_U03",
    "T2R_0016_S4_U02", "T2R_0020_S4_U04",
}
EXPECTED_FINAL_REASSIGNMENTS = {
    "T2R_0001_S1_U04": [2, "exact_utterance", 1],
    "T2R_0001_S3_U04": [2, "structural_real", 2],
    "T2R_0002_S1_U04": [2, "exact_utterance", 1],
    "T2R_0007_S3_U04": [2, "partial_utterance", 1],
    "T2R_0007_S4_U04": [1, "structural_real", 2],
    "T2R_0008_S1_U04": [1, "partial_utterance", 1],
    "T2R_0008_S3_U04": [1, "partial_utterance", 2],
    "T2R_0008_S4_U01": [1, "structural_real", 3],
    "T2R_0009_S3_U04": [1, "structural_real", 1],
    "T2R_0010_S1_U04": [2, "exact_utterance", 1],
    "T2R_0010_S4_U04": [2, "partial_utterance", 2],
    "T2R_0012_S1_U04": [1, "partial_utterance", 2],
    "T2R_0013_S3_U04": [1, "exact_utterance", 2],
    "T2R_0016_S2_U04": [2, "structural_real", 1],
    "T2R_0017_S1_U04": [2, "partial_utterance", 2],
    "T2R_0018_S4_U04": [1, "structural_real", 2],
    "T2R_0020_S1_U04": [2, "exact_utterance", 2],
}
EXPECTED_SOURCE_PATHS = {
    "v1_manifest": "track2_dataset/generated_data/track2_role_ipc_test_v1/dataset/manifest.json",
    "scene_cards": "track2_dataset/track2_role_ipc_scene_cards_v3.csv",
    "candidates": "track2_dataset/track2_role_ipc_utterance_candidate_pool_v2.csv",
    "absent_utterances": "track2_dataset/track2_role_ipc_absent_utterance_candidate_pool_v2.csv",
    "selected_scenes": "track2_dataset/track2_role_ipc_selected_reel_scenes_test_v1.csv",
    "scene_metadata": "track2_dataset/track2_role_ipc_scene_candidates_scene_filter_v1.csv",
    "human_audit": "track2_human_audit/audit_results/track2_human_audit_progress_2026-07-20T04-52-43-406Z.json",
    "absent_card_reviews": "track2_human_audit/audit_results/track2_absent_scene_card_reviews.csv",
    "scene_card_revision": "track2_human_audit/audit_results/track2_scene_card_revision_review.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Separately acquired Seamless Interaction datasets directory.")
    parser.add_argument("--v1-audio-dir", type=Path, default=V1_DIR / "episodes")
    parser.add_argument("--write-report", action="store_true")
    parser.add_argument("--rebuild-check", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def original_assignment(row: dict[str, str]) -> list[Any] | None:
    if row["assigned_role"] == "reserve":
        return None
    return [int(row["assigned_question_set"]), row["assigned_role"], int(row["assigned_role_index"])]


def expected_partial(text: str) -> tuple[str, list[int]]:
    tokens = audit.tokens_from_text(text)
    middle = [
        index for index, token in enumerate(tokens)
        if 2 <= index < len(tokens) - 2
        and token not in audit.STOP_OR_FILLER
        and any(character.isalpha() for character in token)
    ]
    if len(tokens) < 10 or len(middle) < 2:
        return "", []
    mask_count = max(2, round(len(middle) * 0.33))
    indices = middle[:mask_count]
    return " ".join("[MASK]" if index in indices else token for index, token in enumerate(tokens)), indices


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contains_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, dict):
        return forbidden in value or any(contains_key(child, forbidden) for child in value.values())
    if isinstance(value, list):
        return any(contains_key(child, forbidden) for child in value)
    return False


def require_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys - set(value)
    if missing:
        raise ValueError(f"{context}: missing fields {sorted(missing)}")


def validate_span(start: Any, end: Any, episode: dict[str, Any], context: str) -> None:
    start_value, end_value = float(start), float(end)
    if not 0 <= start_value < end_value <= float(episode["duration_seconds"]):
        raise ValueError(f"{context}: invalid span {start_value}-{end_value}")
    if not any(float(scene["start_seconds"]) <= start_value < end_value <= float(scene["end_seconds"]) for scene in episode["scenes"]):
        raise ValueError(f"{context}: span crosses or falls outside scene boundaries")


def artifact_hashes(dataset_dir: Path, include_audio: bool = True) -> dict[str, str]:
    names = ["manifest.json", "task_instances.jsonl", "task_answers.jsonl", "candidate_provenance.jsonl"]
    paths = [dataset_dir / name for name in names]
    if include_audio:
        paths.extend(sorted((dataset_dir / "episodes").glob("*.wav")))
    return {str(path.relative_to(dataset_dir)): sha256_file(path) for path in paths}


def validate_dataset(dataset_dir: Path, dataset_root: Path, v1_audio_dir: Path = V1_DIR / "episodes") -> dict[str, Any]:
    manifest = load_json(dataset_dir / "manifest.json")
    tasks = load_jsonl(dataset_dir / "task_instances.jsonl")
    answer_rows = load_jsonl(dataset_dir / "task_answers.jsonl")
    provenance = load_jsonl(dataset_dir / "candidate_provenance.jsonl")
    checks: list[str] = []

    if manifest.get("dataset_id") != "track2_role_ipc_test_v2" or manifest.get("track_id") != "2":
        raise ValueError("Invalid dataset identity")
    if manifest.get("episode_count") != 22 or len(manifest.get("episodes", [])) != 22:
        raise ValueError("Expected 22 episodes")
    if (manifest.get("global_longest_scene_duration"), manifest.get("boundary_margin"), manifest.get("max_single_query_seconds")) != (354.0, 30.0, 384.0):
        raise ValueError("Query-length constants do not match the frozen design")
    checks.append("manifest_identity_and_constants")

    source_artifacts = manifest.get("source_artifacts", {})
    if set(source_artifacts) != set(EXPECTED_SOURCE_PATHS):
        raise ValueError("Manifest source-artifact names do not match the frozen input set")
    for name, expected_relative_path in EXPECTED_SOURCE_PATHS.items():
        record = source_artifacts[name]
        if record.get("path") != expected_relative_path:
            raise ValueError(f"{name}: unexpected source path {record.get('path')}")
        source_path = (PROJECT_ROOT / expected_relative_path).resolve()
        if not source_path.is_relative_to(PROJECT_ROOT.resolve()):
            raise ValueError(f"{name}: source path escapes the project root")
        if sha256_file(source_path) != record.get("sha256"):
            raise ValueError(f"{name}: source artifact SHA-256 mismatch")
    checks.append("recorded_source_artifacts_match_current_inputs")

    episodes = {episode["reel_id"]: episode for episode in manifest["episodes"]}
    if len(episodes) != 22 or any(len(episode["scenes"]) != 4 for episode in episodes.values()):
        raise ValueError("Expected 22 unique reels with four scenes each")
    v1_manifest = load_json(V1_DIR / "manifest.json")
    v1_episodes = {episode["reel_id"]: episode for episode in v1_manifest["episodes"]}
    if set(v1_episodes) != set(episodes):
        raise ValueError("v1 and v2 reel IDs differ")
    for reel_id, episode in episodes.items():
        audio_path = dataset_dir / episode["audio_path"]
        v1_episode = v1_episodes[reel_id]
        v1_audio_path = v1_audio_dir / Path(v1_episode["audio_path"]).name
        release_hash = sha256_file(audio_path) if audio_path.is_file() else None
        v1_hash = sha256_file(v1_audio_path) if v1_audio_path.is_file() else None
        if release_hash != episode["sha256_audio"] or release_hash != v1_episode["sha256_audio"] or release_hash != v1_hash:
            raise ValueError(f"{reel_id}: missing audio or SHA-256 mismatch")
    checks.append("audio_matches_release_manifest_and_v1")

    if len(tasks) != 418 or len(answer_rows) != 418:
        raise ValueError("Expected 418 public tasks and 418 answers")
    task_ids = [task.get("task_id") for task in tasks]
    if None in task_ids or len(task_ids) != len(set(task_ids)):
        raise ValueError("Task IDs are missing or duplicated")
    answers = {row.get("task_id"): row.get("answer") for row in answer_rows}
    if set(answers) != set(task_ids):
        raise ValueError("Public task IDs and answer task IDs differ")
    family_counts = Counter(task["family"] for task in tasks)
    per_reel = Counter((task["reel_id"], task["family"]) for task in tasks)
    if family_counts != Counter(EXPECTED_COUNTS):
        raise ValueError(f"Unexpected family counts: {family_counts}")
    for reel_id in episodes:
        for family, expected in EXPECTED_PER_REEL.items():
            if per_reel[(reel_id, family)] != expected:
                raise ValueError(f"{reel_id}: expected {expected} tasks for {family}")
    if any(contains_key(task, "speaker") for task in tasks) or any(contains_key(row, "speaker") for row in answer_rows):
        raise ValueError("Speaker field leaked into public tasks or scored answers")
    checks.append("task_counts_ids_and_no_speaker")

    present_provenance = [row for row in provenance if row.get("candidate_kind") == "present_utterance"]
    absent_provenance = [row for row in provenance if row.get("candidate_kind") == "absent_utterance"]
    if len(present_provenance) != 352 or len(absent_provenance) != 44:
        raise ValueError("Unexpected candidate provenance counts")

    raw_candidates = load_csv(PROJECT_ROOT / EXPECTED_SOURCE_PATHS["candidates"])
    raw_by_id = {row["utterance_candidate_id"]: row for row in raw_candidates}
    human_audit = load_json(PROJECT_ROOT / EXPECTED_SOURCE_PATHS["human_audit"])["state"]
    candidate_reviews = {
        candidate_id: review
        for scene in human_audit["scenes"].values()
        for candidate_id, review in scene["candidate_reviews"].items()
    }
    provenance_by_id = {row["utterance_candidate_id"]: row for row in present_provenance}
    if set(provenance_by_id) != set(raw_by_id) or set(candidate_reviews) != set(raw_by_id):
        raise ValueError("Candidate pool, audit, and provenance IDs differ")
    exact_exclusions = OVERLAP_EXCLUSIONS | AMBIGUOUS_EXCLUSIONS
    for candidate_id, raw in raw_by_id.items():
        record = provenance_by_id[candidate_id]
        review = candidate_reviews[candidate_id]
        expected_text = review.get("corrected_transcript_text", "").strip() or raw["text"]
        correction = review.get("corrections", {}).get("utterance", {})
        expected_start = float(correction.get("corrected_start_seconds", raw["start_seconds"]))
        expected_end = float(correction.get("corrected_end_seconds", raw["end_seconds"]))
        expected_normalized = " ".join(audit.tokens_from_text(expected_text))
        partial_text, masked_indices = expected_partial(expected_text)
        expected_original = original_assignment(raw)
        if candidate_id in exact_exclusions:
            expected_final = None
            expected_reason = "overlapping_speech" if candidate_id in OVERLAP_EXCLUSIONS else "transcript_ambiguity"
        else:
            expected_final = EXPECTED_FINAL_REASSIGNMENTS.get(candidate_id, expected_original)
            expected_reason = None
        expected_values = {
            "text": expected_text,
            "normalized_text": expected_normalized,
            "word_count": len(audit.tokens_from_text(expected_text)),
            "partial_text": partial_text,
            "masked_token_indices": masked_indices,
            "partial_maskable": bool(partial_text),
            "audit_candidate_status": review["candidate_status"],
            "audit_timestamp_status": review["timestamp_status"],
            "audit_corrected_text": bool(review.get("corrected_transcript_text", "").strip()),
            "audit_corrected_interval": bool(correction),
            "excluded": candidate_id in exact_exclusions,
            "exclusion_reason": expected_reason,
            "original_assignment": expected_original,
            "final_assignment": expected_final,
            "promoted_or_reassigned": expected_final is not None and expected_final != expected_original,
        }
        for field, expected in expected_values.items():
            if record.get(field) != expected:
                raise ValueError(f"{candidate_id}: {field} differs from audited source; expected {expected!r}, got {record.get(field)!r}")
        if float(record["start_seconds"]) != expected_start or float(record["end_seconds"]) != expected_end:
            raise ValueError(f"{candidate_id}: corrected interval differs from human audit")
        if float(record["duration_seconds"]) != round(expected_end - expected_start, 3):
            raise ValueError(f"{candidate_id}: corrected duration is inconsistent")

    raw_absent = {row["absent_candidate_id"]: row for row in load_csv(PROJECT_ROOT / EXPECTED_SOURCE_PATHS["absent_utterances"])}
    absent_by_id = {row["absent_candidate_id"]: row for row in absent_provenance}
    absent_reviews = human_audit["absent_candidate_reviews"]
    if set(raw_absent) != set(absent_by_id) or set(absent_reviews) != set(raw_absent):
        raise ValueError("Absent candidate pool, audit, and provenance IDs differ")
    for candidate_id, raw in raw_absent.items():
        record, review = absent_by_id[candidate_id], absent_reviews[candidate_id]
        if review["candidate_status"] != "ok":
            raise ValueError(f"{candidate_id}: absent candidate was not accepted")
        expected_text = review.get("corrected_transcript_text", "").strip() or raw["text"]
        expected_normalized = " ".join(audit.tokens_from_text(expected_text))
        if record["text"] != expected_text or record["normalized_text"] != expected_normalized:
            raise ValueError(f"{candidate_id}: absent transcript correction mismatch")
        if int(record["word_count"]) != len(audit.tokens_from_text(expected_text)) or record["audit_corrected_text"] != bool(review.get("corrected_transcript_text", "").strip()):
            raise ValueError(f"{candidate_id}: absent correction metadata mismatch")
    checks.append("exact_audit_corrections_exclusions_and_assignments")

    excluded = [row for row in present_provenance if row["excluded"]]
    if len(excluded) != EXPECTED_EXCLUSIONS or any(row["final_assignment"] is not None for row in excluded):
        raise ValueError("Excluded candidate count or final assignment is invalid")
    active = [row for row in present_provenance if row["final_assignment"] is not None]
    if len(active) != 308:
        raise ValueError(f"Expected 308 active candidates, got {len(active)}")
    active_by_reel = defaultdict(list)
    for row in active:
        active_by_reel[row["reel_id"]].append(row)
        normalized = " ".join(audit.tokens_from_text(row["text"]))
        if normalized != row["normalized_text"] or len(normalized.split()) != int(row["word_count"]):
            raise ValueError(f"{row['utterance_candidate_id']}: normalized text or word count mismatch")
        if row["final_assignment"][1] == "partial_utterance" and (not row["partial_maskable"] or not row["masked_token_indices"]):
            raise ValueError(f"{row['utterance_candidate_id']}: invalid partial assignment")
    for reel_id, rows in active_by_reel.items():
        if len(rows) != 14:
            raise ValueError(f"{reel_id}: expected 14 active candidates")
        for question_set in (1, 2):
            subset = [row for row in rows if int(row["final_assignment"][0]) == question_set]
            roles = Counter(row["final_assignment"][1] for row in subset)
            if roles != Counter({"structural_real": 3, "exact_utterance": 2, "partial_utterance": 2}):
                raise ValueError(f"{reel_id} set {question_set}: invalid role counts")
            structural_scenes = {int(row["scene_rank"]) for row in subset if row["final_assignment"][1] == "structural_real"}
            if len(structural_scenes) != 3:
                raise ValueError(f"{reel_id} set {question_set}: structural scenes are not distinct")
    checks.append("candidate_exclusions_assignments_and_text")

    active_by_key = {(row["reel_id"], float(row["start_seconds"]), float(row["end_seconds"])): row for row in active}
    absent_text_by_reel = defaultdict(set)
    for row in absent_provenance:
        absent_text_by_reel[row["reel_id"]].add(row["text"])
    card_text_to_scenes = defaultdict(list)
    scene_to_reel = {}
    scene_intervals = {}
    for episode in episodes.values():
        for scene in episode["scenes"]:
            scene_to_reel[scene["scene_id"]] = episode["reel_id"]
            scene_intervals[scene["scene_id"]] = (float(scene["start_seconds"]), float(scene["end_seconds"]), int(scene["scene_rank"]))
    scene_cards_path = PROJECT_ROOT / manifest["source_artifacts"]["scene_cards"]["path"]
    with scene_cards_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = EXPECTED_T2R_0009_CARD if row["scene_id"] == "track2_role_ipc_test_S2035_I00000996_P1292A_P1293A" else row["scene_card_text"]
            card_text_to_scenes[text].append(row["scene_id"])

    for task in tasks:
        task_id, reel_id, family = task["task_id"], task["reel_id"], task["family"]
        if reel_id not in episodes or task["audio_path"] != f"episodes/{reel_id}.wav":
            raise ValueError(f"{task_id}: invalid reel or audio path")
        answer = answers[task_id]
        episode = episodes[reel_id]
        reel_scene_ids = {scene["scene_id"] for scene in episode["scenes"]}

        def resolve_present_card(text: str) -> str:
            matches = reel_scene_ids & set(card_text_to_scenes[text])
            if len(matches) != 1:
                raise ValueError(f"{task_id}: card text does not resolve to exactly one reel scene")
            return next(iter(matches))

        if family == "2A.1":
            require_keys(answer, {"status", "card_id", "scene_rank", "start_seconds", "end_seconds"}, task_id)
            if task["variant"] == "present":
                if answer["status"] != "PRESENT":
                    raise ValueError(f"{task_id}: present card marked absent")
                scene = episode["scenes"][int(answer["scene_rank"]) - 1]
                if (float(answer["start_seconds"]), float(answer["end_seconds"])) != (float(scene["start_seconds"]), float(scene["end_seconds"])):
                    raise ValueError(f"{task_id}: scene interval mismatch")
                if resolve_present_card(task["card"]["text"]) != scene["scene_id"]:
                    raise ValueError(f"{task_id}: card content and gold scene disagree")
            elif answer != {"status": "ABSENT", "card_id": "C_TARGET", "scene_rank": None, "start_seconds": None, "end_seconds": None}:
                raise ValueError(f"{task_id}: invalid absent scene answer")
            elif reel_scene_ids & set(card_text_to_scenes[task["card"]["text"]]):
                raise ValueError(f"{task_id}: absent card resolves to a present reel scene")
        elif family == "2A.2":
            validate_span(answer["start_seconds"], answer["end_seconds"], episode, task_id)
            key = (reel_id, float(answer["start_seconds"]), float(answer["end_seconds"]))
            if key not in active_by_key:
                raise ValueError(f"{task_id}: span does not map to an active candidate")
            candidate = active_by_key[key]
            expected_text = candidate["text"] if task["quote_form"] == "exact" else candidate["partial_text"]
            if task["target_text"] != expected_text:
                raise ValueError(f"{task_id}: quotation does not match candidate provenance")
            if task["search_scope"] == "card_conditioned" and resolve_present_card(task["scene_card"]["text"]) != candidate["scene_id"]:
                raise ValueError(f"{task_id}: conditioned card does not match target scene")
        elif family == "2A.3":
            if answer != {"status": "ABSENT", "occurrences": []} or task["target_text"] not in absent_text_by_reel[reel_id]:
                raise ValueError(f"{task_id}: invalid absent utterance task")
        elif family == "2B.1":
            card_ids = {card["card_id"] for card in task["cards"]}
            if len(card_ids) != len(task["cards"]) or not set(answer["ordered_card_ids"]) <= card_ids:
                raise ValueError(f"{task_id}: invalid card IDs")
            expected_size = 5 if task["variant"] == "extra_count" else 4
            if len(card_ids) != expected_size or len(answer["ordered_card_ids"]) != 4:
                raise ValueError(f"{task_id}: invalid card counts")
            if task["variant"] == "extra_count" and answer["absent_card_id"] not in card_ids - set(answer["ordered_card_ids"]):
                raise ValueError(f"{task_id}: absent card not omitted from order")
            if task["variant"] == "exact_count" and answer["absent_card_id"] is not None:
                raise ValueError(f"{task_id}: exact-count task has absent card")
            id_to_scene = {}
            absent_ids = set()
            for card in task["cards"]:
                matches = reel_scene_ids & set(card_text_to_scenes[card["text"]])
                if len(matches) == 1:
                    id_to_scene[card["card_id"]] = next(iter(matches))
                elif not matches:
                    absent_ids.add(card["card_id"])
                else:
                    raise ValueError(f"{task_id}: ambiguous card-to-scene mapping")
            expected_order = [
                card_id for card_id, scene_id in sorted(
                    id_to_scene.items(), key=lambda item: scene_intervals[item[1]][2]
                )
            ]
            if answer["ordered_card_ids"] != expected_order:
                raise ValueError(f"{task_id}: ordered card IDs do not follow scene chronology")
            expected_absent_ids = {answer["absent_card_id"]} if task["variant"] == "extra_count" else set()
            if absent_ids != expected_absent_ids:
                raise ValueError(f"{task_id}: absent card identity mismatch")
        elif family == "2B.2":
            if len(task["utterance_targets"]) != 4 or any(target["quote_form"] != "exact" for target in task["utterance_targets"]):
                raise ValueError(f"{task_id}: 2B.2 targets must be four exact quotations")
            by_id = {target["utterance_id"]: target for target in task["utterance_targets"]}
            answer_by_id = {target["utterance_id"]: target for target in answer["utterances"]}
            if set(by_id) != set(answer_by_id) or Counter(target["status"] for target in answer["utterances"]) != Counter({"PRESENT": 3, "ABSENT": 1}):
                raise ValueError(f"{task_id}: invalid present/absent structure")
            present = []
            task_card_scene = {
                card["card_id"]: resolve_present_card(card["text"])
                for card in task["scene_cards"]
            }
            if len(task_card_scene) != 4 or set(task_card_scene.values()) != reel_scene_ids:
                raise ValueError(f"{task_id}: scene-card set does not cover the reel exactly")
            for utterance_id, item in answer_by_id.items():
                if item["status"] == "ABSENT":
                    if by_id[utterance_id]["text"] not in absent_text_by_reel[reel_id] or any(item[field] is not None for field in ("start_seconds", "end_seconds", "scene_card_id")):
                        raise ValueError(f"{task_id}: invalid absent structural target")
                else:
                    validate_span(item["start_seconds"], item["end_seconds"], episode, task_id)
                    candidate = active_by_key[(reel_id, float(item["start_seconds"]), float(item["end_seconds"]))]
                    if candidate["final_assignment"][1] != "structural_real" or by_id[utterance_id]["text"] != candidate["text"]:
                        raise ValueError(f"{task_id}: invalid structural candidate")
                    if task_card_scene.get(item["scene_card_id"]) != candidate["scene_id"]:
                        raise ValueError(f"{task_id}: structural target assigned to wrong scene card")
                    present.append((float(item["start_seconds"]), utterance_id, int(candidate["scene_rank"])))
            if len({scene for _, _, scene in present}) != 3 or answer["ordered_real_utterance_ids"] != [item[1] for item in sorted(present)]:
                raise ValueError(f"{task_id}: invalid structural scenes or chronology")
        else:
            raise ValueError(f"{task_id}: unexpected family {family}")
    checks.append("task_schemas_spans_and_provenance")

    absent_cards = manifest.get("absent_scene_cards", [])
    if len(absent_cards) != 22 or any(row["audit_status"] != "accept" for row in absent_cards):
        raise ValueError("Expected 22 accepted absent scene cards")
    for row in absent_cards:
        if row["target_reel_id"] == row["donor_reel_id"] or scene_to_reel[row["absent_scene_id"]] != row["donor_reel_id"]:
            raise ValueError(f"{row['target_reel_id']}: invalid absent scene donor")
        if row["absent_scene_id"] in {scene["scene_id"] for scene in episodes[row["target_reel_id"]]["scenes"]}:
            raise ValueError(f"{row['target_reel_id']}: absent scene occurs in target reel")
    if manifest["scene_card_revisions"][0]["proposed_scene_card_text"] != EXPECTED_T2R_0009_CARD:
        raise ValueError("T2R_0009 accepted scene card is not frozen exactly")
    if any("placeholder" in json.dumps(task).lower() for task in tasks):
        raise ValueError("Placeholder content remains in public tasks")
    checks.append("scene_cards_reviews_and_no_placeholders")

    # Source-transcript absence verification.
    selected_scenes_path = SCRIPT_DIR / "track2_role_ipc_selected_reel_scenes_test_v1.csv"
    scene_metadata_path = SCRIPT_DIR / "track2_role_ipc_scene_candidates_scene_filter_v1.csv"
    selected_scenes = pd.read_csv(selected_scenes_path, keep_default_na=False)
    scene_metadata = pd.read_csv(scene_metadata_path, keep_default_na=False).set_index("scene_id", drop=False)
    turns_by_reel = defaultdict(list)
    for scene in selected_scenes.itertuples(index=False):
        transcript = audit.load_scene_transcript(scene_metadata.loc[str(scene.scene_id)], dataset_root)
        turns_by_reel[str(scene.reel_id)].extend(turn.normalized_text for turn in transcript.local_turns)
    for row in absent_provenance:
        if any(row["normalized_text"] in turn for turn in turns_by_reel[row["reel_id"]]):
            raise ValueError(f"{row['absent_candidate_id']}: absent text occurs in source transcript")
    checks.append("absent_utterances_not_in_source_transcripts")

    t8 = next(row for row in present_provenance if row["utterance_candidate_id"] == "T2R_0008_S4_U01")
    if t8["final_assignment"] != [1, "structural_real", 3]:
        raise ValueError("T2R_0008 special reassignment regression")
    t10 = next(row for row in present_provenance if row["utterance_candidate_id"] == "T2R_0010_S2_U01")
    if (float(t10["start_seconds"]), float(t10["end_seconds"]), t10["text"]) != (
        359.89, 364.95,
        "And so he was like, oh, babe, you need to work, you need to come out over here. So that's what, you know, like, ugh.",
    ):
        raise ValueError("T2R_0010_S2_U01 accepted correction regression")
    checks.append("named_audit_regressions")

    hashes = artifact_hashes(dataset_dir)
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return {
        "dataset_id": manifest["dataset_id"], "status": "passed", "checks": checks,
        "episode_count": len(episodes), "task_count": len(tasks),
        "task_counts_by_family": dict(sorted(family_counts.items())),
        "artifact_sha256": hashes, "deterministic_build_fingerprint": fingerprint,
    }


def rebuild_check(release_dir: Path, dataset_root: Path, v1_audio_dir: Path = V1_DIR / "episodes") -> None:
    with tempfile.TemporaryDirectory(prefix="track2_v2_rebuild_a_") as first, tempfile.TemporaryDirectory(prefix="track2_v2_rebuild_b_") as second:
        for output in (first, second):
            subprocess.run(
                [sys.executable, str(BUILDER), "--output-dir", output, "--dataset-root", str(dataset_root), "--audio-mode", "copy", "--v1-audio-dir", str(v1_audio_dir)],
                check=True, cwd=PROJECT_ROOT,
            )
        first_report = validate_dataset(Path(first), dataset_root, v1_audio_dir)
        second_report = validate_dataset(Path(second), dataset_root, v1_audio_dir)
        if first_report["artifact_sha256"] != second_report["artifact_sha256"]:
            raise ValueError("Two clean builds produced different artifact hashes")
        release_hashes = artifact_hashes(release_dir)
        if first_report["artifact_sha256"] != release_hashes:
            raise ValueError("Clean rebuild artifacts differ from the supplied release")


def main() -> None:
    args = parse_args()
    report = validate_dataset(args.dataset_dir.resolve(), args.dataset_root.resolve(), args.v1_audio_dir.resolve())
    if args.rebuild_check:
        rebuild_check(args.dataset_dir.resolve(), args.dataset_root.resolve(), args.v1_audio_dir.resolve())
        report["checks"].append("release_matches_two_clean_builds")
    if args.write_report:
        path = args.dataset_dir.resolve() / "validation_report.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    print(json.dumps({"status": report["status"], "task_count": report["task_count"], "checks": report["checks"]}, indent=2))


if __name__ == "__main__":
    main()
