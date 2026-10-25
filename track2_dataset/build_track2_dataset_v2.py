#!/usr/bin/env python3
"""Build the audited Track 2 v2 dataset release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

import audit_reel_task_availability as audit
from build_utterance_candidate_pool_v2 import partial_prompt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
V1_DIR = SCRIPT_DIR / "generated_data" / "track2_role_ipc_test_v1" / "dataset"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "generated_data" / "track2_role_ipc_test_v2" / "dataset"
AUDIT_DIR = PROJECT_ROOT / "track2_human_audit" / "audit_results"
DATASET_ID = "track2_role_ipc_test_v2"
DATASET_SEED = 20260720
GLOBAL_LONGEST_SCENE_DURATION = 354.0
BOUNDARY_MARGIN = 30.0
MAX_SINGLE_QUERY_SECONDS = 384.0

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
EXCLUSIONS = OVERLAP_EXCLUSIONS | AMBIGUOUS_EXCLUSIONS
ROLE_ORDER = {"exact_utterance": 0, "partial_utterance": 1, "structural_real": 2}
EXPECTED_PER_REEL = {"2A.1": 5, "2A.2": 8, "2A.3": 2, "2B.1": 2, "2B.2": 2}
FINAL_T2R_0009_CARD = (
    "Two people are discussing their experiences with recovery, sharing personal stories about "
    "challenges they have faced and what gives them hope. They compare perspectives on resilience "
    "and offer encouragement and opinions while trying to connect over similar struggles."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Separately acquired Seamless Interaction datasets directory.")
    parser.add_argument("--v1-audio-dir", type=Path, default=V1_DIR / "episodes")
    parser.add_argument("--audio-mode", choices=("copy", "hardlink", "symlink"), default="copy")
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_text(text: str) -> str:
    return " ".join(audit.tokens_from_text(text))


def regenerate_partial(text: str, row: dict[str, Any]) -> tuple[str, list[int]]:
    tokens = tuple(audit.tokens_from_text(text))
    turn = audit.Turn(
        scene_id=str(row["scene_id"]), scene_rank=int(row["scene_rank"]),
        speaker="participant1", start=float(row["start_seconds"]),
        end=float(row["end_seconds"]), text=text, tokens=tokens,
        overlap_ratio=float(row["overlap_ratio"]),
    )
    partial, indices = partial_prompt(turn)
    return partial, list(indices)


def shuffled(values: list[Any], key: str) -> list[Any]:
    result = list(values)
    seed = int.from_bytes(hashlib.sha256(f"{DATASET_SEED}:{key}".encode()).digest()[:8], "big")
    random.Random(seed).shuffle(result)
    return result


def slot(row: dict[str, Any]) -> tuple[int, str, int] | None:
    if row["assigned_role"] == "reserve":
        return None
    return (int(row["assigned_question_set"]), str(row["assigned_role"]), int(row["assigned_role_index"]))


def slot_sort_key(value: tuple[int, str, int]) -> tuple[int, int, int]:
    return (value[0], ROLE_ORDER[value[1]], value[2])


def valid_assignment(assigned: dict[tuple[int, str, int], dict[str, Any]]) -> bool:
    if len(assigned) != 14 or len({r["utterance_candidate_id"] for r in assigned.values()}) != 14:
        return False
    for question_set in (1, 2):
        structural = [r for s, r in assigned.items() if s[0] == question_set and s[1] == "structural_real"]
        if len(structural) != 3 or len({int(r["scene_rank"]) for r in structural}) != 3:
            return False
        partials = [r for s, r in assigned.items() if s[0] == question_set and s[1] == "partial_utterance"]
        if len(partials) != 2 or not all(r["partial_maskable"] for r in partials):
            return False
        exacts = [r for s, r in assigned.items() if s[0] == question_set and s[1] == "exact_utterance"]
        if len(exacts) != 2:
            return False
    return True


def assign_reel(rows: list[dict[str, Any]]) -> dict[str, tuple[int, str, int] | None]:
    reel_id = rows[0]["reel_id"]
    usable = [row for row in rows if row["utterance_candidate_id"] not in EXCLUSIONS]
    assigned = {slot(row): row for row in usable if slot(row) is not None}

    if reel_id == "T2R_0008":
        moved_id = "T2R_0008_S4_U01"
        moved = next(row for row in usable if row["utterance_candidate_id"] == moved_id)
        old_slot = slot(moved)
        structural_slot = (1, "structural_real", 3)
        if old_slot is None or structural_slot in assigned:
            raise ValueError("T2R_0008 special reassignment preconditions failed")
        del assigned[old_slot]
        assigned[structural_slot] = moved

    required = {
        (question_set, role, index)
        for question_set in (1, 2)
        for role, count in (("exact_utterance", 2), ("partial_utterance", 2), ("structural_real", 3))
        for index in range(1, count + 1)
    }
    open_slots = sorted(required - set(assigned), key=slot_sort_key)
    used_ids = {row["utterance_candidate_id"] for row in assigned.values()}
    available = [row for row in usable if row["utterance_candidate_id"] not in used_ids]
    solutions: list[tuple[tuple[Any, ...], dict[tuple[int, str, int], dict[str, Any]]]] = []
    for chosen in itertools.permutations(available, len(open_slots)):
        proposal = dict(assigned)
        proposal.update(zip(open_slots, chosen))
        if not valid_assignment(proposal):
            continue
        changed_active = sum(
            1 for candidate_slot, row in proposal.items()
            if slot(row) is not None and slot(row) != candidate_slot
        )
        cost = (
            changed_active,
            sum(int(row["quality_rank_in_scene"]) for row in chosen),
            tuple(row["utterance_candidate_id"] for row in chosen),
        )
        solutions.append((cost, proposal))
    if not solutions:
        raise ValueError(f"{reel_id}: no valid audited candidate assignment")
    final = min(solutions, key=lambda item: item[0])[1]
    result = {row["utterance_candidate_id"]: None for row in rows}
    for candidate_slot, row in final.items():
        result[row["utterance_candidate_id"]] = candidate_slot
    return result


def apply_audits(
    raw_rows: list[dict[str, str]], audit_state: dict[str, Any]
) -> list[dict[str, Any]]:
    reviews = {
        candidate_id: review
        for scene in audit_state["scenes"].values()
        for candidate_id, review in scene["candidate_reviews"].items()
    }
    if set(reviews) != {row["utterance_candidate_id"] for row in raw_rows}:
        raise ValueError("Candidate pool and human audit candidate IDs differ")
    result = []
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        for name in ("scene_rank", "quality_rank_in_scene", "assigned_role_index", "word_count"):
            row[name] = int(row[name])
        for name in ("start_seconds", "end_seconds", "duration_seconds", "overlap_ratio"):
            row[name] = float(row[name])
        row["partial_maskable"] = str(row["partial_maskable"]).lower() == "true"
        row["masked_token_indices"] = json.loads(row["masked_token_indices"])
        review = reviews[row["utterance_candidate_id"]]
        corrected_text = review.get("corrected_transcript_text", "").strip()
        if corrected_text:
            row["text"] = corrected_text
        correction = review.get("corrections", {}).get("utterance", {})
        if correction:
            row["start_seconds"] = float(correction["corrected_start_seconds"])
            row["end_seconds"] = float(correction["corrected_end_seconds"])
        row["duration_seconds"] = round(row["end_seconds"] - row["start_seconds"], 3)
        row["normalized_text"] = normalized_text(row["text"])
        row["word_count"] = len(audit.tokens_from_text(row["text"]))
        row["partial_text"], row["masked_token_indices"] = regenerate_partial(row["text"], row)
        row["partial_maskable"] = bool(row["partial_text"])
        row["audit_candidate_status"] = review["candidate_status"]
        row["audit_timestamp_status"] = review["timestamp_status"]
        row["audit_corrected_text"] = bool(corrected_text)
        row["audit_corrected_interval"] = bool(correction)
        result.append(row)
    return result


def apply_absent_audits(raw_rows: list[dict[str, str]], audit_state: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    reviews = audit_state["absent_candidate_reviews"]
    for raw in raw_rows:
        row: dict[str, Any] = dict(raw)
        review = reviews[row["absent_candidate_id"]]
        if review["candidate_status"] != "ok":
            raise ValueError(f"Unaccepted absent utterance: {row['absent_candidate_id']}")
        corrected = review.get("corrected_transcript_text", "").strip()
        if corrected:
            row["text"] = corrected
        row["normalized_text"] = normalized_text(row["text"])
        row["word_count"] = len(audit.tokens_from_text(row["text"]))
        row["assigned_question_set"] = int(row["assigned_question_set"])
        for name in ("source_scene_rank", "word_count"):
            row[name] = int(row[name])
        for name in ("local_start_seconds", "local_end_seconds", "duration_seconds"):
            row[name] = float(row[name])
        row["audit_corrected_text"] = bool(corrected)
        result.append(row)
    return result


def card_payload(card_id: str, row: dict[str, Any]) -> dict[str, Any]:
    return {"card_id": card_id, "text": row["scene_card_text"]}


def span_answer(row: dict[str, Any]) -> dict[str, Any]:
    return {"status": "PRESENT", "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"]}


def task_common(task_id: str, family: str, variant: str, reel_id: str) -> dict[str, Any]:
    return {
        "task_id": task_id, "track": "2", "family": family, "variant": variant,
        "reel_id": reel_id, "audio_path": f"episodes/{reel_id}.wav",
    }


def build_tasks(
    manifest: dict[str, Any], cards: list[dict[str, Any]], assignments: dict[str, tuple[int, str, int] | None],
    candidates: list[dict[str, Any]], absent_utterances: list[dict[str, Any]], absent_cards: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    public: list[dict[str, Any]] = []
    answers: list[dict[str, Any]] = []
    cards_by_reel = defaultdict(list)
    for row in cards:
        cards_by_reel[row["reel_id"]].append(row)
    candidates_by_slot = {
        (row["reel_id"], *candidate_slot): row
        for row in candidates
        if (candidate_slot := assignments[row["utterance_candidate_id"]]) is not None
    }
    absent_utterance_by_set = {(r["reel_id"], r["assigned_question_set"]): r for r in absent_utterances}
    absent_card_by_reel = {r["target_reel_id"]: r for r in absent_cards}
    card_by_scene = {row["scene_id"]: row for row in cards}

    def add(task: dict[str, Any], answer: dict[str, Any]) -> None:
        public.append(task)
        answers.append({"task_id": task["task_id"], "answer": answer})

    for episode in sorted(manifest["episodes"], key=lambda item: item["reel_id"]):
        reel_id = episode["reel_id"]
        reel_cards = sorted(cards_by_reel[reel_id], key=lambda row: int(row["scene_rank"]))
        scene_by_id = {scene["scene_id"]: scene for scene in episode["scenes"]}
        absent_card_review = absent_card_by_reel[reel_id]
        donor_card = card_by_scene[absent_card_review["absent_scene_id"]]

        for index, card in enumerate(reel_cards, start=1):
            task_id = f"T2V2_2A1_{reel_id}_present_{index:02d}"
            task = task_common(task_id, "2A.1", "present", reel_id)
            task.update({"card": card_payload("C_TARGET", card), "question_text": "Locate the scene described by the card. Return its rank and temporal interval, or ABSENT."})
            scene = scene_by_id[card["scene_id"]]
            add(task, {"status": "PRESENT", "card_id": "C_TARGET", "scene_rank": int(scene["scene_rank"]), "start_seconds": scene["start_seconds"], "end_seconds": scene["end_seconds"]})
        task_id = f"T2V2_2A1_{reel_id}_absent_01"
        task = task_common(task_id, "2A.1", "absent", reel_id)
        task.update({"card": card_payload("C_TARGET", donor_card), "question_text": "Locate the scene described by the card. Return its rank and temporal interval, or ABSENT."})
        add(task, {"status": "ABSENT", "card_id": "C_TARGET", "scene_rank": None, "start_seconds": None, "end_seconds": None})

        for question_set in (1, 2):
            for role, quote_kind in (("exact_utterance", "exact"), ("partial_utterance", "partial")):
                for role_index in (1, 2):
                    row = candidates_by_slot[(reel_id, question_set, role, role_index)]
                    search_scope = "global" if role_index == 1 else "card_conditioned"
                    task_id = f"T2V2_2A2_{reel_id}_{quote_kind}_{search_scope}_set{question_set}_{role_index:02d}"
                    task = task_common(task_id, "2A.2", f"{quote_kind}_{search_scope}", reel_id)
                    query_text = row["text"] if quote_kind == "exact" else row["partial_text"]
                    task.update({"target_id": "U_TARGET", "target_text": query_text, "quote_form": quote_kind, "search_scope": search_scope})
                    if search_scope == "card_conditioned":
                        task["scene_card"] = card_payload("C_CONTEXT", card_by_scene[row["scene_id"]])
                    task["question_text"] = "Find the complete utterance interval matching the supplied quotation."
                    add(task, span_answer(row))

            absent = absent_utterance_by_set[(reel_id, question_set)]
            task_id = f"T2V2_2A3_{reel_id}_set{question_set}_01"
            task = task_common(task_id, "2A.3", "absent_utterance", reel_id)
            task.update({"target_id": "U_TARGET", "target_text": absent["text"], "question_text": "Find this utterance in the reel, or return ABSENT if it does not occur."})
            add(task, {"status": "ABSENT", "occurrences": []})

        for variant in ("exact_count", "extra_count"):
            task_id = f"T2V2_2B1_{reel_id}_{variant}_01"
            source_cards = list(reel_cards) + ([donor_card] if variant == "extra_count" else [])
            presented = shuffled(source_cards, task_id)
            id_by_scene = {row["scene_id"]: f"C_{chr(65 + i)}" for i, row in enumerate(presented)}
            task = task_common(task_id, "2B.1", variant, reel_id)
            task.update({"cards": [card_payload(id_by_scene[row["scene_id"]], row) for row in presented], "question_text": "Return the chronological order of the cards whose scenes occur in the reel. Omit any absent card."})
            add(task, {"ordered_card_ids": [id_by_scene[row["scene_id"]] for row in reel_cards], "absent_card_id": id_by_scene[donor_card["scene_id"]] if variant == "extra_count" else None})

        for question_set in (1, 2):
            task_id = f"T2V2_2B2_{reel_id}_set{question_set}_01"
            structural = [candidates_by_slot[(reel_id, question_set, "structural_real", i)] for i in (1, 2, 3)]
            absent = absent_utterance_by_set[(reel_id, question_set)]
            target_sources = [("present", row) for row in structural] + [("absent", absent)]
            presented_targets = shuffled(target_sources, task_id + ":targets")
            target_id_by_key = {
                (kind, row.get("utterance_candidate_id", row.get("absent_candidate_id"))): f"U_{chr(65 + i)}"
                for i, (kind, row) in enumerate(presented_targets)
            }
            presented_cards = shuffled(reel_cards, task_id + ":cards")
            card_id_by_scene = {row["scene_id"]: f"C_{chr(65 + i)}" for i, row in enumerate(presented_cards)}
            task = task_common(task_id, "2B.2", "structural_relational_retrieval", reel_id)
            task.update({
                "scene_cards": [card_payload(card_id_by_scene[row["scene_id"]], row) for row in presented_cards],
                "utterance_targets": [
                    {"utterance_id": target_id_by_key[(kind, row.get("utterance_candidate_id", row.get("absent_candidate_id")))], "text": row["text"], "quote_form": "exact"}
                    for kind, row in presented_targets
                ],
                "question_text": "Locate the three present utterances, identify the absent target, order the present utterances, and assign each present utterance to its scene card.",
            })
            answer_targets = []
            for kind, row in presented_targets:
                key = row.get("utterance_candidate_id", row.get("absent_candidate_id"))
                utterance_id = target_id_by_key[(kind, key)]
                if kind == "absent":
                    answer_targets.append({"utterance_id": utterance_id, "status": "ABSENT", "start_seconds": None, "end_seconds": None, "scene_card_id": None})
                else:
                    answer_targets.append({"utterance_id": utterance_id, "status": "PRESENT", "start_seconds": row["start_seconds"], "end_seconds": row["end_seconds"], "scene_card_id": card_id_by_scene[row["scene_id"]]})
            ordered = sorted((row for row in structural), key=lambda row: (row["start_seconds"], row["end_seconds"]))
            add(task, {
                "ordered_real_utterance_ids": [target_id_by_key[("present", row["utterance_candidate_id"])] for row in ordered],
                "utterances": answer_targets,
            })
    return public, answers


def main() -> None:
    args = parse_args()
    inputs = {
        "v1_manifest": V1_DIR / "manifest.json",
        "scene_cards": SCRIPT_DIR / "track2_role_ipc_scene_cards_v3.csv",
        "candidates": SCRIPT_DIR / "track2_role_ipc_utterance_candidate_pool_v2.csv",
        "absent_utterances": SCRIPT_DIR / "track2_role_ipc_absent_utterance_candidate_pool_v2.csv",
        "selected_scenes": SCRIPT_DIR / "track2_role_ipc_selected_reel_scenes_test_v1.csv",
        "scene_metadata": SCRIPT_DIR / "track2_role_ipc_scene_candidates_scene_filter_v1.csv",
        "human_audit": AUDIT_DIR / "track2_human_audit_progress_2026-07-20T04-52-43-406Z.json",
        "absent_card_reviews": AUDIT_DIR / "track2_absent_scene_card_reviews.csv",
        "scene_card_revision": AUDIT_DIR / "track2_scene_card_revision_review.csv",
    }
    for path in inputs.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)

    v1_manifest = read_json(inputs["v1_manifest"])
    audit_export = read_json(inputs["human_audit"])
    candidates = apply_audits(read_csv(inputs["candidates"]), audit_export["state"])
    absent_utterances = apply_absent_audits(read_csv(inputs["absent_utterances"]), audit_export["state"])
    absent_cards = read_csv(inputs["absent_card_reviews"])
    revisions = read_csv(inputs["scene_card_revision"])
    if len(absent_cards) != 22 or any(row["audit_status"] != "accept" for row in absent_cards):
        raise ValueError("Expected 22 accepted absent scene-card reviews")
    if len(revisions) != 1 or revisions[0]["audit_status"] != "accept" or revisions[0]["proposed_scene_card_text"] != FINAL_T2R_0009_CARD:
        raise ValueError("T2R_0009 final scene-card review does not match the accepted text")

    cards: list[dict[str, Any]] = read_csv(inputs["scene_cards"])
    revised_scene_id = revisions[0]["scene_id"]
    for card in cards:
        if card["scene_id"] == revised_scene_id:
            card["scene_card_text"] = FINAL_T2R_0009_CARD
            card["scene_card_word_count"] = len(audit.tokens_from_text(FINAL_T2R_0009_CARD))

    candidate_ids = {row["utterance_candidate_id"] for row in candidates}
    if not EXCLUSIONS <= candidate_ids:
        raise ValueError(f"Missing excluded candidates: {sorted(EXCLUSIONS - candidate_ids)}")
    assignments: dict[str, tuple[int, str, int] | None] = {}
    for reel_id in sorted({row["reel_id"] for row in candidates}):
        reel_rows = [row for row in candidates if row["reel_id"] == reel_id]
        assignments.update(assign_reel(reel_rows))

    # Verify absent utterance strings against complete source turns.
    selected_scenes = pd.read_csv(inputs["selected_scenes"], keep_default_na=False)
    scene_metadata = pd.read_csv(inputs["scene_metadata"], keep_default_na=False).set_index("scene_id", drop=False)
    reel_turn_texts: dict[str, list[str]] = defaultdict(list)
    for scene in selected_scenes.itertuples(index=False):
        transcript = audit.load_scene_transcript(scene_metadata.loc[str(scene.scene_id)], args.dataset_root)
        reel_turn_texts[str(scene.reel_id)].extend(turn.normalized_text for turn in transcript.local_turns)
    for row in absent_utterances:
        if any(row["normalized_text"] in turn for turn in reel_turn_texts[row["reel_id"]]):
            raise ValueError(f"Absent utterance occurs in target reel: {row['absent_candidate_id']}")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    episodes_dir = output / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    for source_episode in sorted(v1_manifest["episodes"], key=lambda item: item["reel_id"]):
        episode = {k: v for k, v in source_episode.items() if not k.endswith("_candidate_count")}
        reel_id = episode["reel_id"]
        source_audio = args.v1_audio_dir / Path(source_episode["audio_path"]).name
        target_audio = episodes_dir / f"{reel_id}.wav"
        if target_audio.exists() or target_audio.is_symlink():
            raise FileExistsError(f"Refusing to replace existing audio: {target_audio}")
        if args.audio_mode == "copy":
            shutil.copyfile(source_audio, target_audio)
        elif args.audio_mode == "hardlink":
            target_audio.hardlink_to(source_audio)
        else:
            target_audio.symlink_to(source_audio)
        if sha256_file(target_audio) != source_episode["sha256_audio"]:
            raise ValueError(f"{reel_id}: copied audio hash mismatch")
        episode["audio_path"] = f"episodes/{reel_id}.wav"
        episodes.append(episode)

    manifest = {
        "dataset_id": DATASET_ID, "track_id": "2", "split": "test", "episode_count": 22,
        "gap_seconds": v1_manifest["gap_seconds"], "dataset_seed": DATASET_SEED,
        "global_longest_scene_duration": GLOBAL_LONGEST_SCENE_DURATION,
        "boundary_margin": BOUNDARY_MARGIN, "max_single_query_seconds": MAX_SINGLE_QUERY_SECONDS,
        "task_count": 418, "task_counts_by_family": {k: v * 22 for k, v in EXPECTED_PER_REEL.items()},
        "episodes": episodes,
        "source_artifacts": {name: {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256_file(path)} for name, path in inputs.items()},
        "absent_scene_cards": absent_cards,
        "scene_card_revisions": revisions,
    }
    public, answers = build_tasks(manifest, cards, assignments, candidates, absent_utterances, absent_cards)

    provenance = []
    for row in sorted(candidates, key=lambda item: item["utterance_candidate_id"]):
        candidate_id = row["utterance_candidate_id"]
        final_slot = assignments[candidate_id]
        original_slot = slot(row)
        record = dict(row)
        record.update({
            "candidate_kind": "present_utterance", "excluded": candidate_id in EXCLUSIONS,
            "exclusion_reason": "overlapping_speech" if candidate_id in OVERLAP_EXCLUSIONS else ("transcript_ambiguity" if candidate_id in AMBIGUOUS_EXCLUSIONS else None),
            "original_assignment": list(original_slot) if original_slot else None,
            "final_assignment": list(final_slot) if final_slot else None,
            "promoted_or_reassigned": final_slot is not None and final_slot != original_slot,
        })
        provenance.append(record)
    for row in sorted(absent_utterances, key=lambda item: item["absent_candidate_id"]):
        provenance.append({"candidate_kind": "absent_utterance", **row})

    if len(public) != 418 or len(answers) != 418:
        raise ValueError(f"Expected 418 tasks, got {len(public)} public and {len(answers)} answers")
    write_json(output / "manifest.json", manifest)
    write_jsonl(output / "task_instances.jsonl", public)
    write_jsonl(output / "task_answers.jsonl", answers)
    write_jsonl(output / "candidate_provenance.jsonl", provenance)
    print(f"wrote {output}")
    print(f"tasks {len(public)}")
    print(f"families {dict(sorted(Counter(row['family'] for row in public).items()))}")


if __name__ == "__main__":
    main()
