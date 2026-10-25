from __future__ import annotations

import json
import math
import re
from copy import deepcopy
from typing import Any, Mapping, Sequence


TRACK2_LEDGER_STATUSES = {"NOT_FOUND_YET", "FOUND", "LIKELY_ABSENT"}
TRACK2_FINAL_STATUSES = {"PRESENT", "ABSENT"}
_ACTION_RE = re.compile(
    r"^ACTION\s+(?P<action>QUERY|STOP)(?:\s+(?P<start>-?\d+(?:\.\d+)?)\s+(?P<end>-?\d+(?:\.\d+)?))?\s*$",
    re.IGNORECASE,
)
_SET_RE = re.compile(
    r"^SET\s+(?P<target>\S+)\s+(?P<status>NOT_FOUND_YET|FOUND|LIKELY_ABSENT)"
    r"(?:\s+(?P<start>-?\d+(?:\.\d+)?)\s+(?P<end>-?\d+(?:\.\d+)?)(?:\s+(?P<card>\S+))?)?\s*$",
    re.IGNORECASE,
)
def track2_target_ids(task: Mapping[str, Any]) -> list[str]:
    family = str(task["family"])
    if family == "2A.1":
        return [str(task["card"]["card_id"])]
    if family in {"2A.2", "2A.3"}:
        return [str(task["target_id"])]
    if family == "2B.1":
        return [str(card["card_id"]) for card in task["cards"]]
    if family == "2B.2":
        return [str(target["utterance_id"]) for target in task["utterance_targets"]]
    raise ValueError(f"Unsupported Track 2 family: {family}")


def track2_scene_card_ids(task: Mapping[str, Any]) -> set[str]:
    if task["family"] == "2B.2":
        return {str(card["card_id"]) for card in task["scene_cards"]}
    return set()


def initialize_track2_ledger(task: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        target_id: {
            "target_id": target_id,
            "status": "NOT_FOUND_YET",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
            "last_updated_step": None,
        }
        for target_id in track2_target_ids(task)
    }


def canonical_ledger_text(ledger: Mapping[str, Mapping[str, Any]]) -> str:
    lines: list[str] = []
    for target_id, entry in ledger.items():
        status = str(entry["status"])
        if status != "FOUND":
            lines.append(f"{target_id} {status}")
            continue
        line = (
            f"{target_id} FOUND {float(entry['start_seconds']):.3f} "
            f"{float(entry['end_seconds']):.3f}"
        )
        if entry.get("scene_card_id") is not None:
            line += f" {entry['scene_card_id']}"
        lines.append(line)
    return "\n".join(lines)


def parse_track2_action_response(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    result: dict[str, Any] = {
        "format_valid": False,
        "updates": [],
        "action": None,
        "start_seconds": None,
        "end_seconds": None,
        "reason": "",
    }
    if not lines:
        result["reason"] = "empty_response"
        return result

    action_lines = [line for line in lines if _ACTION_RE.match(line)]
    if len(action_lines) != 1:
        result["reason"] = "expected_exactly_one_action"
        return result
    action_line = action_lines[0]
    if action_line != lines[-1]:
        result["reason"] = "action_must_be_last"
        return result

    update_lines = lines[:-1]
    if update_lines == ["NO_UPDATE"] or update_lines == ["no_update"]:
        update_lines = []
    elif any(line.upper() == "NO_UPDATE" for line in update_lines):
        result["reason"] = "no_update_cannot_accompany_set"
        return result

    updates: list[dict[str, Any]] = []
    for line in update_lines:
        match = _SET_RE.match(line)
        if match is None:
            result["reason"] = "invalid_set_line"
            return result
        status = match.group("status").upper()
        start = float(match.group("start")) if match.group("start") is not None else None
        end = float(match.group("end")) if match.group("end") is not None else None
        card = match.group("card")
        if status == "FOUND" and (start is None or end is None):
            result["reason"] = "found_requires_interval"
            return result
        if status != "FOUND" and (start is not None or end is not None or card is not None):
            result["reason"] = "non_found_cannot_have_interval"
            return result
        updates.append(
            {
                "target_id": match.group("target"),
                "status": status,
                "start_seconds": start,
                "end_seconds": end,
                "scene_card_id": card,
                "raw_line": line,
            }
        )

    action_match = _ACTION_RE.match(action_line)
    assert action_match is not None
    action = action_match.group("action").upper()
    start = float(action_match.group("start")) if action_match.group("start") is not None else None
    end = float(action_match.group("end")) if action_match.group("end") is not None else None
    if action == "QUERY" and (start is None or end is None):
        result["reason"] = "query_requires_interval"
        return result
    if action == "STOP" and (start is not None or end is not None):
        result["reason"] = "stop_cannot_have_interval"
        return result
    result.update(
        {
            "format_valid": True,
            "updates": updates,
            "action": action,
            "start_seconds": start,
            "end_seconds": end,
        }
    )
    return result


def validate_and_merge_track2_updates(
    task: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
    *,
    reel_duration_seconds: float,
    step_index: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    merged = deepcopy(dict(ledger))
    valid_targets = set(merged)
    valid_cards = track2_scene_card_ids(task)
    outcomes: list[dict[str, Any]] = []
    for update in updates:
        target_id = str(update["target_id"])
        status = str(update["status"])
        reason = ""
        if target_id not in valid_targets:
            reason = "unknown_target_id"
        elif status not in TRACK2_LEDGER_STATUSES:
            reason = "invalid_status"
        elif status == "FOUND":
            start = update.get("start_seconds")
            end = update.get("end_seconds")
            if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (start, end)):
                reason = "non_finite_interval"
            elif not 0.0 <= float(start) < float(end) <= float(reel_duration_seconds):
                reason = "interval_out_of_range"
            elif task["family"] == "2B.2" and update.get("scene_card_id") not in valid_cards:
                reason = "invalid_scene_card_id"
            elif task["family"] != "2B.2" and update.get("scene_card_id") is not None:
                reason = "scene_card_id_not_allowed"
        if reason:
            outcomes.append({"target_id": target_id, "valid": False, "reason": reason, "update": dict(update)})
            continue
        entry = merged[target_id]
        entry["status"] = status
        entry["last_updated_step"] = int(step_index)
        if status == "FOUND":
            entry["start_seconds"] = float(update["start_seconds"])
            entry["end_seconds"] = float(update["end_seconds"])
            entry["scene_card_id"] = update.get("scene_card_id")
        else:
            entry["start_seconds"] = None
            entry["end_seconds"] = None
            entry["scene_card_id"] = None
        outcomes.append({"target_id": target_id, "valid": True, "reason": "", "update": dict(update)})
    return merged, outcomes


def validate_track2_query(
    start_seconds: Any,
    end_seconds: Any,
    *,
    reel_duration_seconds: float,
    max_single_query_seconds: float,
    listened_seconds: float,
    listening_budget_seconds: float,
) -> tuple[bool, str]:
    if not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in (start_seconds, end_seconds)):
        return False, "non_finite_query"
    start = float(start_seconds)
    end = float(end_seconds)
    if not 0.0 <= start < end <= float(reel_duration_seconds):
        return False, "query_out_of_range"
    duration = end - start
    if duration > float(max_single_query_seconds) + 1e-9:
        return False, "query_exceeds_max_single_query_seconds"
    if float(listened_seconds) + duration > float(listening_budget_seconds) + 1e-9:
        return False, "query_exceeds_listening_budget"
    return True, ""


def _extract_json_object(raw: str) -> Any:
    text = str(raw).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("response_must_be_exact_json") from exc


_JSON_FENCE_RE = re.compile(
    r"^```(?:json)?[ \t]*\r?\n(?P<payload>\{.*\})\r?\n```[ \t]*$",
    re.IGNORECASE | re.DOTALL,
)
_SPECIAL_TOKEN_SUFFIX_RE = re.compile(r"^(?:<\|[A-Za-z0-9_:.\-]+\|>\s*)+$")


def _required_final_answer_keys(task: Mapping[str, Any]) -> set[str]:
    family = str(task["family"])
    if family == "2A.1":
        return {
            "status",
            "card_id",
            "scene_rank",
            "start_seconds",
            "end_seconds",
            "confidence",
        }
    if family == "2A.2":
        return {"start_seconds", "end_seconds", "confidence"}
    if family == "2A.3":
        return {"status", "occurrences", "confidence"}
    if family == "2B.1":
        return {"ordered_card_ids", "absent_card_id"}
    if family == "2B.2":
        return {"ordered_real_utterance_ids", "utterances", "confidence_complete"}
    raise ValueError(f"Unsupported Track 2 family: {family}")


def _rescue_has_all_nested_fields(payload: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    family = str(task["family"])
    if family == "2A.3":
        occurrences = payload.get("occurrences")
        return isinstance(occurrences, list) and all(
            isinstance(item, Mapping)
            and {"start_seconds", "end_seconds"}.issubset(item)
            for item in occurrences
        )
    if family == "2B.2":
        utterances = payload.get("utterances")
        required = {
            "utterance_id",
            "status",
            "start_seconds",
            "end_seconds",
            "scene_card_id",
        }
        return isinstance(utterances, list) and all(
            isinstance(item, Mapping) and required.issubset(item)
            for item in utterances
        )
    return True


def _rescue_single_json_object(raw: str, task: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Recover one complete object from superficial, unambiguous wrappers only."""
    text = str(raw).strip()
    fence_match = _JSON_FENCE_RE.fullmatch(text)
    if fence_match is not None:
        candidate = fence_match.group("payload")
        rescue_kind = "markdown_json_fence"
    else:
        decoder = json.JSONDecoder()
        try:
            decoded, end_index = decoder.raw_decode(text)
        except json.JSONDecodeError as exc:
            raise ValueError("response_must_be_exact_json") from exc
        suffix = text[end_index:].strip()
        if not suffix or _SPECIAL_TOKEN_SUFFIX_RE.fullmatch(suffix) is None:
            raise ValueError("response_must_be_exact_json")
        candidate = text[:end_index]
        rescue_kind = "special_token_suffix"
        if not isinstance(decoded, dict):
            raise ValueError("answer_must_be_object")

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError("response_must_be_exact_json") from exc
    if not isinstance(payload, dict):
        raise ValueError("answer_must_be_object")
    if not _required_final_answer_keys(task).issubset(payload):
        raise ValueError("rescue_requires_all_required_fields")
    if not _rescue_has_all_nested_fields(payload, task):
        raise ValueError("rescue_requires_all_required_fields")
    return dict(payload), rescue_kind


def parse_track2_final_answer(raw: str, task: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = _extract_json_object(raw)
        if not isinstance(payload, dict):
            raise ValueError("answer_must_be_object")
        return {
            "format_valid": True,
            "strict_format_valid": True,
            "rescue_applied": False,
            "rescue_kind": None,
            "answer": dict(payload),
            "reason": "",
        }
    except ValueError as strict_exc:
        try:
            payload, rescue_kind = _rescue_single_json_object(raw, task)
            return {
                "format_valid": True,
                "strict_format_valid": False,
                "rescue_applied": True,
                "rescue_kind": rescue_kind,
                "answer": payload,
                "reason": "",
            }
        except ValueError as rescue_exc:
            return {
                "format_valid": False,
                "strict_format_valid": False,
                "rescue_applied": False,
                "rescue_kind": None,
                "answer": None,
                "reason": str(rescue_exc or strict_exc),
            }


def task_objective_text(task: Mapping[str, Any]) -> str:
    family = str(task["family"])
    lines = [f"Task family: {family}", str(task["question_text"])]
    if family == "2A.1":
        lines.append(f"Card {task['card']['card_id']}: {task['card']['text']}")
    elif family in {"2A.2", "2A.3"}:
        lines.append(f"Target {task['target_id']}: {task['target_text']}")
        if task.get("scene_card"):
            lines.append(f"Context card {task['scene_card']['card_id']}: {task['scene_card']['text']}")
    elif family == "2B.1":
        lines.append("Cards:\n" + "\n".join(f"{card['card_id']}: {card['text']}" for card in task["cards"]))
    elif family == "2B.2":
        lines.append("Scene cards:\n" + "\n".join(f"{card['card_id']}: {card['text']}" for card in task["scene_cards"]))
        lines.append(
            "Utterance targets:\n"
            + "\n".join(f"{target['utterance_id']}: {target['text']}" for target in task["utterance_targets"])
        )
    return "\n".join(lines)


def action_grammar_text(task: Mapping[str, Any]) -> str:
    found = "SET <target_id> FOUND <start_seconds> <end_seconds>"
    if task["family"] == "2B.2":
        found += " <scene_card_id>"
    return (
        "Return plain text using exactly this grammar:\n"
        f"{found}\n"
        "SET <target_id> NOT_FOUND_YET\n"
        "SET <target_id> LIKELY_ABSENT\n"
        "NO_UPDATE\n"
        "ACTION QUERY <start_seconds> <end_seconds>\n"
        "ACTION STOP\n"
        "Use zero or more SET lines, or NO_UPDATE when there are no SET lines, followed by exactly one ACTION line. "
        "Choose QUERY or STOP; never return both. The ACTION must be the final line. "
        "Do not add explanations, Markdown, role labels, or another turn."
    )


def ledger_snapshot_grammar_text(task: Mapping[str, Any]) -> str:
    found = "<target_id> FOUND <start_seconds> <end_seconds>"
    if task["family"] == "2B.2":
        found += " <scene_card_id>"
    return (
        "Return plain text with exactly one line per target using these forms:\n"
        f"{found}\n"
        "<target_id> NOT_FOUND_YET\n"
        "<target_id> LIKELY_ABSENT"
    )


def parse_track2_probe_response(raw: str, task: Mapping[str, Any]) -> dict[str, Any]:
    lines = [line.strip() for line in str(raw).splitlines() if line.strip()]
    if not lines:
        return {"format_valid": False, "reason": "empty_response"}
    if any(
        line.upper() == "NO_UPDATE"
        or line.upper().startswith("SET ")
        or line.upper().startswith("ACTION ")
        for line in lines
    ):
        return {"format_valid": False, "reason": "probe_requires_bare_snapshot_lines"}
    parsed = parse_track2_action_response(
        "\n".join(f"SET {line}" for line in lines) + "\nACTION STOP"
    )
    if not parsed.get("format_valid"):
        return parsed
    ids = [str(update["target_id"]) for update in parsed["updates"]]
    if len(ids) != len(set(ids)) or set(ids) != set(track2_target_ids(task)):
        return {
            "format_valid": False,
            "reason": "probe_snapshot_must_include_every_target_once",
        }
    return parsed


def final_schema_text(task: Mapping[str, Any]) -> str:
    family = str(task["family"])
    if family == "2A.1":
        card_id = str(task["card"]["card_id"])
        return (
            "Required keys: status, card_id, scene_rank, start_seconds, end_seconds, confidence.\n"
            f'card_id must be "{card_id}". status must be exactly "PRESENT" or "ABSENT". '
            "For PRESENT, scene_rank is a positive integer and both endpoints are numbers. "
            "For ABSENT, scene_rank and both endpoints are null. confidence is a number from 0 to 1."
        )
    if family == "2A.2":
        return (
            "Required keys: start_seconds, end_seconds, confidence. Both endpoints are numbers, "
            "end_seconds is greater than start_seconds, and confidence is a number from 0 to 1. "
            'An optional status key, if included, must be exactly "PRESENT".'
        )
    if family == "2A.3":
        return (
            "Required keys: status, occurrences, confidence. status must be exactly \"PRESENT\" or \"ABSENT\". "
            "occurrences is an array of objects, each with numeric start_seconds and end_seconds. "
            "It must be nonempty for PRESENT and empty for ABSENT. confidence is a number from 0 to 1."
        )
    if family == "2B.1":
        valid_ids = ", ".join(track2_target_ids(task))
        return (
            "Required keys: ordered_card_ids, absent_card_id. ordered_card_ids is an array containing "
            f"the present card IDs in chronological order. Valid card IDs: {valid_ids}. "
            "absent_card_id is the one absent card ID, or null when none is absent."
        )
    target_ids = ", ".join(track2_target_ids(task))
    card_ids = ", ".join(sorted(track2_scene_card_ids(task)))
    return (
        "Required keys: ordered_real_utterance_ids, utterances, confidence_complete. "
        "utterances must contain exactly one object for every target; each object has utterance_id, status, "
        "start_seconds, end_seconds, and scene_card_id. status must be exactly \"PRESENT\" or \"ABSENT\". "
        "For PRESENT, endpoints are numbers and scene_card_id is valid; for ABSENT, all three are null. "
        "ordered_real_utterance_ids contains exactly the PRESENT target IDs in chronological order. "
        f"Target IDs: {target_ids}. Scene card IDs: {card_ids}. "
        "confidence_complete is a number from 0 to 1."
    )


def build_track2_action_prompt(
    task: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    *,
    reel_duration_seconds: float,
    max_queries: int,
    completed_queries: int,
    listening_budget_seconds: float,
    listened_seconds: float,
    query_history: Sequence[Mapping[str, Any]],
    probe_history: Sequence[Mapping[str, Any]],
    current_crop: Mapping[str, Any] | None,
) -> str:
    history = "\n".join(
        f"step {item['step_index']}: {float(item['start_seconds']):.3f}-{float(item['end_seconds']):.3f}"
        for item in query_history
    ) or "none"
    crop = (
        "No episode audio is attached. Choose the first query."
        if current_crop is None
        else (
            "The attached audio is only the current crop and corresponds to absolute reel interval "
            f"{float(current_crop['start_seconds']):.3f}-{float(current_crop['end_seconds']):.3f} seconds."
        )
    )
    probes = "\n".join(str(item.get("canonical_ledger", "")) for item in probe_history) or "none"
    return (
        f"{task_objective_text(task)}\n\n"
        f"Reel duration: {float(reel_duration_seconds):.3f} seconds.\n"
        f"Remaining queries: {max_queries - completed_queries}.\n"
        f"Remaining listening budget: {listening_budget_seconds - listened_seconds:.3f} seconds.\n"
        f"Maximum one-query duration: 384.000 seconds.\n"
        f"Queried intervals:\n{history}\n\n"
        f"CURRENT BELIEF\n{canonical_ledger_text(ledger)}\n\n"
        f"Recent periodic snapshots:\n{probes}\n\n"
        f"{crop}\n\n{action_grammar_text(task)}"
    )


def build_track2_probe_prompt(task: Mapping[str, Any], ledger: Mapping[str, Mapping[str, Any]]) -> str:
    return (
        f"{task_objective_text(task)}\n\n"
        f"Current canonical ledger:\n{canonical_ledger_text(ledger)}\n\n"
        f"{ledger_snapshot_grammar_text(task)}\n"
        "Do not return an ACTION and do not add explanations."
    )


def build_track2_final_prompt(
    task: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]] | None,
    *,
    reel_duration_seconds: float,
    full_context: bool,
) -> str:
    evidence = "The complete reel is attached." if full_context else "No audio is attached; answer from the acquired evidence and canonical ledger."
    ledger_text = "" if ledger is None else f"\nFinal canonical ledger:\n{canonical_ledger_text(ledger)}\n"
    return (
        f"{task_objective_text(task)}\n\nReel duration: {float(reel_duration_seconds):.3f} seconds.\n"
        f"{evidence}{ledger_text}\nReturn exactly one JSON object satisfying this contract:\n"
        f"{final_schema_text(task)}\nThe first character must be {{ and the last character must be }}. "
        "Do not use Markdown fences, explanations, or role labels."
    )


def build_track2_repair_prompt(raw: str, grammar: str, reason: str) -> str:
    return (
        "Convert the response below to the required syntax without changing its semantic claims. "
        "Do not reconsider the task and do not add evidence. Return only the converted response, with no "
        "Markdown fences, explanation, preamble, role label, or subsequent turn. If JSON is required, the "
        "first character must be { and the last character must be }.\n"
        f"Parse failure: {reason}\nRequired syntax:\n{grammar}\nOriginal response:\n{raw}"
    )


__all__ = [
    "TRACK2_FINAL_STATUSES",
    "TRACK2_LEDGER_STATUSES",
    "action_grammar_text",
    "build_track2_action_prompt",
    "build_track2_final_prompt",
    "build_track2_probe_prompt",
    "build_track2_repair_prompt",
    "canonical_ledger_text",
    "final_schema_text",
    "initialize_track2_ledger",
    "ledger_snapshot_grammar_text",
    "parse_track2_action_response",
    "parse_track2_final_answer",
    "parse_track2_probe_response",
    "task_objective_text",
    "track2_scene_card_ids",
    "track2_target_ids",
    "validate_and_merge_track2_updates",
    "validate_track2_query",
]
