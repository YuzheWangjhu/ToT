from __future__ import annotations

import concurrent.futures
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import soundfile as sf

from .adapters import (
    API_ERROR_HARD,
    APIAdapterError,
    BaseAdapter,
    UnsupportedAdapterCapability,
)
from .core import (
    API_HARD_MAX_ATTEMPTS,
    API_RETRY_MAX_SLEEP_SECONDS,
    API_SOFT_MAX_RETRIES,
    ProgressReporter,
)
from .track2_protocol import (
    TRACK2_FINAL_STATUSES,
    action_grammar_text,
    build_track2_action_prompt,
    build_track2_final_prompt,
    build_track2_probe_prompt,
    build_track2_repair_prompt,
    canonical_ledger_text,
    final_schema_text,
    initialize_track2_ledger,
    ledger_snapshot_grammar_text,
    parse_track2_action_response,
    parse_track2_final_answer,
    parse_track2_probe_response,
    track2_scene_card_ids,
    track2_target_ids,
    validate_and_merge_track2_updates,
    validate_track2_query,
)
from .track2_scoring import (
    aggregate_track2_results,
    formatting_metrics,
    score_track2_answer,
    score_track2_belief,
    score_track2_evidence,
)


TRACK2 = "2"
TRACK2_CONDITIONS = ["full_context", "active", "active_same_session_periodic_probe"]
TRACK2_DEFAULT_CONDITIONS = ["full_context", "active"]
TRACK2_FAMILIES = ["2A.1", "2A.2", "2A.3", "2B.1", "2B.2"]
TRACK2_DATASET_ID = "track2_role_ipc_test_v2"
TRACK2_DATASET_FINGERPRINT = "0bffa118888b587758506c0a553c0221035d7619adbf88fd73992036bdc45819"
TRACK2_MAX_SINGLE_QUERY_SECONDS = 384.0
TRACK2_CROP_SAMPLE_RATE = 16000
TRACK2_RESULT_SCHEMA_VERSION = 5
TRACK2_SYSTEM_PROMPT = (
    "You are participating in a long-horizon audio navigation benchmark. "
    "Follow the requested output grammar exactly and do not add explanations."
)


@dataclass(frozen=True)
class Track2Budget:
    listening_budget_seconds: float
    max_queries: int


TRACK2_BUDGETS: dict[tuple[str, str | None], Track2Budget] = {
    ("2A.1", "present"): Track2Budget(768.0, 6),
    ("2A.1", "absent"): Track2Budget(1200.0, 10),
    ("2A.2", None): Track2Budget(768.0, 8),
    ("2A.3", None): Track2Budget(1200.0, 10),
    ("2B.1", None): Track2Budget(1200.0, 10),
    ("2B.2", None): Track2Budget(1584.0, 12),
}


@dataclass(frozen=True)
class Track2Config:
    max_single_query_seconds: float = TRACK2_MAX_SINGLE_QUERY_SECONDS
    probe_every_n: int = 2
    action_output_tokens: int = 512
    final_output_tokens: int = 1024
    api_retry_sleep_seconds: float = 60.0


def track2_budget_for_task(task: Mapping[str, Any]) -> Track2Budget:
    family = str(task["family"])
    exact = TRACK2_BUDGETS.get((family, str(task["variant"])))
    if exact is not None:
        return exact
    return TRACK2_BUDGETS[(family, None)]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}:{line_number}: expected JSON object")
        records.append(value)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_track2_dataset(dataset_root: Path) -> dict[str, Any]:
    root = Path(dataset_root)
    dataset_dir = root / "dataset"
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    validation_report = json.loads(
        (dataset_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    if manifest.get("dataset_id") != TRACK2_DATASET_ID or str(manifest.get("track_id")) != TRACK2:
        raise RuntimeError("Expected frozen track2_role_ipc_test_v2 dataset.")
    if float(manifest.get("max_single_query_seconds", -1)) != TRACK2_MAX_SINGLE_QUERY_SECONDS:
        raise RuntimeError("Track 2 manifest must freeze max_single_query_seconds at 384.")
    if (
        validation_report.get("status") != "passed"
        or validation_report.get("dataset_id") != TRACK2_DATASET_ID
        or validation_report.get("deterministic_build_fingerprint") != TRACK2_DATASET_FINGERPRINT
    ):
        raise RuntimeError("Track 2 validation report does not match the frozen v2 fingerprint.")
    expected_hashes = validation_report.get("artifact_sha256")
    if not isinstance(expected_hashes, dict) or not expected_hashes:
        raise RuntimeError("Track 2 validation report is missing frozen artifact hashes.")
    actual_hashes: dict[str, str] = {}
    resolved_dataset_dir = dataset_dir.resolve()
    for relative_path, expected_hash in expected_hashes.items():
        artifact_path = (dataset_dir / str(relative_path)).resolve()
        if not artifact_path.is_relative_to(resolved_dataset_dir) or not artifact_path.is_file():
            raise RuntimeError(f"Invalid or missing Track 2 artifact: {relative_path}")
        actual_hashes[str(relative_path)] = _sha256_file(artifact_path)
        if actual_hashes[str(relative_path)] != str(expected_hash):
            raise RuntimeError(f"Track 2 artifact hash mismatch: {relative_path}")
    actual_fingerprint = hashlib.sha256(
        json.dumps(actual_hashes, sort_keys=True).encode()
    ).hexdigest()
    if actual_fingerprint != TRACK2_DATASET_FINGERPRINT:
        raise RuntimeError("Track 2 artifacts do not match the frozen v2 fingerprint.")
    tasks = _read_jsonl(dataset_dir / "task_instances.jsonl")
    answer_records = _read_jsonl(dataset_dir / "task_answers.jsonl")
    answers = {str(record["task_id"]): record["answer"] for record in answer_records}
    if len(answers) != len(answer_records):
        raise RuntimeError("Duplicate task IDs in Track 2 answers.")
    task_ids = [str(task["task_id"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)) or set(task_ids) != set(answers):
        raise RuntimeError("Track 2 public tasks and gold answers do not join one-to-one.")
    if len(tasks) != int(manifest["task_count"]):
        raise RuntimeError("Track 2 task count does not match manifest.")
    episodes = {str(episode["reel_id"]): episode for episode in manifest["episodes"]}
    if len(episodes) != int(manifest["episode_count"]):
        raise RuntimeError("Track 2 episode count does not match manifest.")
    for task in tasks:
        if task.get("family") not in TRACK2_FAMILIES:
            raise RuntimeError(f"Unsupported Track 2 family in dataset: {task.get('family')}")
        reel_id = str(task["reel_id"])
        if reel_id not in episodes:
            raise RuntimeError(f"Task references unknown reel: {reel_id}")
        if str(task["audio_path"]) != str(episodes[reel_id]["audio_path"]):
            raise RuntimeError(f"Task audio path differs from manifest: {task['task_id']}")
        if any("speaker" in key.lower() for key in task):
            raise RuntimeError(f"Track 2 public task exposes speaker data: {task['task_id']}")
    return {
        "root": root,
        "dataset_dir": dataset_dir,
        "manifest": manifest,
        "tasks": tasks,
        "answers": answers,
        "episodes": episodes,
    }


def _result_path(results_root: Path, task: Mapping[str, Any], condition: str) -> Path:
    return (
        Path(results_root)
        / condition
        / str(task["family"])
        / str(task["variant"])
        / f"{task['task_id']}.json"
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _send_track2_with_retry(
    adapter: BaseAdapter,
    prompt: str,
    audio_paths: Sequence[Path],
    *,
    max_output_tokens: int,
    strict_text_only: bool,
    config: Track2Config,
) -> tuple[str | None, list[dict[str, Any]], bool]:
    attempts: list[dict[str, Any]] = []
    hard_attempts = 0
    soft_retries = 0
    attempt_index = 0
    while True:
        attempt_index += 1
        session = adapter.create_session(TRACK2_SYSTEM_PROMPT)
        try:
            raw = adapter.send_track2(
                session,
                prompt,
                audio_paths,
                max_output_tokens=max_output_tokens,
                strict_text_only=strict_text_only,
            )
            attempts.append({"attempt": attempt_index, "status": "success"})
            return raw, attempts, False
        except UnsupportedAdapterCapability as exc:
            attempts.append({"attempt": attempt_index, "status": "unsupported", "error": str(exc)})
            return None, attempts, True
        except APIAdapterError as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "status": "error",
                    "error": str(exc),
                    "error_class": exc.error_class,
                    "status_code": exc.status_code,
                }
            )
            if exc.error_class == API_ERROR_HARD:
                hard_attempts += 1
                if hard_attempts >= API_HARD_MAX_ATTEMPTS:
                    return None, attempts, False
                continue
            if soft_retries >= API_SOFT_MAX_RETRIES:
                return None, attempts, False
            soft_retries += 1
            delay = exc.retry_after_seconds if exc.retry_after_seconds is not None else config.api_retry_sleep_seconds
            time.sleep(min(max(0.0, float(delay)), API_RETRY_MAX_SLEEP_SECONDS))
        finally:
            adapter.close_session(session)


def _parse_with_repair(
    adapter: BaseAdapter,
    raw: str,
    parse_fn: Callable[[str], dict[str, Any]],
    grammar: str,
    *,
    max_output_tokens: int,
    config: Track2Config,
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = parse_fn(raw)
    record: dict[str, Any] = {
        "raw_output": raw,
        "raw_parse_valid": bool(
            parsed.get("strict_format_valid", parsed.get("format_valid"))
        ),
        "raw_rescue_applied": bool(parsed.get("rescue_applied")),
        "raw_rescue_kind": parsed.get("rescue_kind"),
        "repair_attempted": False,
        "repair_unsupported": False,
        "final_output": raw,
        "final_parse_valid": bool(parsed.get("format_valid")),
        "final_rescue_applied": bool(parsed.get("rescue_applied")),
        "final_rescue_kind": parsed.get("rescue_kind"),
        "reason": parsed.get("reason", ""),
        "api_attempts": [],
        "unrecovered_api_error": False,
    }
    if parsed.get("format_valid"):
        return parsed, record
    record["repair_attempted"] = True
    repaired, attempts, unsupported = _send_track2_with_retry(
        adapter,
        build_track2_repair_prompt(raw, grammar, str(parsed.get("reason", "parse_failure"))),
        [],
        max_output_tokens=max_output_tokens,
        strict_text_only=True,
        config=config,
    )
    record["api_attempts"] = attempts
    record["repair_unsupported"] = unsupported
    if repaired is None:
        record["unrecovered_api_error"] = not unsupported
        return parsed, record
    repaired_parse = parse_fn(repaired)
    record["final_output"] = repaired
    record["final_parse_valid"] = bool(repaired_parse.get("format_valid"))
    record["final_rescue_applied"] = bool(repaired_parse.get("rescue_applied"))
    record["final_rescue_kind"] = repaired_parse.get("rescue_kind")
    record["reason"] = repaired_parse.get("reason", "")
    return repaired_parse, record


def _capability_failure(
    adapter: BaseAdapter,
    condition: str,
    audio_path: Path,
    reel_duration_seconds: float,
    config: Track2Config,
) -> str | None:
    capabilities = adapter.track2_capabilities
    if condition not in capabilities.supported_conditions:
        return "condition_not_declared_supported"
    if not capabilities.supports_output_token_override:
        return "output_token_override_not_declared_supported"
    if capabilities.max_audio_seconds is None or capabilities.max_audio_bytes is None:
        return "audio_capabilities_unknown"
    required_seconds = reel_duration_seconds if condition == "full_context" else config.max_single_query_seconds
    if capabilities.max_audio_seconds + 1e-9 < required_seconds:
        return "audio_duration_capability_too_small"
    if condition != "full_context" and not capabilities.supports_text_only:
        return "strict_text_only_not_supported"
    if condition == "full_context":
        required_bytes = audio_path.stat().st_size
    else:
        required_bytes = int(
            math.ceil(
                config.max_single_query_seconds * TRACK2_CROP_SAMPLE_RATE * 2 + 44
            )
        )
    if capabilities.max_audio_bytes < required_bytes:
        return "audio_byte_capability_too_small"
    return None


def _unsupported_result(
    task: Mapping[str, Any], condition: str, adapter: BaseAdapter, reason: str
) -> dict[str, Any]:
    return {
        "track2_result_schema_version": TRACK2_RESULT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "reel_id": task["reel_id"],
        "family": task["family"],
        "variant": task["variant"],
        "condition": condition,
        "model_id": adapter.model_id,
        "run_status": "unsupported",
        "unsupported_reason": reason,
        "answer_metrics": {},
        "evidence_metrics": {},
        "belief_metrics": {},
        "format_metrics": {},
        "confidence": None,
    }


def _api_error_result(
    task: Mapping[str, Any],
    condition: str,
    adapter: BaseAdapter,
    attempts: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    prior_errors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "track2_result_schema_version": TRACK2_RESULT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "reel_id": task["reel_id"],
        "family": task["family"],
        "variant": task["variant"],
        "condition": condition,
        "model_id": adapter.model_id,
        "run_status": "api_error",
        "api_attempts": list(attempts),
        "unrecovered_api_errors": [
            *[dict(error) for error in prior_errors],
            {"stage": stage, "api_attempts": list(attempts)},
        ],
        "answer_metrics": {},
        "evidence_metrics": {},
        "belief_metrics": {},
        "format_metrics": {},
        "confidence": None,
    }


def _mark_unscored_api_error(
    result: dict[str, Any], errors: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    result["run_status"] = "api_error"
    result["unrecovered_api_errors"] = [dict(error) for error in errors]
    if "final_answer" in result:
        result["diagnostic_final_answer"] = result.pop("final_answer")
    if "parsed_final_answer" in result:
        result["diagnostic_parsed_final_answer"] = result.pop("parsed_final_answer")
    if "final_answer_semantic_valid" in result:
        result["diagnostic_final_answer_semantic_valid"] = result.pop(
            "final_answer_semantic_valid"
        )
    result["answer_metrics"] = {}
    result["evidence_metrics"] = {}
    result["belief_metrics"] = {}
    result["format_metrics"] = {}
    result["confidence"] = None
    return result


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name}_must_be_finite_number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name}_must_be_finite_number")
    return result


def _confidence(answer: Mapping[str, Any] | None, family: str) -> float | None:
    if answer is None:
        return None
    key = "confidence_complete" if family == "2B.2" else "confidence"
    if family not in {"2A.1", "2A.2", "2A.3", "2B.2"}:
        return None
    try:
        value = _finite_number(answer.get(key), key)
    except ValueError:
        return None
    return value if 0.0 <= value <= 1.0 else None


def _validate_final_answer_semantics(
    task: Mapping[str, Any], answer: Mapping[str, Any], reel_duration_seconds: float
) -> tuple[bool, str, dict[str, Any] | None]:
    def interval(start_value: Any, end_value: Any) -> tuple[float, float]:
        start = _finite_number(start_value, "start_seconds")
        end = _finite_number(end_value, "end_seconds")
        if not 0.0 <= start < end <= float(reel_duration_seconds):
            raise ValueError("final_interval_out_of_range")
        return start, end

    def confidence(value: Any, name: str) -> float:
        result = _finite_number(value, name)
        if not 0.0 <= result <= 1.0:
            raise ValueError(f"{name}_out_of_range")
        return result

    try:
        family = str(task["family"])
        if family == "2A.1":
            status = str(answer.get("status", "")).upper()
            if status not in TRACK2_FINAL_STATUSES:
                raise ValueError("invalid_status")
            if str(answer.get("card_id")) != str(task["card"]["card_id"]):
                raise ValueError("invalid_card_id")
            normalized = {
                "status": status,
                "card_id": str(answer.get("card_id")),
                "scene_rank": answer.get("scene_rank"),
                "start_seconds": answer.get("start_seconds"),
                "end_seconds": answer.get("end_seconds"),
                "confidence": confidence(answer.get("confidence"), "confidence"),
            }
            if status == "PRESENT":
                if (
                    isinstance(normalized["scene_rank"], bool)
                    or not isinstance(normalized["scene_rank"], int)
                    or normalized["scene_rank"] < 1
                ):
                    raise ValueError("invalid_scene_rank")
                start, end = interval(normalized["start_seconds"], normalized["end_seconds"])
                normalized["start_seconds"], normalized["end_seconds"] = start, end
            elif any(
                normalized[key] is not None
                for key in ("scene_rank", "start_seconds", "end_seconds")
            ):
                raise ValueError("absent_scene_fields_must_be_null")
        elif family == "2A.2":
            status = str(answer.get("status", "PRESENT")).upper()
            if status != "PRESENT":
                raise ValueError("invalid_present_status")
            start, end = interval(answer.get("start_seconds"), answer.get("end_seconds"))
            normalized = {
                "status": status,
                "start_seconds": start,
                "end_seconds": end,
                "confidence": confidence(answer.get("confidence"), "confidence"),
            }
        elif family == "2A.3":
            status = str(answer.get("status", "")).upper()
            occurrences = answer.get("occurrences")
            if status not in TRACK2_FINAL_STATUSES:
                raise ValueError("invalid_status")
            if not isinstance(occurrences, list):
                raise ValueError("occurrences_must_be_array")
            normalized_occurrences: list[dict[str, float]] = []
            for occurrence in occurrences:
                if not isinstance(occurrence, Mapping):
                    raise ValueError("occurrence_must_be_object")
                start, end = interval(
                    occurrence.get("start_seconds"), occurrence.get("end_seconds")
                )
                normalized_occurrences.append(
                    {"start_seconds": start, "end_seconds": end}
                )
            if status == "ABSENT" and normalized_occurrences:
                raise ValueError("absent_occurrences_must_be_empty")
            if status == "PRESENT" and not normalized_occurrences:
                raise ValueError("present_occurrences_must_be_nonempty")
            normalized = {
                "status": status,
                "occurrences": normalized_occurrences,
                "confidence": confidence(answer.get("confidence"), "confidence"),
            }
        elif family == "2B.1":
            ordered_value = answer.get("ordered_card_ids")
            if not isinstance(ordered_value, list):
                raise ValueError("ordered_card_ids_must_be_array")
            ordered = [str(item) for item in ordered_value]
            valid_ids = set(track2_target_ids(task))
            if any(item not in valid_ids for item in ordered):
                raise ValueError("invalid_ordered_card_ids")
            if len(ordered) != len(set(ordered)):
                raise ValueError("duplicate_ordered_card_ids")
            absent_value = answer.get("absent_card_id")
            absent = None if absent_value is None else str(absent_value)
            if absent is not None and absent not in valid_ids:
                raise ValueError("invalid_absent_card_id")
            if task["variant"] == "exact_count" and (
                absent is not None or len(ordered) != len(task["cards"])
            ):
                raise ValueError("exact_count_requires_every_card_and_no_absent_id")
            if task["variant"] == "extra_count" and (
                absent is None
                or absent in ordered
                or len(ordered) != len(task["cards"]) - 1
            ):
                raise ValueError("extra_count_requires_one_omitted_absent_card")
            normalized = {"ordered_card_ids": ordered, "absent_card_id": absent}
        elif family == "2B.2":
            ordered_value = answer.get("ordered_real_utterance_ids")
            items = answer.get("utterances")
            if not isinstance(ordered_value, list) or not isinstance(items, list):
                raise ValueError("invalid_structural_answer")
            valid_targets = set(track2_target_ids(task))
            valid_cards = track2_scene_card_ids(task)
            normalized_items: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    raise ValueError("utterance_must_be_object")
                target_id = str(item.get("utterance_id"))
                status = str(item.get("status", "")).upper()
                if target_id not in valid_targets or target_id in seen:
                    raise ValueError("invalid_utterance_id")
                if status not in TRACK2_FINAL_STATUSES:
                    raise ValueError("invalid_utterance_status")
                seen.add(target_id)
                normalized_item: dict[str, Any] = {
                    "utterance_id": target_id,
                    "status": status,
                }
                if status == "PRESENT":
                    start, end = interval(
                        item.get("start_seconds"), item.get("end_seconds")
                    )
                    scene_card_id = str(item.get("scene_card_id"))
                    if scene_card_id not in valid_cards:
                        raise ValueError("invalid_scene_card_id")
                    normalized_item.update(
                        {
                            "start_seconds": start,
                            "end_seconds": end,
                            "scene_card_id": scene_card_id,
                        }
                    )
                else:
                    if any(
                        item.get(key) is not None
                        for key in ("start_seconds", "end_seconds", "scene_card_id")
                    ):
                        raise ValueError("absent_utterance_fields_must_be_null")
                    normalized_item.update(
                        {
                            "start_seconds": None,
                            "end_seconds": None,
                            "scene_card_id": None,
                        }
                    )
                normalized_items.append(normalized_item)
            if seen != valid_targets:
                raise ValueError("utterance_entries_incomplete")
            if sum(item["status"] == "ABSENT" for item in normalized_items) != 1:
                raise ValueError("2B.2_requires_exactly_one_absent_target")
            ordered = [str(item) for item in ordered_value]
            present_ids = {
                item["utterance_id"]
                for item in normalized_items
                if item["status"] == "PRESENT"
            }
            if len(ordered) != len(set(ordered)) or set(ordered) != present_ids:
                raise ValueError("ordered_ids_do_not_match_present_targets")
            normalized = {
                "ordered_real_utterance_ids": ordered,
                "utterances": normalized_items,
                "confidence_complete": confidence(
                    answer.get("confidence_complete"), "confidence_complete"
                ),
            }
        else:
            raise ValueError(f"unsupported_family_{family}")
        return True, "", normalized
    except (KeyError, TypeError, ValueError) as exc:
        return False, str(exc), None


def _base_complete_result(
    task: Mapping[str, Any],
    condition: str,
    adapter: BaseAdapter,
    answer_parse: Mapping[str, Any],
    gold: Mapping[str, Any],
    episode: Mapping[str, Any],
    format_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parsed_answer = answer_parse.get("answer") if answer_parse.get("format_valid") else None
    semantic_valid, semantic_reason, normalized_answer = (
        _validate_final_answer_semantics(
            task, parsed_answer, float(episode["duration_seconds"])
        )
        if parsed_answer is not None
        else (False, "final_answer_parse_invalid", None)
    )
    answer = normalized_answer if semantic_valid else None
    if format_records:
        format_records[-1]["semantic_valid"] = semantic_valid
        format_records[-1]["semantic_reason"] = semantic_reason
    answer_metrics = score_track2_answer(task, gold, answer, episode)
    if parsed_answer is not None and not semantic_valid:
        try:
            parsed_metrics = score_track2_answer(task, gold, parsed_answer, episode)
        except (KeyError, TypeError, ValueError):
            parsed_metrics = {}
        if "false_localization_rate" in parsed_metrics:
            answer_metrics["false_localization_rate"] = parsed_metrics["false_localization_rate"]
    return {
        "track2_result_schema_version": TRACK2_RESULT_SCHEMA_VERSION,
        "task_id": task["task_id"],
        "reel_id": task["reel_id"],
        "family": task["family"],
        "variant": task["variant"],
        "condition": condition,
        "model_id": adapter.model_id,
        "run_status": "complete",
        "unrecovered_api_errors": [],
        "final_answer": answer,
        "parsed_final_answer": parsed_answer,
        "final_answer_semantic_valid": semantic_valid,
        "answer_metrics": answer_metrics,
        "format_metrics": formatting_metrics(format_records),
        "format_records": list(format_records),
        "confidence": _confidence(parsed_answer, str(task["family"])),
    }


def _run_full_context(
    dataset: Mapping[str, Any],
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    episode: Mapping[str, Any],
    adapter: BaseAdapter,
    config: Track2Config,
) -> dict[str, Any]:
    audio_path = Path(dataset["dataset_dir"]) / str(task["audio_path"])
    prompt = build_track2_final_prompt(
        task, None, reel_duration_seconds=float(episode["duration_seconds"]), full_context=True
    )
    raw, api_attempts, unsupported = _send_track2_with_retry(
        adapter,
        prompt,
        [audio_path],
        max_output_tokens=config.final_output_tokens,
        strict_text_only=False,
        config=config,
    )
    if unsupported:
        return _unsupported_result(task, "full_context", adapter, "full_context_call_unsupported")
    if raw is None:
        return _api_error_result(
            task,
            "full_context",
            adapter,
            api_attempts,
            stage="full_context_final",
        )
    parse_fn = lambda value: parse_track2_final_answer(value, task)
    parsed, format_record = _parse_with_repair(
        adapter,
        raw,
        parse_fn,
        final_schema_text(task),
        max_output_tokens=config.final_output_tokens,
        config=config,
    )
    format_record["api_attempts"] = [*api_attempts, *format_record.get("api_attempts", [])]
    result = _base_complete_result(
        task, "full_context", adapter, parsed, gold, episode, [format_record]
    )
    result.update(
        {
            "stop_reason": "full_context",
            "query_history": [],
            "evidence_metrics": {
                "full_context_duration_seconds": float(episode["duration_seconds"]),
                "full_context_reference": 1.0,
            },
            "belief_metrics": {},
        }
    )
    if format_record.get("unrecovered_api_error"):
        return _mark_unscored_api_error(
            result,
            [
                {
                    "stage": "full_context_final_repair",
                    "api_attempts": list(format_record.get("api_attempts", [])),
                }
            ],
        )
    return result


def _write_query_crop(
    audio_path: Path,
    runtime_root: Path,
    task_id: str,
    step_index: int,
    start_seconds: float,
    end_seconds: float,
) -> Path:
    info = sf.info(str(audio_path))
    start_frame = int(round(float(start_seconds) * info.samplerate))
    frames = int(round((float(end_seconds) - float(start_seconds)) * info.samplerate))
    audio, samplerate = sf.read(
        str(audio_path), start=start_frame, frames=frames, dtype="float32", always_2d=False
    )
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = audio.mean(axis=1)
    if samplerate != TRACK2_CROP_SAMPLE_RATE:
        try:
            from scipy.signal import resample_poly
        except Exception as exc:
            raise RuntimeError(
                "Track 2 crop generation requires scipy for 16 kHz resampling."
            ) from exc
        divisor = math.gcd(int(samplerate), TRACK2_CROP_SAMPLE_RATE)
        audio = resample_poly(
            np.asarray(audio, dtype=np.float32),
            TRACK2_CROP_SAMPLE_RATE // divisor,
            int(samplerate) // divisor,
        ).astype(np.float32, copy=False)
        samplerate = TRACK2_CROP_SAMPLE_RATE
    crop_path = runtime_root / f"{task_id}_step_{step_index:02d}.wav"
    crop_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(crop_path), np.asarray(audio, dtype=np.float32), samplerate, subtype="PCM_16")
    return crop_path


def _run_probe(
    adapter: BaseAdapter,
    task: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    episode: Mapping[str, Any],
    step_index: int,
    config: Track2Config,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    raw, attempts, unsupported = _send_track2_with_retry(
        adapter,
        build_track2_probe_prompt(task, ledger),
        [],
        max_output_tokens=config.action_output_tokens,
        strict_text_only=True,
        config=config,
    )
    if raw is None:
        record = {
            "raw_parse_valid": False,
            "final_parse_valid": False,
            "repair_attempted": False,
            "repair_unsupported": unsupported,
            "semantic_valid": False,
            "api_attempts": attempts,
            "unrecovered_api_error": not unsupported,
            "api_failure_stage": "probe",
        }
        return dict(ledger), {"step_index": step_index, "canonical_ledger": canonical_ledger_text(ledger)}, record
    parsed, record = _parse_with_repair(
        adapter,
        raw,
        lambda value: parse_track2_probe_response(value, task),
        ledger_snapshot_grammar_text(task),
        max_output_tokens=config.action_output_tokens,
        config=config,
    )
    record["api_attempts"] = [*attempts, *record.get("api_attempts", [])]
    if record.get("unrecovered_api_error"):
        record["api_failure_stage"] = "probe_repair"
    if not parsed.get("format_valid"):
        record["semantic_valid"] = False
        return dict(ledger), {"step_index": step_index, "canonical_ledger": canonical_ledger_text(ledger)}, record
    merged, outcomes = validate_and_merge_track2_updates(
        task,
        ledger,
        parsed["updates"],
        reel_duration_seconds=float(episode["duration_seconds"]),
        step_index=step_index,
    )
    for outcome in outcomes:
        outcome["step_index"] = step_index
        outcome["source"] = "periodic_probe"
    record["semantic_valid"] = all(outcome["valid"] for outcome in outcomes)
    return merged, {
        "step_index": step_index,
        "canonical_ledger": canonical_ledger_text(merged),
        "update_outcomes": outcomes,
    }, record


def _run_active(
    dataset: Mapping[str, Any],
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    episode: Mapping[str, Any],
    adapter: BaseAdapter,
    config: Track2Config,
    results_root: Path,
    *,
    periodic_probe: bool,
) -> dict[str, Any]:
    condition = "active_same_session_periodic_probe" if periodic_probe else "active"
    budget = track2_budget_for_task(task)
    ledger = initialize_track2_ledger(task)
    query_history: list[dict[str, Any]] = []
    update_history: list[dict[str, Any]] = []
    probe_history: list[dict[str, Any]] = []
    format_records: list[dict[str, Any]] = []
    unrecovered_api_errors: list[dict[str, Any]] = []
    listened_seconds = 0.0
    stop_reason = ""
    current_crop: dict[str, Any] | None = None
    current_crop_path: Path | None = None
    audio_path = Path(dataset["dataset_dir"]) / str(task["audio_path"])
    runtime_root = Path(results_root) / "runtime" / condition / str(task["task_id"])

    while True:
        prompt = build_track2_action_prompt(
            task,
            ledger,
            reel_duration_seconds=float(episode["duration_seconds"]),
            max_queries=budget.max_queries,
            completed_queries=len(query_history),
            listening_budget_seconds=budget.listening_budget_seconds,
            listened_seconds=listened_seconds,
            query_history=query_history,
            probe_history=probe_history,
            current_crop=current_crop,
        )
        raw, attempts, unsupported = _send_track2_with_retry(
            adapter,
            prompt,
            [] if current_crop_path is None else [current_crop_path],
            max_output_tokens=config.action_output_tokens,
            strict_text_only=current_crop_path is None,
            config=config,
        )
        if unsupported:
            return _unsupported_result(task, condition, adapter, "active_call_unsupported")
        if raw is None:
            stop_reason = "api_error:action"
            unrecovered_api_errors.append(
                {"stage": "action", "api_attempts": list(attempts)}
            )
            format_records.append(
                {
                    "raw_parse_valid": False,
                    "final_parse_valid": False,
                    "repair_attempted": False,
                    "repair_unsupported": False,
                    "semantic_valid": False,
                    "api_attempts": attempts,
                    "unrecovered_api_error": True,
                    "api_failure_stage": "action",
                }
            )
            break
        parsed, record = _parse_with_repair(
            adapter,
            raw,
            parse_track2_action_response,
            action_grammar_text(task),
            max_output_tokens=config.action_output_tokens,
            config=config,
        )
        record["api_attempts"] = [*attempts, *record.get("api_attempts", [])]
        format_records.append(record)
        if not parsed.get("format_valid"):
            record["semantic_valid"] = False
            if record.get("unrecovered_api_error"):
                stop_reason = "api_error:action_repair"
                unrecovered_api_errors.append(
                    {
                        "stage": "action_repair",
                        "api_attempts": list(record.get("api_attempts", [])),
                    }
                )
            else:
                stop_reason = "invalid_action:parse_failure"
            break

        step_index = len(query_history)
        ledger, update_outcomes = validate_and_merge_track2_updates(
            task,
            ledger,
            parsed["updates"],
            reel_duration_seconds=float(episode["duration_seconds"]),
            step_index=step_index,
        )
        for outcome in update_outcomes:
            outcome["step_index"] = step_index
            update_history.append(outcome)
        record["semantic_valid"] = all(outcome["valid"] for outcome in update_outcomes)

        observed_step = int(current_crop["step_index"]) if current_crop is not None else 0
        if periodic_probe and observed_step > 0 and observed_step % config.probe_every_n == 0:
            ledger, probe_record, probe_format = _run_probe(
                adapter, task, ledger, episode, observed_step, config
            )
            probe_history.append(probe_record)
            update_history.extend(probe_record.get("update_outcomes", []))
            format_records.append(probe_format)
            if probe_format.get("unrecovered_api_error"):
                stage = str(probe_format.get("api_failure_stage", "probe"))
                stop_reason = f"api_error:{stage}"
                unrecovered_api_errors.append(
                    {
                        "stage": stage,
                        "api_attempts": list(probe_format.get("api_attempts", [])),
                    }
                )
                break
        if current_crop is not None and len(query_history) >= budget.max_queries:
            stop_reason = "max_queries_reached"
            break
        if current_crop is not None and listened_seconds >= budget.listening_budget_seconds - 1e-9:
            stop_reason = "listening_budget_reached"
            break

        if parsed["action"] == "STOP":
            stop_reason = "model_stop"
            break
        query_valid, query_reason = validate_track2_query(
            parsed["start_seconds"],
            parsed["end_seconds"],
            reel_duration_seconds=float(episode["duration_seconds"]),
            max_single_query_seconds=config.max_single_query_seconds,
            listened_seconds=listened_seconds,
            listening_budget_seconds=budget.listening_budget_seconds,
        )
        if not query_valid:
            record["semantic_valid"] = False
            record["query_validation_reason"] = query_reason
            stop_reason = f"invalid_action:{query_reason}"
            break

        start = float(parsed["start_seconds"])
        end = float(parsed["end_seconds"])
        step_index = len(query_history) + 1
        current_crop_path = _write_query_crop(
            audio_path, runtime_root, str(task["task_id"]), step_index, start, end
        )
        current_crop = {
            "step_index": step_index,
            "start_seconds": start,
            "end_seconds": end,
            "duration_seconds": end - start,
            "audio_path": str(current_crop_path),
        }
        query_history.append(dict(current_crop))
        listened_seconds += end - start


    final_raw, final_attempts, final_unsupported = _send_track2_with_retry(
        adapter,
        build_track2_final_prompt(
            task,
            ledger,
            reel_duration_seconds=float(episode["duration_seconds"]),
            full_context=False,
        ),
        [],
        max_output_tokens=config.final_output_tokens,
        strict_text_only=True,
        config=config,
    )
    if final_unsupported:
        return _unsupported_result(task, condition, adapter, "final_text_call_unsupported")
    if final_raw is None:
        result = _api_error_result(
            task,
            condition,
            adapter,
            final_attempts,
            stage="final",
            prior_errors=unrecovered_api_errors,
        )
        result.update({"query_history": query_history, "stop_reason": stop_reason})
        return result
    final_parse, final_format = _parse_with_repair(
        adapter,
        final_raw,
        lambda value: parse_track2_final_answer(value, task),
        final_schema_text(task),
        max_output_tokens=config.final_output_tokens,
        config=config,
    )
    final_format["api_attempts"] = [*final_attempts, *final_format.get("api_attempts", [])]
    format_records.append(final_format)
    if final_format.get("unrecovered_api_error"):
        unrecovered_api_errors.append(
            {
                "stage": "final_repair",
                "api_attempts": list(final_format.get("api_attempts", [])),
            }
        )
    result = _base_complete_result(
        task, condition, adapter, final_parse, gold, episode, format_records
    )
    final_answer = result["final_answer"]
    task_for_scoring = dict(task)
    task_for_scoring["_episode_scenes"] = episode["scenes"]
    task_for_scoring["_reel_duration_seconds"] = float(episode["duration_seconds"])
    result.update(
        {
            "stop_reason": stop_reason,
            "query_history": query_history,
            "update_history": update_history,
            "probe_history": probe_history,
            "final_ledger": ledger,
            "evidence_metrics": score_track2_evidence(
                task_for_scoring,
                gold,
                query_history,
                reel_duration_seconds=float(episode["duration_seconds"]),
                listening_budget_seconds=budget.listening_budget_seconds,
                stop_reason=stop_reason,
            ),
            "belief_metrics": score_track2_belief(
                task_for_scoring, gold, query_history, update_history, ledger, final_answer
            ),
        }
    )
    if unrecovered_api_errors:
        return _mark_unscored_api_error(result, unrecovered_api_errors)
    return result


def _write_flat_metrics(results: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("run_status") != "complete":
            continue
        row = {
            "task_id": result["task_id"],
            "reel_id": result["reel_id"],
            "family": result["family"],
            "variant": result["variant"],
            "condition": result["condition"],
            "model_id": result["model_id"],
            "run_status": result["run_status"],
            "stop_reason": result.get("stop_reason"),
        }
        for group_name in ("answer_metrics", "evidence_metrics", "belief_metrics", "format_metrics"):
            for key, value in dict(result.get(group_name) or {}).items():
                if isinstance(value, (str, int, float)) or value is None:
                    row[f"{group_name}.{key}"] = value
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rebuild_track2_outputs(results_root: Path) -> list[dict[str, Any]]:
    """Rebuild aggregate artifacts from existing per-task Track 2 results."""
    root = Path(results_root)
    result_paths = sorted(root.glob("*/*/*/*.json"))
    results = [json.loads(path.read_text(encoding="utf-8")) for path in result_paths]
    incompatible_results = [
        path
        for path, result in zip(result_paths, results)
        if not isinstance(result, Mapping)
        or result.get("track2_result_schema_version") != TRACK2_RESULT_SCHEMA_VERSION
    ]
    if incompatible_results:
        raise ValueError(
            f"Cannot rebuild Track 2 outputs: {len(incompatible_results)} per-task result(s) "
            f"do not use schema version {TRACK2_RESULT_SCHEMA_VERSION}. "
            "Rerun the benchmark with the current scorer."
        )
    _write_flat_metrics(results, root / "episode_metrics.csv")
    _write_json(root / "summary.json", aggregate_track2_results(results))
    return results


def _is_reusable_track2_result(result: Any) -> bool:
    return (
        isinstance(result, Mapping)
        and result.get("track2_result_schema_version") == TRACK2_RESULT_SCHEMA_VERSION
        and result.get("run_status") == "complete"
        and not result.get("unrecovered_api_errors")
    )


def run_track2_benchmark(
    *,
    dataset_root: Path,
    results_root: Path,
    adapter: BaseAdapter,
    task_ids: Sequence[str] | None = None,
    reel_ids: Sequence[str] | None = None,
    families: Sequence[str] | None = None,
    variants: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    reprocess: bool = False,
    parallelism: int = 1,
    config: Track2Config | None = None,
) -> list[dict[str, Any]]:
    resolved_config = config or Track2Config()
    if resolved_config.max_single_query_seconds != TRACK2_MAX_SINGLE_QUERY_SECONDS:
        raise RuntimeError("Track 2 max_single_query_seconds must remain frozen at 384.")
    if resolved_config.probe_every_n != 2:
        raise RuntimeError("Track 2 probe_every_n must remain frozen at 2.")
    if resolved_config.action_output_tokens != 512:
        raise RuntimeError("Track 2 action_output_tokens must remain frozen at 512.")
    if resolved_config.final_output_tokens != 1024:
        raise RuntimeError("Track 2 final_output_tokens must remain frozen at 1024.")
    dataset = load_track2_dataset(dataset_root)
    all_tasks = list(dataset["tasks"])
    filters = {
        "task_id": None if task_ids is None else set(map(str, task_ids)),
        "reel_id": None if reel_ids is None else set(map(str, reel_ids)),
        "family": None if families is None else set(map(str, families)),
        "variant": None if variants is None else set(map(str, variants)),
    }
    for key, allowed in filters.items():
        if allowed is not None:
            available = {str(task[key]) for task in all_tasks}
            unknown = sorted(allowed - available)
            if unknown:
                raise RuntimeError(f"Unknown Track 2 {key} filters: {unknown}")
    selected_tasks = [
        task
        for task in all_tasks
        if all(allowed is None or str(task[key]) in allowed for key, allowed in filters.items())
    ]
    selected_conditions = list(TRACK2_DEFAULT_CONDITIONS if conditions is None else map(str, conditions))
    unknown_conditions = sorted(set(selected_conditions) - set(TRACK2_CONDITIONS))
    if unknown_conditions:
        raise RuntimeError(f"Unknown Track 2 conditions: {unknown_conditions}")
    run_specs = [(task, condition) for task in selected_tasks for condition in selected_conditions]
    if not run_specs:
        raise RuntimeError("Selected Track 2 benchmark run is empty.")
    resolved_parallelism = max(1, int(parallelism))
    print(
        f"track2-run start total={len(run_specs)} parallelism={resolved_parallelism}",
        flush=True,
    )
    progress = ProgressReporter(len(run_specs), label="track2-run")

    def run_one(spec: tuple[dict[str, Any], str]) -> dict[str, Any]:
        task, condition = spec
        path = _result_path(results_root, task, condition)
        if path.is_file() and not reprocess:
            cached_result = json.loads(path.read_text(encoding="utf-8"))
            if _is_reusable_track2_result(cached_result):
                return cached_result
        episode = dataset["episodes"][str(task["reel_id"])]
        gold = dataset["answers"][str(task["task_id"])]
        audio_path = Path(dataset["dataset_dir"]) / str(task["audio_path"])
        capability_failure = _capability_failure(
            adapter, condition, audio_path, float(episode["duration_seconds"]), resolved_config
        )
        if capability_failure:
            result = _unsupported_result(task, condition, adapter, capability_failure)
        elif condition == "full_context":
            result = _run_full_context(dataset, task, gold, episode, adapter, resolved_config)
        else:
            result = _run_active(
                dataset,
                task,
                gold,
                episode,
                adapter,
                resolved_config,
                Path(results_root),
                periodic_probe=condition == "active_same_session_periodic_probe",
            )
        _write_json(path, result)
        return result

    def progress_detail(result: Mapping[str, Any]) -> str:
        return (
            f"condition={result.get('condition')} family={result.get('family')} "
            f"task={result.get('task_id')} status={result.get('run_status')}"
        )

    if resolved_parallelism == 1:
        results = []
        for index, spec in enumerate(run_specs, start=1):
            result = run_one(spec)
            results.append(result)
            progress.update(index, progress_detail(result))
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(resolved_parallelism, len(run_specs))
        ) as executor:
            results = []
            for index, result in enumerate(executor.map(run_one, run_specs), start=1):
                results.append(result)
                progress.update(index, progress_detail(result))
    _write_flat_metrics(results, Path(results_root) / "episode_metrics.csv")
    _write_json(Path(results_root) / "summary.json", aggregate_track2_results(results))
    return results


__all__ = [
    "TRACK2",
    "TRACK2_BUDGETS",
    "TRACK2_CONDITIONS",
    "TRACK2_CROP_SAMPLE_RATE",
    "TRACK2_DEFAULT_CONDITIONS",
    "TRACK2_FAMILIES",
    "TRACK2_RESULT_SCHEMA_VERSION",
    "Track2Budget",
    "Track2Config",
    "load_track2_dataset",
    "rebuild_track2_outputs",
    "run_track2_benchmark",
    "track2_budget_for_task",
]
