from __future__ import annotations

import math
from collections import defaultdict
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SCENE_ENDPOINT_TOLERANCE_SECONDS = 10.0
UTTERANCE_ENDPOINT_TOLERANCE_SECONDS = 1.0
SCENE_LOCALIZATION_FAMILIES = frozenset({"2A.1", "2B.1"})
UTTERANCE_LOCALIZATION_FAMILIES = frozenset({"2A.2", "2A.3", "2B.2"})


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def merge_intervals(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(start), float(end)) for start, end in intervals if float(end) > float(start))
    merged: list[list[float]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def unique_interval_duration(intervals: Iterable[tuple[float, float]]) -> float:
    return sum(end - start for start, end in merge_intervals(intervals))


def interval_metrics(
    predicted: tuple[float, float],
    gold: tuple[float, float],
    *,
    endpoint_tolerance_seconds: float = 0.0,
) -> dict[str, float]:
    pred_start, pred_end = map(float, predicted)
    gold_start, gold_end = map(float, gold)
    overlap = interval_overlap((pred_start, pred_end), (gold_start, gold_end))
    pred_duration = max(0.0, pred_end - pred_start)
    gold_duration = max(0.0, gold_end - gold_start)
    union = pred_duration + gold_duration - overlap
    dice_denom = pred_duration + gold_duration
    snapped_start = gold_start if abs(pred_start - gold_start) <= endpoint_tolerance_seconds else pred_start
    snapped_end = gold_end if abs(pred_end - gold_end) <= endpoint_tolerance_seconds else pred_end
    snapped_overlap = interval_overlap((snapped_start, snapped_end), (gold_start, gold_end))
    snapped_duration = max(0.0, snapped_end - snapped_start)
    snapped_union = snapped_duration + gold_duration - snapped_overlap
    return {
        "overlap_seconds": overlap,
        "target_coverage": overlap / gold_duration if gold_duration else 0.0,
        "predicted_interval_precision": overlap / pred_duration if pred_duration else 0.0,
        "temporal_f1": 2.0 * overlap / dice_denom if dice_denom else 0.0,
        "temporal_iou": overlap / union if union else 0.0,
        "start_boundary_error_seconds": abs(pred_start - gold_start),
        "end_boundary_error_seconds": abs(pred_end - gold_end),
        "center_error_seconds": abs((pred_start + pred_end - gold_start - gold_end) / 2.0),
        "excess_predicted_duration_seconds": max(0.0, pred_duration - overlap),
        "tolerant_temporal_iou": snapped_overlap / snapped_union if snapped_union else 0.0,
        "localization_pass": float(
            (snapped_overlap / snapped_union if snapped_union else 0.0) >= 0.5
        ),
    }


def _localization_interval_metrics(
    family: str,
    predicted: tuple[float, float],
    gold: tuple[float, float],
) -> dict[str, float]:
    if family in SCENE_LOCALIZATION_FAMILIES:
        return interval_metrics(
            predicted,
            gold,
            endpoint_tolerance_seconds=SCENE_ENDPOINT_TOLERANCE_SECONDS,
        )
    if family in UTTERANCE_LOCALIZATION_FAMILIES:
        return interval_metrics(
            predicted,
            gold,
            endpoint_tolerance_seconds=UTTERANCE_ENDPOINT_TOLERANCE_SECONDS,
        )
    raise ValueError(f"Unsupported Track 2 localization family: {family}")


def pairwise_order_accuracy(predicted: Sequence[str], gold: Sequence[str]) -> float:
    gold_ids = list(gold)
    if len(gold_ids) < 2:
        return 1.0
    predicted_positions = {value: index for index, value in enumerate(predicted)}
    correct = 0
    total = 0
    for left_index, left in enumerate(gold_ids):
        for right in gold_ids[left_index + 1 :]:
            total += 1
            if left in predicted_positions and right in predicted_positions:
                correct += int(predicted_positions[left] < predicted_positions[right])
    return correct / total


def _empty_answer_metrics(prefix: str = "") -> dict[str, float | None]:
    return {
        f"{prefix}answer_complete": 0.0,
        f"{prefix}primary_correct": 0.0,
    }


def _failed_interval_metrics() -> dict[str, float | None]:
    return {
        "overlap_seconds": 0.0,
        "target_coverage": 0.0,
        "predicted_interval_precision": 0.0,
        "temporal_f1": 0.0,
        "temporal_iou": 0.0,
        "start_boundary_error_seconds": None,
        "end_boundary_error_seconds": None,
        "center_error_seconds": None,
        "excess_predicted_duration_seconds": 0.0,
        "tolerant_temporal_iou": 0.0,
        "localization_pass": 0.0,
    }


def _scene_for_interval(predicted: tuple[float, float], scenes: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    ranked = sorted(
        scenes,
        key=lambda scene: (
            -interval_overlap(predicted, (float(scene["start_seconds"]), float(scene["end_seconds"]))),
            int(scene["scene_rank"]),
        ),
    )
    if not ranked:
        return None
    best = ranked[0]
    return best if interval_overlap(predicted, (float(best["start_seconds"]), float(best["end_seconds"]))) > 0 else None


def score_track2_answer(
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    parsed_answer: Mapping[str, Any] | None,
    episode: Mapping[str, Any],
) -> dict[str, float | None]:
    family = str(task["family"])
    if parsed_answer is None:
        metrics = _empty_answer_metrics()
        if family == "2A.1":
            metrics.update(
                {
                    "present_absent_accuracy": 0.0,
                    "false_localization_rate": 0.0,
                    "scene_identity_accuracy": 0.0 if gold["status"] == "PRESENT" else None,
                    "scene_rank_accuracy": 0.0 if gold["status"] == "PRESENT" else None,
                }
            )
            if gold["status"] == "PRESENT":
                metrics.update(_failed_interval_metrics())
        elif family == "2A.2":
            metrics.update(_failed_interval_metrics())
            metrics["present_absent_accuracy"] = 0.0
        elif family == "2A.3":
            metrics.update({"present_absent_accuracy": 0.0, "false_localization_rate": 0.0})
        elif family == "2B.1":
            metrics.update(
                {
                    "exact_order_accuracy": 0.0,
                    "pairwise_order_accuracy": 0.0,
                    "kendall_tau": -1.0,
                    "card_to_scene_assignment_accuracy": 0.0,
                    "absent_card_accuracy": 0.0,
                }
            )
        elif family == "2B.2":
            metrics.update(
                {
                    "target_status_accuracy": 0.0,
                    "utterance_localization_accuracy": 0.0,
                    "scene_card_assignment_accuracy": 0.0,
                    "false_localization_rate": 0.0,
                    "exact_order_accuracy": 0.0,
                    "pairwise_order_accuracy": 0.0,
                    "all_components_complete_accuracy": 0.0,
                }
            )
            metrics.update({f"mean_{key}": value for key, value in _failed_interval_metrics().items()})
        return metrics
    answer = parsed_answer
    metrics: dict[str, float | None] = {"answer_complete": 1.0}

    if family == "2A.1":
        gold_status = str(gold["status"])
        status_correct = float(answer["status"] == gold_status)
        metrics["present_absent_accuracy"] = status_correct
        if gold_status == "ABSENT":
            metrics.update(
                {
                    "false_localization_rate": float(answer["status"] != "ABSENT"),
                    "scene_identity_accuracy": None,
                    "scene_rank_accuracy": None,
                    "primary_correct": status_correct,
                }
            )
        elif answer["status"] == "PRESENT":
            predicted = (float(answer["start_seconds"]), float(answer["end_seconds"]))
            gold_interval = (float(gold["start_seconds"]), float(gold["end_seconds"]))
            interval_scores = _localization_interval_metrics(family, predicted, gold_interval)
            metrics.update(interval_scores)
            predicted_scene = _scene_for_interval(predicted, episode["scenes"])
            scene_identity = float(
                predicted_scene is not None and int(predicted_scene["scene_rank"]) == int(gold["scene_rank"])
            )
            rank_accuracy = float(int(answer["scene_rank"]) == int(gold["scene_rank"]))
            metrics["scene_identity_accuracy"] = scene_identity
            metrics["scene_rank_accuracy"] = rank_accuracy
            metrics["primary_correct"] = float(status_correct and scene_identity and interval_scores["localization_pass"])
        else:
            metrics.update(
                {
                    "scene_identity_accuracy": 0.0,
                    "scene_rank_accuracy": 0.0,
                    "localization_pass": 0.0,
                    "primary_correct": 0.0,
                }
            )
            metrics.update(_failed_interval_metrics())
    elif family == "2A.2":
        predicted = (float(answer["start_seconds"]), float(answer["end_seconds"]))
        gold_interval = (float(gold["start_seconds"]), float(gold["end_seconds"]))
        metrics.update(_localization_interval_metrics(family, predicted, gold_interval))
        metrics["present_absent_accuracy"] = float(answer.get("status", "PRESENT") == "PRESENT")
        metrics["primary_correct"] = metrics["localization_pass"]
    elif family == "2A.3":
        status_correct = float(answer["status"] == gold["status"])
        false_localization = float(gold["status"] == "ABSENT" and bool(answer["occurrences"]))
        metrics.update(
            {
                "present_absent_accuracy": status_correct,
                "false_localization_rate": false_localization,
                "primary_correct": float(status_correct and not false_localization),
            }
        )
    elif family == "2B.1":
        predicted_order = list(answer["ordered_card_ids"])
        gold_order = list(gold["ordered_card_ids"])
        exact = float(predicted_order == gold_order)
        pairwise = pairwise_order_accuracy(predicted_order, gold_order)
        absent_correct = float(answer.get("absent_card_id") == gold.get("absent_card_id"))
        metrics.update(
            {
                "exact_order_accuracy": exact,
                "pairwise_order_accuracy": pairwise,
                "kendall_tau": 2.0 * pairwise - 1.0,
                "card_to_scene_assignment_accuracy": mean(
                    float(
                        card_id in predicted_order
                        and predicted_order.index(card_id) == gold_rank
                    )
                    for gold_rank, card_id in enumerate(gold_order)
                ),
                "absent_card_accuracy": absent_correct,
                "primary_correct": float(exact and absent_correct),
            }
        )
    elif family == "2B.2":
        predicted_by_id = {item["utterance_id"]: item for item in answer["utterances"]}
        gold_by_id = {item["utterance_id"]: item for item in gold["utterances"]}
        status_scores: list[float] = []
        localization_scores: list[float] = []
        card_scores: list[float] = []
        false_localizations: list[float] = []
        interval_score_groups: defaultdict[str, list[float]] = defaultdict(list)
        components_complete = True
        for target_id, gold_item in gold_by_id.items():
            predicted_item = predicted_by_id[target_id]
            status_correct = float(predicted_item["status"] == gold_item["status"])
            status_scores.append(status_correct)
            components_complete = components_complete and bool(status_correct)
            if gold_item["status"] == "ABSENT":
                false_localizations.append(float(predicted_item["status"] != "ABSENT"))
                continue
            if predicted_item["status"] != "PRESENT":
                localization_scores.append(0.0)
                card_scores.append(0.0)
                components_complete = False
                for key, value in _failed_interval_metrics().items():
                    interval_score_groups[key].append(value)
                continue
            scores = _localization_interval_metrics(
                family,
                (float(predicted_item["start_seconds"]), float(predicted_item["end_seconds"])),
                (float(gold_item["start_seconds"]), float(gold_item["end_seconds"])),
            )
            for key, value in scores.items():
                interval_score_groups[key].append(value)
            localization_scores.append(scores["localization_pass"])
            card_correct = float(predicted_item["scene_card_id"] == gold_item["scene_card_id"])
            card_scores.append(card_correct)
            components_complete = components_complete and bool(scores["localization_pass"]) and bool(card_correct)
        exact_order = float(answer["ordered_real_utterance_ids"] == gold["ordered_real_utterance_ids"])
        pairwise = pairwise_order_accuracy(answer["ordered_real_utterance_ids"], gold["ordered_real_utterance_ids"])
        components_complete = components_complete and bool(exact_order)
        metrics.update(
            {
                "target_status_accuracy": mean(status_scores),
                "utterance_localization_accuracy": mean(localization_scores),
                "scene_card_assignment_accuracy": mean(card_scores),
                "false_localization_rate": mean(false_localizations) if false_localizations else 0.0,
                "exact_order_accuracy": exact_order,
                "pairwise_order_accuracy": pairwise,
                "all_components_complete_accuracy": float(components_complete),
                "primary_correct": float(components_complete),
            }
        )
        for key, values in interval_score_groups.items():
            numeric_values = [float(value) for value in values if isinstance(value, (int, float))]
            metrics[f"mean_{key}"] = mean(numeric_values) if numeric_values else None
    else:
        raise ValueError(f"Unsupported Track 2 family: {family}")
    return metrics


def _gold_target_intervals(task: Mapping[str, Any], gold: Mapping[str, Any]) -> dict[str, list[tuple[float, float]]]:
    family = str(task["family"])
    if family == "2A.1":
        return {
            str(task["card"]["card_id"]): []
            if gold["status"] == "ABSENT"
            else [(float(gold["start_seconds"]), float(gold["end_seconds"]))]
        }
    if family == "2A.2":
        return {str(task["target_id"]): [(float(gold["start_seconds"]), float(gold["end_seconds"]))]}
    if family == "2A.3":
        return {
            str(task["target_id"]): [
                (float(item["start_seconds"]), float(item["end_seconds"])) for item in gold.get("occurrences", [])
            ]
        }
    if family == "2B.1":
        ordered = list(gold["ordered_card_ids"])
        scene_by_rank = {index + 1: scene for index, scene in enumerate(sorted(task["_episode_scenes"], key=lambda row: row["scene_rank"]))}
        result = {str(card["card_id"]): [] for card in task["cards"]}
        for rank, card_id in enumerate(ordered, start=1):
            scene = scene_by_rank[rank]
            result[str(card_id)] = [(float(scene["start_seconds"]), float(scene["end_seconds"]))]
        return result
    result = {}
    for item in gold["utterances"]:
        result[str(item["utterance_id"])] = [] if item["status"] == "ABSENT" else [
            (float(item["start_seconds"]), float(item["end_seconds"]))
        ]
    return result


def score_track2_evidence(
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    query_history: Sequence[Mapping[str, Any]],
    *,
    reel_duration_seconds: float,
    listening_budget_seconds: float,
    stop_reason: str,
) -> dict[str, float | str | None]:
    intervals = [(float(item["start_seconds"]), float(item["end_seconds"])) for item in query_history]
    total = sum(end - start for start, end in intervals)
    unique = unique_interval_duration(intervals)
    task_with_scenes = dict(task)
    task_with_scenes.setdefault("_episode_scenes", task.get("_episode_scenes", []))
    targets = _gold_target_intervals(task_with_scenes, gold)
    contact_costs: list[float] = []
    contact_steps: list[float] = []
    target_coverages: dict[str, float] = {}
    for target_id, gold_intervals in targets.items():
        if not gold_intervals:
            continue
        gold_duration = unique_interval_duration(gold_intervals)
        covered_duration = sum(
            interval_overlap(queried, target)
            for queried in merge_intervals(intervals)
            for target in gold_intervals
        )
        target_coverages[target_id] = covered_duration / gold_duration if gold_duration else 0.0
        cumulative = 0.0
        found = False
        for step, queried in enumerate(intervals, start=1):
            duration = queried[1] - queried[0]
            cumulative += duration
            if any(interval_overlap(queried, target) > 0 for target in gold_intervals):
                contact_costs.append(cumulative)
                contact_steps.append(float(step))
                found = True
                break
        if not found:
            contact_costs.append(math.nan)
            contact_steps.append(math.nan)
    finite_costs = [value for value in contact_costs if math.isfinite(value)]
    finite_steps = [value for value in contact_steps if math.isfinite(value)]
    present_targets = {target_id for target_id, values in targets.items() if values}
    contacted: set[str] = set()
    cumulative_cost = 0.0
    all_target_contact_cost: float | None = None
    for queried in intervals:
        cumulative_cost += queried[1] - queried[0]
        for target_id in present_targets - contacted:
            if any(interval_overlap(queried, target) > 0 for target in targets[target_id]):
                contacted.add(target_id)
        if present_targets and contacted == present_targets:
            all_target_contact_cost = cumulative_cost
            break
    return {
        "total_audio_inspected_seconds": total,
        "unique_audio_inspected_seconds": unique,
        "overlap_replay_seconds": total - unique,
        "unique_reel_coverage": unique / float(reel_duration_seconds),
        "listening_budget_used_fraction": total / float(listening_budget_seconds),
        "remaining_listening_budget_seconds": float(listening_budget_seconds) - total,
        "successful_query_count": float(len(query_history)),
        "mean_cost_to_first_target_contact_seconds": mean(finite_costs) if finite_costs else None,
        "mean_step_to_first_target_contact": mean(finite_steps) if finite_steps else None,
        "target_contact_rate": (
            len(finite_costs) / len(contact_costs) if contact_costs else None
        ),
        "target_evidence_coverage": target_coverages,
        "mean_target_evidence_coverage": (
            mean(target_coverages.values()) if target_coverages else None
        ),
        "all_present_targets_contacted": (
            float(contacted == present_targets) if present_targets else None
        ),
        "cost_to_all_present_target_contact_seconds": all_target_contact_cost,
        "stop_reason": stop_reason,
        "step_budget_exhausted": float(stop_reason == "max_queries_reached"),
        "listening_budget_exhausted": float(stop_reason == "listening_budget_reached"),
        "voluntary_stop": float(stop_reason == "model_stop"),
    }


def score_track2_belief(
    task: Mapping[str, Any],
    gold: Mapping[str, Any],
    query_history: Sequence[Mapping[str, Any]],
    update_history: Sequence[Mapping[str, Any]],
    final_ledger: Mapping[str, Mapping[str, Any]],
    final_answer: Mapping[str, Any] | None,
) -> dict[str, float | None | list[dict[str, float]]]:
    task_with_scenes = dict(task)
    targets = _gold_target_intervals(task_with_scenes, gold)
    gold_card_by_target = (
        {str(item["utterance_id"]): item.get("scene_card_id") for item in gold["utterances"]}
        if task["family"] == "2B.2"
        else {}
    )
    contacts: dict[str, int | None] = {}
    for target_id, gold_intervals in targets.items():
        contacts[target_id] = next(
            (
                index
                for index, query in enumerate(query_history, start=1)
                if any(
                    interval_overlap(
                        (float(query["start_seconds"]), float(query["end_seconds"])), interval
                    )
                    > 0
                    for interval in gold_intervals
                )
            ),
            None,
        )
    unsupported_found = 0
    valid_found_updates = 0
    delays: list[float] = []
    commitment_costs: list[float] = []
    absence_claims: list[dict[str, Any]] = []
    transitions_by_target: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    committed_targets: set[str] = set()
    commitment_step_by_target: dict[str, int] = {}
    post_contact_status_scores: list[float] = []
    for record in update_history:
        if not record.get("valid"):
            continue
        step = int(record.get("step_index", 0))
        update = record["update"]
        target_id = str(update["target_id"])
        if (
            targets.get(target_id)
            and contacts.get(target_id) is not None
            and step >= int(contacts[target_id])
        ):
            post_contact_status_scores.append(float(update["status"] == "FOUND"))
        observed_intervals = [
            (float(query["start_seconds"]), float(query["end_seconds"]))
            for query in query_history[:step]
        ]
        if update["status"] == "FOUND":
            valid_found_updates += 1
            gold_intervals = targets.get(target_id, [])
            localization_correct = bool(gold_intervals) and any(
                _localization_interval_metrics(
                    str(task["family"]),
                    (float(update["start_seconds"]), float(update["end_seconds"])),
                    interval,
                )["localization_pass"]
                for interval in gold_intervals
            )
            if task["family"] == "2B.2":
                localization_correct = localization_correct and (
                    update.get("scene_card_id") == gold_card_by_target[target_id]
                )
            transitions_by_target[target_id].append(
                {"step": step, "status": "FOUND", "correct": localization_correct}
            )
            if contacts.get(target_id) is None or step < int(contacts[target_id]):
                unsupported_found += 1
            elif localization_correct and target_id not in committed_targets:
                committed_targets.add(target_id)
                commitment_step_by_target[target_id] = step
                delays.append(float(step - int(contacts[target_id])))
                commitment_costs.append(sum(end - start for start, end in observed_intervals))
        elif update["status"] == "LIKELY_ABSENT":
            truth_is_absent = not targets.get(target_id)
            transitions_by_target[target_id].append(
                {
                    "step": step,
                    "status": "LIKELY_ABSENT",
                    "correct": truth_is_absent,
                }
            )
            unique_coverage = unique_interval_duration(observed_intervals)
            absence_claims.append(
                {
                    "source": "ledger",
                    "target_id": target_id,
                    "step_index": float(step),
                    "unique_coverage_seconds": unique_coverage,
                    "unique_reel_coverage_fraction": unique_coverage
                    / float(task["_reel_duration_seconds"]),
                    "listening_cost_seconds": sum(end - start for start, end in observed_intervals),
                    "truth_is_absent": truth_is_absent,
                    "false_absence_claim": not truth_is_absent,
                }
            )
        else:
            transitions_by_target[target_id].append(
                {"step": step, "status": "NOT_FOUND_YET", "correct": None}
            )
    preservation_scores: list[float] = []
    final_ledger_correctness: list[float] = []
    for target_id, entry in final_ledger.items():
        should_be_found = bool(targets.get(target_id))
        expected_status = "FOUND" if should_be_found else "LIKELY_ABSENT"
        preservation_scores.append(float(entry["status"] == expected_status))
        if not should_be_found:
            final_ledger_correctness.append(float(entry["status"] == "LIKELY_ABSENT"))
            continue
        localization_correct = entry["status"] == "FOUND" and any(
            _localization_interval_metrics(
                str(task["family"]),
                (float(entry["start_seconds"]), float(entry["end_seconds"])),
                interval,
            )["localization_pass"]
            for interval in targets[target_id]
        )
        if task["family"] == "2B.2":
            localization_correct = localization_correct and (
                entry.get("scene_card_id") == gold_card_by_target[target_id]
            )
        final_ledger_correctness.append(float(localization_correct))
    revision_opportunities = 0
    successful_revisions = 0
    landmark_preservation: list[float] = []
    for records in transitions_by_target.values():
        first_correct_found_index = next(
            (
                index
                for index, record in enumerate(records)
                if record["status"] == "FOUND" and record["correct"] is True
            ),
            None,
        )
        for index, record in enumerate(records):
            if record["correct"] is not False:
                continue
            revision_opportunities += 1
            successful_revisions += int(
                any(
                    later["correct"] is True or later["status"] == "NOT_FOUND_YET"
                    for later in records[index + 1 :]
                )
            )
        if first_correct_found_index is not None:
            landmark_preservation.append(
                float(
                    all(
                        later["status"] == "FOUND" and later["correct"] is True
                        for later in records[first_correct_found_index:]
                    )
                )
            )
    ledger_answer_consistency = _ledger_answer_consistency(task, final_ledger, final_answer)
    if final_answer is not None:
        if task["family"] in {"2A.1", "2A.3"}:
            final_absent_ids = (
                [next(iter(targets))] if final_answer["status"] == "ABSENT" else []
            )
        elif task["family"] == "2B.1":
            final_absent_ids = (
                [str(final_answer["absent_card_id"])]
                if final_answer.get("absent_card_id") is not None
                else []
            )
        elif task["family"] == "2B.2":
            final_absent_ids = [
                str(item["utterance_id"])
                for item in final_answer["utterances"]
                if item["status"] == "ABSENT"
            ]
        else:
            final_absent_ids = []
        all_queried = [
            (float(query["start_seconds"]), float(query["end_seconds"]))
            for query in query_history
        ]
        for target_id in final_absent_ids:
            truth_is_absent = not targets.get(target_id)
            unique_coverage = unique_interval_duration(all_queried)
            absence_claims.append(
                {
                    "source": "final_answer",
                    "target_id": target_id,
                    "step_index": float(len(query_history)),
                    "unique_coverage_seconds": unique_coverage,
                    "unique_reel_coverage_fraction": unique_coverage
                    / float(task["_reel_duration_seconds"]),
                    "listening_cost_seconds": sum(end - start for start, end in all_queried),
                    "truth_is_absent": truth_is_absent,
                    "false_absence_claim": not truth_is_absent,
                }
            )
    false_absence_claims = [
        claim for claim in absence_claims if claim["false_absence_claim"]
    ]
    present_target_ids = {target_id for target_id, intervals in targets.items() if intervals}
    all_commitment_cost: float | None = None
    if present_target_ids and present_target_ids <= set(commitment_step_by_target):
        last_commitment_step = max(commitment_step_by_target[target_id] for target_id in present_target_ids)
        all_commitment_cost = sum(
            float(query["end_seconds"]) - float(query["start_seconds"])
            for query in query_history[:last_commitment_step]
        )
    return {
        "unsupported_found_claim_count": float(unsupported_found),
        "unsupported_found_claim_rate": unsupported_found / valid_found_updates if valid_found_updates else 0.0,
        "mean_contact_to_correct_update_steps": mean(delays) if delays else None,
        "mean_cost_to_correct_commitment_seconds": mean(commitment_costs) if commitment_costs else None,
        "cost_to_all_correct_commitments_seconds": all_commitment_cost,
        "target_status_after_contact_accuracy": (
            mean(post_contact_status_scores) if post_contact_status_scores else None
        ),
        "final_ledger_status_accuracy": mean(preservation_scores) if preservation_scores else None,
        "final_ledger_target_accuracy": (
            mean(final_ledger_correctness) if final_ledger_correctness else None
        ),
        "final_ledger_complete_accuracy": (
            float(all(final_ledger_correctness)) if final_ledger_correctness else None
        ),
        "successful_revision_rate": (
            successful_revisions / revision_opportunities if revision_opportunities else None
        ),
        "found_landmark_preservation_rate": (
            mean(landmark_preservation) if landmark_preservation else None
        ),
        "final_ledger_answer_consistency": ledger_answer_consistency,
        "absence_claim_coverage_history": absence_claims,
        "false_absence_claim_count": float(len(false_absence_claims)),
        "false_absence_claim_rate": (
            len(false_absence_claims) / len(absence_claims) if absence_claims else 0.0
        ),
    }


def _ledger_answer_consistency(
    task: Mapping[str, Any],
    ledger: Mapping[str, Mapping[str, Any]],
    answer: Mapping[str, Any] | None,
) -> float | None:
    if answer is None:
        return None

    def same_interval(entry: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
        return all(
            entry.get(key) is not None
            and value.get(key) is not None
            and math.isclose(float(entry[key]), float(value[key]), abs_tol=1e-6)
            for key in ("start_seconds", "end_seconds")
        )

    def same_status_and_evidence(entry: Mapping[str, Any], value: Mapping[str, Any]) -> bool:
        if value["status"] == "ABSENT":
            return entry["status"] == "LIKELY_ABSENT"
        if entry["status"] != "FOUND" or not same_interval(entry, value):
            return False
        return task["family"] != "2B.2" or entry.get("scene_card_id") == value.get("scene_card_id")

    family = str(task["family"])
    if family == "2A.1":
        entry = ledger[str(task["card"]["card_id"])]
        return float(same_status_and_evidence(entry, answer))
    if family == "2A.2":
        entry = ledger[str(task["target_id"])]
        return float(entry["status"] == "FOUND" and same_interval(entry, answer))
    if family == "2A.3":
        entry = ledger[str(task["target_id"])]
        if answer["status"] == "ABSENT":
            return float(entry["status"] == "LIKELY_ABSENT")
        return float(
            entry["status"] == "FOUND"
            and any(same_interval(entry, occurrence) for occurrence in answer["occurrences"])
        )
    if family == "2B.1":
        found = sorted(
            (entry for entry in ledger.values() if entry["status"] == "FOUND"),
            key=lambda entry: float(entry["start_seconds"]),
        )
        likely_absent = [
            target_id for target_id, entry in ledger.items() if entry["status"] == "LIKELY_ABSENT"
        ]
        expected_absent = [] if answer.get("absent_card_id") is None else [answer["absent_card_id"]]
        return float(
            [entry["target_id"] for entry in found] == answer["ordered_card_ids"]
            and likely_absent == expected_absent
        )
    answer_by_id = {item["utterance_id"]: item for item in answer["utterances"]}
    per_target_consistent = all(
        same_status_and_evidence(entry, answer_by_id[target_id])
        for target_id, entry in ledger.items()
    )
    found_order = [
        entry["target_id"]
        for entry in sorted(
            (entry for entry in ledger.values() if entry["status"] == "FOUND"),
            key=lambda entry: float(entry["start_seconds"]),
        )
    ]
    return float(per_target_consistent and found_order == answer["ordered_real_utterance_ids"])


def formatting_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    if not records:
        return {
            "raw_parse_success": 1.0,
            "universal_rescue_rate": 0.0,
            "repair_attempt_rate": 0.0,
            "repair_success_rate": 0.0,
            "semantic_validity_rate": 1.0,
            "repair_unsupported_rate": 0.0,
        }
    repair_attempts = [record for record in records if record.get("repair_attempted")]
    return {
        "raw_parse_success": mean(float(bool(record.get("raw_parse_valid"))) for record in records),
        "universal_rescue_rate": mean(
            float(bool(record.get("final_rescue_applied"))) for record in records
        ),
        "repair_attempt_rate": len(repair_attempts) / len(records),
        "repair_success_rate": (
            mean(float(bool(record.get("final_parse_valid"))) for record in repair_attempts)
            if repair_attempts
            else 0.0
        ),
        "semantic_validity_rate": mean(float(bool(record.get("semantic_valid", True))) for record in records),
        "repair_unsupported_rate": mean(float(bool(record.get("repair_unsupported"))) for record in records),
    }


def calibration_metrics(confidence_correct_pairs: Sequence[tuple[float, float]]) -> dict[str, float | None]:
    if not confidence_correct_pairs:
        return {"brier_score": None, "ece_10_bin": None}
    pairs = [(float(confidence), float(correct)) for confidence, correct in confidence_correct_pairs]
    brier = mean((confidence - correct) ** 2 for confidence, correct in pairs)
    ece = 0.0
    for bin_index in range(10):
        lower = bin_index / 10.0
        upper = (bin_index + 1) / 10.0
        bucket = [pair for pair in pairs if lower <= pair[0] < upper or (bin_index == 9 and pair[0] == 1.0)]
        if bucket:
            ece += len(bucket) / len(pairs) * abs(mean(pair[0] for pair in bucket) - mean(pair[1] for pair in bucket))
    return {"brier_score": brier, "ece_10_bin": ece}


def aggregate_track2_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: defaultdict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    condition_family_groups: defaultdict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    reel_groups: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        groups[(str(result["condition"]), str(result["family"]), str(result["variant"]))].append(result)
        condition_family_groups[(str(result["condition"]), str(result["family"]))].append(result)
        reel_groups[str(result["reel_id"])].append(result)

    def aggregate_group(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        scored_items = [item for item in items if item.get("run_status") == "complete"]
        output: dict[str, Any] = {
            "task_count": len(items),
            "scored_task_count": len(scored_items),
        }
        for group_name in ("answer_metrics", "evidence_metrics", "belief_metrics", "format_metrics"):
            keys = sorted({key for item in scored_items for key, value in dict(item.get(group_name) or {}).items() if isinstance(value, (int, float))})
            output[group_name] = {
                key: mean(float(item[group_name][key]) for item in scored_items if isinstance((item.get(group_name) or {}).get(key), (int, float)))
                for key in keys
            }
        calibration_pairs = [
            (float(item["confidence"]), float(item["answer_metrics"]["primary_correct"]))
            for item in scored_items
            if isinstance(item.get("confidence"), (int, float))
            and isinstance((item.get("answer_metrics") or {}).get("primary_correct"), (int, float))
        ]
        output["calibration_metrics"] = calibration_metrics(calibration_pairs)
        output["status_counts"] = {
            status: sum(item.get("run_status") == status for item in items)
            for status in sorted({str(item.get("run_status")) for item in items})
        }
        return output

    return {
        "task_count": len(results),
        "groups": {
            f"{condition}/{family}/{variant}": aggregate_group(items)
            for (condition, family, variant), items in sorted(groups.items())
        },
        "condition_family_groups": {
            f"{condition}/{family}": aggregate_group(items)
            for (condition, family), items in sorted(condition_family_groups.items())
        },
        "reel_groups": {
            reel_id: aggregate_group(items) for reel_id, items in sorted(reel_groups.items())
        },
        "overall": aggregate_group(results),
    }


__all__ = [
    "aggregate_track2_results",
    "calibration_metrics",
    "formatting_metrics",
    "interval_metrics",
    "interval_overlap",
    "merge_intervals",
    "pairwise_order_accuracy",
    "score_track2_answer",
    "score_track2_belief",
    "score_track2_evidence",
    "unique_interval_duration",
]
