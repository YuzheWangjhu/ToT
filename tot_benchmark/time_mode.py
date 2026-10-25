from __future__ import annotations

import math
import re
from typing import Any, Sequence

TIME_MODE_BINNED = "binned"
TIME_MODE_CONTINUOUS = "continuous"
TIME_MODES = [TIME_MODE_BINNED, TIME_MODE_CONTINUOUS]
CONTINUOUS_OBSERVATION_TOLERANCE_SEC = 0.05
CONTINUOUS_EXACT_THRESHOLD_SEC = 0.25


def validate_time_mode(time_mode: str) -> str:
    normalized = str(time_mode).strip().lower()
    if normalized not in TIME_MODES:
        raise RuntimeError(f"Unknown time_mode: {time_mode}")
    return normalized


def format_seconds(value: float) -> str:
    return f"{float(value):.2f}"


def parse_two_decimal_seconds(value: str) -> float | None:
    if not re.fullmatch(r"\d+\.\d{2}", str(value)):
        return None
    return float(value)


def parse_continuous_action(raw_text: str) -> dict[str, Any]:
    stripped = str(raw_text).replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0].strip()
    if stripped == "STOP":
        return {"format_valid": True, "action": "STOP"}
    match = re.fullmatch(r"QUERY\s+(\d+\.\d{2})\s+(\d+\.\d{2})", stripped)
    if not match:
        return {"format_valid": False, "reason": "invalid_action_format"}
    start = float(match.group(1))
    end = float(match.group(2))
    return {
        "format_valid": True,
        "action": "QUERY",
        "start_time_seconds": start,
        "end_time_seconds": end,
        "duration_seconds": round(end - start, 6),
    }


def validate_continuous_query(
    parsed: dict[str, Any],
    *,
    audio_duration_seconds: float,
    max_single_query_seconds: float,
    total_listening_budget_seconds: float,
    listened_seconds_total: float,
) -> tuple[bool, str]:
    start = float(parsed["start_time_seconds"])
    end = float(parsed["end_time_seconds"])
    duration = end - start
    epsilon = 1e-9
    if start < -epsilon:
        return False, "start_seconds_out_of_range"
    if end <= start + epsilon:
        return False, "end_seconds_must_exceed_start_seconds"
    if end > float(audio_duration_seconds) + epsilon:
        return False, "query_range_out_of_range"
    if duration > float(max_single_query_seconds) + epsilon:
        return False, "duration_seconds_exceeds_max_query_seconds"
    if float(listened_seconds_total) + duration > float(total_listening_budget_seconds) + epsilon:
        return False, "query_exceeds_total_seconds_budget"
    return True, ""


def merge_intervals(intervals: Sequence[tuple[float, float]], *, tolerance: float = 0.0) -> list[tuple[float, float]]:
    cleaned = [
        (float(start), float(end))
        for start, end in intervals
        if float(end) > float(start)
    ]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def interval_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return float(sum(max(0.0, float(end) - float(start)) for start, end in intervals))


def interval_overlap(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(float(left[1]), float(right[1])) - max(float(left[0]), float(right[0])))


def interval_iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = interval_overlap(left, right)
    union = max(float(left[1]), float(right[1])) - min(float(left[0]), float(right[0]))
    return float(overlap / union) if union > 0.0 else 0.0


def interval_contained(
    container_intervals: Sequence[tuple[float, float]],
    contained: tuple[float, float],
    *,
    tolerance: float = CONTINUOUS_OBSERVATION_TOLERANCE_SEC,
) -> bool:
    start, end = float(contained[0]), float(contained[1])
    for merged_start, merged_end in merge_intervals(container_intervals, tolerance=tolerance):
        if merged_start <= start + tolerance and merged_end + tolerance >= end:
            return True
    return False


def interval_set_intersection_duration(
    left: Sequence[tuple[float, float]],
    right: Sequence[tuple[float, float]],
) -> float:
    total = 0.0
    for left_interval in merge_intervals(left):
        for right_interval in merge_intervals(right):
            total += interval_overlap(left_interval, right_interval)
    return float(total)


def interval_set_unique_duration(intervals: Sequence[tuple[float, float]]) -> float:
    return interval_duration(merge_intervals(intervals))


def continuous_redundancy_counts(
    query_intervals: Sequence[tuple[float, float]],
) -> tuple[float, float, int, int]:
    listened = interval_duration(query_intervals)
    unique = 0.0
    repeated = 0.0
    redundant_count = 0
    partial_redundant_count = 0
    seen: list[tuple[float, float]] = []
    for query in query_intervals:
        query_duration = interval_duration([query])
        overlap = interval_set_intersection_duration([query], seen)
        if query_duration > 0.0 and overlap >= query_duration - 1e-9:
            redundant_count += 1
        elif overlap > 1e-9:
            partial_redundant_count += 1
        repeated += min(query_duration, overlap)
        seen.append(query)
        unique = interval_set_unique_duration(seen)
    return float(listened), float(unique), redundant_count, partial_redundant_count


def parse_continuous_localization_answer(
    raw_text: str,
    *,
    audio_duration_seconds: float,
    allow_unexplored: bool = False,
    unknown_token: str | None = None,
) -> dict[str, Any]:
    stripped = str(raw_text).replace("\r\n", "\n").replace("\r", "\n").split("\n", 1)[0].strip().upper()
    if allow_unexplored and stripped == "UNEXPLORED":
        return {"format_valid": True, "answer": "UNEXPLORED"}
    if unknown_token is not None and stripped == unknown_token:
        return {"format_valid": True, "answer": unknown_token}
    match = re.fullmatch(r"START\s+(\d+\.\d{2})\s+END\s+(\d+\.\d{2})", stripped)
    if not match:
        return {"format_valid": False, "reason": "invalid_localization_answer"}
    start = float(match.group(1))
    end = float(match.group(2))
    if start < 0.0 or end <= start or end > float(audio_duration_seconds) + 1e-9:
        return {"format_valid": False, "reason": "localization_out_of_range"}
    return {"format_valid": True, "start_time_seconds": start, "end_time_seconds": end}


def parse_continuous_segments(
    raw_text: str,
    *,
    allowed_labels: Sequence[str],
    audio_duration_seconds: float,
    allow_multi_label: bool,
    require_full_coverage: bool = True,
) -> dict[str, Any]:
    text = str(raw_text).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return {"format_valid": False, "reason": "empty_segments", "segments": None}
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    allowed = set(allowed_labels)
    segments: list[dict[str, Any]] = []
    previous_end: float | None = None
    for line in lines:
        match = re.fullmatch(r"START\s+(\d+\.\d{2})\s+END\s+(\d+\.\d{2})\s+LABEL\s+(.+)", line, re.IGNORECASE)
        if not match:
            return {"format_valid": False, "reason": "invalid_segment_row", "segments": None}
        start = float(match.group(1))
        end = float(match.group(2))
        if start < 0.0 or end <= start or end > float(audio_duration_seconds) + 0.01:
            return {"format_valid": False, "reason": "segment_out_of_range", "segments": None}
        if previous_end is not None and abs(start - previous_end) > 0.01:
            return {"format_valid": False, "reason": "segment_timeline_not_contiguous", "segments": None}
        tokens = match.group(3).strip().split()
        if len(tokens) != len(set(tokens)):
            return {"format_valid": False, "reason": "duplicate_segment_label", "segments": None}
        special = {"00", "??"} & set(tokens)
        if special:
            if len(tokens) != 1:
                return {"format_valid": False, "reason": "mixed_special_segment_label", "segments": None}
            labels: list[str] | None = None if tokens[0] == "??" else []
        else:
            if any(token not in allowed for token in tokens):
                return {"format_valid": False, "reason": "invalid_segment_label", "segments": None}
            if not allow_multi_label and len(tokens) != 1:
                return {"format_valid": False, "reason": "multiple_segment_labels_not_allowed", "segments": None}
            labels = sorted(tokens)
        segments.append(
            {
                "start_time_seconds": start,
                "end_time_seconds": end,
                "labels": labels,
            }
        )
        previous_end = end
    rounded_duration = round(float(audio_duration_seconds) + 1e-9, 2)
    if require_full_coverage:
        if abs(segments[0]["start_time_seconds"] - 0.0) > 0.01:
            return {"format_valid": False, "reason": "segment_timeline_must_start_at_zero", "segments": None}
        if abs(segments[-1]["end_time_seconds"] - rounded_duration) > 0.01:
            return {"format_valid": False, "reason": "segment_timeline_must_end_at_audio_duration", "segments": None}
    return {"format_valid": True, "segments": segments}


def label_set_at_time(segments: Sequence[dict[str, Any]], start: float, end: float) -> list[str] | None:
    midpoint = (float(start) + float(end)) / 2.0
    for segment in segments:
        if float(segment["start_time_seconds"]) - 1e-9 <= midpoint < float(segment["end_time_seconds"]) + 1e-9:
            labels = segment.get("labels")
            return None if labels is None else list(labels)
    return []


def segments_from_event_spans(
    event_spans: dict[str, dict[str, float]],
    *,
    audio_duration_seconds: float,
    observed_intervals: Sequence[tuple[float, float]] | None = None,
    unexplored_label: bool = False,
) -> list[dict[str, Any]]:
    breakpoints = {0.0, round(float(audio_duration_seconds), 2)}
    for span in event_spans.values():
        breakpoints.add(round(float(span["start_time_seconds"]), 2))
        breakpoints.add(round(float(span["end_time_seconds"]), 2))
    for start, end in observed_intervals or []:
        breakpoints.add(round(float(start), 2))
        breakpoints.add(round(float(end), 2))
    points = sorted(point for point in breakpoints if 0.0 <= point <= round(float(audio_duration_seconds), 2))
    segments: list[dict[str, Any]] = []
    for start, end in zip(points[:-1], points[1:]):
        if end <= start:
            continue
        if unexplored_label and not interval_contained(observed_intervals or [], (start, end), tolerance=0.01):
            labels: list[str] | None = None
        else:
            labels = sorted(
                label
                for label, span in event_spans.items()
                if interval_overlap(
                    (start, end),
                    (float(span["start_time_seconds"]), float(span["end_time_seconds"])),
                )
                > 1e-9
            )
        if segments and segments[-1]["labels"] == labels and abs(segments[-1]["end_time_seconds"] - start) <= 0.01:
            segments[-1]["end_time_seconds"] = end
        else:
            segments.append({"start_time_seconds": start, "end_time_seconds": end, "labels": labels})
    return segments


def segment_text(segments: Sequence[dict[str, Any]]) -> str:
    lines = []
    for segment in segments:
        labels = segment.get("labels")
        if labels is None:
            label_text = "??"
        elif not labels:
            label_text = "00"
        else:
            label_text = " ".join(str(label) for label in labels)
        lines.append(
            f"START {format_seconds(float(segment['start_time_seconds']))} "
            f"END {format_seconds(float(segment['end_time_seconds']))} LABEL {label_text}"
        )
    return "\n".join(lines)


def spans_from_segments(segments: Sequence[dict[str, Any]]) -> dict[str, dict[str, float]]:
    spans: dict[str, dict[str, float]] = {}
    for segment in segments:
        labels = segment.get("labels") or []
        for label in labels:
            start = float(segment["start_time_seconds"])
            end = float(segment["end_time_seconds"])
            if label not in spans:
                spans[label] = {"start_time_seconds": start, "end_time_seconds": end}
            else:
                spans[label]["start_time_seconds"] = min(spans[label]["start_time_seconds"], start)
                spans[label]["end_time_seconds"] = max(spans[label]["end_time_seconds"], end)
    return spans


def localization_exact_within_threshold(
    predicted: tuple[float, float],
    gold: tuple[float, float],
    *,
    threshold_seconds: float = CONTINUOUS_EXACT_THRESHOLD_SEC,
) -> float:
    return float(
        abs(float(predicted[0]) - float(gold[0])) <= threshold_seconds
        and abs(float(predicted[1]) - float(gold[1])) <= threshold_seconds
    )


def segment_time_accuracy(
    predicted_segments: Sequence[dict[str, Any]],
    expected_segments: Sequence[dict[str, Any]],
    *,
    audio_duration_seconds: float,
    observed_intervals: Sequence[tuple[float, float]] | None = None,
    credit_unexplored: float = 0.5,
) -> float:
    breakpoints = {0.0, round(float(audio_duration_seconds), 2)}
    for segment in list(predicted_segments) + list(expected_segments):
        breakpoints.add(round(float(segment["start_time_seconds"]), 2))
        breakpoints.add(round(float(segment["end_time_seconds"]), 2))
    for start, end in observed_intervals or []:
        breakpoints.add(round(float(start), 2))
        breakpoints.add(round(float(end), 2))
    points = sorted(point for point in breakpoints if 0.0 <= point <= round(float(audio_duration_seconds), 2))
    total = 0.0
    score = 0.0
    for start, end in zip(points[:-1], points[1:]):
        duration = max(0.0, end - start)
        if duration <= 0.0:
            continue
        predicted = label_set_at_time(predicted_segments, start, end)
        expected = label_set_at_time(expected_segments, start, end)
        if predicted == expected:
            slice_score = 1.0
        elif predicted is None and observed_intervals is not None and not interval_contained(observed_intervals, (start, end), tolerance=0.01):
            slice_score = credit_unexplored
        else:
            slice_score = 0.0
        total += duration
        score += duration * slice_score
    return float(score / total) if total > 0.0 else 0.0


def segment_exact_match_within_threshold(
    predicted_segments: Sequence[dict[str, Any]],
    expected_segments: Sequence[dict[str, Any]],
    *,
    threshold_seconds: float = CONTINUOUS_EXACT_THRESHOLD_SEC,
) -> float:
    if len(predicted_segments) != len(expected_segments):
        return 0.0
    for predicted, expected in zip(predicted_segments, expected_segments):
        if predicted.get("labels") != expected.get("labels"):
            return 0.0
        if abs(float(predicted["start_time_seconds"]) - float(expected["start_time_seconds"])) > threshold_seconds:
            return 0.0
        if abs(float(predicted["end_time_seconds"]) - float(expected["end_time_seconds"])) > threshold_seconds:
            return 0.0
    return 1.0


def label_iou_from_segments(
    predicted_segments: Sequence[dict[str, Any]],
    expected_segments: Sequence[dict[str, Any]],
    label: str,
) -> float | None:
    pred_intervals = [
        (float(segment["start_time_seconds"]), float(segment["end_time_seconds"]))
        for segment in predicted_segments
        if label in (segment.get("labels") or [])
    ]
    gold_intervals = [
        (float(segment["start_time_seconds"]), float(segment["end_time_seconds"]))
        for segment in expected_segments
        if label in (segment.get("labels") or [])
    ]
    pred_duration = interval_set_unique_duration(pred_intervals)
    gold_duration = interval_set_unique_duration(gold_intervals)
    if pred_duration <= 0.0 and gold_duration <= 0.0:
        return None
    intersection = interval_set_intersection_duration(pred_intervals, gold_intervals)
    union = pred_duration + gold_duration - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def mean_label_iou(
    predicted_segments: Sequence[dict[str, Any]],
    expected_segments: Sequence[dict[str, Any]],
    labels: Sequence[str],
) -> float:
    values = [value for label in labels for value in [label_iou_from_segments(predicted_segments, expected_segments, label)] if value is not None]
    return float(sum(values) / len(values)) if values else 0.0


def relation_from_spans(left_span: dict[str, float], right_span: dict[str, float]) -> str:
    return "BEFORE" if float(left_span["start_time_seconds"]) < float(right_span["start_time_seconds"]) else "AFTER"


def gap_from_spans(
    left_span: dict[str, float],
    right_span: dict[str, float],
    *,
    tolerance: float = CONTINUOUS_OBSERVATION_TOLERANCE_SEC,
) -> str:
    gap = float(right_span["start_time_seconds"]) - float(left_span["end_time_seconds"])
    return "ADJACENT" if abs(gap) <= tolerance else "GAP"


def max_concurrency_from_segments(segments: Sequence[dict[str, Any]]) -> int:
    return max((len(segment.get("labels") or []) for segment in segments), default=0)

