from __future__ import annotations

import base64
import io
import json
import math
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import soundfile as sf

from tests.audio_fixtures import write_audio_fixture

from tot_benchmark.adapters import (
    API_ERROR_SOFT,
    APIAdapterError,
    AdapterSession,
    DummyAdapter,
    GeminiAudioAdapter,
    LocalQwenOmniAdapter,
    OpenAIResponsesAudioAdapter,
    adapter_from_config,
    load_api_key_from_env_file,
)
from tot_benchmark import track1c
from tot_benchmark.audio_model_adapters import (
    AudioFlamingoNextAdapter,
    GPTAudioMiniAdapter,
    KimiAudioAdapter,
    MOSSAudioAdapter,
    MiMoAudioAdapter,
    _normalized_output,
    _strip_leading_think_blocks,
)
from tot_benchmark.core import (
    _build_behavioral_plan,
    _explored_events_in_order,
    _score_behavioral_outputs,
    _score_uncertainty_metrics,
    ACTIVE_SAME_SESSION_PERIODIC_PROBE,
    BenchmarkConfig,
    EpisodeEnvironment,
    build_arg_parser,
    build_datasets_for_tracks,
    build_dataset,
    build_probe_repair_prompt,
    parse_action,
    parse_gap_answer,
    parse_localization_answer,
    parse_pairwise_answer,
    parse_raster,
    parse_sequence_answer,
    run_benchmark,
    score_belief_metrics,
    score_generic_master_metrics,
)
from tot_benchmark.continuous_tracks import (
    ContinuousEpisodeEnvironment,
    score_continuous_belief_metrics,
)
from tot_benchmark.time_mode import (
    TIME_MODE_CONTINUOUS,
    continuous_redundancy_counts,
    interval_contained,
    interval_iou,
    merge_intervals,
    parse_continuous_action,
    parse_continuous_localization_answer,
    parse_continuous_segments,
    segment_text,
    segment_time_accuracy,
    validate_continuous_query,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

# Build disposable non-speech assets so tests do not require unreleased TTS audio.
_AUDIO_TEMP: tempfile.TemporaryDirectory | None = None
AUDIO_ROOT: Path


def setUpModule() -> None:
    global _AUDIO_TEMP, AUDIO_ROOT
    _AUDIO_TEMP = tempfile.TemporaryDirectory(prefix="attend-test-audio-")
    AUDIO_ROOT = write_audio_fixture(Path(_AUDIO_TEMP.name) / "assets")


def tearDownModule() -> None:
    assert _AUDIO_TEMP is not None
    _AUDIO_TEMP.cleanup()


def raster_text_from_bins(bins: list[list[str]]) -> str:
    lines = []
    for index, bin_labels in enumerate(bins):
        contents = " ".join(bin_labels) if bin_labels else "00"
        lines.append(f"{index}: {contents}")
    return "\n".join(lines)


def unresolved_raster_text(num_bins: int) -> str:
    return "\n".join(f"{index}: ??" for index in range(num_bins))


def interval_overlap_sec(left: tuple[float, float], right: tuple[float, float]) -> float:
    return max(0.0, min(left[1], right[1]) - max(left[0], right[0]))


def interval_contains(container: tuple[float, float], contained: tuple[float, float]) -> bool:
    return container[0] <= contained[0] and contained[1] <= container[1]


class DatasetBuilderTests(unittest.TestCase):
    def test_build_dataset_is_deterministic_and_uses_subsecond_assets(self) -> None:
        config = BenchmarkConfig(num_episodes=2, seed=1234)
        with tempfile.TemporaryDirectory() as tmpdir_a, tempfile.TemporaryDirectory() as tmpdir_b:
            manifest_a = build_dataset(AUDIO_ROOT, Path(tmpdir_a), config)
            manifest_b = build_dataset(AUDIO_ROOT, Path(tmpdir_b), config)

            self.assertEqual(manifest_a["episodes"], manifest_b["episodes"])
            self.assertEqual(len(manifest_a["episodes"]), 2)

            for episode in manifest_a["episodes"]:
                audio_path = Path(tmpdir_a) / episode["audio_path"]
                info = sf.info(str(audio_path))
                self.assertAlmostEqual(info.duration, config.episode_duration_sec, places=3)
                self.assertEqual(len(episode["ground_truth_raster"]), config.num_bins)
                for bin_labels in episode["ground_truth_raster"]:
                    self.assertIsInstance(bin_labels, list)
                    self.assertNotIn("??", bin_labels)
                    self.assertLessEqual(len(bin_labels), 1)
                self.assertEqual(len(set(episode["labels_present"])), len(episode["labels_present"]))
                self.assertEqual(len(episode["behavioral_queries"]["pairwise_order"]), 2)
                self.assertEqual(len(episode["behavioral_queries"]["localization"]), 1)
                self.assertEqual(len(episode["behavioral_queries"]["gap_adjacency"]), 1)
                for provenance in episode["asset_provenance"].values():
                    self.assertLess(provenance["duration_sec"], 1.0)

    def test_build_track_1b_has_overlap_and_multibin_events(self) -> None:
        config = BenchmarkConfig(track_id="1b", num_episodes=2, seed=5678)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)

            self.assertEqual(manifest["benchmark_defaults"]["track_id"], "1b")
            for episode in manifest["episodes"]:
                info = sf.info(str(Path(tmpdir) / episode["audio_path"]))
                self.assertAlmostEqual(info.duration, config.episode_duration_sec, places=3)
                self.assertTrue(any(len(bin_labels) == 2 for bin_labels in episode["ground_truth_raster"]))
                self.assertTrue(
                    any(item["end_bin"] > item["start_bin"] for item in episode["event_bin_assignments"])
                )
                self.assertTrue(all(len(bin_labels) <= 2 for bin_labels in episode["ground_truth_raster"]))
                for item in episode["event_bin_assignments"]:
                    duration_sec = episode["asset_provenance"][item["label"]]["duration_sec"]
                    expected_end_bin = item["start_bin"] + math.ceil(duration_sec / config.bin_duration_sec) - 1
                    self.assertEqual(item["end_bin"], expected_end_bin)
                    for bin_index in range(item["start_bin"], item["end_bin"] + 1):
                        self.assertIn(item["label"], episode["ground_truth_raster"][bin_index])
                        bin_start = bin_index * config.bin_duration_sec
                        bin_end = bin_start + config.bin_duration_sec
                        event_start = item["start_bin"] * config.bin_duration_sec
                        event_end = event_start + duration_sec
                        coverage_sec = max(0.0, min(event_end, bin_end) - max(event_start, bin_start))
                        self.assertGreaterEqual(coverage_sec, 0.30 * config.bin_duration_sec)

                intervals = {}
                durations = {}
                for item in episode["event_bin_assignments"]:
                    duration_sec = episode["asset_provenance"][item["label"]]["duration_sec"]
                    start_sec = item["start_bin"] * config.bin_duration_sec
                    intervals[item["label"]] = (start_sec, start_sec + duration_sec)
                    durations[item["label"]] = duration_sec
                labels = list(intervals)
                for left_index, left_label in enumerate(labels):
                    for right_label in labels[left_index + 1 :]:
                        left_interval = intervals[left_label]
                        right_interval = intervals[right_label]
                        overlap_sec = interval_overlap_sec(left_interval, right_interval)
                        if overlap_sec <= 0:
                            continue
                        if interval_contains(left_interval, right_interval) or interval_contains(right_interval, left_interval):
                            continue
                        self.assertGreaterEqual(overlap_sec, 0.25 * durations[left_label])
                        self.assertGreaterEqual(overlap_sec, 0.25 * durations[right_label])

    def test_build_datasets_for_tracks_uses_subdirectories_for_multiple_tracks(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=2468)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifests = build_datasets_for_tracks(
                audio_root=AUDIO_ROOT,
                output_root=root,
                tracks=["1a", "1b"],
                config=config,
            )

            self.assertEqual(sorted(manifests), ["1a", "1b"])
            self.assertTrue((root / "track_1a" / "dataset" / "manifest.json").is_file())
            self.assertTrue((root / "track_1b" / "dataset" / "manifest.json").is_file())

    def test_binned_manifest_defaults_do_not_gain_time_mode_field(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=1357)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)

        self.assertNotIn("time_mode", manifest["benchmark_defaults"])

    def test_build_continuous_track_1a_has_noninteger_time_spans(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=9753, time_mode=TIME_MODE_CONTINUOUS)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)

        episode = manifest["episodes"][0]
        self.assertEqual(manifest["time_mode"], TIME_MODE_CONTINUOUS)
        self.assertEqual(episode["track_display_id"], "1a-H")
        self.assertIn("ground_truth_event_spans", episode)
        self.assertIn("ground_truth_segments", episode)
        self.assertNotIn("ground_truth_raster", episode)
        starts = [span["start_time_seconds"] for span in episode["ground_truth_event_spans"].values()]
        self.assertTrue(any(abs(start - round(start)) > 1e-9 for start in starts))

    def test_build_continuous_track_1b_has_overlap_metadata(self) -> None:
        config = BenchmarkConfig(track_id="1b", num_episodes=1, seed=8642, time_mode=TIME_MODE_CONTINUOUS)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)

        episode = manifest["episodes"][0]
        self.assertEqual(episode["track_display_id"], "1b-H")
        self.assertTrue(episode["ground_truth_overlap_relations"])
        self.assertGreaterEqual(episode["ground_truth_max_concurrency"], 2)


class ParserAndScoringTests(unittest.TestCase):
    def test_generic_master_score_matches_track1c_weight_shape(self) -> None:
        format_metrics = {
            "action_format_valid_rate": 1.0,
            "action_semantic_valid_rate": 1.0,
        }
        exploration_metrics = {
            "event_coverage_rate": 1.0,
            "event_bin_coverage_rate": 1.0,
            "repeated_bin_total": 0.0,
        }
        belief_metrics = {
            "belief_bin_score_mean": 1.0,
            "belief_sequence_exact": 1.0,
            "belief_pairwise_accuracy": 1.0,
            "belief_localization_mean_iou": 1.0,
            "belief_gap_accuracy": 1.0,
        }
        core_metrics = {
            "core_behavior_sequence_exact": 1.0,
            "core_behavior_pairwise_accuracy": 1.0,
            "core_behavior_localization_iou": 1.0,
            "core_behavior_localization_exact": 1.0,
            "core_behavior_gap_accuracy": 1.0,
        }
        uncertainty_metrics = {
            "uncertainty_pairwise_accuracy": 1.0,
            "uncertainty_pairwise_count": 1.0,
            "uncertainty_localization_accuracy": 1.0,
            "uncertainty_localization_count": 1.0,
            "uncertainty_gap_accuracy": 1.0,
            "uncertainty_gap_count": 1.0,
        }

        active = score_generic_master_metrics(
            track_id="1a",
            condition="active",
            stop_reason="model_stop",
            listened_bin_total=3,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
            belief_metrics=belief_metrics,
            core_metrics=core_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )
        invalid = score_generic_master_metrics(
            track_id="1a",
            condition="active",
            stop_reason="invalid_action:range",
            listened_bin_total=3,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
            belief_metrics=belief_metrics,
            core_metrics=core_metrics,
            uncertainty_metrics=uncertainty_metrics,
        )
        full_context = score_generic_master_metrics(
            track_id="1b",
            condition="full_context",
            stop_reason="full_context",
            listened_bin_total=10,
            format_metrics={"probe_valid_rate": 1.0},
            exploration_metrics={},
            belief_metrics=belief_metrics,
            core_metrics=core_metrics,
            uncertainty_metrics={
                "uncertainty_pairwise_count": 0.0,
                "uncertainty_localization_count": 0.0,
                "uncertainty_gap_count": 0.0,
            },
        )

        self.assertEqual(active["track1a_active_protocol_score"], 1.0)
        self.assertEqual(active["track1a_active_exploration_score"], 1.0)
        self.assertEqual(active["track1a_active_answer_score"], 1.0)
        self.assertEqual(active["track1a_active_master_score"], 100.0)
        self.assertEqual(invalid["track1a_active_protocol_score"], 0.0)
        self.assertEqual(invalid["track1a_active_master_score"], 0.0)
        self.assertEqual(full_context["track1b_full_context_answer_score"], 1.0)
        self.assertEqual(full_context["track1b_full_context_master_score"], 100.0)

    def test_parsers_accept_valid_formats(self) -> None:
        vocabulary = ["red", "blue", "delta"]
        self.assertTrue(parse_action("QUERY 2 3")["format_valid"])
        self.assertTrue(parse_action("STOP")["format_valid"])
        parsed_raster = parse_raster(
            "\n".join(
                [
                    "0: red",
                    "1: 00",
                    "2: blue red",
                    "3: ??",
                    "4: delta",
                ]
            ),
            vocabulary,
            5,
        )
        self.assertTrue(parsed_raster["format_valid"])
        self.assertEqual(parsed_raster["bins"], [["red"], [], ["blue", "red"], None, ["delta"]])
        parsed_raster_with_trailing_text = parse_raster(
            "\n".join(
                [
                    "0: red",
                    "1: 00",
                    "2: blue red",
                    "3: ??",
                    "4: delta",
                    "Human: ignore this overgenerated continuation",
                ]
            ),
            vocabulary,
            5,
        )
        self.assertTrue(parsed_raster_with_trailing_text["format_valid"])
        self.assertEqual(parsed_raster_with_trailing_text["bins"], parsed_raster["bins"])
        self.assertTrue(parse_sequence_answer("red -> blue -> delta", vocabulary)["format_valid"])
        self.assertTrue(parse_pairwise_answer("before")["format_valid"])
        self.assertTrue(parse_pairwise_answer("unexplored", allow_unexplored=True)["format_valid"])
        self.assertTrue(parse_localization_answer("START 2 END 2", 10)["format_valid"])
        self.assertTrue(parse_localization_answer("UNEXPLORED", 10, allow_unexplored=True)["format_valid"])
        self.assertTrue(parse_gap_answer("gap")["format_valid"])
        self.assertTrue(parse_gap_answer("UNEXPLORED", allow_unexplored=True)["format_valid"])

    def test_continuous_action_parser_is_strict_two_decimal(self) -> None:
        parsed = parse_continuous_action("QUERY 2.00 5.25")

        self.assertTrue(parsed["format_valid"])
        self.assertEqual(parsed["start_time_seconds"], 2.0)
        self.assertEqual(parsed["end_time_seconds"], 5.25)
        self.assertFalse(parse_continuous_action("QUERY 2 5")["format_valid"])
        self.assertFalse(parse_continuous_action("QUERY 2.0 5.00")["format_valid"])
        self.assertTrue(parse_continuous_action("STOP")["format_valid"])
        self.assertEqual(
            validate_continuous_query(
                parsed,
                audio_duration_seconds=10.0,
                max_single_query_seconds=3.5,
                total_listening_budget_seconds=8.0,
                listened_seconds_total=0.0,
            ),
            (True, ""),
        )
        self.assertEqual(
            validate_continuous_query(
                parsed,
                audio_duration_seconds=10.0,
                max_single_query_seconds=3.0,
                total_listening_budget_seconds=8.0,
                listened_seconds_total=0.0,
            )[1],
            "duration_seconds_exceeds_max_query_seconds",
        )

    def test_continuous_interval_and_segment_scoring(self) -> None:
        self.assertEqual(merge_intervals([(0.0, 1.0), (0.9, 2.0), (3.0, 4.0)]), [(0.0, 2.0), (3.0, 4.0)])
        self.assertTrue(interval_contained([(0.0, 1.0), (1.0, 2.0)], (0.05, 1.95)))
        self.assertAlmostEqual(interval_iou((0.0, 2.0), (1.0, 3.0)), 1.0 / 3.0)
        listened, unique, redundant, partial = continuous_redundancy_counts([(0.0, 1.0), (0.5, 1.5), (0.0, 1.5)])
        self.assertAlmostEqual(listened, 3.5)
        self.assertAlmostEqual(unique, 1.5)
        self.assertEqual(redundant, 1)
        self.assertEqual(partial, 1)

        expected = [
            {"start_time_seconds": 0.0, "end_time_seconds": 1.0, "labels": ["red"]},
            {"start_time_seconds": 1.0, "end_time_seconds": 2.0, "labels": []},
        ]
        parsed = parse_continuous_segments(
            segment_text(expected),
            allowed_labels=["red"],
            audio_duration_seconds=2.0,
            allow_multi_label=False,
        )
        self.assertTrue(parsed["format_valid"])
        self.assertEqual(segment_time_accuracy(parsed["segments"], expected, audio_duration_seconds=2.0), 1.0)

    def test_continuous_localization_parser_accepts_unexplored(self) -> None:
        parsed = parse_continuous_localization_answer("START 1.25 END 2.50", audio_duration_seconds=3.0)

        self.assertTrue(parsed["format_valid"])
        self.assertFalse(
            parse_continuous_localization_answer("START 1 END 2.50", audio_duration_seconds=3.0)["format_valid"]
        )
        self.assertTrue(
            parse_continuous_localization_answer(
                "UNEXPLORED",
                audio_duration_seconds=3.0,
                allow_unexplored=True,
            )["format_valid"]
        )

    def test_parse_raster_rejects_invalid_vertical_formats(self) -> None:
        vocabulary = ["red", "blue", "delta"]
        invalid_inputs = [
            "red 00 ?? blue 00",
            "0: red\n1: 00\n2: blue\n3: 00",
            "0: red\n1: 00\n1: blue\n3: 00\n4: 00",
            "0: red\n2: blue\n1: 00\n3: 00\n4: 00",
            "0: red\n1: unknown\n2: blue\n3: 00\n4: 00",
            "0: red\n1: 00 red\n2: blue\n3: 00\n4: 00",
            "0: red\n1: ?? red\n2: blue\n3: 00\n4: 00",
            "0: red\n1: blue blue\n2: delta\n3: 00\n4: 00",
        ]
        for raw_text in invalid_inputs:
            with self.subTest(raw_text=raw_text):
                self.assertFalse(parse_raster(raw_text, vocabulary, 5)["format_valid"])

    def test_probe_repair_prompt_uses_vertical_raster_format(self) -> None:
        prompt = build_probe_repair_prompt("wrong_row_count", BenchmarkConfig(num_bins=3))

        self.assertIn("Return exactly 3 lines", prompt)
        self.assertIn("0: ??\n1: ??\n2: ??", prompt)
        self.assertNotIn("space-separated tokens", prompt)

    def test_multiline_normalization_preserves_probe_raster(self) -> None:
        raw = "0: ??\n1: ??\n2: ??"

        self.assertEqual(_normalized_output(raw), "0: ??")
        self.assertEqual(_normalized_output(raw, preserve_multiline=True), raw)

    def test_strip_leading_think_blocks_returns_only_final_answer(self) -> None:
        raw = "<think>Inspect events.\nCompare timestamps.</think>\n0: ??\n1: bell"

        self.assertEqual(_strip_leading_think_blocks(raw), "0: ??\n1: bell")

    def test_strip_leading_think_blocks_preserves_malformed_output(self) -> None:
        answerless = "<think>Still reasoning.</think>"
        truncated = "<think>Truncated reasoning"

        self.assertEqual(_strip_leading_think_blocks(answerless), answerless)
        self.assertEqual(_strip_leading_think_blocks(truncated), truncated)

    def test_score_belief_metrics_handles_missing_and_exact_predictions(self) -> None:
        episode = {
            "ground_truth_raster": [[], ["red"], [], ["blue"], [], ["delta"], [], [], [], []],
            "ground_truth_sequence": ["red", "blue", "delta"],
            "ground_truth_spans": {
                "red": {"start_bin": 1, "end_bin": 1},
                "blue": {"start_bin": 3, "end_bin": 3},
                "delta": {"start_bin": 5, "end_bin": 5},
            },
            "ground_truth_gap_relations": [
                {"left_label": "red", "right_label": "blue", "relation": "GAP"},
                {"left_label": "blue", "right_label": "delta", "relation": "GAP"},
            ],
        }
        exact = score_belief_metrics(
            {"format_valid": True, "bins": episode["ground_truth_raster"]},
            episode,
            observed_bins={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        )
        self.assertEqual(exact["belief_sequence_exact"], 1.0)
        self.assertEqual(exact["belief_pairwise_accuracy"], 1.0)
        self.assertEqual(exact["belief_localization_exact"], 1.0)
        self.assertEqual(exact["belief_gap_accuracy"], 1.0)

        missing = score_belief_metrics(
            {"format_valid": True, "bins": [None] * 10},
            episode,
            observed_bins=set(),
        )
        self.assertLess(missing["belief_bin_score_mean"], 1.0)
        self.assertEqual(missing["belief_sequence_exact"], 0.0)

    def test_score_continuous_belief_metrics_handles_segments_and_overlap(self) -> None:
        episode = {
            "track_id": "1b",
            "audio_duration_seconds": 3.0,
            "ground_truth_sequence": ["red", "blue"],
            "ground_truth_event_spans": {
                "red": {"start_time_seconds": 0.50, "end_time_seconds": 2.00},
                "blue": {"start_time_seconds": 1.00, "end_time_seconds": 2.50},
            },
            "ground_truth_segments": [
                {"start_time_seconds": 0.0, "end_time_seconds": 0.5, "labels": []},
                {"start_time_seconds": 0.5, "end_time_seconds": 1.0, "labels": ["red"]},
                {"start_time_seconds": 1.0, "end_time_seconds": 2.0, "labels": ["blue", "red"]},
                {"start_time_seconds": 2.0, "end_time_seconds": 2.5, "labels": ["blue"]},
                {"start_time_seconds": 2.5, "end_time_seconds": 3.0, "labels": []},
            ],
            "ground_truth_gap_relations": [],
            "ground_truth_overlap_relations": [
                {"label_a": "red", "label_b": "blue", "start_time_seconds": 1.0, "end_time_seconds": 2.0}
            ],
            "ground_truth_max_concurrency": 2,
        }
        parsed = parse_continuous_segments(
            segment_text(episode["ground_truth_segments"]),
            allowed_labels=["red", "blue"],
            audio_duration_seconds=3.0,
            allow_multi_label=True,
        )

        metrics = score_continuous_belief_metrics(parsed, episode, [(0.0, 3.0)])

        self.assertEqual(metrics["belief_time_score_mean"], 1.0)
        self.assertEqual(metrics["belief_sequence_exact"], 1.0)
        self.assertEqual(metrics["belief_overlap_pair_f1"], 1.0)
        self.assertEqual(metrics["belief_overlap_interval_iou"], 1.0)

    def test_behavioral_plan_skips_overlapping_gap_pairs(self) -> None:
        episode = {
            "ground_truth_sequence": ["red", "blue", "delta"],
            "ground_truth_spans": {
                "red": {"start_bin": 1, "end_bin": 3},
                "blue": {"start_bin": 2, "end_bin": 4},
                "delta": {"start_bin": 6, "end_bin": 7},
            },
        }
        plan = _build_behavioral_plan(episode, {1, 2, 3, 4, 6, 7})

        self.assertEqual(plan["core"]["pairwise_order"][0]["answer"], "BEFORE")
        self.assertEqual(plan["core"]["localization"][0]["answer"], "START 1 END 3")
        self.assertEqual(plan["core"]["gap_adjacency"][0]["left_label"], "blue")
        self.assertEqual(plan["core"]["gap_adjacency"][0]["right_label"], "delta")
        self.assertEqual(plan["core"]["gap_adjacency"][0]["answer"], "GAP")

    def test_behavioral_plan_tracks_explored_and_unexplored_events(self) -> None:
        episode = {
            "ground_truth_sequence": ["red", "blue", "delta"],
            "ground_truth_spans": {
                "red": {"start_bin": 1, "end_bin": 1},
                "blue": {"start_bin": 3, "end_bin": 3},
                "delta": {"start_bin": 5, "end_bin": 5},
            },
        }
        observed_bins = {1}

        self.assertEqual(_explored_events_in_order(episode, observed_bins), ["red"])
        plan = _build_behavioral_plan(episode, observed_bins)

        self.assertEqual(plan["explored_events"], ["red"])
        self.assertEqual(plan["unexplored_events"], ["blue", "delta"])
        self.assertEqual(plan["coverage_metrics"]["event_coverage_count"], 1)
        self.assertEqual(plan["coverage_metrics"]["pairwise_core_evaluable"], 0.0)
        self.assertEqual(plan["coverage_metrics"]["localization_core_evaluable"], 1.0)
        self.assertEqual(plan["coverage_metrics"]["gap_core_evaluable"], 0.0)
        self.assertEqual(plan["core"]["sequence"]["answer"], "red")
        self.assertEqual(plan["core"]["localization"][0]["label"], "red")
        self.assertEqual(plan["uncertainty"]["pairwise_order"][0]["answer"], "UNEXPLORED")
        self.assertEqual(plan["uncertainty"]["localization"][0]["label"], "blue")
        self.assertEqual(plan["uncertainty"]["gap_adjacency"][0]["left_label"], "red")

    def test_behavioral_scoring_separates_core_and_uncertainty(self) -> None:
        episode = {
            "ground_truth_sequence": ["red", "blue", "delta"],
            "ground_truth_spans": {
                "red": {"start_bin": 1, "end_bin": 1},
                "blue": {"start_bin": 3, "end_bin": 3},
                "delta": {"start_bin": 5, "end_bin": 5},
            },
        }
        plan = _build_behavioral_plan(episode, {1})
        core_outputs = {
            "sequence": {"asked": True, "parsed": {"format_valid": True, "tokens": ["red"]}, "repair_attempts": []},
            "pairwise_order": {},
            "localization": {
                "core_localization_1": {
                    "asked": True,
                    "parsed": {"format_valid": True, "start_bin": 1, "end_bin": 1},
                    "repair_attempts": [],
                }
            },
            "gap_adjacency": {},
        }
        uncertainty_outputs = {
            "pairwise_order": {
                "uncertainty_pairwise_1": {
                    "asked": True,
                    "parsed": {"format_valid": True, "answer": "UNEXPLORED"},
                    "repair_attempts": [],
                }
            },
            "localization": {
                "uncertainty_localization_1": {
                    "asked": True,
                    "parsed": {"format_valid": True, "answer": "UNEXPLORED"},
                    "repair_attempts": [],
                }
            },
            "gap_adjacency": {
                "uncertainty_gap_1": {
                    "asked": True,
                    "parsed": {"format_valid": True, "answer": "UNEXPLORED"},
                    "repair_attempts": [],
                }
            },
        }

        core_metrics = _score_behavioral_outputs(core_outputs, plan["core"], "core")
        uncertainty_metrics = _score_uncertainty_metrics(uncertainty_outputs, plan["uncertainty"])

        self.assertEqual(core_metrics["core_behavior_sequence_exact"], 1.0)
        self.assertIsNone(core_metrics["core_behavior_pairwise_accuracy"])
        self.assertEqual(core_metrics["core_behavior_localization_exact"], 1.0)
        self.assertEqual(core_metrics["core_behavior_localization_iou"], 1.0)
        self.assertIsNone(core_metrics["core_behavior_gap_accuracy"])
        self.assertEqual(uncertainty_metrics["uncertainty_pairwise_accuracy"], 1.0)
        self.assertEqual(uncertainty_metrics["uncertainty_localization_accuracy"], 1.0)
        self.assertEqual(uncertainty_metrics["uncertainty_gap_accuracy"], 1.0)

        empty_uncertainty_metrics = _score_uncertainty_metrics(
            {"pairwise_order": {}, "localization": {}, "gap_adjacency": {}},
            {"pairwise_order": [], "localization": [], "gap_adjacency": []},
        )
        self.assertIsNone(empty_uncertainty_metrics["uncertainty_pairwise_accuracy"])
        self.assertIsNone(empty_uncertainty_metrics["uncertainty_localization_accuracy"])
        self.assertIsNone(empty_uncertainty_metrics["uncertainty_gap_accuracy"])
        self.assertEqual(empty_uncertainty_metrics["uncertainty_pairwise_count"], 0.0)

        wrong_uncertainty_metrics = _score_uncertainty_metrics(
            {
                "pairwise_order": {
                    "uncertainty_pairwise_1": {
                        "parsed": {"format_valid": True, "answer": "BEFORE"},
                    }
                },
                "localization": {},
                "gap_adjacency": {},
            },
            {
                "pairwise_order": plan["uncertainty"]["pairwise_order"],
                "localization": [],
                "gap_adjacency": [],
            },
        )
        self.assertEqual(wrong_uncertainty_metrics["uncertainty_pairwise_accuracy"], 0.0)


class GeminiAdapterTests(unittest.TestCase):
    def test_adapter_from_config_builds_gemini_adapter(self) -> None:
        adapter = adapter_from_config(
            {
                "adapter_type": "gemini",
                "model_id": "gemini-2.5-flash",
                "api_key_env_path": "/tmp/fake.env",
                "api_key_names": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
            }
        )

        self.assertIsInstance(adapter, GeminiAudioAdapter)
        self.assertEqual(adapter.model_id, "gemini-2.5-flash")

    def test_gemini_adapter_loads_api_key_from_supported_env_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "keys.env"
            env_path.write_text("GEMINI_API_KEY=gemini-key\n", encoding="utf-8")

            self.assertEqual(
                load_api_key_from_env_file(
                    env_path=env_path,
                    key_names=["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                ),
                "gemini-key",
            )

    def test_gemini_adapter_send_text_and_audio_with_mock_sdk(self) -> None:
        calls: dict[str, object] = {}

        class FakeResponse:
            text = "QUERY 0 1"

        class FakeModels:
            def generate_content(self, **kwargs):
                calls.update(kwargs)
                return FakeResponse()

        class FakeClient:
            def __init__(self, api_key: str) -> None:
                calls["api_key"] = api_key
                self.models = FakeModels()

        class FakePart:
            def __init__(self, *, text: str | None = None, audio: bytes | None = None, mime_type: str | None = None):
                self.text = text
                self.audio = audio
                self.mime_type = mime_type

            @staticmethod
            def from_bytes(*, data: bytes, mime_type: str):
                calls["part_data"] = data
                calls["part_mime_type"] = mime_type
                return FakePart(audio=data, mime_type=mime_type)

        class FakeContent:
            def __init__(self, *, role: str, parts: list[FakePart]) -> None:
                self.role = role
                self.parts = parts

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_genai = types.SimpleNamespace(Client=FakeClient)
        fake_types = types.SimpleNamespace(
            Part=FakePart,
            UserContent=lambda *, parts: FakeContent(role="user", parts=parts),
            ModelContent=lambda *, parts: FakeContent(role="model", parts=parts),
            GenerateContentConfig=FakeGenerateContentConfig,
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        fake_google_genai = types.ModuleType("google.genai")
        fake_google_genai.types = fake_types

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            env_path = tmp_root / "keys.env"
            env_path.write_text("GOOGLE_API_KEY=test-key\n", encoding="utf-8")
            audio_path = tmp_root / "clip.wav"
            audio_path.write_bytes(b"wav")
            adapter = GeminiAudioAdapter(
                model_id="gemini-2.5-flash",
                api_key_env_path=env_path,
            )
            with patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.genai": fake_google_genai,
                },
            ):
                session = adapter.create_session("system prompt")
                text = adapter.send(session, "user prompt", audio_paths=[audio_path])

        self.assertEqual(text, "QUERY 0 1")
        self.assertEqual(calls["api_key"], "test-key")
        self.assertEqual(calls["part_data"], b"wav")
        self.assertEqual(calls["part_mime_type"], "audio/wav")
        self.assertEqual(calls["model"], "gemini-2.5-flash")
        contents = calls["contents"]
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "user prompt")
        self.assertEqual(contents[0].parts[1].audio, b"wav")
        self.assertEqual(contents[0].parts[1].mime_type, "audio/wav")
        self.assertEqual(calls["config"].kwargs["system_instruction"], "system prompt")
        self.assertEqual(session.state["messages"][-1], {"role": "model", "text": "QUERY 0 1"})

    def test_gemini_adapter_replays_full_message_history_with_audio(self) -> None:
        calls: list[dict[str, object]] = []

        class FakeResponse:
            def __init__(self, text: str) -> None:
                self.text = text

        class FakeModels:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse("QUERY 0 1" if len(calls) == 1 else "STOP")

        class FakeClient:
            def __init__(self, api_key: str) -> None:
                self.models = FakeModels()

        class FakePart:
            def __init__(self, *, text: str | None = None, audio: bytes | None = None, mime_type: str | None = None):
                self.text = text
                self.audio = audio
                self.mime_type = mime_type

            @staticmethod
            def from_bytes(*, data: bytes, mime_type: str):
                return FakePart(audio=data, mime_type=mime_type)

        class FakeContent:
            def __init__(self, *, role: str, parts: list[FakePart]) -> None:
                self.role = role
                self.parts = parts

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_genai = types.SimpleNamespace(Client=FakeClient)
        fake_types = types.SimpleNamespace(
            Part=FakePart,
            UserContent=lambda *, parts: FakeContent(role="user", parts=parts),
            ModelContent=lambda *, parts: FakeContent(role="model", parts=parts),
            GenerateContentConfig=FakeGenerateContentConfig,
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        fake_google_genai = types.ModuleType("google.genai")
        fake_google_genai.types = fake_types

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            env_path = tmp_root / "keys.env"
            env_path.write_text("GOOGLE_API_KEY=test-key\n", encoding="utf-8")
            first_audio = tmp_root / "first.wav"
            second_audio = tmp_root / "second.wav"
            first_audio.write_bytes(b"first")
            second_audio.write_bytes(b"second")
            adapter = GeminiAudioAdapter(model_id="gemini-2.5-flash", api_key_env_path=env_path)
            with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_google_genai}):
                session = adapter.create_session("system")
                self.assertEqual(adapter.send(session, "Turn 1", audio_paths=[first_audio]), "QUERY 0 1")
                self.assertEqual(adapter.send(session, "Turn 2", audio_paths=[second_audio]), "STOP")

        self.assertEqual(len(calls), 2)
        second_contents = calls[1]["contents"]
        self.assertEqual([content.role for content in second_contents], ["user", "model", "user"])
        self.assertEqual(second_contents[0].parts[0].text, "Turn 1")
        self.assertEqual(second_contents[0].parts[1].audio, b"first")
        self.assertEqual(second_contents[1].parts[0].text, "QUERY 0 1")
        self.assertEqual(second_contents[2].parts[0].text, "Turn 2")
        self.assertEqual(second_contents[2].parts[1].audio, b"second")
        self.assertEqual(
            session.state["messages"],
            [
                {"role": "user", "text": "Turn 1", "audio_paths": [str(first_audio)]},
                {"role": "model", "text": "QUERY 0 1"},
                {"role": "user", "text": "Turn 2", "audio_paths": [str(second_audio)]},
                {"role": "model", "text": "STOP"},
            ],
        )

    def test_gemini_adapter_raises_api_error_on_empty_or_failed_response(self) -> None:
        class EmptyResponse:
            text = ""

        class EmptyModels:
            def generate_content(self, **kwargs):
                return EmptyResponse()

        class FailingModels:
            def generate_content(self, **kwargs):
                raise RuntimeError("boom")

        class FakeClient:
            def __init__(self, api_key: str) -> None:
                self.models = EmptyModels()

        class FakePart:
            def __init__(self, *, text: str | None = None) -> None:
                self.text = text

        class FakeContent:
            def __init__(self, *, role: str, parts: list[FakePart]) -> None:
                self.role = role
                self.parts = parts

        class FakeGenerateContentConfig:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        fake_genai = types.SimpleNamespace(Client=FakeClient)
        fake_types = types.SimpleNamespace(
            Part=FakePart,
            UserContent=lambda *, parts: FakeContent(role="user", parts=parts),
            ModelContent=lambda *, parts: FakeContent(role="model", parts=parts),
            GenerateContentConfig=FakeGenerateContentConfig,
        )
        fake_google = types.ModuleType("google")
        fake_google.genai = fake_genai
        fake_google_genai = types.ModuleType("google.genai")
        fake_google_genai.types = fake_types

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "keys.env"
            env_path.write_text("GOOGLE_API_KEY=test-key\n", encoding="utf-8")
            adapter = GeminiAudioAdapter(model_id="gemini-2.5-flash", api_key_env_path=env_path)
            with patch.dict(sys.modules, {"google": fake_google, "google.genai": fake_google_genai}):
                session = adapter.create_session("system")
                with self.assertRaises(APIAdapterError):
                    adapter.send(session, "prompt")
                adapter._runtime["client"].models = FailingModels()
                with self.assertRaises(APIAdapterError):
                    adapter.send(session, "prompt")


class AudioModelAdapterTests(unittest.TestCase):
    def test_submission_model_configs_are_complete_and_constructible(self) -> None:
        expected = {
            "gpt_audio", "gpt_audio_mini", "gemini_2_flash", "gemini_3_flash",
            "audio_flamingo_next", "audio_flamingo_next_think", "kimi_audio",
            "mimo_audio", "moss_audio", "qwen25_omni",
        }
        configs = {path.parent.name: path for path in REPO_ROOT.glob("configs/*/config.json")}
        self.assertEqual(set(configs), expected)
        for name, path in configs.items():
            with self.subTest(model=name):
                config = json.loads(path.read_text())
                adapter = adapter_from_config(config)
                self.assertEqual(adapter.model_id, config["model_id"])
                self.assertEqual(
                    adapter.track2_capabilities.supported_conditions,
                    frozenset(config["track2_supported_conditions"]),
                )
        with self.assertRaisesRegex(ValueError, "Unknown adapter_type"):
            adapter_from_config({"adapter_type": "unsupported_model"})

    def test_api_model_configs_share_track2_active_contract(self) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        expected_capabilities = {
            "gpt_audio": (
                {"full_context", "active", "active_same_session_periodic_probe"},
                1200.0,
                120_000_000,
            ),
            "gpt_audio_mini": (
                {"full_context", "active", "active_same_session_periodic_probe"},
                1200.0,
                120_000_000,
            ),
            "gemini_2_flash": (
                {"full_context", "active", "active_same_session_periodic_probe"},
                1200.0,
                120_000_000,
            ),
            "gemini_3_flash": (
                {"full_context", "active", "active_same_session_periodic_probe"},
                1200.0,
                120_000_000,
            ),
        }
        for model_dir, (conditions, max_seconds, max_bytes) in expected_capabilities.items():
            with self.subTest(model_dir=model_dir):
                config = json.loads(
                    (repo_root / "configs" / model_dir / "config.json").read_text(encoding="utf-8")
                )
                adapter = adapter_from_config(config)
                capabilities = adapter.track2_capabilities
                self.assertEqual(
                    capabilities.supported_conditions,
                    frozenset(conditions),
                )
                self.assertTrue(capabilities.supports_text_only)
                self.assertTrue(capabilities.supports_output_token_override)
                self.assertEqual(capabilities.max_audio_seconds, max_seconds)
                self.assertEqual(capabilities.max_audio_bytes, max_bytes)

    def test_adapter_from_config_builds_new_audio_adapters(self) -> None:
        cases = [
            ({"adapter_type": "moss_audio"}, MOSSAudioAdapter, "OpenMOSS-Team/MOSS-Audio-8B-Instruct"),
            ({"adapter_type": "mimo_audio"}, MiMoAudioAdapter, "XiaomiMiMo/MiMo-Audio-7B-Instruct"),
            ({"adapter_type": "kimi_audio"}, KimiAudioAdapter, "moonshotai/Kimi-Audio-7B-Instruct"),
            ({"adapter_type": "gpt_audio_mini"}, GPTAudioMiniAdapter, "gpt-audio-mini"),
            ({"adapter_type": "audio_flamingo_next"}, AudioFlamingoNextAdapter, "nvidia/audio-flamingo-next-hf"),
        ]
        for config, expected_cls, model_id in cases:
            with self.subTest(adapter_type=config["adapter_type"]):
                adapter = adapter_from_config(config)
                self.assertIsInstance(adapter, expected_cls)
                self.assertEqual(adapter.model_id, model_id)

        think_adapter = adapter_from_config(
            {
                "adapter_type": "audio_flamingo_next",
                "model_id": "nvidia/audio-flamingo-next-think-hf",
                "max_new_tokens": 4096,
                "repetition_penalty": 1.2,
                "strip_thinking": True,
            }
        )
        self.assertEqual(think_adapter.max_new_tokens, 4096)
        self.assertEqual(think_adapter.repetition_penalty, 1.2)
        self.assertTrue(think_adapter.strip_thinking)

    def test_qwen_adapter_cleans_padded_temp_audio_on_session_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "short.wav"
            with wave.open(str(source_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00")

            adapter = LocalQwenOmniAdapter(model_id="dummy")
            session = adapter.create_session("system")
            padded_path = adapter._runtime_audio_path(session, source_path)

            self.assertNotEqual(padded_path, source_path)
            self.assertTrue(padded_path.exists())
            self.assertEqual(session.state["temp_audio_paths"], [str(padded_path)])

            adapter.close_session(session)

            self.assertFalse(padded_path.exists())
            self.assertNotIn("temp_audio_paths", session.state)

    def test_qwen_adapter_truncates_generated_followup_turn(self) -> None:
        raw = 'NO_UPDATE\nACTION QUERY 0 384 Human: continue the conversation'
        self.assertEqual(
            LocalQwenOmniAdapter._truncate_generated_turn(raw),
            "NO_UPDATE\nACTION QUERY 0 384",
        )
        multiline = 'SET target FOUND 1 2\nACTION STOP'
        self.assertEqual(LocalQwenOmniAdapter._truncate_generated_turn(multiline), multiline)

    def test_gpt_audio_mini_formats_chat_audio_request(self) -> None:
        calls: dict[str, object] = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"choices": [{"message": {"content": "STOP"}}]}

        def fake_post(url, headers, data, timeout):
            calls["url"] = url
            calls["headers"] = headers
            calls["payload"] = json.loads(data)
            calls["timeout"] = timeout
            return FakeResponse()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            env_path = root / "keys.env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            audio_path = root / "clip.wav"
            with wave.open(str(audio_path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x01\x00" * 160)
            adapter = GPTAudioMiniAdapter(
                model_id="gpt-audio-mini",
                api_key_env_path=env_path,
                api_base_url="https://example.test/v1",
                request_timeout=12.0,
                max_output_tokens=9,
            )
            with patch("tot_benchmark.audio_model_adapters.requests.post", side_effect=fake_post):
                session = adapter.create_session("system")
                text = adapter.send(session, "prompt", audio_paths=[audio_path])

        self.assertEqual(text, "STOP")
        self.assertEqual(calls["url"], "https://example.test/v1/chat/completions")
        self.assertEqual(calls["timeout"], 12.0)
        self.assertEqual(calls["headers"]["Authorization"], "Bearer test-key")
        payload = calls["payload"]
        self.assertEqual(payload["model"], "gpt-audio-mini")
        self.assertEqual(payload["max_completion_tokens"], 9)
        content = payload["messages"][1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "prompt"})
        self.assertEqual(content[1]["type"], "input_audio")
        padded_wav = base64.b64decode(content[1]["input_audio"]["data"])
        with wave.open(io.BytesIO(padded_wav), "rb") as handle:
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnframes(), 4000)
            frames = handle.readframes(handle.getnframes())
        self.assertEqual(frames[:320], b"\x01\x00" * 160)
        self.assertEqual(frames[320:], b"\x00" * (8000 - 320))

    def test_gpt_audio_adapters_make_strict_track2_calls_without_silence(self) -> None:
        class FakeResponse:
            status_code = 200
            headers: dict[str, str] = {}

            def __init__(self, *, audio_transcript: bool) -> None:
                self.audio_transcript = audio_transcript

            def json(self):
                message = (
                    {
                        "content": None,
                        "audio": {
                            "id": "audio-test",
                            "data": "discarded-audio-data",
                            "transcript": "NO_UPDATE\nACTION STOP",
                        },
                    }
                    if self.audio_transcript
                    else {"content": "NO_UPDATE\nACTION STOP"}
                )
                return {"choices": [{"message": message}]}

        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "keys.env"
            env_path.write_text("OPENAI_API_KEY=test-key\n", encoding="utf-8")
            cases = (
                (
                    OpenAIResponsesAudioAdapter(
                        model_id="gpt-audio-2025-08-28",
                        api_key_env_path=env_path,
                        request_timeout=17.0,
                    ),
                    "tot_benchmark.adapters.requests.post",
                    True,
                ),
                (
                    GPTAudioMiniAdapter(
                        model_id="gpt-audio-mini",
                        api_key_env_path=env_path,
                        request_timeout=19.0,
                    ),
                    "tot_benchmark.audio_model_adapters.requests.post",
                    True,
                ),
            )
            for adapter, request_target, uses_audio_output in cases:
                with self.subTest(model_id=adapter.model_id):
                    adapter.configure_track2_capabilities(
                        {
                            "track2_supported_conditions": ["active"],
                            "track2_supports_text_only": True,
                            "track2_supports_output_token_override": True,
                            "track2_max_audio_seconds": 384,
                            "track2_max_audio_bytes": 13_000_000,
                        }
                    )
                    captured: dict[str, object] = {}

                    def fake_post(url, headers, data, timeout):
                        captured["payload"] = json.loads(data)
                        captured["timeout"] = timeout
                        return FakeResponse(audio_transcript=uses_audio_output)

                    with patch(request_target, side_effect=fake_post):
                        session = adapter.create_session("system")
                        output = adapter.send_track2(
                            session,
                            "prompt",
                            max_output_tokens=512,
                            strict_text_only=True,
                        )

                    self.assertEqual(output, "NO_UPDATE\nACTION STOP")
                    payload = captured["payload"]
                    self.assertEqual(payload["max_completion_tokens"], 512)
                    content = payload["messages"][1]["content"]
                    self.assertEqual(content, [{"type": "text", "text": "prompt"}])
                    self.assertEqual(
                        payload["modalities"],
                        ["text", "audio"] if uses_audio_output else ["text"],
                    )
                    if uses_audio_output:
                        self.assertEqual(
                            payload["audio"],
                            {"voice": "alloy", "format": "wav"},
                        )
                    else:
                        self.assertNotIn("audio", payload)
                    self.assertNotEqual(captured["timeout"], 120.0)

    def test_gpt_audio_extractors_keep_strict_transcripts_separate_from_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "ordinary content",
                        "audio": {"transcript": "strict transcript"},
                    }
                }
            ]
        }
        for adapter_class in (OpenAIResponsesAudioAdapter, GPTAudioMiniAdapter):
            with self.subTest(adapter=adapter_class.__name__):
                self.assertEqual(adapter_class._extract_text(response), "ordinary content")
                self.assertEqual(
                    adapter_class._extract_audio_transcript(response),
                    "strict transcript",
                )
                for transcript in (None, 42, ""):
                    malformed = {
                        "choices": [
                            {
                                "message": {
                                    "content": "must not be used",
                                    "audio": {"transcript": transcript},
                                }
                            }
                        ]
                    }
                    with self.assertRaises(APIAdapterError) as raised:
                        adapter_class._extract_audio_transcript(malformed)
                    self.assertEqual(raised.exception.error_class, API_ERROR_SOFT)

    def test_mimo_audio_concatenates_multi_clip_history_and_cleans_temp_file(self) -> None:
        class FakeMiMo:
            def __init__(self) -> None:
                self.audio_calls: list[dict[str, object]] = []
                self.text_calls: list[dict[str, object]] = []

            def audio_understanding_sft(self, audio_path: str, prompt: str, thinking: bool = False) -> str:
                self.audio_calls.append(
                    {
                        "audio_path": audio_path,
                        "prompt": prompt,
                        "thinking": thinking,
                        "exists_during_call": Path(audio_path).is_file(),
                    }
                )
                return "QUERY 0 1\nignored"

            def text_dialogue_sft_multiturn(self, messages, thinking: bool = False) -> str:
                self.text_calls.append({"messages": messages, "thinking": thinking})
                return "STOP"

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first_audio = root / "first.wav"
            second_audio = root / "second.wav"
            first_audio.write_bytes(b"first")
            second_audio.write_bytes(b"second")
            concat_path = root / "concat.wav"
            concat_calls: list[list[Path]] = []

            def fake_concat(paths):
                concat_calls.append([Path(path) for path in paths])
                concat_path.write_bytes(b"concat")
                return concat_path

            model = FakeMiMo()
            adapter = MiMoAudioAdapter(thinking=True)
            with patch.object(adapter, "_ensure_runtime", return_value={"model": model}):
                with patch("tot_benchmark.audio_model_adapters._write_concatenated_wav", side_effect=fake_concat):
                    session = adapter.create_session("system prompt")
                    self.assertEqual(adapter.send(session, "first turn", audio_paths=[first_audio]), "QUERY 0 1")
                    self.assertEqual(adapter.send(session, "second turn", audio_paths=[second_audio]), "QUERY 0 1")

        self.assertEqual(concat_calls, [[first_audio], [first_audio, second_audio]])
        self.assertEqual(len(model.audio_calls), 2)
        self.assertTrue(model.audio_calls[-1]["exists_during_call"])
        self.assertFalse(Path(model.audio_calls[-1]["audio_path"]).exists())
        self.assertIn("system prompt", model.audio_calls[-1]["prompt"])
        self.assertIn("Audio evidence clips attached to this turn: 1", model.audio_calls[-1]["prompt"])
        self.assertEqual(model.text_calls, [])

    def test_local_adapters_support_text_only_turn_formatting(self) -> None:
        class FakeMossProcessor:
            config = types.SimpleNamespace(mel_sr=16000)

        moss = MOSSAudioAdapter()
        with patch.object(moss, "_ensure_runtime", return_value={"processor": FakeMossProcessor()}):
            session = moss.create_session("system")
            session.state["messages"].append({"role": "user", "text": "prompt", "audio_paths": []})
            prompt, audio_arrays = moss._build_prompt_and_audio(session)
        self.assertEqual(audio_arrays, [])
        self.assertIn("system", prompt)
        self.assertNotIn("<|AUDIO|>", prompt)

        class FakeNoGrad:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeTorch:
            def no_grad(self):
                return FakeNoGrad()

        class FakeKimiModel:
            def __init__(self) -> None:
                self.messages = None

            def generate(self, messages, **kwargs):
                self.messages = messages
                return None, "STOP\nignored"

        kimi_model = FakeKimiModel()
        kimi = KimiAudioAdapter(max_new_tokens=7)
        with patch.object(kimi, "_ensure_runtime", return_value={"torch": FakeTorch(), "model": kimi_model}):
            session = kimi.create_session("system")
            self.assertEqual(kimi.send(session, "prompt"), "STOP")
        self.assertTrue(any(message["message_type"] == "audio" for message in kimi_model.messages))
        self.assertFalse(any(message.get("audio_paths") for message in session.state["messages"]))
        self.assertEqual(kimi_model.messages[0]["content"], "system")

        class FakeWhisperModel:
            use_whisper_feature = True

        class FakeStrictKimiModel(FakeKimiModel):
            def __init__(self) -> None:
                super().__init__()
                self.alm = types.SimpleNamespace(model=FakeWhisperModel())
                self.use_whisper_during_generate = None

            def generate(self, messages, **kwargs):
                self.messages = messages
                self.use_whisper_during_generate = self.alm.model.use_whisper_feature
                return None, "NO_UPDATE\nACTION STOP<|im_msg_end|><|im_msg_end|>"

        strict_model = FakeStrictKimiModel()
        strict_kimi = KimiAudioAdapter(max_new_tokens=7)
        strict_kimi.configure_track2_capabilities(
            {
                "track2_supported_conditions": ["active"],
                "track2_supports_text_only": True,
                "track2_supports_output_token_override": True,
                "track2_max_audio_seconds": 384,
                "track2_max_audio_bytes": 13_000_000,
            }
        )
        with patch.object(
            strict_kimi,
            "_ensure_runtime",
            return_value={"torch": FakeTorch(), "model": strict_model},
        ):
            session = strict_kimi.create_session("system")
            self.assertEqual(
                strict_kimi.send_track2(
                    session,
                    "prompt",
                    max_output_tokens=512,
                    strict_text_only=True,
                ),
                "NO_UPDATE\nACTION STOP",
            )
        self.assertTrue(strict_model.use_whisper_during_generate)
        self.assertTrue(strict_model.alm.model.use_whisper_feature)
        self.assertTrue(
            any(message["message_type"] == "audio" for message in strict_model.messages)
        )
        self.assertFalse(any(message.get("audio_paths") for message in session.state["messages"]))

        afn = AudioFlamingoNextAdapter()
        session = afn.create_session("system")
        session.state["messages"].append({"role": "user", "text": "prompt", "audio_paths": []})
        temp_paths: list[Path] = []
        conversation = afn._conversation(session, temp_paths)
        self.assertEqual(temp_paths, [])
        self.assertEqual(conversation[0][0]["role"], "user")
        self.assertEqual(conversation[0][0]["content"][0]["type"], "text")
        self.assertFalse(any(item.get("type") == "audio" for item in conversation[0][0]["content"]))


def make_track1c_dataset(root: Path) -> dict[str, object]:
    dataset_dir = root / "dataset"
    audio_dir = dataset_dir / "episodes" / "T01"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / "version_1.wav"
    sf.write(str(audio_path), [0.0] * (4 * 16000), 16000)
    episode = {
        "track_id": "1c",
        "conversation_id": "T01",
        "variant_id": "version_1",
        "audio_file_path": "episodes/T01/version_1.wav",
        "audio_duration_seconds": 4.0,
        "samplerate": 16000,
        "channels": 1,
        "source_row": {
            "category": "A",
            "category_label": "Correction and confirmation",
            "target_evidence_sentences": "T01_S01,T01_S03",
        },
        "sentences": [
            {
                "sentence_id": "T01_S01",
                "speaker": "A",
                "text": "The pickup color is blue.",
                "sentence_start_time": 0.0,
                "sentence_end_time": 1.0,
            },
            {
                "sentence_id": "T01_S02",
                "speaker": "B",
                "text": "I heard blue.",
                "sentence_start_time": 1.2,
                "sentence_end_time": 2.2,
            },
            {
                "sentence_id": "T01_S03",
                "speaker": "A",
                "text": "Correction, the pickup color is green.",
                "sentence_start_time": 3.0,
                "sentence_end_time": 3.8,
            },
        ],
        "pauses": [],
        "parsed_metadata": {
            "generic_questions": [
                {
                    "question_id": "T01_GEN_Q1",
                    "question_type": "speaker_handoff_path",
                    "prompt": "Return the speaker handoff path.",
                }
            ],
            "generic_answer_rules": [
                {
                    "question_id": "T01_GEN_Q1",
                    "ask_after": ["T01_S01", "T01_S02", "T01_S03", "END"],
                    "final_answer_full_evidence": "A>B>A",
                    "evidence_rules": [{"compute_from_observed_sentence_ids": "generic"}],
                }
            ],
            "category_specific_questions": [
                {
                    "question_id": "T01_CAT_Q1",
                    "question_type": "slot_value",
                    "prompt": "What is the pickup color? Return exactly one of: BLUE, GREEN, UNKNOWN.",
                    "allowed_answers": ["BLUE", "GREEN", "UNKNOWN"],
                }
            ],
            "category_specific_answer_rules": [
                {
                    "question_id": "T01_CAT_Q1",
                    "ask_after": ["T01_S01", "T01_S03", "END"],
                    "final_answer_full_evidence": "GREEN",
                    "evidence_rules": [
                        {"answer": "GREEN", "if_observed_all": ["T01_S03"]},
                        {"answer": "BLUE", "if_observed_all": ["T01_S01"]},
                        {"default": "UNKNOWN"},
                    ],
                }
            ],
            "conversation_content_specific_questions": [
                {
                    "question_id": "T01_CONTENT_Q1",
                    "question_type": "speaker_content_attribution",
                    "prompt": "Which speaker first says green? Return A, B, or UNKNOWN.",
                    "allowed_answers": ["A", "B", "UNKNOWN"],
                }
            ],
            "conversation_content_specific_answer_rules": [
                {
                    "question_id": "T01_CONTENT_Q1",
                    "ask_after": ["T01_S03", "END"],
                    "final_answer_full_evidence": "A",
                    "evidence_rules": [
                        {"answer": "A", "if_observed_all": ["T01_S03"]},
                        {"default": "UNKNOWN"},
                    ],
                }
            ],
            "timing_questions": [
                {
                    "question_id": "T01_TIMING_Q1",
                    "question_type": "semantic_evidence_localization",
                    "prompt": "When is the evidence for pickup color heard? Return START <i> END <j>, or UNKNOWN.",
                },
                {
                    "question_id": "T01_TIMING_Q2",
                    "question_type": "speaker_activity_raster",
                    "prompt": "Return the speaker/silence raster for bins heard so far.",
                },
            ],
        },
    }
    manifest = {"track_id": "1c", "episodes": [episode], "bin_metadata_pending": True}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class Track1cHelperTests(unittest.TestCase):
    def test_manifest_normalization_and_scoring_helpers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = make_track1c_dataset(Path(tmpdir))
            episode = track1c.normalize_track1c_manifest(manifest)[0]

        self.assertEqual(episode["episode_id"], "T01__version_1")
        self.assertEqual(episode["num_bins"], 4)
        config = track1c._track1c_config_for_episode(episode)
        self.assertEqual(config.max_query_bins, 3)
        self.assertEqual(config.max_total_queried_bins, 8)
        self.assertEqual(config.max_steps, 4)

        observed = track1c.observed_sentence_ids(episode, [(3.0, 4.0), (0.0, 1.0)])
        self.assertEqual(observed, ["T01_S01", "T01_S03"])
        crossed = track1c.newly_crossed_checkpoints(episode, ["T01_S01"], observed)
        self.assertEqual(crossed, ["T01_S03"])

        semantic_items = track1c.semantic_questions_for_checkpoint(episode, "T01_S03")
        expected = [
            track1c.expected_semantic_answer(episode, group, question, rule, observed, full_context=False)
            for group, question, rule in semantic_items
        ]
        self.assertEqual(expected, ["A", "GREEN", "A"])
        self.assertEqual(track1c.expected_timing_localization_answer(episode, observed), "START 0 END 3")
        self.assertEqual(
            track1c.expected_speaker_raster_tokens(episode, {0, 3}),
            ["A", "??", "??", "A"],
        )
        self.assertTrue(track1c.parse_track1c_speaker_raster("A,??,00,B", 4)["format_valid"])
        self.assertFalse(track1c.parse_track1c_speaker_raster("A,NOPE,00,B", 4)["format_valid"])

    def test_track1c_continuous_timing_and_speaker_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = make_track1c_dataset(Path(tmpdir))
            episode = track1c.normalize_track1c_manifest(manifest)[0]

        observed = track1c.observed_sentence_ids(episode, [(0.0, 1.0), (3.0, 3.8)])
        self.assertEqual(
            track1c.expected_continuous_timing_localization_answer(episode, observed),
            "START 0.00 END 3.80",
        )
        expected_segments = track1c.expected_speaker_segments(
            episode,
            [(0.0, 1.0), (3.0, 3.8)],
            full_context=False,
        )
        parsed = track1c.parse_track1c_speaker_segments(
            track1c._speaker_segment_text(expected_segments),
            float(episode["audio_duration_seconds"]),
        )
        self.assertTrue(parsed["format_valid"])
        self.assertTrue(any(segment["labels"] is None for segment in expected_segments))


class Track1cRunTests(unittest.TestCase):
    def test_track1c_master_score_excludes_raster_and_penalizes_failures(self) -> None:
        base_track_metrics = {
            "track1c_generic_accuracy": 1.0,
            "track1c_category_accuracy": 1.0,
            "track1c_content_accuracy": 1.0,
            "track1c_non_unknown_accuracy": 1.0,
            "track1c_unknown_accuracy": 1.0,
            "track1c_update_success_rate": 1.0,
            "track1c_timing_localization_iou": 1.0,
            "track1c_speaker_raster_token_accuracy": 0.0,
            "track1c_speaker_raster_exact": 0.0,
        }
        format_metrics = {
            "action_format_valid_rate": 1.0,
            "action_semantic_valid_rate": 1.0,
        }
        exploration_metrics = {
            "target_evidence_sentence_recall": 1.0,
            "sentence_coverage_rate": 1.0,
            "repeated_bin_total": 0.0,
            "prefix_contiguous_observed_rate": 1.0,
        }
        score_with_bad_raster = track1c.score_track1c_master_metrics(
            condition="active",
            stop_reason="model_stop",
            listened_bin_total=4,
            track1c_metrics=base_track_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
        )
        score_with_good_raster = track1c.score_track1c_master_metrics(
            condition="active",
            stop_reason="model_stop",
            listened_bin_total=4,
            track1c_metrics={
                **base_track_metrics,
                "track1c_speaker_raster_token_accuracy": 1.0,
                "track1c_speaker_raster_exact": 1.0,
            },
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
        )
        self.assertEqual(score_with_bad_raster, score_with_good_raster)
        self.assertEqual(score_with_bad_raster["track1c_active_master_score"], 100.0)

        invalid_score = track1c.score_track1c_master_metrics(
            condition="active",
            stop_reason="invalid_action:length_bin_exceeds_max_query_bins",
            listened_bin_total=4,
            track1c_metrics=base_track_metrics,
            format_metrics=format_metrics,
            exploration_metrics=exploration_metrics,
        )
        self.assertEqual(invalid_score["track1c_active_protocol_score"], 0.0)
        self.assertEqual(invalid_score["track1c_active_master_score"], 0.0)

        no_exploration_score = track1c.score_track1c_master_metrics(
            condition="active_same_session_periodic_probe",
            stop_reason="max_steps_reached",
            listened_bin_total=4,
            track1c_metrics=base_track_metrics,
            format_metrics=format_metrics,
            exploration_metrics={
                **exploration_metrics,
                "target_evidence_sentence_recall": 0.0,
                "sentence_coverage_rate": 0.0,
            },
        )
        self.assertEqual(no_exploration_score["track1c_active_same_session_exploration_score"], 0.0)
        self.assertEqual(no_exploration_score["track1c_active_same_session_master_score"], 0.0)

    def test_track1c_full_context_run_emits_metrics_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "track_1c"
            make_track1c_dataset(root)
            outputs = ["A>B>A", "GREEN", "A", "START 0 END 3", "A,B,B,A"]
            results = track1c.run_track1c_benchmark(
                dataset_root=root,
                results_root=Path(tmpdir) / "results",
                adapter=DummyAdapter(outputs, model_id="dummy-1c"),
                conditions=["full_context"],
                regimes=["Discovery"],
                reprocess=True,
            )

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["episode_id"], "T01__version_1")
        self.assertEqual(result["conversation_id"], "T01")
        self.assertEqual(result["variant_id"], "version_1")
        self.assertEqual(result["track1c_metrics"]["track1c_semantic_accuracy"], 1.0)
        self.assertEqual(result["track1c_metrics"]["track1c_timing_localization_exact"], 1.0)
        self.assertEqual(result["track1c_metrics"]["track1c_speaker_raster_exact"], 1.0)

    def test_core_run_benchmark_dispatches_track1c_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "track_1c"
            make_track1c_dataset(root)
            outputs = ["A>B>A", "GREEN", "A", "START 0 END 3", "A,B,B,A"]
            results = run_benchmark(
                dataset_root=root,
                results_root=Path(tmpdir) / "results",
                adapter=DummyAdapter(outputs, model_id="dummy-1c-core"),
                conditions=["full_context"],
                regimes=["Discovery"],
                reprocess=True,
            )

        self.assertEqual(results[0]["track_id"], "1c")
        self.assertIn("track1c_metrics", results[0])

    def test_track1c_active_run_fires_checkpoint_questions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "track_1c"
            make_track1c_dataset(root)
            outputs = [
                "QUERY 0 1",
                "A",
                "BLUE",
                "START 0 END 0",
                "A,??,??,??",
                "QUERY 3 1",
                "A",
                "GREEN",
                "A",
                "START 0 END 3",
                "A,??,??,A",
                "STOP",
            ]
            results = track1c.run_track1c_benchmark(
                dataset_root=root,
                results_root=Path(tmpdir) / "results",
                adapter=DummyAdapter(outputs, model_id="dummy-1c-active"),
                episode_ids=["T01__version_1"],
                conditions=["active"],
                regimes=["Discovery"],
                reprocess=True,
            )

        result = results[0]
        self.assertEqual(result["observed_sentence_ids"], ["T01_S01", "T01_S03"])
        self.assertEqual(len(result["checkpoint_outputs"]), 9)
        self.assertEqual(result["track1c_metrics"]["track1c_semantic_accuracy"], 1.0)
        self.assertEqual(result["exploration_metrics"]["checkpoint_crossed_count"], 2.0)
        self.assertEqual(result["exploration_metrics"]["sentence_acquisition_inversion_count"], 0.0)

    def test_track1c_same_session_uses_one_session(self) -> None:
        class RecordingAdapter:
            model_id = "recording-1c"

            def __init__(self, outputs: list[str]) -> None:
                self.outputs = list(outputs)
                self.sessions: list[AdapterSession] = []
                self.records: list[dict[str, object]] = []

            def create_session(self, system_prompt: str) -> AdapterSession:
                session = AdapterSession(system_prompt=system_prompt, state={"turns": []})
                self.sessions.append(session)
                return session

            def send(self, session: AdapterSession, user_text: str, audio_paths=None) -> str:
                self.records.append({"prompt": user_text, "audio_paths": [str(path) for path in audio_paths or []]})
                if not self.outputs:
                    raise RuntimeError("RecordingAdapter ran out of outputs.")
                return self.outputs.pop(0)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "track_1c"
            make_track1c_dataset(root)
            outputs = [
                "QUERY 0 1",
                "A",
                "BLUE",
                "START 0 END 0",
                "A,??,??,??",
                "QUERY 3 1",
                "A",
                "GREEN",
                "A",
                "START 0 END 3",
                "A,??,??,A",
                "STOP",
            ]
            adapter = RecordingAdapter(outputs)
            results = track1c.run_track1c_benchmark(
                dataset_root=root,
                results_root=Path(tmpdir) / "results",
                adapter=adapter,
                episode_ids=["T01__version_1"],
                conditions=[ACTIVE_SAME_SESSION_PERIODIC_PROBE],
                regimes=["Discovery"],
                reprocess=True,
            )

        self.assertEqual(len(adapter.sessions), 1)
        self.assertEqual(results[0]["condition"], ACTIVE_SAME_SESSION_PERIODIC_PROBE)
        audio_records = [record for record in adapter.records if record["audio_paths"]]
        self.assertEqual(len(audio_records), 2)
        self.assertTrue(any("Checkpoint: T01_S01" in str(record["prompt"]) for record in audio_records))


class EnvironmentTests(unittest.TestCase):
    def test_episode_environment_tracks_requeries_and_clip_metadata(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)
            episode = manifest["episodes"][0]
            run_dir = Path(tmpdir) / "runtime"
            env = EpisodeEnvironment(Path(tmpdir), episode, config, run_dir)

            first_clip, first_meta = env.clip_for_query(1, 0, 1)
            second_clip, second_meta = env.clip_for_query(2, 0, 1)

            self.assertTrue(first_clip.is_file())
            self.assertTrue(second_clip.is_file())
            self.assertEqual(first_meta["start_bin"], 0)
            self.assertEqual(second_meta["length_bin"], 1)
            self.assertEqual(env.listened_bin_total, 2)
            self.assertEqual(env.observed_bins, {0})

    def test_continuous_environment_tracks_seconds_and_no_bin_metadata(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=7, time_mode=TIME_MODE_CONTINUOUS)
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = build_dataset(AUDIO_ROOT, Path(tmpdir), config)
            episode = manifest["episodes"][0]
            run_dir = Path(tmpdir) / "runtime"
            env = ContinuousEpisodeEnvironment(Path(tmpdir), episode, config, run_dir)

            clip, metadata = env.clip_for_query(1, 0.25, 1.25)

            self.assertTrue(clip.is_file())
            self.assertEqual(metadata["start_time_seconds"], 0.25)
            self.assertEqual(metadata["end_time_seconds"], 1.25)
            self.assertNotIn("start_bin", metadata)
            self.assertAlmostEqual(env.listened_seconds_total, 1.0)
            self.assertEqual(env.query_intervals, [(0.25, 1.25)])


class SmokeRunTests(unittest.TestCase):
    def test_run_benchmark_writes_outputs_for_dummy_adapter(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=20260413)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_root"
            manifest = build_dataset(AUDIO_ROOT, dataset_root, config)

            outputs: list[str] = []
            for episode in manifest["episodes"]:
                ground_truth_raster = raster_text_from_bins(episode["ground_truth_raster"])
                sequence_answer = episode["behavioral_answers"]["sequence"]["answer"]
                first_label = episode["ground_truth_sequence"][0]
                last_label = episode["ground_truth_sequence"][-1]
                full_pair_answer = "BEFORE" if len(episode["ground_truth_sequence"]) >= 2 else "AFTER"
                full_localization_answer = (
                    f"START {episode['ground_truth_spans'][first_label]['start_bin']} "
                    f"END {episode['ground_truth_spans'][first_label]['end_bin']}"
                )
                second_label = episode["ground_truth_sequence"][1]
                full_gap_answer = (
                    "ADJACENT"
                    if episode["ground_truth_spans"][second_label]["start_bin"]
                    == episode["ground_truth_spans"][first_label]["end_bin"] + 1
                    else "GAP"
                )
                for _regime in episode["regime_variants"]:
                    outputs.extend(
                        [
                            "STOP",
                            unresolved_raster_text(config.num_bins),
                            "UNEXPLORED",
                            "UNEXPLORED",
                            "UNEXPLORED",
                            ground_truth_raster,
                            sequence_answer,
                            full_pair_answer,
                            full_localization_answer,
                            full_gap_answer,
                            "STOP",
                            unresolved_raster_text(config.num_bins),
                            "UNEXPLORED",
                            "UNEXPLORED",
                            "UNEXPLORED",
                        ]
                    )

            adapter = DummyAdapter(outputs, model_id="dummy-smoke")
            results_root = Path(tmpdir) / "results"
            results = run_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=adapter,
                config=config,
            )

            self.assertEqual(len(results), 9)
            self.assertTrue((results_root / "episode_metrics.csv").is_file())
            self.assertTrue((results_root / "summary.json").is_file())
            self.assertTrue(
                (
                    results_root
                    / "episode_results"
                    / "active"
                    / "Discovery"
                    / f"{manifest['episodes'][0]['episode_id']}.json"
                ).is_file()
            )
            summary = json.loads((results_root / "summary.json").read_text(encoding="utf-8"))
            self.assertIn("metrics", summary)
            active_result = next(result for result in results if result["condition"] == "active")
            full_result = next(result for result in results if result["condition"] == "full_context")
            self.assertEqual(active_result["behavioral_protocol_version"], 2)
            self.assertIn("coverage_metrics", active_result)
            self.assertIn("exploration_metrics", active_result)
            self.assertIn("master_metrics", active_result)
            self.assertIn("uncertainty_outputs", active_result)
            self.assertIn("core_behavioral_outputs", active_result)
            self.assertIn("master_metrics", full_result)
            self.assertEqual(full_result["uncertainty_metrics"]["uncertainty_pairwise_count"], 0.0)
            self.assertIn("track1a_active_master_score", summary["metrics"])
            self.assertIn("track1a_full_context_master_score", summary["metrics"])
            self.assertIn("track1a_active_same_session_master_score", summary["metrics"])

            rerun = run_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=DummyAdapter([], model_id="unused"),
                config=config,
                reprocess=False,
            )
            self.assertEqual(len(rerun), 9)

    def test_same_session_periodic_probe_condition_uses_one_session_without_audio_replay(self) -> None:
        class RecordingAdapter:
            model_id = "recording-same-session"

            def __init__(self, outputs: list[str]) -> None:
                self.outputs = list(outputs)
                self.records: list[dict[str, object]] = []
                self.sessions: list[AdapterSession] = []

            def create_session(self, system_prompt: str) -> AdapterSession:
                session = AdapterSession(system_prompt=system_prompt, state={"turns": []})
                self.sessions.append(session)
                return session

            def send(
                self,
                session: AdapterSession,
                user_text: str,
                audio_paths=None,
                preserve_multiline: bool = False,
            ) -> str:
                paths = [str(path) for path in audio_paths or []]
                session.state.setdefault("turns", []).append({"role": "user", "text": user_text, "audio_paths": paths})
                self.records.append(
                    {
                        "prompt": user_text,
                        "audio_paths": paths,
                        "preserve_multiline": bool(preserve_multiline),
                    }
                )
                if not self.outputs:
                    raise RuntimeError("RecordingAdapter ran out of outputs.")
                output = self.outputs.pop(0)
                session.state["turns"].append({"role": "assistant", "text": output})
                return output

        config = BenchmarkConfig(num_episodes=1, seed=20260413, probe_every_n=2)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_root"
            manifest = build_dataset(AUDIO_ROOT, dataset_root, config)
            episode = manifest["episodes"][0]
            first_label = episode["ground_truth_sequence"][0]
            first_start = episode["ground_truth_spans"][first_label]["start_bin"]
            second_start = (first_start + 1) % config.num_bins
            observed_bins = {first_start, second_start}
            plan = _build_behavioral_plan(episode, observed_bins)
            outputs = [
                f"QUERY {first_start} 1",
                f"QUERY {second_start} 1",
                raster_text_from_bins(episode["ground_truth_raster"]),
                "STOP",
                raster_text_from_bins(episode["ground_truth_raster"]),
            ]
            for query in plan["core"]["localization"]:
                outputs.append(query["answer"])
            for query in plan["core"]["pairwise_order"]:
                outputs.append(query["answer"])
            for query in plan["core"]["gap_adjacency"]:
                outputs.append(query["answer"])
            if plan["core"]["sequence"] is not None:
                outputs.append(plan["core"]["sequence"]["answer"])
            for query in plan["uncertainty"]["localization"]:
                outputs.append(query["answer"])
            for query in plan["uncertainty"]["pairwise_order"]:
                outputs.append(query["answer"])
            for query in plan["uncertainty"]["gap_adjacency"]:
                outputs.append(query["answer"])

            adapter = RecordingAdapter(outputs)
            results_root = Path(tmpdir) / "results"
            results = run_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=adapter,
                config=config,
                episode_ids=[episode["episode_id"]],
                conditions=[ACTIVE_SAME_SESSION_PERIODIC_PROBE],
                regimes=["Discovery"],
                reprocess=True,
            )
            result_path_exists = (
                results_root
                / "episode_results"
                / ACTIVE_SAME_SESSION_PERIODIC_PROBE
                / "Discovery"
                / f"{episode['episode_id']}.json"
            ).is_file()
            summary_metrics = json.loads((results_root / "summary.json").read_text(encoding="utf-8"))["metrics"]

        self.assertEqual(len(adapter.sessions), 1)
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["condition"], ACTIVE_SAME_SESSION_PERIODIC_PROBE)
        self.assertEqual([probe["probe_kind"] for probe in result["probe_outputs"]], ["periodic", "final"])
        self.assertEqual(result["probe_outputs"][0]["checkpoint_step"], 2)
        self.assertEqual(result["probe_outputs"][1]["checkpoint_step"], 2)
        self.assertIn("same_session_behavioral_metrics", result)
        self.assertIn("same_session_core_behavioral_metrics", result)
        self.assertIn("same_session_uncertainty_metrics", result)
        self.assertNotIn("behavioral_metrics", result)
        self.assertEqual(result["exploration_metrics"]["query_count"], 2.0)
        self.assertTrue(result_path_exists)
        self.assertIn("same_session_behavior_sequence_exact", summary_metrics)

        audio_records = [record for record in adapter.records if record["audio_paths"]]
        self.assertEqual(len(audio_records), 2)
        flattened_audio_paths = [path for record in audio_records for path in record["audio_paths"]]
        self.assertEqual(len(flattened_audio_paths), len(set(flattened_audio_paths)))
        self.assertIn("You inspected bins", str(audio_records[0]["prompt"]))
        self.assertIn("periodic checkpoint belief probe", str(audio_records[1]["prompt"]))
        self.assertTrue(audio_records[1]["preserve_multiline"])
        final_probe_index = next(
            index
            for index, record in enumerate(adapter.records)
            if "final terminal belief probe" in str(record["prompt"])
        )
        self.assertTrue(adapter.records[final_probe_index]["preserve_multiline"])
        self.assertFalse(adapter.records[0]["preserve_multiline"])
        self.assertFalse(adapter.records[1]["preserve_multiline"])
        self.assertFalse(adapter.records[final_probe_index]["audio_paths"])
        self.assertTrue(all(not record["audio_paths"] for record in adapter.records[final_probe_index + 1 :]))

    def test_invalid_action_repair_does_not_trigger_periodic_same_session_probe(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=20260413, probe_every_n=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_root"
            manifest = build_dataset(AUDIO_ROOT, dataset_root, config)
            episode = manifest["episodes"][0]
            plan = _build_behavioral_plan(episode, set())
            outputs = ["QUERY 0 99", "STOP", unresolved_raster_text(config.num_bins)]
            for query in plan["uncertainty"]["localization"]:
                outputs.append(query["answer"])
            for query in plan["uncertainty"]["pairwise_order"]:
                outputs.append(query["answer"])
            for query in plan["uncertainty"]["gap_adjacency"]:
                outputs.append(query["answer"])

            results = run_benchmark(
                dataset_root=dataset_root,
                results_root=Path(tmpdir) / "results",
                adapter=DummyAdapter(outputs, model_id="dummy-invalid-same-session"),
                config=config,
                episode_ids=[episode["episode_id"]],
                conditions=[ACTIVE_SAME_SESSION_PERIODIC_PROBE],
                regimes=["Discovery"],
                reprocess=True,
            )

        result = results[0]
        self.assertEqual([probe["probe_kind"] for probe in result["probe_outputs"]], ["final"])
        self.assertEqual(result["exploration_metrics"]["query_count"], 0.0)
        self.assertEqual(result["exploration_metrics"]["repair_attempted_action_count"], 1.0)

    def test_run_parser_accepts_condition_selection(self) -> None:
        args = build_arg_parser().parse_args(
            [
                "run",
                "--results-root",
                "results",
                "--adapter-config",
                "config.json",
                "--conditions",
                ACTIVE_SAME_SESSION_PERIODIC_PROBE,
                "--time-mode",
                TIME_MODE_CONTINUOUS,
            ]
        )

        self.assertEqual(args.conditions, [ACTIVE_SAME_SESSION_PERIODIC_PROBE])
        self.assertEqual(args.time_mode, TIME_MODE_CONTINUOUS)

    def test_api_adapter_errors_mark_episode_invalid_without_crashing(self) -> None:
        class FailingAPIAdapter:
            model_id = "failing-api"

            def create_session(self, system_prompt: str) -> AdapterSession:
                return AdapterSession(system_prompt=system_prompt)

            def send(self, session: AdapterSession, user_text: str, audio_paths=None) -> str:
                raise APIAdapterError("temporary API failure")

        config = BenchmarkConfig(num_episodes=1, seed=20260413, api_retry_sleep_sec=0)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_root"
            manifest = build_dataset(AUDIO_ROOT, dataset_root, config)
            results = run_benchmark(
                dataset_root=dataset_root,
                results_root=Path(tmpdir) / "results",
                adapter=FailingAPIAdapter(),
                config=config,
                episode_ids=[manifest["episodes"][0]["episode_id"]],
                conditions=["active"],
                regimes=["Discovery"],
                reprocess=True,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["stop_reason"], "invalid_action:api_error")
        self.assertEqual(results[0]["format_metrics"]["action_format_valid_rate"], 0.0)

    def test_run_benchmark_supports_filtered_subset(self) -> None:
        config = BenchmarkConfig(num_episodes=1, seed=20260413)
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "dataset_root"
            manifest = build_dataset(AUDIO_ROOT, dataset_root, config)
            episode = manifest["episodes"][0]
            sequence_answer = episode["behavioral_answers"]["sequence"]["answer"]
            ground_truth_raster = raster_text_from_bins(episode["ground_truth_raster"])
            first_label = episode["ground_truth_sequence"][0]
            first_span = episode["ground_truth_spans"][first_label]
            second_label = episode["ground_truth_sequence"][1]
            full_gap_answer = (
                "ADJACENT"
                if episode["ground_truth_spans"][second_label]["start_bin"] == first_span["end_bin"] + 1
                else "GAP"
            )

            scripted_outputs = [
                "STOP",
                unresolved_raster_text(config.num_bins),
                "UNEXPLORED",
                "UNEXPLORED",
                "UNEXPLORED",
                ground_truth_raster,
                sequence_answer,
                "BEFORE",
                f"START {first_span['start_bin']} END {first_span['end_bin']}",
                full_gap_answer,
            ]
            adapter = DummyAdapter(scripted_outputs, model_id="dummy-filtered")
            results_root = Path(tmpdir) / "results"

            results = run_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=adapter,
                config=config,
                episode_ids=[episode["episode_id"]],
                conditions=["active", "full_context"],
                regimes=["Discovery"],
            )

            self.assertEqual(len(results), 2)
            self.assertTrue(
                (
                    results_root
                    / "episode_results"
                    / "active"
                    / "Discovery"
                    / f"{episode['episode_id']}.json"
                ).is_file()
            )
            self.assertTrue(
                (
                    results_root
                    / "episode_results"
                    / "full_context"
                    / "Discovery"
                    / f"{episode['episode_id']}.json"
                ).is_file()
            )
            self.assertTrue((results_root / "episode_metrics.csv").is_file())
            self.assertTrue((results_root / "summary.json").is_file())
            active_result = next(result for result in results if result["condition"] == "active")
            self.assertEqual(active_result["uncertainty_metrics"]["uncertainty_pairwise_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
