from __future__ import annotations

import io
import json
import hashlib
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from tot_benchmark.adapters import API_ERROR_HARD, APIAdapterError, DummyAdapter, adapter_from_config
from tot_benchmark.track2 import (
    TRACK2_CROP_SAMPLE_RATE,
    TRACK2_RESULT_SCHEMA_VERSION,
    Track2Config,
    _is_reusable_track2_result,
    _write_query_crop,
    load_track2_dataset as _load_track2_dataset,
    rebuild_track2_outputs,
    run_track2_benchmark as _run_track2_benchmark,
    track2_budget_for_task,
)
from tot_benchmark.track2_protocol import (
    build_track2_probe_prompt,
    initialize_track2_ledger,
    parse_track2_action_response,
    parse_track2_final_answer,
    parse_track2_probe_response,
    validate_and_merge_track2_updates,
    validate_track2_query,
)
from tot_benchmark.track2_scoring import (
    aggregate_track2_results,
    interval_metrics,
    pairwise_order_accuracy,
    score_track2_answer,
    score_track2_belief,
    score_track2_evidence,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _refresh_synthetic_validation_report(root: Path) -> str:
    dataset_dir = root / "dataset"
    relative_paths = [
        "manifest.json",
        "task_instances.jsonl",
        "task_answers.jsonl",
        "candidate_provenance.jsonl",
        "episodes/R1.wav",
    ]
    hashes = {
        relative_path: hashlib.sha256((dataset_dir / relative_path).read_bytes()).hexdigest()
        for relative_path in relative_paths
    }
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    (dataset_dir / "validation_report.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "dataset_id": "track2_role_ipc_test_v2",
                "artifact_sha256": hashes,
                "deterministic_build_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return fingerprint


def load_track2_dataset(root: Path) -> dict:
    fingerprint = json.loads(
        (root / "dataset" / "validation_report.json").read_text(encoding="utf-8")
    )["deterministic_build_fingerprint"]
    with patch("tot_benchmark.track2.TRACK2_DATASET_FINGERPRINT", fingerprint):
        return _load_track2_dataset(root)


def run_track2_benchmark(**kwargs):
    root = Path(kwargs["dataset_root"])
    fingerprint = json.loads(
        (root / "dataset" / "validation_report.json").read_text(encoding="utf-8")
    )["deterministic_build_fingerprint"]
    with patch("tot_benchmark.track2.TRACK2_DATASET_FINGERPRINT", fingerprint):
        return _run_track2_benchmark(**kwargs)


def _synthetic_dataset(root: Path, tasks: list[dict], answers: list[dict]) -> Path:
    dataset_dir = root / "dataset"
    episodes_dir = dataset_dir / "episodes"
    episodes_dir.mkdir(parents=True)
    sf.write(episodes_dir / "R1.wav", np.zeros(32000, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = {
        "dataset_id": "track2_role_ipc_test_v2",
        "track_id": "2",
        "task_count": len(tasks),
        "episode_count": 1,
        "max_single_query_seconds": 384,
        "episodes": [
            {
                "reel_id": "R1",
                "audio_path": "episodes/R1.wav",
                "duration_seconds": 2.0,
                "scenes": [
                    {
                        "scene_id": "S1",
                        "scene_rank": 1,
                        "start_seconds": 0.0,
                        "end_seconds": 1.0,
                    },
                    {
                        "scene_id": "S2",
                        "scene_rank": 2,
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                    },
                ],
            }
        ],
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _write_jsonl(dataset_dir / "task_instances.jsonl", tasks)
    _write_jsonl(dataset_dir / "task_answers.jsonl", answers)
    _write_jsonl(dataset_dir / "candidate_provenance.jsonl", [])
    _refresh_synthetic_validation_report(root)
    return root


def _scene_task(task_id: str = "T1") -> dict:
    return {
        "task_id": task_id,
        "track": "2",
        "family": "2A.1",
        "variant": "present",
        "reel_id": "R1",
        "audio_path": "episodes/R1.wav",
        "question_text": "Locate the scene.",
        "card": {"card_id": "C_TARGET", "text": "A scene."},
    }


class Track2ProtocolTests(unittest.TestCase):
    def test_parse_sparse_updates_and_action(self) -> None:
        parsed = parse_track2_action_response(
            "SET U_A FOUND 1.0 2.0 C_A\nSET U_B LIKELY_ABSENT\nACTION QUERY 2 3"
        )
        self.assertTrue(parsed["format_valid"])
        self.assertEqual(parsed["action"], "QUERY")
        self.assertEqual(len(parsed["updates"]), 2)

    def test_action_parser_rejects_ambiguous_or_mixed_no_update(self) -> None:
        for raw in (
            "NO_UPDATE\nSET X NOT_FOUND_YET\nACTION STOP",
            "NO_UPDATE\nACTION STOP\nACTION STOP",
            "ACTION QUERY 1",
            "SET X FOUND from 1 to 2\nACTION STOP",
            "X: FOUND 1-2\nACTION STOP",
        ):
            with self.subTest(raw=raw):
                self.assertFalse(parse_track2_action_response(raw)["format_valid"])

    def test_probe_parser_accepts_documented_bare_complete_snapshot(self) -> None:
        task = {
            "family": "2B.2",
            "utterance_targets": [{"utterance_id": "U1"}, {"utterance_id": "U2"}],
            "scene_cards": [{"card_id": "C1"}],
        }
        parsed = parse_track2_probe_response(
            "U1 FOUND 1 2 C1\nU2 LIKELY_ABSENT",
            task,
        )
        self.assertTrue(parsed["format_valid"])
        self.assertEqual([update["target_id"] for update in parsed["updates"]], ["U1", "U2"])
        self.assertFalse(
            parse_track2_probe_response(
                "SET U1 FOUND 1 2 C1\nSET U2 LIKELY_ABSENT",
                task,
            )["format_valid"]
        )

    def test_merge_rejects_invalid_updates_independently(self) -> None:
        task = {
            "family": "2B.2",
            "utterance_targets": [{"utterance_id": "U1"}, {"utterance_id": "U2"}],
            "scene_cards": [{"card_id": "C1"}],
        }
        ledger = initialize_track2_ledger(task)
        merged, outcomes = validate_and_merge_track2_updates(
            task,
            ledger,
            [
                {"target_id": "U1", "status": "FOUND", "start_seconds": 1, "end_seconds": 2, "scene_card_id": "C1"},
                {"target_id": "U2", "status": "FOUND", "start_seconds": 3, "end_seconds": 2, "scene_card_id": "C1"},
            ],
            reel_duration_seconds=4,
            step_index=1,
        )
        self.assertEqual(merged["U1"]["status"], "FOUND")
        self.assertEqual(merged["U2"]["status"], "NOT_FOUND_YET")
        self.assertEqual([item["valid"] for item in outcomes], [True, False])

    def test_query_validation_is_fail_fast_without_clamping(self) -> None:
        self.assertEqual(
            validate_track2_query(
                0,
                385,
                reel_duration_seconds=1000,
                max_single_query_seconds=384,
                listened_seconds=0,
                listening_budget_seconds=1000,
            ),
            (False, "query_exceeds_max_single_query_seconds"),
        )
        self.assertEqual(
            validate_track2_query(
                0,
                10,
                reel_duration_seconds=1000,
                max_single_query_seconds=384,
                listened_seconds=995,
                listening_budget_seconds=1000,
            ),
            (False, "query_exceeds_listening_budget"),
        )

    def test_final_parser_separates_exact_json_from_semantics(self) -> None:
        task = _scene_task()
        valid = parse_track2_final_answer(
            '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,"start_seconds":null,"end_seconds":null,"confidence":0.7}',
            task,
        )
        self.assertTrue(valid["format_valid"])
        semantically_invalid = parse_track2_final_answer(
            '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":null,"end_seconds":null,"confidence":1.2}',
            task,
        )
        self.assertTrue(semantically_invalid["format_valid"])
        wrapped = parse_track2_final_answer(
            'Answer: {"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,'
            '"start_seconds":null,"end_seconds":null,"confidence":0.7}',
            task,
        )
        self.assertFalse(wrapped["format_valid"])

    def test_final_parser_conservatively_rescues_superficial_wrappers(self) -> None:
        task = _scene_task()
        payload = (
            '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,'
            '"start_seconds":null,"end_seconds":null,"confidence":0.7}'
        )
        fenced = parse_track2_final_answer(f"```json\n{payload}\n```", task)
        self.assertTrue(fenced["format_valid"])
        self.assertFalse(fenced["strict_format_valid"])
        self.assertTrue(fenced["rescue_applied"])
        self.assertEqual(fenced["rescue_kind"], "markdown_json_fence")

        token_suffixed = parse_track2_final_answer(
            payload + "<|im_msg_end|><|im_msg_end|>", task
        )
        self.assertTrue(token_suffixed["format_valid"])
        self.assertEqual(token_suffixed["rescue_kind"], "special_token_suffix")

    def test_final_parser_rescue_rejects_incomplete_ambiguous_or_prose_outputs(self) -> None:
        task = _scene_task()
        incomplete = parse_track2_final_answer(
            '```json\n{"status":"ABSENT","card_id":"C_TARGET"}\n```', task
        )
        self.assertFalse(incomplete["format_valid"])
        self.assertEqual(incomplete["reason"], "rescue_requires_all_required_fields")

        payload = (
            '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,'
            '"start_seconds":null,"end_seconds":null,"confidence":0.7}'
        )
        for raw in (
            f"Answer: {payload}",
            f"{payload}\nExplanation: guessed",
            f"{payload}\n{payload}",
            f"```json{payload}```",
            f"```{payload}```",
        ):
            self.assertFalse(parse_track2_final_answer(raw, task)["format_valid"])


class Track2ScoringTests(unittest.TestCase):
    def test_utterance_endpoint_tolerance_snaps_before_iou(self) -> None:
        scores = interval_metrics((9.2, 20.8), (10, 20), endpoint_tolerance_seconds=1.0)
        self.assertEqual(scores["tolerant_temporal_iou"], 1.0)
        self.assertEqual(scores["localization_pass"], 1.0)

    def test_scene_endpoint_tolerance_relaxes_iou_without_invalidating_prior_passes(self) -> None:
        task = _scene_task()
        gold = {
            "status": "PRESENT",
            "card_id": "C_TARGET",
            "scene_rank": 1,
            "start_seconds": 100.0,
            "end_seconds": 300.0,
        }
        episode = {
            "duration_seconds": 600.0,
            "scenes": [
                {"scene_rank": 1, "start_seconds": 100.0, "end_seconds": 300.0},
            ]
        }

        newly_passing = score_track2_answer(
            task,
            gold,
            {
                **gold,
                "start_seconds": 91.0,
                "end_seconds": 500.0,
                "confidence": 1.0,
            },
            episode,
        )
        prior_iou_pass = score_track2_answer(
            task,
            gold,
            {
                **gold,
                "start_seconds": 140.0,
                "end_seconds": 260.0,
                "confidence": 1.0,
            },
            episode,
        )
        still_failing = score_track2_answer(
            task,
            gold,
            {
                **gold,
                "start_seconds": 50.0,
                "end_seconds": 500.0,
                "confidence": 1.0,
            },
            episode,
        )

        self.assertLess(newly_passing["temporal_iou"], 0.5)
        self.assertEqual(newly_passing["tolerant_temporal_iou"], 0.5)
        self.assertEqual(newly_passing["localization_pass"], 1.0)
        self.assertEqual(newly_passing["primary_correct"], 1.0)
        self.assertEqual(prior_iou_pass["temporal_iou"], 0.6)
        self.assertEqual(prior_iou_pass["localization_pass"], 1.0)
        self.assertEqual(prior_iou_pass["primary_correct"], 1.0)
        self.assertLess(still_failing["tolerant_temporal_iou"], 0.5)
        self.assertEqual(still_failing["localization_pass"], 0.0)
        self.assertEqual(still_failing["primary_correct"], 0.0)

    def test_scene_boundary_tolerance_applies_to_active_belief_scoring(self) -> None:
        task = {
            "family": "2B.1",
            "cards": [{"card_id": "C1"}],
            "_episode_scenes": [
                {"scene_rank": 1, "start_seconds": 100.0, "end_seconds": 300.0},
            ],
            "_reel_duration_seconds": 600.0,
        }
        gold = {"ordered_card_ids": ["C1"], "absent_card_id": None}
        newly_passing = {
            "target_id": "C1",
            "status": "FOUND",
            "start_seconds": 91.0,
            "end_seconds": 500.0,
            "scene_card_id": None,
        }
        still_failing = {
            **newly_passing,
            "start_seconds": 50.0,
            "last_updated_step": 1,
        }
        metrics = score_track2_belief(
            task,
            gold,
            [{"start_seconds": 100.0, "end_seconds": 300.0}],
            [{"valid": True, "step_index": 1, "update": newly_passing}],
            {"C1": still_failing},
            None,
        )

        self.assertEqual(metrics["mean_contact_to_correct_update_steps"], 0.0)
        self.assertEqual(metrics["final_ledger_target_accuracy"], 0.0)

    def test_pairwise_order_counts_missing_items_incorrect(self) -> None:
        self.assertEqual(pairwise_order_accuracy(["A", "C"], ["A", "B", "C"]), 1 / 3)

    def test_2b1_card_to_scene_assignment_reports_partial_rank_accuracy(self) -> None:
        task = {
            "family": "2B.1",
            "cards": [{"card_id": value} for value in ("A", "B", "C")],
        }
        metrics = score_track2_answer(
            task,
            {"ordered_card_ids": ["A", "B", "C"], "absent_card_id": None},
            {"ordered_card_ids": ["A", "C", "B"], "absent_card_id": None},
            {},
        )
        self.assertEqual(metrics["card_to_scene_assignment_accuracy"], 1 / 3)

    def test_scene_identity_and_rank_are_separate(self) -> None:
        task = _scene_task()
        gold = {
            "status": "PRESENT",
            "card_id": "C_TARGET",
            "scene_rank": 1,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
        }
        predicted = {
            "status": "PRESENT",
            "card_id": "C_TARGET",
            "scene_rank": 2,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
            "confidence": 0.8,
        }
        episode = {
            "scenes": [
                {"scene_rank": 1, "start_seconds": 0.0, "end_seconds": 1.0},
                {"scene_rank": 2, "start_seconds": 1.0, "end_seconds": 2.0},
            ]
        }
        metrics = score_track2_answer(task, gold, predicted, episode)
        self.assertEqual(metrics["scene_identity_accuracy"], 1.0)
        self.assertEqual(metrics["scene_rank_accuracy"], 0.0)

    def test_failed_present_localization_contributes_zero_to_aggregate(self) -> None:
        task = _scene_task()
        gold = {
            "status": "PRESENT",
            "card_id": "C_TARGET",
            "scene_rank": 1,
            "start_seconds": 0.0,
            "end_seconds": 1.0,
        }
        episode = {"scenes": [{"scene_rank": 1, "start_seconds": 0.0, "end_seconds": 1.0}]}
        perfect = score_track2_answer(
            task,
            gold,
            {**gold, "confidence": 1.0},
            episode,
        )
        missed = score_track2_answer(
            task,
            gold,
            {
                "status": "ABSENT",
                "card_id": "C_TARGET",
                "scene_rank": None,
                "start_seconds": None,
                "end_seconds": None,
                "confidence": 1.0,
            },
            episode,
        )
        base = {
            "condition": "active",
            "family": "2A.1",
            "variant": "present",
            "reel_id": "R1",
            "run_status": "complete",
            "evidence_metrics": {},
            "belief_metrics": {},
            "format_metrics": {},
            "confidence": 1.0,
        }
        aggregate = aggregate_track2_results(
            [{**base, "answer_metrics": perfect}, {**base, "answer_metrics": missed}]
        )
        self.assertEqual(
            aggregate["groups"]["active/2A.1/present"]["answer_metrics"]["temporal_f1"],
            0.5,
        )

    def test_aggregate_excludes_api_errors_from_scores(self) -> None:
        complete = {
            "condition": "active",
            "family": "2A.1",
            "variant": "present",
            "reel_id": "R1",
            "run_status": "complete",
            "answer_metrics": {"primary_correct": 1.0},
            "evidence_metrics": {},
            "belief_metrics": {},
            "format_metrics": {},
            "confidence": 1.0,
        }
        api_error = {
            **complete,
            "run_status": "api_error",
            "answer_metrics": {"primary_correct": 0.0},
            "confidence": 0.0,
        }
        group = aggregate_track2_results([complete, api_error])["overall"]
        self.assertEqual(group["task_count"], 2)
        self.assertEqual(group["scored_task_count"], 1)
        self.assertEqual(group["answer_metrics"]["primary_correct"], 1.0)
        self.assertEqual(group["status_counts"], {"api_error": 1, "complete": 1})

    def test_evidence_reports_target_coverage_and_all_target_contact_cost(self) -> None:
        task = {
            "family": "2B.2",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {
            "utterances": [
                {"utterance_id": "U1", "status": "PRESENT", "start_seconds": 1, "end_seconds": 3},
                {"utterance_id": "U2", "status": "PRESENT", "start_seconds": 7, "end_seconds": 9},
                {"utterance_id": "U3", "status": "ABSENT"},
            ]
        }
        metrics = score_track2_evidence(
            task,
            gold,
            [
                {"start_seconds": 0, "end_seconds": 2},
                {"start_seconds": 7, "end_seconds": 8},
            ],
            reel_duration_seconds=10,
            listening_budget_seconds=20,
            stop_reason="model_stop",
        )
        self.assertEqual(metrics["mean_target_evidence_coverage"], 0.5)
        self.assertEqual(metrics["cost_to_all_present_target_contact_seconds"], 3.0)

    def test_belief_uses_first_commitment_and_requires_later_revision(self) -> None:
        task = {
            "family": "2A.2",
            "target_id": "U1",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {"status": "PRESENT", "start_seconds": 1.0, "end_seconds": 2.0}
        found = {
            "target_id": "U1",
            "status": "FOUND",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "scene_card_id": None,
        }
        false_absent = {
            "target_id": "U1",
            "status": "LIKELY_ABSENT",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
        }
        metrics = score_track2_belief(
            task,
            gold,
            [
                {"start_seconds": 1, "end_seconds": 2},
                {"start_seconds": 3, "end_seconds": 4},
            ],
            [
                {"valid": True, "step_index": 1, "update": found},
                {"valid": True, "step_index": 2, "source": "periodic_probe", "update": found},
                {"valid": True, "step_index": 2, "update": false_absent},
            ],
            {"U1": {**false_absent, "last_updated_step": 2}},
            {"start_seconds": 1.0, "end_seconds": 2.0, "confidence": 1.0},
        )
        self.assertEqual(metrics["mean_contact_to_correct_update_steps"], 0.0)
        self.assertEqual(metrics["false_absence_claim_count"], 1.0)
        self.assertEqual(metrics["successful_revision_rate"], 0.0)
        self.assertEqual(metrics["final_ledger_answer_consistency"], 0.0)

    def test_not_found_yet_is_not_a_false_claim_or_revision_opportunity(self) -> None:
        task = {
            "family": "2A.2",
            "target_id": "U1",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {"status": "PRESENT", "start_seconds": 1.0, "end_seconds": 2.0}
        unresolved = {
            "target_id": "U1",
            "status": "NOT_FOUND_YET",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
        }
        metrics = score_track2_belief(
            task,
            gold,
            [{"start_seconds": 1, "end_seconds": 2}],
            [{"valid": True, "step_index": 1, "update": unresolved}],
            {"U1": {**unresolved, "last_updated_step": 1}},
            {"start_seconds": 1.0, "end_seconds": 2.0, "confidence": 1.0},
        )
        self.assertIsNone(metrics["successful_revision_rate"])
        self.assertEqual(metrics["target_status_after_contact_accuracy"], 0.0)

    def test_not_found_yet_retracts_false_claim_but_breaks_correct_found_landmark(self) -> None:
        task = {
            "family": "2A.2",
            "target_id": "U1",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {"status": "PRESENT", "start_seconds": 1.0, "end_seconds": 2.0}
        false_absent = {
            "target_id": "U1",
            "status": "LIKELY_ABSENT",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
        }
        unresolved = {
            "target_id": "U1",
            "status": "NOT_FOUND_YET",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
        }
        found = {
            "target_id": "U1",
            "status": "FOUND",
            "start_seconds": 1.0,
            "end_seconds": 2.0,
            "scene_card_id": None,
        }
        query_history = [
            {"start_seconds": 1, "end_seconds": 2},
            {"start_seconds": 2, "end_seconds": 3},
            {"start_seconds": 3, "end_seconds": 4},
            {"start_seconds": 4, "end_seconds": 5},
        ]
        revision = score_track2_belief(
            task,
            gold,
            query_history,
            [
                {"valid": True, "step_index": 1, "update": false_absent},
                {"valid": True, "step_index": 2, "update": unresolved},
            ],
            {"U1": {**unresolved, "last_updated_step": 2}},
            None,
        )
        self.assertEqual(revision["successful_revision_rate"], 1.0)

        preservation = score_track2_belief(
            task,
            gold,
            query_history,
            [
                {"valid": True, "step_index": 1, "update": found},
                {"valid": True, "step_index": 2, "update": unresolved},
            ],
            {"U1": {**unresolved, "last_updated_step": 2}},
            None,
        )
        self.assertEqual(preservation["found_landmark_preservation_rate"], 0.0)

    def test_true_absence_claim_reports_zero_coverage_without_arbitrary_cutoff(self) -> None:
        task = {
            "family": "2A.3",
            "target_id": "U1",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {"status": "ABSENT", "occurrences": []}
        absent = {
            "target_id": "U1",
            "status": "LIKELY_ABSENT",
            "start_seconds": None,
            "end_seconds": None,
            "scene_card_id": None,
        }
        metrics = score_track2_belief(
            task,
            gold,
            [],
            [{"valid": True, "step_index": 0, "update": absent}],
            {"U1": {**absent, "last_updated_step": 0}},
            {"status": "ABSENT", "occurrences": [], "confidence": 0.8},
        )
        self.assertEqual(metrics["false_absence_claim_count"], 0.0)
        self.assertEqual(len(metrics["absence_claim_coverage_history"]), 2)
        self.assertTrue(
            all(
                claim["unique_coverage_seconds"] == 0.0
                for claim in metrics["absence_claim_coverage_history"]
            )
        )

    def test_final_ledger_correctness_checks_gold_interval(self) -> None:
        task = {
            "family": "2A.2",
            "target_id": "U1",
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        gold = {"status": "PRESENT", "start_seconds": 1.0, "end_seconds": 2.0}
        wrong = {
            "target_id": "U1",
            "status": "FOUND",
            "start_seconds": 7.0,
            "end_seconds": 8.0,
            "scene_card_id": None,
            "last_updated_step": 1,
        }
        metrics = score_track2_belief(
            task,
            gold,
            [{"start_seconds": 0, "end_seconds": 8}],
            [],
            {"U1": wrong},
            {"start_seconds": 7.0, "end_seconds": 8.0, "confidence": 1.0},
        )
        self.assertEqual(metrics["final_ledger_answer_consistency"], 1.0)
        self.assertEqual(metrics["final_ledger_target_accuracy"], 0.0)
        self.assertEqual(metrics["final_ledger_complete_accuracy"], 0.0)

    def test_absent_task_has_no_all_present_target_contact_metric(self) -> None:
        task = {
            "family": "2A.1",
            "card": {"card_id": "C1"},
            "_episode_scenes": [],
            "_reel_duration_seconds": 10.0,
        }
        metrics = score_track2_evidence(
            task,
            {"status": "ABSENT"},
            [{"start_seconds": 0, "end_seconds": 2}],
            reel_duration_seconds=10,
            listening_budget_seconds=20,
            stop_reason="model_stop",
        )
        self.assertIsNone(metrics["all_present_targets_contacted"])
        self.assertIsNone(metrics["cost_to_all_present_target_contact_seconds"])

    def test_probe_prompt_is_self_contained_for_2b2(self) -> None:
        task = {
            "family": "2B.2",
            "question_text": "Find utterances.",
            "scene_cards": [{"card_id": "C1", "text": "scene"}],
            "utterance_targets": [{"utterance_id": "U1", "text": "quote"}],
        }
        prompt = build_track2_probe_prompt(task, initialize_track2_ledger(task))
        self.assertIn("<target_id> FOUND <start_seconds> <end_seconds> <scene_card_id>", prompt)
        self.assertIn("exactly one line per target", prompt)


class Track2RuntimeTests(unittest.TestCase):
    def test_completed_results_are_reusable_regardless_of_answer_correctness(self) -> None:
        self.assertEqual(TRACK2_RESULT_SCHEMA_VERSION, 5)
        complete = {
            "track2_result_schema_version": TRACK2_RESULT_SCHEMA_VERSION,
            "run_status": "complete",
            "unrecovered_api_errors": [],
        }
        self.assertTrue(
            _is_reusable_track2_result(
                {**complete, "final_answer_semantic_valid": True}
            )
        )
        self.assertTrue(
            _is_reusable_track2_result(
                {**complete, "final_answer_semantic_valid": False}
            )
        )
        self.assertFalse(
            _is_reusable_track2_result(
                {
                    **complete,
                    "track2_result_schema_version": 4,
                }
            )
        )
        self.assertFalse(_is_reusable_track2_result({"run_status": "complete"}))
        self.assertFalse(_is_reusable_track2_result({"run_status": "api_error"}))
        self.assertFalse(_is_reusable_track2_result({"run_status": "unsupported"}))

    def test_rebuild_rejects_results_from_previous_scoring_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result_path = root / "active" / "2A.1" / "present" / "T1.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(
                    {
                        "track2_result_schema_version": 4,
                        "run_status": "complete",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "schema version 5"):
                rebuild_track2_outputs(root)
            self.assertFalse((root / "episode_metrics.csv").exists())
            self.assertFalse((root / "summary.json").exists())

    def test_semantically_invalid_complete_result_is_reused_without_second_attempt(self) -> None:
        class NoCallDummy(DummyAdapter):
            def send_track2(self, *args, **kwargs):
                raise AssertionError("completed result must be reused")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            results_root = base / "results"
            first = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=DummyAdapter(
                    [
                        '{"status":"PRESENT","card_id":"WRONG","scene_rank":1,'
                        '"start_seconds":0,"end_seconds":1,"confidence":0.9}'
                    ]
                ),
                conditions=["full_context"],
            )[0]
            self.assertFalse(first["final_answer_semantic_valid"])
            second = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=results_root,
                adapter=NoCallDummy([]),
                conditions=["full_context"],
            )[0]
            self.assertEqual(second, first)

    def test_progress_is_reported_without_changing_parallel_result_order_or_cache_use(self) -> None:
        class NoCallDummy(DummyAdapter):
            def send_track2(self, *args, **kwargs):
                raise AssertionError("completed results must be reused")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            tasks = [
                _scene_task(task_id="T1"),
                _scene_task(task_id="T2"),
            ]
            answers = [
                {
                    "task_id": task["task_id"],
                    "answer": {
                        "status": "ABSENT",
                        "card_id": task["card"]["card_id"],
                        "scene_rank": None,
                        "start_seconds": None,
                        "end_seconds": None,
                    },
                }
                for task in tasks
            ]
            dataset_root = _synthetic_dataset(base / "data", tasks, answers)
            results_root = base / "results"
            first_stdout = io.StringIO()
            with redirect_stdout(first_stdout):
                first = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=results_root,
                    adapter=DummyAdapter(
                        [
                            json.dumps(
                                {
                                    "status": "ABSENT",
                                    "card_id": task["card"]["card_id"],
                                    "scene_rank": None,
                                    "start_seconds": None,
                                    "end_seconds": None,
                                    "confidence": 1.0,
                                }
                            )
                            for task in tasks
                        ]
                    ),
                    conditions=["full_context"],
                    parallelism=2,
                )
            self.assertEqual([result["task_id"] for result in first], ["T1", "T2"])
            self.assertIn("track2-run start total=2 parallelism=2", first_stdout.getvalue())
            self.assertIn("track2-run [############################] 100% (2/2)", first_stdout.getvalue())

            cached_stdout = io.StringIO()
            with redirect_stdout(cached_stdout):
                cached = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=results_root,
                    adapter=NoCallDummy([]),
                    conditions=["full_context"],
                    parallelism=2,
                )
            self.assertEqual(cached, first)
            self.assertIn("status=complete", cached_stdout.getvalue())

            sequential_stdout = io.StringIO()
            with redirect_stdout(sequential_stdout):
                sequential = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / "sequential",
                    adapter=DummyAdapter(
                        [
                            json.dumps(
                                {
                                    "status": "ABSENT",
                                    "card_id": task["card"]["card_id"],
                                    "scene_rank": None,
                                    "start_seconds": None,
                                    "end_seconds": None,
                                    "confidence": 1.0,
                                }
                            )
                            for task in tasks
                        ]
                    ),
                    conditions=["full_context"],
                    parallelism=1,
                )
            self.assertEqual([result["task_id"] for result in sequential], ["T1", "T2"])
            self.assertIn("parallelism=1", sequential_stdout.getvalue())
            self.assertIn("status=complete", sequential_stdout.getvalue())

            unsupported_adapter = DummyAdapter([])
            unsupported_adapter.configure_track2_capabilities(
                {
                    "track2_supported_conditions": [],
                    "track2_supports_text_only": True,
                    "track2_supports_output_token_override": True,
                    "track2_max_audio_seconds": 384,
                    "track2_max_audio_bytes": 13_000_000,
                }
            )
            unsupported_stdout = io.StringIO()
            with redirect_stdout(unsupported_stdout):
                unsupported = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / "unsupported",
                    adapter=unsupported_adapter,
                    task_ids=["T1"],
                    conditions=["full_context"],
                    parallelism=1,
                )
            self.assertEqual(unsupported[0]["run_status"], "unsupported")
            self.assertIn("status=unsupported", unsupported_stdout.getvalue())

            class FailingDummy(DummyAdapter):
                def send_track2(self, *args, **kwargs):
                    raise APIAdapterError("hard failure", error_class=API_ERROR_HARD)

            api_error_stdout = io.StringIO()
            with redirect_stdout(api_error_stdout):
                api_error = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / "api_error",
                    adapter=FailingDummy([]),
                    task_ids=["T1"],
                    conditions=["full_context"],
                    parallelism=1,
                )
            self.assertEqual(api_error[0]["run_status"], "api_error")
            self.assertIn("status=api_error", api_error_stdout.getvalue())

    def test_query_crops_are_mono_16khz(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.wav"
            sf.write(
                source,
                np.zeros((48000, 2), dtype=np.float32),
                48000,
                subtype="PCM_16",
            )
            crop = _write_query_crop(source, root / "runtime", "T1", 1, 0.25, 0.75)
            info = sf.info(crop)
            self.assertEqual(info.samplerate, TRACK2_CROP_SAMPLE_RATE)
            self.assertEqual(info.channels, 1)
            self.assertAlmostEqual(info.duration, 0.5, places=3)

    def test_loader_joins_public_tasks_and_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            root = _synthetic_dataset(Path(tmpdir), [task], [answer])
            loaded = load_track2_dataset(root)
            self.assertEqual(len(loaded["tasks"]), 1)
            self.assertEqual(loaded["answers"]["T1"]["scene_rank"], 1)

    def test_loader_rejects_artifacts_changed_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            root = _synthetic_dataset(Path(tmpdir), [task], [answer])
            report = json.loads(
                (root / "dataset" / "validation_report.json").read_text(encoding="utf-8")
            )
            with (root / "dataset" / "task_instances.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with patch(
                "tot_benchmark.track2.TRACK2_DATASET_FINGERPRINT",
                report["deterministic_build_fingerprint"],
            ):
                with self.assertRaisesRegex(RuntimeError, "artifact hash mismatch"):
                    _load_track2_dataset(root)

    def test_active_uses_fresh_sessions_and_only_current_crop(self) -> None:
        class RecordingDummy(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.calls = []

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.calls.append(
                    {
                        "session": session,
                        "audio_paths": list(audio_paths or []),
                        "strict_text_only": kwargs["strict_text_only"],
                        "max_output_tokens": kwargs["max_output_tokens"],
                    }
                )
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = RecordingDummy(
                [
                    "NO_UPDATE\nACTION QUERY 0 1",
                    "SET C_TARGET FOUND 0 1\nACTION STOP",
                    '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":0,"end_seconds":1,"confidence":0.9}',
                ]
            )
            results = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                task_ids=["T1"],
                conditions=["active"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )
            self.assertEqual(results[0]["answer_metrics"]["primary_correct"], 1.0)
            self.assertEqual([len(call["audio_paths"]) for call in adapter.calls], [0, 1, 0])
            self.assertEqual([call["strict_text_only"] for call in adapter.calls], [True, False, True])
            self.assertEqual(len({id(call["session"]) for call in adapter.calls}), 3)
            self.assertEqual(adapter.calls[-1]["max_output_tokens"], 1024)

    def test_full_context_attaches_exactly_one_complete_reel(self) -> None:
        class RecordingDummy(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.calls = []

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.calls.append((list(audio_paths or []), kwargs))
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = RecordingDummy(
                ['{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":0,"end_seconds":1,"confidence":1.0}']
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["full_context"],
            )[0]
            self.assertEqual(result["answer_metrics"]["primary_correct"], 1.0)
            self.assertEqual(len(adapter.calls), 1)
            self.assertEqual(len(adapter.calls[0][0]), 1)
            self.assertFalse(adapter.calls[0][1]["strict_text_only"])
            self.assertEqual(adapter.calls[0][1]["max_output_tokens"], 1024)

    def test_unknown_capabilities_write_unsupported_without_calling_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = adapter_from_config({"adapter_type": "gemini", "model_id": "fake"})
            results = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["full_context"],
            )
            self.assertEqual(results[0]["run_status"], "unsupported")
            self.assertEqual(results[0]["unsupported_reason"], "condition_not_declared_supported")

    def test_partial_track2_capability_declaration_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "capability declaration is incomplete"):
            adapter_from_config(
                {
                    "adapter_type": "gemini",
                    "model_id": "fake",
                    "track2_supported_conditions": [],
                }
            )

    def test_missing_output_token_override_capability_is_unsupported(self) -> None:
        class NoCallDummy(DummyAdapter):
            def send_track2(self, *args, **kwargs):
                raise AssertionError("preflight must prevent inference")

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = NoCallDummy([])
            adapter.configure_track2_capabilities(
                {
                    "track2_supported_conditions": ["full_context"],
                    "track2_supports_text_only": False,
                    "track2_supports_output_token_override": False,
                    "track2_max_audio_seconds": 10,
                    "track2_max_audio_bytes": 1_000_000,
                }
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["full_context"],
            )[0]
            self.assertEqual(result["run_status"], "unsupported")
            self.assertEqual(
                result["unsupported_reason"],
                "output_token_override_not_declared_supported",
            )

    def test_periodic_probe_is_text_only_and_runs_after_second_observation(self) -> None:
        class RecordingDummy(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.calls = []

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.calls.append((list(audio_paths or []), kwargs["strict_text_only"]))
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = RecordingDummy(
                [
                    "NO_UPDATE\nACTION QUERY 0 0.5",
                    "NO_UPDATE\nACTION QUERY 0.5 1",
                    "SET C_TARGET FOUND 0 1\nACTION STOP",
                    "C_TARGET FOUND 0 1",
                    '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":0,"end_seconds":1,"confidence":0.9}',
                ]
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active_same_session_periodic_probe"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )[0]
            self.assertEqual(result["answer_metrics"]["primary_correct"], 1.0)
            self.assertEqual(len(result["probe_history"]), 1)
            self.assertEqual([len(call[0]) for call in adapter.calls], [0, 1, 1, 0, 0])
            self.assertEqual([call[1] for call in adapter.calls], [True, False, False, True, True])

    def test_syntax_repair_is_text_only_and_does_not_consume_query(self) -> None:
        class RecordingDummy(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.audio_counts = []

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.audio_counts.append(len(audio_paths or []))
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "ABSENT",
                    "card_id": "C_TARGET",
                    "scene_rank": None,
                    "start_seconds": None,
                    "end_seconds": None,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = RecordingDummy(
                [
                    "please query the beginning",
                    "NO_UPDATE\nACTION STOP",
                    '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,"start_seconds":null,"end_seconds":null,"confidence":0.8}',
                ]
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )[0]
            self.assertEqual(result["query_history"], [])
            self.assertEqual(adapter.audio_counts, [0, 0, 0])
            self.assertEqual(result["format_records"][0]["repair_attempted"], True)
            self.assertEqual(result["format_records"][0]["final_parse_valid"], True)

    def test_invalid_action_still_requests_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {"task_id": "T1", "answer": {"status": "ABSENT", "card_id": "C_TARGET", "scene_rank": None, "start_seconds": None, "end_seconds": None}}
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = DummyAdapter(
                [
                    "NO_UPDATE\nACTION QUERY -1 1",
                    '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,"start_seconds":null,"end_seconds":null,"confidence":0.7}',
                ]
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active"],
            )[0]
            self.assertEqual(result["stop_reason"], "invalid_action:query_out_of_range")
            self.assertEqual(result["query_history"], [])
            self.assertEqual(result["answer_metrics"]["primary_correct"], 1.0)

    def test_semantically_invalid_answer_retains_calibration_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=DummyAdapter(
                    [
                        '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,'
                        '"start_seconds":2,"end_seconds":3,"confidence":0.9}'
                    ]
                ),
                conditions=["full_context"],
            )[0]
            self.assertIsNone(result["final_answer"])
            self.assertEqual(result["confidence"], 0.9)

    def test_universal_rescue_is_scored_without_hiding_strict_format_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "ABSENT",
                    "card_id": "C_TARGET",
                    "scene_rank": None,
                    "start_seconds": None,
                    "end_seconds": None,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=DummyAdapter(
                    [
                        "```json\n"
                        '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,'
                        '"start_seconds":null,"end_seconds":null,"confidence":0.9}'
                        "\n```"
                    ]
                ),
                conditions=["full_context"],
            )[0]
            record = result["format_records"][0]
            self.assertFalse(record["raw_parse_valid"])
            self.assertTrue(record["raw_rescue_applied"])
            self.assertFalse(record["repair_attempted"])
            self.assertTrue(result["final_answer_semantic_valid"])
            self.assertEqual(result["format_metrics"]["raw_parse_success"], 0.0)
            self.assertEqual(result["format_metrics"]["universal_rescue_rate"], 1.0)
            self.assertEqual(result["answer_metrics"]["primary_correct"], 1.0)

    def test_exact_json_semantic_errors_do_not_trigger_repair(self) -> None:
        invalid_answers = (
            '{"status":"PRESENT","card_id":"WRONG","scene_rank":1,'
            '"start_seconds":0,"end_seconds":1,"confidence":0.9}',
            '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":1,'
            '"start_seconds":null,"end_seconds":null,"confidence":0.9}',
        )
        for index, output in enumerate(invalid_answers):
            with self.subTest(output=output), tempfile.TemporaryDirectory() as tmpdir:
                base = Path(tmpdir)
                task = _scene_task()
                answer = {
                    "task_id": "T1",
                    "answer": {
                        "status": "PRESENT",
                        "card_id": "C_TARGET",
                        "scene_rank": 1,
                        "start_seconds": 0,
                        "end_seconds": 1,
                    },
                }
                dataset_root = _synthetic_dataset(base / "data", [task], [answer])
                result = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / f"results_{index}",
                    adapter=DummyAdapter([output]),
                    conditions=["full_context"],
                )[0]
                self.assertTrue(result["format_records"][0]["raw_parse_valid"])
                self.assertFalse(result["format_records"][0]["repair_attempted"])
                self.assertFalse(result["final_answer_semantic_valid"])
                self.assertIsNone(result["final_answer"])
            self.assertEqual(result["answer_metrics"]["primary_correct"], 0.0)

    def test_semantically_invalid_present_claim_retains_false_localization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = {**_scene_task(), "variant": "absent"}
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "ABSENT",
                    "card_id": "C_TARGET",
                    "scene_rank": None,
                    "start_seconds": None,
                    "end_seconds": None,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=DummyAdapter(
                    [
                        '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,'
                        '"start_seconds":2,"end_seconds":3,"confidence":0.9}'
                    ]
                ),
                conditions=["full_context"],
            )[0]
            self.assertFalse(result["final_answer_semantic_valid"])
            self.assertEqual(result["answer_metrics"]["primary_correct"], 0.0)
            self.assertEqual(result["answer_metrics"]["false_localization_rate"], 1.0)

    def test_action_api_failure_is_recorded_and_final_answer_is_still_requested(self) -> None:
        class FailingActionAdapter(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.call_count = 0

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.call_count += 1
                if self.call_count <= 3:
                    raise APIAdapterError("hard failure", error_class=API_ERROR_HARD)
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {"task_id": "T1", "answer": {"status": "ABSENT", "card_id": "C_TARGET", "scene_rank": None, "start_seconds": None, "end_seconds": None}}
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = FailingActionAdapter(
                ['{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,"start_seconds":null,"end_seconds":null,"confidence":0.7}']
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )[0]
            self.assertEqual(result["stop_reason"], "api_error:action")
            self.assertEqual(result["run_status"], "api_error")
            self.assertEqual(result["answer_metrics"], {})
            self.assertEqual(result["diagnostic_final_answer"]["status"], "ABSENT")
            self.assertEqual(result["unrecovered_api_errors"][0]["stage"], "action")
            self.assertEqual(len(result["format_records"][0]["api_attempts"]), 3)

    def test_action_repair_api_failure_makes_episode_unscored(self) -> None:
        class FailingRepairAdapter(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.call_count = 0

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.call_count += 1
                if 2 <= self.call_count <= 4:
                    raise APIAdapterError("repair failure", error_class=API_ERROR_HARD)
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {"task_id": "T1", "answer": {"status": "ABSENT", "card_id": "C_TARGET", "scene_rank": None, "start_seconds": None, "end_seconds": None}}
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = FailingRepairAdapter(
                [
                    "not valid action syntax",
                    '{"status":"ABSENT","card_id":"C_TARGET","scene_rank":null,"start_seconds":null,"end_seconds":null,"confidence":0.7}',
                ]
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )[0]
            self.assertEqual(result["run_status"], "api_error")
            self.assertEqual(result["stop_reason"], "api_error:action_repair")
            self.assertEqual(result["unrecovered_api_errors"][0]["stage"], "action_repair")
            self.assertEqual(result["answer_metrics"], {})

    def test_probe_api_failure_stops_exploration_and_makes_episode_unscored(self) -> None:
        class FailingProbeAdapter(DummyAdapter):
            def __init__(self, outputs):
                super().__init__(outputs)
                self.call_count = 0

            def send_track2(self, session, user_text, audio_paths=None, **kwargs):
                self.call_count += 1
                if 4 <= self.call_count <= 6:
                    raise APIAdapterError("probe failure", error_class=API_ERROR_HARD)
                return super().send_track2(session, user_text, audio_paths, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {"task_id": "T1", "answer": {"status": "PRESENT", "card_id": "C_TARGET", "scene_rank": 1, "start_seconds": 0, "end_seconds": 1}}
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            adapter = FailingProbeAdapter(
                [
                    "NO_UPDATE\nACTION QUERY 0 0.5",
                    "NO_UPDATE\nACTION QUERY 0.5 1",
                    "SET C_TARGET FOUND 0 1\nACTION STOP",
                    '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":0,"end_seconds":1,"confidence":0.8}',
                ]
            )
            result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "results",
                adapter=adapter,
                conditions=["active_same_session_periodic_probe"],
                config=Track2Config(api_retry_sleep_seconds=0),
            )[0]
            self.assertEqual(result["run_status"], "api_error")
            self.assertEqual(result["stop_reason"], "api_error:probe")
            self.assertEqual(result["unrecovered_api_errors"][0]["stage"], "probe")
            self.assertEqual(result["answer_metrics"], {})

    def test_step_and_listening_limits_stop_only_after_latest_crop_is_observed(self) -> None:
        final = '{"status":"PRESENT","card_id":"C_TARGET","scene_rank":1,"start_seconds":0,"end_seconds":1,"confidence":0.9}'
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {"task_id": "T1", "answer": {"status": "PRESENT", "card_id": "C_TARGET", "scene_rank": 1, "start_seconds": 0, "end_seconds": 1}}
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            step_outputs = ["NO_UPDATE\nACTION QUERY 0 0.1"] * 7 + [final]
            step_result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "step_results",
                adapter=DummyAdapter(step_outputs),
                conditions=["active"],
            )[0]
            self.assertEqual(step_result["stop_reason"], "max_queries_reached")
            self.assertEqual(len(step_result["query_history"]), 6)

            manifest_path = dataset_root / "dataset" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["episodes"][0]["duration_seconds"] = 384.0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            _refresh_synthetic_validation_report(dataset_root)
            listening_outputs = [
                "NO_UPDATE\nACTION QUERY 0 384",
                "NO_UPDATE\nACTION QUERY 0 384",
                "NO_UPDATE\nACTION STOP",
                final,
            ]
            listening_result = run_track2_benchmark(
                dataset_root=dataset_root,
                results_root=base / "listening_results",
                adapter=DummyAdapter(listening_outputs),
                conditions=["active"],
            )[0]
            self.assertEqual(listening_result["stop_reason"], "listening_budget_reached")
            self.assertEqual(len(listening_result["query_history"]), 2)
            self.assertEqual(listening_result["evidence_metrics"]["total_audio_inspected_seconds"], 768.0)

    def test_budget_defaults_are_frozen(self) -> None:
        self.assertEqual(track2_budget_for_task(_scene_task()), track2_budget_for_task(_scene_task()))
        self.assertEqual(track2_budget_for_task(_scene_task()).listening_budget_seconds, 768.0)

    def test_protocol_constants_cannot_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            task = _scene_task()
            answer = {
                "task_id": "T1",
                "answer": {
                    "status": "PRESENT",
                    "card_id": "C_TARGET",
                    "scene_rank": 1,
                    "start_seconds": 0,
                    "end_seconds": 1,
                },
            }
            dataset_root = _synthetic_dataset(base / "data", [task], [answer])
            with self.assertRaisesRegex(RuntimeError, "probe_every_n"):
                run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / "results",
                    adapter=DummyAdapter([]),
                    config=Track2Config(probe_every_n=3),
                )

    def test_every_family_runs_end_to_end_in_each_condition(self) -> None:
        tasks = [
            _scene_task("A1"),
            {
                "task_id": "A2", "track": "2", "family": "2A.2", "variant": "exact_global",
                "reel_id": "R1", "audio_path": "episodes/R1.wav", "question_text": "Find it.",
                "target_id": "U_TARGET", "target_text": "words", "quote_form": "exact", "search_scope": "global",
            },
            {
                "task_id": "A3", "track": "2", "family": "2A.3", "variant": "absent_utterance",
                "reel_id": "R1", "audio_path": "episodes/R1.wav", "question_text": "Find it or absent.",
                "target_id": "U_TARGET", "target_text": "missing",
            },
            {
                "task_id": "B1", "track": "2", "family": "2B.1", "variant": "exact_count",
                "reel_id": "R1", "audio_path": "episodes/R1.wav", "question_text": "Order cards.",
                "cards": [{"card_id": "C1", "text": "first"}, {"card_id": "C2", "text": "second"}],
            },
            {
                "task_id": "B2", "track": "2", "family": "2B.2", "variant": "structural_relational_retrieval",
                "reel_id": "R1", "audio_path": "episodes/R1.wav", "question_text": "Retrieve structure.",
                "scene_cards": [{"card_id": "C1", "text": "first"}, {"card_id": "C2", "text": "second"}],
                "utterance_targets": [
                    {"utterance_id": "U1", "text": "words", "quote_form": "exact"},
                    {"utterance_id": "U2", "text": "missing", "quote_form": "exact"},
                ],
            },
        ]
        gold_answers = [
            {"task_id": "A1", "answer": {"status": "PRESENT", "card_id": "C_TARGET", "scene_rank": 1, "start_seconds": 0, "end_seconds": 1}},
            {"task_id": "A2", "answer": {"status": "PRESENT", "start_seconds": 0.2, "end_seconds": 0.4}},
            {"task_id": "A3", "answer": {"status": "ABSENT", "occurrences": []}},
            {"task_id": "B1", "answer": {"ordered_card_ids": ["C1", "C2"], "absent_card_id": None}},
            {
                "task_id": "B2",
                "answer": {
                    "ordered_real_utterance_ids": ["U1"],
                    "utterances": [
                        {"utterance_id": "U1", "status": "PRESENT", "start_seconds": 0.2, "end_seconds": 0.4, "scene_card_id": "C1"},
                        {"utterance_id": "U2", "status": "ABSENT", "start_seconds": None, "end_seconds": None, "scene_card_id": None},
                    ],
                },
            },
        ]
        final_outputs = []
        for task, record in zip(tasks, gold_answers):
            answer = dict(record["answer"])
            if task["family"] in {"2A.1", "2A.2", "2A.3"}:
                answer["confidence"] = 1.0
            if task["family"] == "2B.2":
                answer["confidence_complete"] = 1.0
            final_outputs.append(json.dumps(answer))

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            dataset_root = _synthetic_dataset(base / "data", tasks, gold_answers)
            for condition in ("full_context", "active", "active_same_session_periodic_probe"):
                outputs = (
                    final_outputs
                    if condition == "full_context"
                    else [item for final in final_outputs for item in ("NO_UPDATE\nACTION STOP", final)]
                )
                results = run_track2_benchmark(
                    dataset_root=dataset_root,
                    results_root=base / condition,
                    adapter=DummyAdapter(outputs),
                    conditions=[condition],
                )
                self.assertEqual(len(results), 5)
                self.assertTrue(all(result["answer_metrics"]["primary_correct"] == 1.0 for result in results))


if __name__ == "__main__":
    unittest.main()
