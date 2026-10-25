from __future__ import annotations

import base64
import io
import json
import tempfile
import unittest
import wave
from json import loads
from pathlib import Path
from unittest.mock import patch

import requests

from tot_benchmark.adapters import (
    API_ERROR_HARD,
    API_ERROR_SOFT,
    APIAdapterError,
    AdapterSession,
    BaseAdapter,
    GeminiAudioAdapter,
    OpenAIResponsesAudioAdapter,
    _retry_after_seconds_from_text,
    adapter_from_config,
    classify_api_error,
    wav_bytes_with_min_duration,
)
from tot_benchmark.api_error_migration import archive_unresolved_results
from tot_benchmark.api_error_rerun import (
    _rebuild_track_outputs,
    _run_selected_specs,
    find_failed_run_specs,
    result_has_unrecovered_api_error,
)
from tot_benchmark.audio_model_adapters import GPTAudioMiniAdapter
from tot_benchmark.core import (
    API_STATUS_COMPLETE,
    API_STATUS_HARD_ERROR,
    API_STATUS_SOFT_ERROR,
    _send_with_api_retry,
    _write_summary_json,
    result_api_status,
)
from tot_benchmark.track2 import TRACK2_RESULT_SCHEMA_VERSION


class ApiErrorRerunTest(unittest.TestCase):
    @staticmethod
    def _track2_result(*, task_id: str, error_class: str) -> dict[str, object]:
        return {
            "track2_result_schema_version": 2,
            "task_id": task_id,
            "reel_id": "reel_001",
            "family": "2A.1",
            "variant": "present",
            "condition": "active",
            "model_id": "api-model",
            "run_status": "api_error",
            "unrecovered_api_errors": [
                {
                    "stage": "final",
                    "api_attempts": [
                        {"status": "error", "error_class": error_class}
                    ],
                }
            ],
            "answer_metrics": {},
            "evidence_metrics": {},
            "belief_metrics": {},
            "format_metrics": {},
            "confidence": None,
        }

    def test_track2_discovers_and_reruns_standalone_and_mixed_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            legacy_results_root = model_dir / "results"
            standalone_root = model_dir / "results_track2"
            mixed_root = legacy_results_root / "track_2"
            for root, task_id in (
                (standalone_root, "task_standalone"),
                (mixed_root, "task_mixed"),
            ):
                result_path = root / "active" / "2A.1" / "present" / f"{task_id}.json"
                result_path.parent.mkdir(parents=True)
                result_path.write_text(
                    json.dumps(self._track2_result(task_id=task_id, error_class=API_ERROR_SOFT)),
                    encoding="utf-8",
                )

            specs = find_failed_run_specs(legacy_results_root, ["2"])
            self.assertEqual({spec.task_id for spec in specs}, {"task_standalone", "task_mixed"})
            self.assertTrue(all(spec.regime is None and spec.episode_id is None for spec in specs))
            self.assertEqual(
                {spec.benchmark_results_root for spec in specs},
                {standalone_root, mixed_root},
            )

            calls: list[dict[str, object]] = []

            def fake_run_track2_benchmark(**kwargs):
                calls.append(kwargs)
                for task_id in kwargs["task_ids"]:
                    result_path = (
                        Path(kwargs["results_root"])
                        / kwargs["conditions"][0]
                        / "2A.1"
                        / "present"
                        / f"{task_id}.json"
                    )
                    result = json.loads(result_path.read_text(encoding="utf-8"))
                    result["track2_result_schema_version"] = TRACK2_RESULT_SCHEMA_VERSION
                    result_path.write_text(json.dumps(result), encoding="utf-8")
                return []

            with patch(
                "tot_benchmark.track2.run_track2_benchmark",
                side_effect=fake_run_track2_benchmark,
            ):
                _run_selected_specs(
                    specs=specs,
                    dataset_root=Path(tmpdir) / "unused-track1-dataset",
                    results_root=legacy_results_root,
                    adapter=object(),
                    parallelism=7,
                    time_mode="binned",
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(
                {call["results_root"] for call in calls},
                {standalone_root, mixed_root},
            )
            self.assertEqual(
                {tuple(call["task_ids"]) for call in calls},
                {("task_standalone",), ("task_mixed",)},
            )
            self.assertTrue(all(call["conditions"] == ["active"] for call in calls))
            self.assertTrue(all(call["reprocess"] is True for call in calls))
            self.assertTrue(all(call["parallelism"] == 7 for call in calls))

            _rebuild_track_outputs(
                tracks=["2"],
                dataset_root=Path(tmpdir) / "unused-track1-dataset",
                results_root=legacy_results_root,
                adapter=object(),
                time_mode="binned",
            )
            for root in (standalone_root, mixed_root):
                self.assertTrue((root / "summary.json").is_file())
                self.assertTrue((root / "episode_metrics.csv").is_file())

    def test_track2_discovery_filters_hard_api_errors_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            results_root = Path(tmpdir) / "results_track2"
            result_path = results_root / "active" / "2A.1" / "present" / "task_hard.json"
            result_path.parent.mkdir(parents=True)
            result_path.write_text(
                json.dumps(self._track2_result(task_id="task_hard", error_class=API_ERROR_HARD)),
                encoding="utf-8",
            )
            self.assertEqual(find_failed_run_specs(results_root, ["2"]), [])
            selected = find_failed_run_specs(
                results_root,
                ["2"],
                statuses=(API_STATUS_HARD_ERROR,),
            )
            self.assertEqual([spec.task_id for spec in selected], ["task_hard"])

    def test_selects_terminal_action_api_error(self) -> None:
        self.assertTrue(
            result_has_unrecovered_api_error(
                {
                    "stop_reason": "invalid_action:api_error",
                    "repair_attempts": [],
                }
            )
        )

    def test_selects_unrecovered_prompt_api_error(self) -> None:
        self.assertTrue(
            result_has_unrecovered_api_error(
                {
                    "stop_reason": "full_context",
                    "checkpoint_outputs": [
                        {
                            "prompt": "required question",
                            "raw_output": "",
                            "parsed": {"format_valid": False, "reason": "api_error"},
                        }
                    ],
                }
            )
        )

    def test_ignores_recovered_transient_api_error(self) -> None:
        self.assertFalse(
            result_has_unrecovered_api_error(
                {
                    "stop_reason": "model_stop",
                    "checkpoint_outputs": [
                        {
                            "prompt": "required question",
                            "raw_output": "A",
                            "parsed": {"format_valid": True, "answer": "A"},
                            "repair_attempts": [
                                {"attempt": 1, "reason": "api_error", "error": "429"}
                            ],
                        }
                    ],
                }
            )
        )

    def test_ignores_model_and_protocol_failures(self) -> None:
        for result in (
            {"stop_reason": "invalid_action:invalid_action_format"},
            {"stop_reason": "max_steps_reached"},
            {"stop_reason": "model_stop"},
            {
                "stop_reason": "full_context",
                "checkpoint_outputs": [
                    {
                        "prompt": "required question",
                        "raw_output": "wrong answer",
                        "parsed": {"format_valid": True, "answer": "wrong answer"},
                    }
                ],
            },
        ):
            with self.subTest(result=result):
                self.assertFalse(result_has_unrecovered_api_error(result))

    def test_ignores_intentionally_not_evaluable_placeholder(self) -> None:
        self.assertFalse(
            result_has_unrecovered_api_error(
                {
                    "stop_reason": "model_stop",
                    "core_behavioral_outputs": {
                        "sequence": {
                            "asked": False,
                            "prompt": "",
                            "raw_output": "",
                            "parsed": {"format_valid": False, "reason": "not_evaluable"},
                        }
                    },
                }
            )
        )


class ApiRetryPolicyTest(unittest.TestCase):
    class ScriptedAdapter(BaseAdapter):
        def __init__(self, outcomes):
            self.model_id = "scripted"
            self.outcomes = list(outcomes)
            self.calls = 0

        def create_session(self, system_prompt: str) -> AdapterSession:
            return AdapterSession(system_prompt=system_prompt)

        def send(self, session, user_text, audio_paths=None, preserve_multiline=False):
            outcome = self.outcomes[self.calls]
            self.calls += 1
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def test_hard_error_uses_three_total_attempts(self) -> None:
        adapter = self.ScriptedAdapter(
            [
                APIAdapterError("400 invalid payload", status_code=400),
                APIAdapterError("400 invalid payload", status_code=400),
                APIAdapterError("400 invalid payload", status_code=400),
            ]
        )
        with patch("tot_benchmark.core.time.sleep") as sleep:
            raw, attempts = _send_with_api_retry(
                adapter,
                adapter.create_session("system"),
                "prompt",
                [],
                60.0,
            )
        self.assertIsNone(raw)
        self.assertEqual(adapter.calls, 3)
        self.assertEqual([item["attempt"] for item in attempts], [1, 2, 3])
        self.assertTrue(all(item["error_class"] == API_ERROR_HARD for item in attempts))
        self.assertTrue(all(item["status"] == "error" for item in attempts))
        self.assertTrue(all(item["retry_delay_seconds"] == 0.0 for item in attempts))
        sleep.assert_not_called()

    def test_soft_error_retries_once_after_delay(self) -> None:
        adapter = self.ScriptedAdapter(
            [
                APIAdapterError(
                    "429 quota",
                    status_code=429,
                    retry_after_seconds=2.5,
                ),
                APIAdapterError("503 unavailable", status_code=503),
            ]
        )
        with patch("tot_benchmark.core.time.sleep") as sleep:
            raw, attempts = _send_with_api_retry(
                adapter,
                adapter.create_session("system"),
                "prompt",
                [],
                60.0,
            )
        self.assertIsNone(raw)
        self.assertEqual(adapter.calls, 2)
        self.assertEqual([item["attempt"] for item in attempts], [1, 2])
        self.assertTrue(all(item["error_class"] == API_ERROR_SOFT for item in attempts))
        self.assertTrue(all(item["status"] == "error" for item in attempts))
        self.assertEqual(attempts[0]["retry_delay_seconds"], 2.5)
        self.assertEqual(attempts[1]["retry_delay_seconds"], 0.0)
        sleep.assert_called_once_with(2.5)

    def test_provider_retry_delay_parses_compound_duration_and_caps_sleep(self) -> None:
        message = (
            "Please retry in 11h35m14.642110785s. "
            "'retryDelay': '41714s'"
        )
        self.assertEqual(_retry_after_seconds_from_text(message), 41714.0)
        self.assertAlmostEqual(
            _retry_after_seconds_from_text("Please retry in 11h35m14.642110785s."),
            41714.642110785,
        )
        adapter = self.ScriptedAdapter(
            [
                APIAdapterError(message, status_code=429),
                "STOP",
            ]
        )
        with patch("tot_benchmark.core.time.sleep") as sleep:
            raw, attempts = _send_with_api_retry(
                adapter,
                adapter.create_session("system"),
                "prompt",
                [],
                60.0,
            )
        self.assertEqual(raw, "STOP")
        self.assertEqual(attempts[0]["provider_retry_delay_seconds"], 41714.0)
        self.assertEqual(attempts[0]["retry_delay_seconds"], 600.0)
        sleep.assert_called_once_with(600.0)

    def test_recovered_soft_error_is_complete(self) -> None:
        adapter = self.ScriptedAdapter(
            [APIAdapterError("429 quota", status_code=429), "STOP"]
        )
        with patch("tot_benchmark.core.time.sleep"):
            raw, attempts = _send_with_api_retry(
                adapter,
                adapter.create_session("system"),
                "prompt",
                [],
                0.0,
        )
        self.assertEqual(raw, "STOP")
        self.assertEqual(len(attempts), 2)
        self.assertEqual([item["status"] for item in attempts], ["error", "success"])
        self.assertEqual(attempts[1]["error_class"], None)
        self.assertEqual(attempts[1]["status_code"], None)
        self.assertEqual(attempts[1]["message"], "")
        result = {"stop_reason": "model_stop", "repair_attempts": [{"api_attempts": attempts}]}
        self.assertEqual(result_api_status(result), API_STATUS_COMPLETE)

    def test_openai_extractors_normalize_malformed_responses_as_soft_errors(self) -> None:
        for extractor in (
            OpenAIResponsesAudioAdapter._extract_text,
            GPTAudioMiniAdapter._extract_text,
        ):
            for response in ({"choices": []}, {"choices": None}, [], None):
                with self.subTest(extractor=extractor.__qualname__, response=response):
                    with self.assertRaises(APIAdapterError) as raised:
                        extractor(response)
                    self.assertEqual(raised.exception.error_class, API_ERROR_SOFT)

    def test_gpt_audio_mini_invalid_endpoint_is_hard_configuration_error(self) -> None:
        adapter = GPTAudioMiniAdapter(api_base_url="not-a-url")
        adapter._api_key = "test-key"
        with patch(
            "tot_benchmark.audio_model_adapters.requests.post",
            side_effect=requests.exceptions.MissingSchema("missing scheme"),
        ):
            with self.assertRaises(APIAdapterError) as raised:
                adapter.send(adapter.create_session("system"), "prompt")
        self.assertEqual(raised.exception.error_class, API_ERROR_HARD)

    def test_result_status_distinguishes_soft_and_hard(self) -> None:
        def result(error_class):
            return {
                "stop_reason": "invalid_action:api_error",
                "repair_attempts": [
                    {
                        "final_parse": {"reason": "api_error"},
                        "api_failures": [
                            {
                                "reason": "api_error",
                                "error": "failure",
                                "error_class": error_class,
                            }
                        ],
                    }
                ],
            }

        self.assertEqual(result_api_status(result(API_ERROR_SOFT)), API_STATUS_SOFT_ERROR)
        self.assertEqual(result_api_status(result(API_ERROR_HARD)), API_STATUS_HARD_ERROR)

    def test_track1c_checkpoint_status_uses_api_attempt_history(self) -> None:
        hard_failure = {
            "stop_reason": "model_stop",
            "checkpoint_outputs": [
                {
                    "prompt": "required question",
                    "raw_output": "",
                    "parsed": {"format_valid": False, "reason": "api_error"},
                    "repair_attempts": [],
                    "api_attempts": [
                        {
                            "attempt": 1,
                            "status": "error",
                            "outcome": "error",
                            "reason": "api_error",
                            "error_class": API_ERROR_HARD,
                            "status_code": 400,
                            "message": "invalid payload",
                            "retry_delay_seconds": 0.0,
                        }
                    ],
                }
            ],
        }
        recovered_soft_failure = {
            "stop_reason": "model_stop",
            "checkpoint_outputs": [
                {
                    "prompt": "required question",
                    "raw_output": "A",
                    "parsed": {"format_valid": True, "answer": "A"},
                    "repair_attempts": [],
                    "api_attempts": [
                        {
                            "attempt": 1,
                            "status": "error",
                            "outcome": "error",
                            "reason": "api_error",
                            "error_class": API_ERROR_SOFT,
                            "status_code": 429,
                            "message": "rate limited",
                            "retry_delay_seconds": 1.0,
                        },
                        {
                            "attempt": 2,
                            "status": "success",
                            "outcome": "success",
                            "reason": "api_success",
                            "error_class": None,
                            "status_code": None,
                            "message": "",
                            "retry_delay_seconds": 0.0,
                        },
                    ],
                }
            ],
        }
        self.assertEqual(result_api_status(hard_failure), API_STATUS_HARD_ERROR)
        self.assertEqual(result_api_status(recovered_soft_failure), API_STATUS_COMPLETE)

    def test_provider_error_classification(self) -> None:
        for status in (408, 409, 425, 429, 500, 503, 504):
            with self.subTest(status=status):
                self.assertEqual(classify_api_error("provider failure", status), API_ERROR_SOFT)
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                self.assertEqual(classify_api_error("provider failure", status), API_ERROR_HARD)
        for message in (
            "RESOURCE_EXHAUSTED",
            "Too many requests",
            "TLS alert",
            "response did not contain any text output",
        ):
            with self.subTest(message=message):
                self.assertEqual(classify_api_error(message), API_ERROR_SOFT)

    def test_summary_excludes_unresolved_api_results(self) -> None:
        complete = {
            "api_status": API_STATUS_COMPLETE,
            "format_metrics": {"score": 1.0},
        }
        soft = {
            "api_status": API_STATUS_SOFT_ERROR,
            "format_metrics": {"score": 0.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.json"
            _write_summary_json([complete, soft], path)
            summary = loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["num_rows"], 1)
        self.assertEqual(summary["num_results_total"], 2)
        self.assertEqual(summary["api_status_counts"][API_STATUS_SOFT_ERROR], 1)
        self.assertEqual(summary["metrics"]["score"], 1.0)


class ApiErrorMigrationTest(unittest.TestCase):
    @staticmethod
    def _result(*, episode_id: str, stop_reason: str, error_class: str | None = None):
        result = {
            "episode_id": episode_id,
            "condition": "active",
            "regime": "Discovery",
            "model_id": "model",
            "stop_reason": stop_reason,
            "repair_attempts": [],
        }
        if error_class is not None:
            result["repair_attempts"] = [
                {
                    "final_parse": {"reason": "api_error"},
                    "api_failures": [
                        {
                            "reason": "api_error",
                            "error": "failure",
                            "error_class": error_class,
                        }
                    ],
                }
            ]
        return result

    def test_archive_moves_only_unresolved_api_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "model"
            result_dir = (
                model_dir
                / "results"
                / "track_1a"
                / "episode_results"
                / "active"
                / "Discovery"
            )
            result_dir.mkdir(parents=True)
            failed = result_dir / "episode_000.json"
            complete = result_dir / "episode_001.json"
            failed.write_text(
                __import__("json").dumps(
                    self._result(
                        episode_id="episode_000",
                        stop_reason="invalid_action:api_error",
                        error_class=API_ERROR_HARD,
                    )
                ),
                encoding="utf-8",
            )
            complete.write_text(
                __import__("json").dumps(
                    self._result(episode_id="episode_001", stop_reason="model_stop")
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                find_failed_run_specs(model_dir / "results", ["1a"]),
                [],
            )
            manifest = archive_unresolved_results(
                model_dir=model_dir,
                tracks=["1a"],
                time_mode="binned",
                archive_timestamp="test",
                dry_run=False,
            )

            archived = (
                model_dir
                / "results"
                / "api_error_archive"
                / "retry_policy_v2"
                / "test"
                / "track_1a"
                / "episode_results"
                / "active"
                / "Discovery"
                / "episode_000.json"
            )
            self.assertEqual(manifest["num_archived"], 1)
            self.assertFalse(failed.exists())
            self.assertTrue(archived.exists())
            self.assertTrue(complete.exists())


class GeminiThinkingConfigTest(unittest.TestCase):
    def test_gemini_25_omits_thinking_level_for_model_default(self) -> None:
        adapter = adapter_from_config(
            {
                "adapter_type": "gemini",
                "model_id": "gemini-2.5-flash",
            }
        )
        self.assertIsInstance(adapter, GeminiAudioAdapter)
        self.assertIsNone(adapter.thinking_level)

    def test_gemini_3_flash_uses_explicit_default_high_level(self) -> None:
        config_path = (
            Path(__file__).resolve().parent.parent / "configs" / "gemini_3_flash" / "config.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertNotIn("track2_thinking_level", config)
        adapter = adapter_from_config(config)
        self.assertIsInstance(adapter, GeminiAudioAdapter)
        self.assertEqual(adapter.thinking_level, "high")


class OpenAIAudioTransportTest(unittest.TestCase):
    def test_short_wav_is_silence_padded_without_changing_source_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "short.wav"
            source_frames = b"\x01\x00" * 160
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(source_frames)

            padded = wav_bytes_with_min_duration(path)
            item = OpenAIResponsesAudioAdapter._audio_item(path)

        self.assertEqual(base64.b64decode(item["input_audio"]["data"]), padded)
        with wave.open(io.BytesIO(padded), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 16000)
            self.assertEqual(handle.getnframes(), 4000)
            frames = handle.readframes(handle.getnframes())
        self.assertEqual(frames[: len(source_frames)], source_frames)
        self.assertEqual(frames[len(source_frames) :], b"\x00" * (8000 - len(source_frames)))

    def test_wav_at_minimum_duration_is_not_reencoded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "minimum.wav"
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 4000)
            original = path.read_bytes()
            self.assertEqual(wav_bytes_with_min_duration(path), original)


if __name__ == "__main__":
    unittest.main()
