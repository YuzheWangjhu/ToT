from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from track1c import build_dataset as track1c_build


TEST_VOICE_PAIRS = {
    "pair_1": {"female": "female_1", "male": "male_1"},
    "pair_2": {"female": "female_2", "male": "male_2"},
}


def make_row(conversation_id: str = "T01") -> dict[str, str]:
    sentence_metadata = [
        {
            "sentence_id": f"{conversation_id}_S01",
            "speaker": "A",
            "text": "The pickup color is blue.",
            "abstract_start_bin": 0,
            "abstract_end_bin": 0,
            "observation_rule": "full_sentence_required",
        },
        {
            "sentence_id": f"{conversation_id}_S02",
            "speaker": "B",
            "text": "Correction, the pickup color is green.",
            "abstract_start_bin": 1,
            "abstract_end_bin": 1,
            "observation_rule": "full_sentence_required",
        },
    ]
    generic_questions = [
        {
            "question_id": f"{conversation_id}_GEN_Q1",
            "question_type": "final_heard_speaker",
            "prompt": "Which speaker is final?",
            "allowed_answers": ["A", "B", "UNKNOWN"],
        }
    ]
    generic_answer_rules = [
        {
            "question_id": f"{conversation_id}_GEN_Q1",
            "ask_after": [f"{conversation_id}_S01", f"{conversation_id}_S02", "END"],
            "sentence_observation_rule": "A sentence is observed only when the full sentence has been explored.",
            "prefix_answer_trace": {
                "START": "UNKNOWN",
                f"{conversation_id}_S01": "A",
                f"{conversation_id}_S02": "B",
            },
            "final_answer_full_evidence": "B",
            "evidence_rules": [{"compute_from_observed_sentence_ids": "test"}],
        }
    ]
    category_questions = [
        {
            "question_id": f"{conversation_id}_CAT_Q1",
            "question_type": "slot_value",
            "prompt": "What is the pickup color?",
            "allowed_answers": ["BLUE", "GREEN", "UNKNOWN"],
        }
    ]
    category_rules = [
        {
            "question_id": f"{conversation_id}_CAT_Q1",
            "ask_after": [f"{conversation_id}_S01", f"{conversation_id}_S02", "END"],
            "sentence_observation_rule": "A sentence is observed only when the full sentence has been explored.",
            "prefix_answer_trace": {
                "START": "UNKNOWN",
                f"{conversation_id}_S01": "BLUE",
                f"{conversation_id}_S02": "GREEN",
            },
            "final_answer_full_evidence": "GREEN",
            "evidence_rules": [{"if_observed_all": [f"{conversation_id}_S02"], "answer": "GREEN"}],
        }
    ]
    content_questions = [
        {
            "question_id": f"{conversation_id}_CONTENT_Q1",
            "question_type": "speaker_content_attribution",
            "prompt": "Who says green?",
            "allowed_answers": ["A", "B", "UNKNOWN"],
        }
    ]
    content_rules = [
        {
            "question_id": f"{conversation_id}_CONTENT_Q1",
            "ask_after": [f"{conversation_id}_S02", "END"],
            "sentence_observation_rule": "A sentence is observed only when the full sentence has been explored.",
            "prefix_answer_trace": {
                "START": "UNKNOWN",
                f"{conversation_id}_S01": "UNKNOWN",
                f"{conversation_id}_S02": "B",
            },
            "final_answer_full_evidence": "B",
            "evidence_rules": [{"if_observed_all": [f"{conversation_id}_S02"], "answer": "B"}],
        }
    ]
    timing_questions = [
        {
            "question_id": f"{conversation_id}_TIMING_Q1",
            "question_type": "semantic_evidence_localization",
            "prompt": "When is evidence heard?",
            "allowed_answer_format": "START <i> END <j> | UNKNOWN",
            "answer_provided": False,
        }
    ]
    return {
        "category": "A",
        "category_label": "Correction and confirmation",
        "conversation_id": conversation_id,
        "turn_count": "2",
        "speaker_sequence": "A -> B",
        "transcript": "A: The pickup color is blue.\nB: Correction, the pickup color is green.",
        "transcript_with_pause_recommendations": (
            "A: The pickup color is blue. (short)\n"
            "B: Correction, the pickup color is green. (long)"
        ),
        "sentence_metadata": json.dumps(sentence_metadata, separators=(",", ":")),
        "generic_questions": json.dumps(generic_questions, separators=(",", ":")),
        "generic_answer_rules": json.dumps(generic_answer_rules, separators=(",", ":")),
        "category_specific_questions": json.dumps(category_questions, separators=(",", ":")),
        "category_specific_answer_rules": json.dumps(category_rules, separators=(",", ":")),
        "conversation_content_specific_questions": json.dumps(content_questions, separators=(",", ":")),
        "conversation_content_specific_answer_rules": json.dumps(content_rules, separators=(",", ":")),
        "timing_questions": json.dumps(timing_questions, separators=(",", ":")),
        "timing_answers": "",
        "start_bin": "0",
        "end_bin": "1",
        "sentence_bin_map": f"{conversation_id}_S01=0; {conversation_id}_S02=1",
        "turn_bin_map": "T1=0; T2=1",
        "target_evidence_sentences": f"{conversation_id}_S01,{conversation_id}_S02",
        "critical_span": "bins 0-1",
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fake_synthesize(text: str, output_path: Path, **_kwargs) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), np.full(200, 0.2, dtype=np.float32), 1000, subtype="PCM_16")
    return output_path


class Track1cParsingTests(unittest.TestCase):
    def test_csv_row_parsing_extracts_sentences_and_pause_labels(self) -> None:
        conversation = track1c_build.parse_conversation_row(make_row("T01"))

        self.assertEqual(conversation.conversation_id, "T01")
        self.assertEqual([sentence.sentence_id for sentence in conversation.sentences], ["T01_S01", "T01_S02"])
        self.assertEqual([sentence.speaker for sentence in conversation.sentences], ["A", "B"])
        self.assertEqual([sentence.pause_label for sentence in conversation.sentences], ["short", "long"])
        self.assertEqual(conversation.parsed_json["category_specific_answer_rules"][0]["final_answer_full_evidence"], "GREEN")

    def test_missing_or_malformed_pause_label_raises(self) -> None:
        row = make_row("T02")
        row["transcript_with_pause_recommendations"] = (
            "A: The pickup color is blue.\n"
            "B: Correction, the pickup color is green. (long)"
        )

        with self.assertRaisesRegex(RuntimeError, "Malformed pause recommendation"):
            track1c_build.parse_conversation_row(row)

    def test_variant_mapping_assigns_expected_voices(self) -> None:
        self.assertEqual(track1c_build.voices_for_variant("version_1", TEST_VOICE_PAIRS), {"A": "female_1", "B": "male_1"})
        self.assertEqual(track1c_build.voices_for_variant("version_2", TEST_VOICE_PAIRS), {"A": "male_1", "B": "female_1"})
        self.assertEqual(track1c_build.voices_for_variant("version_3", TEST_VOICE_PAIRS), {"A": "female_2", "B": "male_2"})
        self.assertEqual(track1c_build.voices_for_variant("version_4", TEST_VOICE_PAIRS), {"A": "male_2", "B": "female_2"})

    def test_placeholder_voices_raise_before_api_call(self) -> None:
        conversation = track1c_build.parse_conversation_row(make_row("T03"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "Placeholder voice"):
                track1c_build.build_variant(
                    conversation,
                    Path(tmpdir),
                    "version_1",
                    voice_pairs={"pair_1": {"female": "TODO_FEMALE", "male": "TODO_MALE"}, "pair_2": {"female": "female_2", "male": "male_2"}},
                    synthesize_fn=fake_synthesize,
                )


class Track1cGenerationTests(unittest.TestCase):
    def test_concatenation_inserts_between_sentence_pause_and_timestamps(self) -> None:
        conversation = track1c_build.parse_conversation_row(make_row("T04"))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            metadata = track1c_build.build_variant(
                conversation,
                output_root,
                "version_1",
                voice_pairs=TEST_VOICE_PAIRS,
                synthesize_fn=fake_synthesize,
            )
            audio, samplerate = sf.read(str(output_root / metadata["audio_file_path"]), dtype="float32")

        self.assertEqual(samplerate, 1000)
        self.assertEqual(audio.shape[0], 550)
        self.assertEqual(metadata["sentences"][0]["sentence_start_time"], 0.0)
        self.assertEqual(metadata["sentences"][0]["sentence_end_time"], 0.2)
        self.assertEqual(metadata["pauses"][0]["pause_start_time"], 0.2)
        self.assertEqual(metadata["pauses"][0]["pause_end_time"], 0.35)
        self.assertEqual(metadata["sentences"][1]["sentence_start_time"], 0.35)
        self.assertEqual(metadata["sentences"][1]["sentence_end_time"], 0.55)

    def test_sentence_clips_are_loudness_normalized(self) -> None:
        conversation = track1c_build.parse_conversation_row(make_row("T08"))

        def varied_synthesize(text: str, output_path: Path, **_kwargs) -> Path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            amplitude = 0.01 if "blue" in text else 0.2
            sf.write(str(output_path), np.full(200, amplitude, dtype=np.float32), 1000, subtype="PCM_16")
            return output_path

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            metadata = track1c_build.build_variant(
                conversation,
                output_root,
                "version_1",
                voice_pairs=TEST_VOICE_PAIRS,
                synthesize_fn=varied_synthesize,
            )
            rms_values = []
            for sentence in metadata["sentences"]:
                audio, _samplerate = sf.read(str(output_root / sentence["audio_file_path"]), dtype="float32")
                rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float32) ** 2)))
                rms_values.append(rms)
                self.assertIn("applied_gain_db", sentence["loudness_normalization"])

        self.assertAlmostEqual(rms_values[0], rms_values[1], places=3)
        target_rms = 10 ** (track1c_build.LOUDNESS_NORMALIZATION_TARGET_DBFS / 20.0)
        self.assertAlmostEqual(rms_values[0], target_rms, places=3)

    def test_existing_complete_variant_is_skipped_without_reprocess(self) -> None:
        conversation = track1c_build.parse_conversation_row(make_row("T05"))
        calls = 0

        def counted_synthesize(text: str, output_path: Path, **kwargs) -> Path:
            nonlocal calls
            calls += 1
            return fake_synthesize(text, output_path, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            first = track1c_build.build_variant(
                conversation,
                output_root,
                "version_1",
                voice_pairs=TEST_VOICE_PAIRS,
                synthesize_fn=counted_synthesize,
            )
            calls = 0
            second = track1c_build.build_variant(
                conversation,
                output_root,
                "version_1",
                voice_pairs=TEST_VOICE_PAIRS,
                synthesize_fn=counted_synthesize,
            )
            self.assertEqual(calls, 0)
            self.assertEqual(first["audio_file_path"], second["audio_file_path"])

            track1c_build.build_variant(
                conversation,
                output_root,
                "version_1",
                voice_pairs=TEST_VOICE_PAIRS,
                reprocess=True,
                synthesize_fn=counted_synthesize,
            )

        self.assertEqual(calls, 2)

    def test_manifest_preserves_source_metadata_and_pending_bin_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "input.csv"
            write_csv(csv_path, [make_row("T06")])
            manifest = track1c_build.build_dataset(
                csv_path=csv_path,
                output_root=root / "dataset",
                conversation_id="T06",
                voice_pairs=TEST_VOICE_PAIRS,
                synthesize_fn=fake_synthesize,
            )

            manifest_path = root / "dataset" / "manifest.json"
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(manifest["episodes"]), 4)
        self.assertEqual(len(saved_manifest["episodes"]), 4)
        episode = saved_manifest["episodes"][0]
        self.assertTrue(episode["bin_metadata_pending"])
        self.assertIsNone(episode["sentence_bin_map"])
        self.assertIsNone(episode["turn_bin_map"])
        self.assertEqual(episode["source_row"]["conversation_id"], "T06")
        self.assertEqual(episode["parsed_metadata"]["sentence_metadata"][0]["sentence_id"], "T06_S01")

    def test_cli_smoke_generates_four_variants_with_mocked_tts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            csv_path = root / "input.csv"
            output_root = root / "dataset"
            write_csv(csv_path, [make_row("T07")])
            original_pairs = track1c_build.VOICE_PAIRS
            original_synthesize = track1c_build.synthesize_speech
            try:
                track1c_build.VOICE_PAIRS = TEST_VOICE_PAIRS
                track1c_build.synthesize_speech = fake_synthesize
                exit_code = track1c_build.main(
                    [
                        "--csv-path",
                        str(csv_path),
                        "--output-root",
                        str(output_root),
                        "--conversation-id",
                        "T07",
                        "--reprocess",
                        "false",
                    ]
                )
            finally:
                track1c_build.VOICE_PAIRS = original_pairs
                track1c_build.synthesize_speech = original_synthesize

            manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(manifest["episodes"]), 4)
            for variant_id in track1c_build.VARIANT_DEFINITIONS:
                self.assertTrue((output_root / "episodes" / "T07" / f"{variant_id}.wav").is_file())
                self.assertTrue((output_root / "metadata" / "T07" / f"{variant_id}.json").is_file())


if __name__ == "__main__":
    unittest.main()
