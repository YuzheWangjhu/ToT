from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

import openai_tts


class ParseEnvFileTests(unittest.TestCase):
    def test_parse_env_file_reads_openai_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "api_keys.env"
            env_path.write_text(
                "# Comment\nOPENAI_API_KEY=test-key\nOTHER_KEY=other\n",
                encoding="utf-8",
            )

            values = openai_tts.parse_env_file(env_path)

        self.assertEqual(values["OPENAI_API_KEY"], "test-key")

    def test_load_openai_api_key_requires_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / "api_keys.env"
            env_path.write_text("OTHER_KEY=value\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY is missing or empty"):
                openai_tts.load_openai_api_key(env_path)


class ValidationTests(unittest.TestCase):
    def test_validate_generated_wav_rejects_silent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "silent.wav"
            sf.write(str(output_path), [0.0, 0.0, 0.0], 24000, subtype="PCM_16")

            with self.assertRaisesRegex(RuntimeError, "silent"):
                openai_tts.validate_generated_wav(output_path)

    def test_main_check_output_rejects_silent_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "silent.wav"
            sf.write(str(output_path), [0.0, 0.0, 0.0], 24000, subtype="PCM_16")

            with self.assertRaises(SystemExit) as exc:
                openai_tts.main(["--check-output", str(output_path)])

        self.assertEqual(exc.exception.code, 1)

    def test_main_rejects_empty_text(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            openai_tts.main(["--text", "   ", "--output", "out.wav"])

        self.assertEqual(exc.exception.code, 1)

    def test_main_rejects_non_wav_output(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            openai_tts.main(["--text", "hello", "--output", "out.mp3"])

        self.assertEqual(exc.exception.code, 1)

    def test_main_rejects_speed_out_of_range(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            openai_tts.main(["--text", "hello", "--output", "out.wav", "--speed", "10"])

        self.assertEqual(exc.exception.code, 1)

    def test_trim_generated_wav_edges_preserves_padding_around_speech(self) -> None:
        samplerate = 1000
        leading = np.zeros(300, dtype=np.float32)
        speech = np.full(200, 0.2, dtype=np.float32)
        trailing = np.zeros(300, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "speech.wav"
            sf.write(str(output_path), np.concatenate([leading, speech, trailing]), samplerate)

            openai_tts.validate_generated_wav(output_path)
            result = openai_tts.trim_generated_wav_edges(output_path)
            trimmed, _samplerate = sf.read(str(result), dtype="float32")

        self.assertEqual(result, output_path)
        self.assertEqual(len(trimmed), 400)
        self.assertAlmostEqual(float(np.max(np.abs(trimmed[:100]))), 0.0)
        self.assertGreater(float(np.max(np.abs(trimmed[100:300]))), 0.1)
        self.assertAlmostEqual(float(np.max(np.abs(trimmed[300:]))), 0.0)


class SynthesizeSpeechTests(unittest.TestCase):
    def test_synthesize_speech_calls_openai_with_wav_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "speech.wav"

            streamed_response = mock.Mock()
            context_manager = mock.Mock()
            context_manager.__enter__ = mock.Mock(return_value=streamed_response)
            context_manager.__exit__ = mock.Mock(return_value=False)

            client = mock.Mock()
            client.audio.speech.with_streaming_response.create.return_value = context_manager

            def _write_valid_wav(target_path: Path) -> None:
                sf.write(
                    str(target_path),
                    np.concatenate(
                        [
                            np.zeros(4800, dtype=np.float32),
                            np.full(2400, 0.2, dtype=np.float32),
                            np.zeros(4800, dtype=np.float32),
                        ]
                    ),
                    24000,
                    subtype="PCM_16",
                )

            streamed_response.stream_to_file.side_effect = _write_valid_wav

            result = openai_tts.synthesize_speech(
                "Hello world",
                output_path,
                voice="ash",
                speed=1.5,
                instructions="Speak slowly and clearly.",
                client=client,
            )
            trimmed, samplerate = sf.read(str(output_path), dtype="float32")

        self.assertEqual(result, output_path)
        self.assertEqual(samplerate, 24000)
        self.assertEqual(len(trimmed), 7200)
        client.audio.speech.with_streaming_response.create.assert_called_once_with(
            model="gpt-4o-mini-tts",
            voice="ash",
            input="Hello world",
            response_format="wav",
            speed=1.5,
            instructions="Speak slowly and clearly.",
        )
        streamed_response.stream_to_file.assert_called_once_with(output_path)

    def test_synthesize_speech_regenerates_once_when_first_output_is_silent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "speech.wav"

            streamed_response = mock.Mock()
            context_manager = mock.Mock()
            context_manager.__enter__ = mock.Mock(return_value=streamed_response)
            context_manager.__exit__ = mock.Mock(return_value=False)

            client = mock.Mock()
            client.audio.speech.with_streaming_response.create.return_value = context_manager

            call_count = 0

            def _write_attempt(target_path: Path) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    sf.write(str(target_path), [0.0, 0.0, 0.0], 24000, subtype="PCM_16")
                else:
                    sf.write(
                        str(target_path),
                        np.concatenate(
                            [
                                np.zeros(4800, dtype=np.float32),
                                np.full(2400, 0.2, dtype=np.float32),
                                np.zeros(4800, dtype=np.float32),
                            ]
                        ),
                        24000,
                        subtype="PCM_16",
                    )

            streamed_response.stream_to_file.side_effect = _write_attempt

            result = openai_tts.synthesize_speech(
                "Hello world",
                output_path,
                voice="ash",
                speed=1.0,
                client=client,
            )

        self.assertEqual(result, output_path)
        self.assertEqual(client.audio.speech.with_streaming_response.create.call_count, 2)
        self.assertEqual(streamed_response.stream_to_file.call_count, 2)

    def test_synthesize_speech_retries_twice_and_removes_bad_output_on_final_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "speech.wav"

            streamed_response = mock.Mock()
            context_manager = mock.Mock()
            context_manager.__enter__ = mock.Mock(return_value=streamed_response)
            context_manager.__exit__ = mock.Mock(return_value=False)

            client = mock.Mock()
            client.audio.speech.with_streaming_response.create.return_value = context_manager

            def _write_silent_wav(target_path: Path) -> None:
                sf.write(str(target_path), [0.0, 0.0, 0.0], 24000, subtype="PCM_16")

            streamed_response.stream_to_file.side_effect = _write_silent_wav

            with self.assertRaisesRegex(RuntimeError, "after 2 retries"):
                openai_tts.synthesize_speech(
                    "Hello world",
                    output_path,
                    voice="ash",
                    speed=1.0,
                    client=client,
                )

        self.assertEqual(client.audio.speech.with_streaming_response.create.call_count, 3)
        self.assertEqual(streamed_response.stream_to_file.call_count, 3)
        self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
