#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


ENV_FILE = Path(__file__).resolve().parent / ".env" / "api_keys.env"
MODEL_NAME = "gpt-4o-mini-tts"
DEFAULT_VOICE = "alloy"
OUTPUT_FORMAT = "wav"
DEFAULT_SPEED = 1.0
SILENCE_THRESHOLD = 1e-4
MAX_GENERATION_ATTEMPTS = 3
TRIM_FRAME_MS = 10
TRIM_THRESHOLD_DBFS = -50.0
TRIM_MIN_SPEECH_RUN_MS = 30
TRIM_PRESERVE_PADDING_MS = 100


def parse_env_file(env_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid env line in {env_path}: {line!r}")
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_openai_api_key(env_path: Path | None = None) -> str:
    resolved_env_path = ENV_FILE if env_path is None else Path(env_path)
    if not resolved_env_path.is_file():
        raise RuntimeError(f"Missing env file: {resolved_env_path}")
    values = parse_env_file(resolved_env_path)
    api_key = values.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(f"OPENAI_API_KEY is missing or empty in {resolved_env_path}")
    return api_key


def validate_text(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        raise ValueError("Text input must not be empty.")
    return normalized


def validate_output_path(output: str) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output path must end with .wav")
    return output_path


def validate_speed(speed: float) -> float:
    if not 0.25 <= speed <= 4.0:
        raise ValueError("Speed must be between 0.25 and 4.0")
    return speed


def validate_generated_wav(output_path: Path) -> Path:
    path = validate_output_path(str(output_path))
    if not path.is_file():
        raise RuntimeError(f"Generated WAV is missing: {path}")

    try:
        audio, _samplerate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise RuntimeError(f"Generated WAV is corrupted: {path}") from exc

    samples = np.asarray(audio, dtype=np.float32)
    if samples.size == 0:
        raise RuntimeError(f"Generated WAV is empty: {path}")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError(f"Generated WAV contains non-finite samples: {path}")
    if float(np.max(np.abs(samples))) <= SILENCE_THRESHOLD:
        raise RuntimeError(f"Generated WAV is silent: {path}")
    return path


def trim_generated_wav_edges(output_path: Path) -> Path:
    path = validate_output_path(str(output_path))
    audio, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise RuntimeError(f"Generated WAV is empty: {path}")

    frame_samples = max(1, round(samplerate * TRIM_FRAME_MS / 1000))
    min_speech_frames = max(1, round(TRIM_MIN_SPEECH_RUN_MS / TRIM_FRAME_MS))
    padding_samples = max(0, round(samplerate * TRIM_PRESERVE_PADDING_MS / 1000))
    threshold = 10 ** (TRIM_THRESHOLD_DBFS / 20.0)

    mono_abs = np.max(np.abs(audio), axis=1)
    frame_is_speech = []
    for start in range(0, mono_abs.shape[0], frame_samples):
        frame = mono_abs[start : start + frame_samples]
        rms = float(np.sqrt(np.mean(frame * frame)))
        frame_is_speech.append(rms >= threshold)

    first_speech_frame = None
    for index in range(0, len(frame_is_speech) - min_speech_frames + 1):
        if all(frame_is_speech[index : index + min_speech_frames]):
            first_speech_frame = index
            break
    if first_speech_frame is None:
        raise RuntimeError(f"Generated WAV has no sustained speech above trim threshold: {path}")

    last_speech_frame_end = None
    for index in range(len(frame_is_speech) - min_speech_frames, -1, -1):
        if all(frame_is_speech[index : index + min_speech_frames]):
            last_speech_frame_end = index + min_speech_frames
            break
    assert last_speech_frame_end is not None

    trim_start = max(0, first_speech_frame * frame_samples - padding_samples)
    trim_end = min(audio.shape[0], last_speech_frame_end * frame_samples + padding_samples)
    if trim_start == 0 and trim_end == audio.shape[0]:
        return path

    trimmed = audio[trim_start:trim_end]
    sf.write(str(path), trimmed, samplerate)
    return path


def _get_openai_client_class():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `openai` package is required. Install it with `pip install -r requirements.txt`."
        ) from exc
    return OpenAI


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    instructions: str | None = None,
    env_path: Path | None = None,
    client=None,
) -> Path:
    normalized_text = validate_text(text)
    validated_output_path = validate_output_path(str(output_path))
    validated_speed = validate_speed(speed)

    if client is None:
        api_key = load_openai_api_key(env_path)
        client = _get_openai_client_class()(api_key=api_key)

    request_kwargs = {
        "model": MODEL_NAME,
        "voice": voice,
        "input": normalized_text,
        "response_format": OUTPUT_FORMAT,
        "speed": validated_speed,
    }
    if instructions is not None:
        request_kwargs["instructions"] = instructions

    last_error: RuntimeError | None = None
    for attempt_index in range(MAX_GENERATION_ATTEMPTS):
        with client.audio.speech.with_streaming_response.create(**request_kwargs) as response:
            response.stream_to_file(validated_output_path)
        try:
            validate_generated_wav(validated_output_path)
            return trim_generated_wav_edges(validated_output_path)
        except RuntimeError as exc:
            last_error = exc
            if validated_output_path.exists():
                validated_output_path.unlink()
            if attempt_index + 1 >= MAX_GENERATION_ATTEMPTS:
                raise RuntimeError(
                    f"{exc} after {MAX_GENERATION_ATTEMPTS - 1} retries."
                ) from exc

    raise RuntimeError(f"Failed to generate valid WAV output: {validated_output_path}") from last_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate WAV speech from text using OpenAI gpt-4o-mini-tts."
    )
    parser.add_argument("--text", help="Text to synthesize.")
    parser.add_argument("--output", help="Output WAV file path.")
    parser.add_argument(
        "--check-output",
        help="Validate an existing WAV file and exit nonzero if it is missing, corrupted, or silent.",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Voice name to use.")
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Playback speed from 0.25 to 4.0.",
    )
    parser.add_argument(
        "--instructions",
        help="Optional style instructions for the TTS model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check_output is not None:
            validate_generated_wav(validate_output_path(args.check_output))
        else:
            if args.text is None or args.output is None:
                parser.exit(status=1, message="Error: --text and --output are required unless --check-output is used.\n")
            synthesize_speech(
                args.text,
                validate_output_path(args.output),
                voice=args.voice,
                speed=validate_speed(args.speed),
                instructions=args.instructions,
            )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
