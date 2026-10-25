#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV_PATH = REPO_ROOT / "track1c" / "task_1c_conversation_transcripts_v10_5.csv"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "generated_data" / "theory_of_time_v1" / "track_1c" / "dataset"
ENV_FILE = REPO_ROOT / ".env" / "api_keys.env"

MODEL_NAME = "gpt-4o-mini-tts"
OUTPUT_FORMAT = "wav"
DEFAULT_SPEED = 1.0
TTS_INSTRUCTION = (
    "Speak clearly in a neutral conversational tone, at a normal speed. "
    "Do not add extra words."
)

VOICE_PAIRS = {
    "pair_1": {"female": "marin", "male": "ash"},
    "pair_2": {"female": "sage", "male": "echo"},
}

VARIANT_DEFINITIONS = {
    "version_1": {"voice_pair_id": "pair_1", "speaker_A_gender": "female", "speaker_B_gender": "male"},
    "version_2": {"voice_pair_id": "pair_1", "speaker_A_gender": "male", "speaker_B_gender": "female"},
    "version_3": {"voice_pair_id": "pair_2", "speaker_A_gender": "female", "speaker_B_gender": "male"},
    "version_4": {"voice_pair_id": "pair_2", "speaker_A_gender": "male", "speaker_B_gender": "female"},
}

PAUSE_DURATIONS_SECONDS = {"short": 0.15, "natural": 1.0, "long": 10.0}
JSON_COLUMNS = [
    "sentence_metadata",
    "generic_questions",
    "generic_answer_rules",
    "category_specific_questions",
    "category_specific_answer_rules",
    "conversation_content_specific_questions",
    "conversation_content_specific_answer_rules",
    "timing_questions",
]

SILENCE_THRESHOLD = 1e-4
MAX_GENERATION_ATTEMPTS = 3
TRIM_FRAME_MS = 10
TRIM_THRESHOLD_DBFS = -50.0
TRIM_MIN_SPEECH_RUN_MS = 30
TRIM_PRESERVE_PADDING_MS = 100
LOUDNESS_NORMALIZATION_TARGET_DBFS = -24.0
LOUDNESS_NORMALIZATION_PEAK_CEILING = 0.95
LOUDNESS_NORMALIZATION_SILENCE_THRESHOLD = 1e-4
PAUSE_RE = re.compile(r"^\s*([AB])\s*:\s*(.*?)\s*\((short|natural|long)\)\s*$")

@dataclass(frozen=True)
class SentenceSpec:
    sentence_id: str
    speaker: str
    text: str
    abstract_start_bin: int
    abstract_end_bin: int
    observation_rule: str
    pause_label: str


@dataclass(frozen=True)
class ConversationSpec:
    conversation_id: str
    source_row: dict[str, str]
    parsed_json: dict[str, Any]
    sentences: list[SentenceSpec]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got: {value}")


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


def validate_output_path(output: str | Path) -> Path:
    output_path = Path(output)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output path must end with .wav")
    return output_path


def validate_speed(speed: float) -> float:
    if not 0.25 <= speed <= 4.0:
        raise ValueError("Speed must be between 0.25 and 4.0")
    return speed


def validate_generated_wav(output_path: Path) -> Path:
    path = validate_output_path(output_path)
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
    path = validate_output_path(output_path)
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
    sf.write(str(path), audio[trim_start:trim_end], samplerate)
    return path


def _get_openai_client_class():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The `openai` package is required. Install it with: pip install -r requirements.txt"
        ) from exc
    return OpenAI


def synthesize_speech(
    text: str,
    output_path: Path,
    *,
    voice: str,
    speed: float = DEFAULT_SPEED,
    instructions: str = TTS_INSTRUCTION,
    env_path: Path | None = None,
    client=None,
) -> Path:
    normalized_text = validate_text(text)
    validated_output_path = validate_output_path(output_path)
    validated_speed = validate_speed(speed)
    validated_output_path.parent.mkdir(parents=True, exist_ok=True)

    if client is None:
        api_key = load_openai_api_key(env_path)
        client = _get_openai_client_class()(api_key=api_key)

    request_kwargs = {
        "model": MODEL_NAME,
        "voice": voice,
        "input": normalized_text,
        "response_format": OUTPUT_FORMAT,
        "speed": validated_speed,
        "instructions": instructions,
    }
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
                raise RuntimeError(f"{exc} after {MAX_GENERATION_ATTEMPTS - 1} retries.") from exc
    raise RuntimeError(f"Failed to generate valid WAV output: {validated_output_path}") from last_error


def _normalize_text_for_compare(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip())


def _load_json_cell(row: dict[str, str], column: str) -> Any:
    raw = row.get(column, "")
    try:
        return json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        conversation_id = row.get("conversation_id", "<unknown>")
        raise RuntimeError(f"Malformed JSON in {column} for {conversation_id}: {exc}") from exc


def parse_pause_recommendations(row: dict[str, str], sentences: Sequence[dict[str, Any]]) -> list[str]:
    lines = [line.strip() for line in row["transcript_with_pause_recommendations"].splitlines() if line.strip()]
    conversation_id = row["conversation_id"]
    _assert(
        len(lines) == len(sentences),
        f"Pause recommendation count mismatch for {conversation_id}: {len(lines)} != {len(sentences)}",
    )
    pause_labels: list[str] = []
    for index, (line, sentence) in enumerate(zip(lines, sentences), start=1):
        match = PAUSE_RE.fullmatch(line)
        if not match:
            if index == len(sentences):
                final_match = re.fullmatch(r"\s*([AB])\s*:\s*(.*?)\s*", line)
                if not final_match:
                    raise RuntimeError(f"Malformed pause recommendation for {conversation_id} sentence {index}: {line!r}")
                speaker, text = final_match.groups()
                pause_label = "none"
            else:
                raise RuntimeError(f"Malformed pause recommendation for {conversation_id} sentence {index}: {line!r}")
        else:
            speaker, text, pause_label = match.groups()
        if speaker != sentence["speaker"]:
            raise RuntimeError(
                f"Pause speaker mismatch for {conversation_id} {sentence['sentence_id']}: "
                f"{speaker} != {sentence['speaker']}"
            )
        if _normalize_text_for_compare(text) != _normalize_text_for_compare(sentence["text"]):
            raise RuntimeError(f"Pause text mismatch for {conversation_id} {sentence['sentence_id']}")
        pause_labels.append(pause_label)
    return pause_labels


def parse_conversation_row(row: dict[str, str]) -> ConversationSpec:
    conversation_id = row.get("conversation_id", "").strip()
    _assert(bool(conversation_id), "Missing conversation_id in CSV row.")
    parsed_json = {column: _load_json_cell(row, column) for column in JSON_COLUMNS}
    sentence_metadata = parsed_json["sentence_metadata"]
    _assert(isinstance(sentence_metadata, list) and sentence_metadata, f"No sentence metadata for {conversation_id}")
    pause_labels = parse_pause_recommendations(row, sentence_metadata)
    sentences = []
    for sentence, pause_label in zip(sentence_metadata, pause_labels):
        speaker = str(sentence["speaker"])
        if speaker not in {"A", "B"}:
            raise RuntimeError(f"Unknown speaker {speaker!r} in {conversation_id} {sentence['sentence_id']}")
        if sentence.get("observation_rule") != "full_sentence_required":
            raise RuntimeError(
                f"Unsupported observation_rule for {conversation_id} {sentence['sentence_id']}: "
                f"{sentence.get('observation_rule')}"
            )
        sentences.append(
            SentenceSpec(
                sentence_id=str(sentence["sentence_id"]),
                speaker=speaker,
                text=str(sentence["text"]),
                abstract_start_bin=int(sentence["abstract_start_bin"]),
                abstract_end_bin=int(sentence["abstract_end_bin"]),
                observation_rule=str(sentence["observation_rule"]),
                pause_label=pause_label,
            )
        )
    return ConversationSpec(
        conversation_id=conversation_id,
        source_row=dict(row),
        parsed_json=parsed_json,
        sentences=sentences,
    )


def load_conversations(csv_path: Path) -> list[ConversationSpec]:
    resolved_path = Path(csv_path)
    if not resolved_path.is_file():
        raise RuntimeError(f"Missing CSV: {resolved_path}")
    with resolved_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    conversations = [parse_conversation_row(row) for row in rows]
    ids = [item.conversation_id for item in conversations]
    duplicate_ids = sorted({conversation_id for conversation_id in ids if ids.count(conversation_id) > 1})
    _assert(not duplicate_ids, f"Duplicate conversation_id values: {duplicate_ids}")
    return conversations


def validate_voice_pairs(voice_pairs: dict[str, dict[str, str]]) -> None:
    for variant_id, variant in VARIANT_DEFINITIONS.items():
        pair_id = variant["voice_pair_id"]
        if pair_id not in voice_pairs:
            raise RuntimeError(f"{variant_id} references missing voice pair: {pair_id}")
        for gender in {variant["speaker_A_gender"], variant["speaker_B_gender"]}:
            voice = str(voice_pairs[pair_id].get(gender, "")).strip()
            if not voice:
                raise RuntimeError(f"Missing {gender} voice for {pair_id}")
            if voice.startswith("TODO_"):
                raise RuntimeError(f"Placeholder voice must be replaced before generation: {pair_id}.{gender}={voice}")


def voices_for_variant(
    variant_id: str,
    voice_pairs: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    pairs = VOICE_PAIRS if voice_pairs is None else voice_pairs
    if variant_id not in VARIANT_DEFINITIONS:
        raise RuntimeError(f"Unknown variant_id: {variant_id}")
    variant = VARIANT_DEFINITIONS[variant_id]
    pair_id = variant["voice_pair_id"]
    if pair_id not in pairs:
        raise RuntimeError(f"Missing voice pair: {pair_id}")
    return {
        "A": pairs[pair_id][variant["speaker_A_gender"]],
        "B": pairs[pair_id][variant["speaker_B_gender"]],
    }


def request_metadata_hash(sentence: SentenceSpec, voice: str) -> str:
    payload = {
        "sentence_id": sentence.sentence_id,
        "speaker": sentence.speaker,
        "text": sentence.text,
        "voice": voice,
        "tts_model": MODEL_NAME,
        "speed": DEFAULT_SPEED,
        "instruction": TTS_INSTRUCTION,
        "response_format": OUTPUT_FORMAT,
        "trim_frame_ms": TRIM_FRAME_MS,
        "trim_threshold_dbfs": TRIM_THRESHOLD_DBFS,
        "trim_min_speech_run_ms": TRIM_MIN_SPEECH_RUN_MS,
        "trim_preserve_padding_ms": TRIM_PRESERVE_PADDING_MS,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_to_output(path: Path, output_root: Path) -> str:
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    try:
        return str(resolved_path.relative_to(resolved_root))
    except ValueError:
        return str(resolved_path)


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, samplerate = sf.read(str(path), dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        raise RuntimeError(f"Generated WAV is empty: {path}")
    return np.asarray(audio, dtype=np.float32), int(samplerate)


def _dbfs_from_rms(rms: float) -> float:
    if rms <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(rms))


def normalize_audio_loudness(audio: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    samples = np.asarray(audio, dtype=np.float32)
    mono_abs = np.max(np.abs(samples), axis=1) if samples.ndim == 2 else np.abs(samples)
    speech_mask = mono_abs > LOUDNESS_NORMALIZATION_SILENCE_THRESHOLD
    if not np.any(speech_mask):
        raise RuntimeError("Cannot normalize audio with no speech samples above loudness threshold.")

    speech_samples = samples[speech_mask]
    original_rms = float(np.sqrt(np.mean(speech_samples * speech_samples)))
    if original_rms <= 0.0:
        raise RuntimeError("Cannot normalize audio with zero speech RMS.")
    original_peak = float(np.max(np.abs(samples)))
    target_rms = float(10 ** (LOUDNESS_NORMALIZATION_TARGET_DBFS / 20.0))
    desired_gain = target_rms / original_rms
    peak_limited_gain = (LOUDNESS_NORMALIZATION_PEAK_CEILING / original_peak) if original_peak > 0.0 else desired_gain
    applied_gain = min(desired_gain, peak_limited_gain)
    normalized = np.asarray(samples * applied_gain, dtype=np.float32)
    normalized_speech = normalized[speech_mask]
    normalized_rms = float(np.sqrt(np.mean(normalized_speech * normalized_speech)))
    normalized_peak = float(np.max(np.abs(normalized)))
    metadata = {
        "loudness_normalization_target_dbfs": LOUDNESS_NORMALIZATION_TARGET_DBFS,
        "loudness_normalization_peak_ceiling": LOUDNESS_NORMALIZATION_PEAK_CEILING,
        "original_speech_rms_dbfs": round(_dbfs_from_rms(original_rms), 6),
        "normalized_speech_rms_dbfs": round(_dbfs_from_rms(normalized_rms), 6),
        "applied_gain_db": round(float(20.0 * np.log10(applied_gain)), 6),
        "original_peak": round(original_peak, 6),
        "normalized_peak": round(normalized_peak, 6),
    }
    return normalized, metadata


def _write_audio(path: Path, audio: np.ndarray, samplerate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, samplerate, subtype="PCM_16")


def _variant_output_paths(output_root: Path, conversation_id: str, variant_id: str) -> tuple[Path, Path, Path]:
    sentence_dir = output_root / "sentence_clips" / conversation_id / variant_id
    episode_path = output_root / "episodes" / conversation_id / f"{variant_id}.wav"
    metadata_path = output_root / "metadata" / conversation_id / f"{variant_id}.json"
    return sentence_dir, episode_path, metadata_path


def _variant_complete(
    conversation: ConversationSpec,
    output_root: Path,
    variant_id: str,
) -> bool:
    sentence_dir, episode_path, metadata_path = _variant_output_paths(output_root, conversation.conversation_id, variant_id)
    return (
        episode_path.is_file()
        and metadata_path.is_file()
        and all((sentence_dir / f"{sentence.sentence_id}.wav").is_file() for sentence in conversation.sentences)
    )


def _variant_has_any_output(
    conversation: ConversationSpec,
    output_root: Path,
    variant_id: str,
) -> bool:
    sentence_dir, episode_path, metadata_path = _variant_output_paths(output_root, conversation.conversation_id, variant_id)
    return sentence_dir.exists() or episode_path.exists() or metadata_path.exists()


def build_variant(
    conversation: ConversationSpec,
    output_root: Path,
    variant_id: str,
    *,
    voice_pairs: dict[str, dict[str, str]] | None = None,
    reprocess: bool = False,
    env_path: Path | None = None,
    client=None,
    synthesize_fn: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    pairs = VOICE_PAIRS if voice_pairs is None else voice_pairs
    resolved_synthesize_fn = synthesize_speech if synthesize_fn is None else synthesize_fn
    validate_voice_pairs(pairs)
    sentence_dir, episode_path, metadata_path = _variant_output_paths(output_root, conversation.conversation_id, variant_id)
    if not reprocess and _variant_complete(conversation, output_root, variant_id):
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    if not reprocess and _variant_has_any_output(conversation, output_root, variant_id):
        raise RuntimeError(
            f"Partial output already exists for {conversation.conversation_id}/{variant_id}; "
            "rerun with --reprocess true."
        )

    voices = voices_for_variant(variant_id, pairs)
    variant = VARIANT_DEFINITIONS[variant_id]
    generation_timestamp = datetime.now(timezone.utc).isoformat()
    sentence_records: list[dict[str, Any]] = []
    pause_records: list[dict[str, Any]] = []
    audio_segments: list[np.ndarray] = []
    samplerate: int | None = None
    cursor_sec = 0.0

    for index, sentence in enumerate(conversation.sentences):
        sentence_path = sentence_dir / f"{sentence.sentence_id}.wav"
        voice = voices[sentence.speaker]
        resolved_synthesize_fn(
            sentence.text,
            sentence_path,
            voice=voice,
            speed=DEFAULT_SPEED,
            instructions=TTS_INSTRUCTION,
            env_path=env_path,
            client=client,
        )
        validate_generated_wav(sentence_path)
        trim_generated_wav_edges(sentence_path)
        audio, sentence_samplerate = _read_audio(sentence_path)
        if samplerate is None:
            samplerate = sentence_samplerate
        elif sentence_samplerate != samplerate:
            raise RuntimeError(
                f"Sample-rate mismatch for {sentence_path}: {sentence_samplerate} != {samplerate}"
            )
        audio, loudness_metadata = normalize_audio_loudness(audio)
        _write_audio(sentence_path, audio, sentence_samplerate)
        audio, sentence_samplerate = _read_audio(sentence_path)
        duration_sec = audio.shape[0] / float(samplerate)
        sentence_start = cursor_sec
        sentence_end = sentence_start + duration_sec
        audio_segments.append(audio)
        sentence_records.append(
            {
                **asdict(sentence),
                "voice": voice,
                "audio_file_path": _relative_to_output(sentence_path, output_root),
                "sentence_start_time": round(sentence_start, 6),
                "sentence_end_time": round(sentence_end, 6),
                "duration_seconds": round(duration_sec, 6),
                "request_metadata_hash": request_metadata_hash(sentence, voice),
                "loudness_normalization": loudness_metadata,
            }
        )
        cursor_sec = sentence_end
        if index + 1 < len(conversation.sentences):
            pause_duration = PAUSE_DURATIONS_SECONDS[sentence.pause_label]
            pause_start = cursor_sec
            pause_end = pause_start + pause_duration
            pause_frames = int(round(pause_duration * samplerate))
            audio_segments.append(np.zeros((pause_frames, audio.shape[1]), dtype=np.float32))
            pause_records.append(
                {
                    "after_sentence_id": sentence.sentence_id,
                    "pause_label": sentence.pause_label,
                    "pause_start_time": round(pause_start, 6),
                    "pause_end_time": round(pause_end, 6),
                    "duration_seconds": round(pause_duration, 6),
                }
            )
            cursor_sec = pause_end

    assert samplerate is not None
    full_audio = np.concatenate(audio_segments, axis=0)
    _write_audio(episode_path, full_audio, samplerate)
    validate_generated_wav(episode_path)
    metadata = {
        "track_id": "1c",
        "conversation_id": conversation.conversation_id,
        "variant_id": variant_id,
        "voice_pair_id": variant["voice_pair_id"],
        "speaker_A_voice": voices["A"],
        "speaker_B_voice": voices["B"],
        "speaker_A_gender": variant["speaker_A_gender"],
        "speaker_B_gender": variant["speaker_B_gender"],
        "tts_model": MODEL_NAME,
        "tts_instruction": TTS_INSTRUCTION,
        "tts_speed": DEFAULT_SPEED,
        "output_format": OUTPUT_FORMAT,
        "pause_durations_seconds": PAUSE_DURATIONS_SECONDS,
        "audio_file_path": _relative_to_output(episode_path, output_root),
        "audio_duration_seconds": round(full_audio.shape[0] / float(samplerate), 6),
        "samplerate": samplerate,
        "channels": int(full_audio.shape[1]),
        "generation_timestamp": generation_timestamp,
        "sentences": sentence_records,
        "pauses": pause_records,
        "source_row": conversation.source_row,
        "parsed_metadata": conversation.parsed_json,
        "bin_metadata_pending": True,
        "sentence_bin_map": None,
        "turn_bin_map": None,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def build_dataset(
    *,
    csv_path: Path = DEFAULT_CSV_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    conversation_id: str = "A01",
    all_conversations: bool = False,
    reprocess: bool = False,
    api_key_env_path: Path | None = ENV_FILE,
    voice_pairs: dict[str, dict[str, str]] | None = None,
    client=None,
    synthesize_fn: Callable[..., Path] | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root)
    pairs = VOICE_PAIRS if voice_pairs is None else voice_pairs
    conversations = load_conversations(Path(csv_path))
    if all_conversations:
        selected = conversations
    else:
        selected = [item for item in conversations if item.conversation_id == conversation_id]
        _assert(bool(selected), f"Unknown conversation_id: {conversation_id}")

    episode_records = []
    for conversation in selected:
        for variant_id in VARIANT_DEFINITIONS:
            episode_records.append(
                build_variant(
                    conversation,
                    output_root,
                    variant_id,
                    voice_pairs=pairs,
                    reprocess=reprocess,
                    env_path=api_key_env_path,
                    client=client,
                    synthesize_fn=synthesize_fn,
                )
            )

    manifest = {
        "track_id": "1c",
        "source_csv_path": str(Path(csv_path)),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "tts_settings": {
            "model": MODEL_NAME,
            "instruction": TTS_INSTRUCTION,
            "speed": DEFAULT_SPEED,
            "output_format": OUTPUT_FORMAT,
            "trim_frame_ms": TRIM_FRAME_MS,
            "trim_threshold_dbfs": TRIM_THRESHOLD_DBFS,
            "trim_min_speech_run_ms": TRIM_MIN_SPEECH_RUN_MS,
            "trim_preserve_padding_ms": TRIM_PRESERVE_PADDING_MS,
            "loudness_normalization_target_dbfs": LOUDNESS_NORMALIZATION_TARGET_DBFS,
            "loudness_normalization_peak_ceiling": LOUDNESS_NORMALIZATION_PEAK_CEILING,
            "loudness_normalization_silence_threshold": LOUDNESS_NORMALIZATION_SILENCE_THRESHOLD,
        },
        "pause_durations_seconds": PAUSE_DURATIONS_SECONDS,
        "voice_pairs": pairs,
        "variant_definitions": VARIANT_DEFINITIONS,
        "bin_metadata_pending": True,
        "episodes": episode_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the Track 1c synthetic conversation dataset.")
    parser.add_argument("--csv-path", default=str(DEFAULT_CSV_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--conversation-id", default="A01")
    parser.add_argument("--all-conversations", type=parse_bool_arg, default=False)
    parser.add_argument("--reprocess", type=parse_bool_arg, default=False)
    parser.add_argument("--api-key-env-path", default=str(ENV_FILE))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        build_dataset(
            csv_path=Path(args.csv_path),
            output_root=Path(args.output_root),
            conversation_id=str(args.conversation_id),
            all_conversations=bool(args.all_conversations),
            reprocess=bool(args.reprocess),
            api_key_env_path=Path(args.api_key_env_path),
        )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
