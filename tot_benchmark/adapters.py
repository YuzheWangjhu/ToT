from __future__ import annotations

import base64
import json
import io
import math
import os
import re
import sys
import tempfile
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import requests

from openai_tts import load_openai_api_key, parse_env_file
from .text_protocol import normalize_first_line


def _temp_wav_handle(prefix: str):
    temp_dir = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False, dir=temp_dir)


def wav_bytes_with_min_duration(path: Path, min_duration_sec: float = 0.25) -> bytes:
    source_path = Path(path)
    with wave.open(str(source_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        compression_type = source.getcomptype()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)
    if compression_type != "NONE":
        raise ValueError(f"Expected uncompressed PCM WAV: {source_path}")
    min_frames = int(round(float(sample_rate) * float(min_duration_sec)))
    if frame_count >= min_frames:
        return source_path.read_bytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(sample_rate)
        target.writeframes(frames)
        target.writeframes(b"\x00" * sample_width * channels * (min_frames - frame_count))
    return buffer.getvalue()


@dataclass
class AdapterSession:
    system_prompt: str
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Track2AdapterCapabilities:
    supported_conditions: frozenset[str] = frozenset()
    supports_text_only: bool = False
    supports_output_token_override: bool = False
    max_audio_seconds: float | None = None
    max_audio_bytes: int | None = None


class UnsupportedAdapterCapability(RuntimeError):
    pass


def session_output_token_limit(session: AdapterSession, default: int) -> int:
    value = session.state.get("track2_max_output_tokens")
    return int(default if value is None else value)


def session_requires_strict_text_only(session: AdapterSession) -> bool:
    return bool(session.state.get("track2_strict_text_only", False))


class BaseAdapter(ABC):
    model_id: str

    @abstractmethod
    def create_session(self, system_prompt: str) -> AdapterSession:
        raise NotImplementedError

    def close_session(self, session: AdapterSession) -> None:
        return None

    def configure_track2_capabilities(self, config: dict[str, Any]) -> None:
        capability_keys = {
            "track2_supported_conditions",
            "track2_supports_text_only",
            "track2_supports_output_token_override",
            "track2_max_audio_seconds",
            "track2_max_audio_bytes",
        }
        if not isinstance(self, DummyAdapter) and "track2_supported_conditions" in config:
            missing = sorted(capability_keys - set(config))
            if missing:
                raise ValueError(
                    f"Track 2 capability declaration is incomplete; missing keys: {missing}"
                )
        conditions = config.get("track2_supported_conditions")
        if conditions is None and isinstance(self, DummyAdapter):
            conditions = ["full_context", "active", "active_same_session_periodic_probe"]
        if conditions is not None and (not isinstance(conditions, list) or not all(isinstance(item, str) for item in conditions)):
            raise ValueError("track2_supported_conditions must be a list of strings.")
        unknown_conditions = set(conditions or []) - {
            "full_context",
            "active",
            "active_same_session_periodic_probe",
        }
        if unknown_conditions:
            raise ValueError(f"Unknown Track 2 conditions: {sorted(unknown_conditions)}")
        default_text_only = isinstance(self, DummyAdapter)
        default_output_token_override = isinstance(self, DummyAdapter)
        self._track2_capabilities = Track2AdapterCapabilities(
            supported_conditions=frozenset(conditions or []),
            supports_text_only=bool(config.get("track2_supports_text_only", default_text_only)),
            supports_output_token_override=bool(
                config.get(
                    "track2_supports_output_token_override",
                    default_output_token_override,
                )
            ),
            max_audio_seconds=(
                float(config["track2_max_audio_seconds"])
                if config.get("track2_max_audio_seconds") is not None
                else (math.inf if isinstance(self, DummyAdapter) else None)
            ),
            max_audio_bytes=(
                int(config["track2_max_audio_bytes"])
                if config.get("track2_max_audio_bytes") is not None
                else (sys.maxsize if isinstance(self, DummyAdapter) else None)
            ),
        )
        if self._track2_capabilities.max_audio_seconds is not None and self._track2_capabilities.max_audio_seconds <= 0:
            raise ValueError("track2_max_audio_seconds must be positive.")
        if self._track2_capabilities.max_audio_bytes is not None and self._track2_capabilities.max_audio_bytes <= 0:
            raise ValueError("track2_max_audio_bytes must be positive.")

    @property
    def track2_capabilities(self) -> Track2AdapterCapabilities:
        return getattr(self, "_track2_capabilities", Track2AdapterCapabilities())

    def send_track2(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        *,
        max_output_tokens: int,
        strict_text_only: bool,
    ) -> str:
        resolved_audio_paths = [Path(path) for path in audio_paths or []]
        if strict_text_only:
            if resolved_audio_paths:
                raise ValueError("Strict text-only Track 2 calls cannot contain audio paths.")
            if not self.track2_capabilities.supports_text_only:
                raise UnsupportedAdapterCapability("adapter_does_not_support_strict_text_only_calls")
        session.state["track2_max_output_tokens"] = int(max_output_tokens)
        session.state["track2_strict_text_only"] = bool(strict_text_only)
        return self.send(
            session,
            user_text,
            resolved_audio_paths,
            preserve_multiline=True,
        )

    @abstractmethod
    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        raise NotImplementedError


API_ERROR_SOFT = "soft"
API_ERROR_HARD = "hard"
API_ERROR_CLASSES = {API_ERROR_SOFT, API_ERROR_HARD}


def _coerce_status_code(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def classify_api_error(message: str, status_code: int | None = None) -> str:
    normalized = str(message).lower()
    code = _coerce_status_code(status_code)
    if code in {408, 409, 425, 429} or (code is not None and code >= 500):
        return API_ERROR_SOFT
    if code is not None and 400 <= code < 500:
        return API_ERROR_HARD
    soft_markers = (
        "resource_exhausted",
        "too many requests",
        "rate limit",
        "quota exceeded",
        "unavailable",
        "deadline_exceeded",
        "timed out",
        "timeout",
        "connection",
        "tls",
        "ssl",
        "server_error",
        "internal",
        "did not contain any text",
        "did not contain text",
        "not valid json",
    )
    return API_ERROR_SOFT if any(marker in normalized for marker in soft_markers) else API_ERROR_HARD


def _retry_after_seconds_from_text(message: str) -> float | None:
    duration = (
        r"((?:[0-9]+(?:\.[0-9]+)?\s*"
        r"(?:milliseconds?|ms|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\s*)+)"
    )
    match = re.search(rf"retryDelay[^0-9]*{duration}", message, re.I)
    if match is None:
        match = re.search(rf"(?:retry in|retry after)\s+{duration}", message, re.I)
    if match is not None:
        total_seconds = 0.0
        for value_text, unit_text in re.findall(
            r"([0-9]+(?:\.[0-9]+)?)\s*"
            r"(milliseconds?|ms|hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)",
            match.group(1),
            re.I,
        ):
            value = float(value_text)
            unit = unit_text.lower()
            if unit.startswith("h"):
                total_seconds += value * 3600.0
            elif unit.startswith("m") and unit not in {"ms", "millisecond", "milliseconds"}:
                total_seconds += value * 60.0
            elif unit in {"ms", "millisecond", "milliseconds"}:
                total_seconds += value / 1000.0
            else:
                total_seconds += value
        return total_seconds
    seconds_match = re.search(
        r"(?:retry in|retry after)\s+([0-9]+(?:\.[0-9]+)?)",
        message,
        re.I,
    )
    return float(seconds_match.group(1)) if seconds_match is not None else None


class APIAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        error_class: str | None = None,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        resolved_status = _coerce_status_code(status_code)
        resolved_class = error_class or classify_api_error(message, resolved_status)
        if resolved_class not in API_ERROR_CLASSES:
            raise ValueError(f"Unsupported API error class: {resolved_class}")
        self.error_class = resolved_class
        self.status_code = resolved_status
        self.retry_after_seconds = (
            _retry_after_seconds_from_text(message)
            if retry_after_seconds is None
            else max(0.0, float(retry_after_seconds))
        )


class DummyAdapter(BaseAdapter):
    def __init__(self, outputs: Sequence[str], model_id: str = "dummy") -> None:
        self.model_id = model_id
        self._outputs = list(outputs)
        self._cursor = 0
        self.configure_track2_capabilities({})

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        session.state.setdefault("messages", []).append(
            {
                "role": "user",
                "text": user_text,
                "audio_paths": [str(path) for path in audio_paths or []],
            }
        )
        if self._cursor >= len(self._outputs):
            raise RuntimeError("DummyAdapter ran out of scripted outputs.")
        output = self._outputs[self._cursor]
        self._cursor += 1
        session.state["messages"].append({"role": "assistant", "text": output})
        return output


class OpenAIResponsesAudioAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env_path: Path | None = None,
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        request_timeout: float = 3600.0,
    ) -> None:
        self.model_id = model_id
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.request_timeout = float(request_timeout)
        self.api_key_env_path = api_key_env_path
        self._api_key: str | None = None
        self._chat_url = "https://api.openai.com/v1/chat/completions"
        self._silence_b64: str | None = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            self._api_key = load_openai_api_key(self.api_key_env_path)
        return self._api_key

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(
            system_prompt=system_prompt,
            state={"messages": [{"role": "system", "content": system_prompt}]},
        )

    @staticmethod
    def _audio_item(path: Path) -> dict[str, Any]:
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(wav_bytes_with_min_duration(path)).decode("ascii"),
                "format": "wav",
            },
        }

    def _silence_audio_item(self) -> dict[str, Any]:
        if self._silence_b64 is None:
            sample_rate = 16000
            duration_frames = sample_rate // 4
            pcm_bytes = b"\x00\x00" * duration_frames
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm_bytes)
            self._silence_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "input_audio",
            "input_audio": {
                "data": self._silence_b64,
                "format": "wav",
            },
        }

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._get_api_key()}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _extract_text(response_json: Any) -> str:
        if not isinstance(response_json, dict):
            raise APIAdapterError(
                "OpenAI response JSON root was not an object.",
                error_class=API_ERROR_SOFT,
            )
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIAdapterError(
                "OpenAI response did not contain a valid first choice.",
                error_class=API_ERROR_SOFT,
            )
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIAdapterError(
                "OpenAI response choice did not contain a valid message.",
                error_class=API_ERROR_SOFT,
            )
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    value = str(item.get("text", "")).strip()
                    if value:
                        text_parts.append(value)
            if text_parts:
                return "\n".join(text_parts)
        raise APIAdapterError(
            "OpenAI response did not contain any text output.",
            error_class=API_ERROR_SOFT,
        )

    @staticmethod
    def _extract_audio_transcript(response_json: Any) -> str:
        if not isinstance(response_json, dict):
            raise APIAdapterError(
                "OpenAI response JSON root was not an object.",
                error_class=API_ERROR_SOFT,
            )
        choices = response_json.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIAdapterError(
                "OpenAI response did not contain a valid first choice.",
                error_class=API_ERROR_SOFT,
            )
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIAdapterError(
                "OpenAI response choice did not contain a valid message.",
                error_class=API_ERROR_SOFT,
            )
        audio = message.get("audio")
        transcript = audio.get("transcript") if isinstance(audio, dict) else None
        if isinstance(transcript, str) and transcript.strip():
            return transcript.strip()
        raise APIAdapterError(
            "OpenAI audio response did not contain a valid transcript.",
            error_class=API_ERROR_SOFT,
        )

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        resolved_audio_paths = list(audio_paths or [])
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        if not resolved_audio_paths and not session_requires_strict_text_only(session):
            content.insert(
                0,
                {
                    "type": "text",
                    "text": (
                        "No benchmark audio evidence is attached for this turn. "
                        "A silent placeholder audio clip is included only to satisfy the model API. "
                        "Do not treat it as episode evidence."
                    ),
                },
            )
            content.append(self._silence_audio_item())
        for path in resolved_audio_paths:
            content.append(self._audio_item(Path(path)))
        user_message = {"role": "user", "content": content}
        strict_text_only = session_requires_strict_text_only(session)
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [*session.state["messages"], user_message],
            "temperature": self.temperature,
            "n": 1,
            "max_completion_tokens": session_output_token_limit(session, self.max_output_tokens),
            "modalities": ["text", "audio"] if strict_text_only else ["text"],
        }
        if strict_text_only:
            payload["audio"] = {"voice": "alloy", "format": "wav"}
        try:
            response = requests.post(
                self._chat_url,
                headers=self._headers(),
                data=json.dumps(payload),
                timeout=self.request_timeout,
            )
        except requests.RequestException as exc:
            raise APIAdapterError(
                f"OpenAI API request failed: {exc}",
                error_class=API_ERROR_SOFT,
            ) from exc
        if response.status_code != 200:
            try:
                err = response.json().get("error", {})
                err_msg = err.get("message", response.text)
                err_type = err.get("type", "")
            except Exception:
                err_msg = response.text
                err_type = ""
            retry_after = getattr(response, "headers", {}).get("retry-after")
            try:
                retry_after_seconds = float(retry_after) if retry_after else None
            except (TypeError, ValueError):
                retry_after_seconds = None
            raise APIAdapterError(
                f"OpenAI API error {response.status_code}: {err_type} {err_msg}",
                status_code=response.status_code,
                retry_after_seconds=retry_after_seconds,
            )
        try:
            response_json = response.json()
        except ValueError as exc:
            raise APIAdapterError(
                "OpenAI API response was not valid JSON.",
                error_class=API_ERROR_SOFT,
            ) from exc
        text = (
            self._extract_audio_transcript(response_json)
            if strict_text_only
            else self._extract_text(response_json)
        )
        session.state["messages"].append(user_message)
        session.state["messages"].append({"role": "assistant", "content": text})
        return text


def load_api_key_from_env_file(
    *,
    env_path: Path | None,
    key_names: Sequence[str],
) -> str:
    resolved_env_path = Path(env_path) if env_path is not None else Path(".env/api_keys.env")
    if not resolved_env_path.is_file():
        raise RuntimeError(f"Missing env file: {resolved_env_path}")
    values = parse_env_file(resolved_env_path)
    for key_name in key_names:
        api_key = values.get(str(key_name), "").strip()
        if api_key:
            return api_key
    keys_text = ", ".join(str(key_name) for key_name in key_names)
    raise RuntimeError(f"None of {keys_text} is set in {resolved_env_path}")


class GeminiAudioAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str,
        api_key_env_path: Path | None = None,
        api_key_names: Sequence[str] = ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        temperature: float = 0.0,
        max_output_tokens: int = 256,
        top_p: float = 1.0,
        thinking_level: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.api_key_env_path = api_key_env_path
        self.api_key_names = [str(key_name) for key_name in api_key_names]
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.top_p = float(top_p)
        self.thinking_level = None if thinking_level is None else str(thinking_level).strip().lower()
        if self.thinking_level not in {None, "minimal", "low", "medium", "high"}:
            raise ValueError(f"Unsupported Gemini thinking_level: {thinking_level}")
        self._runtime: dict[str, Any] | None = None

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            from google import genai  # type: ignore
            from google.genai import types  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing dependency 'google-genai'. Install with: pip install google-genai") from exc

        api_key = load_api_key_from_env_file(
            env_path=self.api_key_env_path,
            key_names=self.api_key_names,
        )
        client = genai.Client(api_key=api_key)
        self._runtime = {"client": client, "types": types}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    @staticmethod
    def _build_contents_from_messages(types: Any, messages: Sequence[dict[str, Any]]) -> list[Any]:
        contents: list[Any] = []
        for message in messages:
            role = message["role"]
            text = str(message["text"])
            if role == "user":
                parts = [types.Part(text=text)]
                for path in message.get("audio_paths", []):
                    parts.append(
                        types.Part.from_bytes(
                            data=Path(path).read_bytes(),
                            mime_type="audio/wav",
                        )
                    )
                contents.append(types.UserContent(parts=parts))
            elif role == "model":
                contents.append(types.ModelContent(parts=[types.Part(text=text)]))
            else:
                raise ValueError(f"Unknown Gemini message role: {role}")
        return contents

    @staticmethod
    def _collect_text_parts(value: Any, limit: int = 64) -> list[str]:
        texts: list[str] = []

        def visit(item: Any) -> None:
            if len(texts) >= limit:
                return
            if isinstance(item, str):
                stripped = item.strip()
                if stripped:
                    texts.append(stripped)
                return
            if isinstance(item, dict):
                for key, child in item.items():
                    if key in {"text", "output_text"} and isinstance(child, str):
                        visit(child)
                    elif isinstance(child, (dict, list, tuple)):
                        visit(child)
                return
            if isinstance(item, (list, tuple)):
                for child in item:
                    visit(child)
                return
            for attr in ("text", "parts", "content", "candidates"):
                if hasattr(item, attr):
                    try:
                        visit(getattr(item, attr))
                    except Exception:
                        pass

        visit(value)
        return texts

    @classmethod
    def _extract_text(cls, response: Any) -> str:
        text = str(getattr(response, "text", "") or "").strip()
        if text:
            return text
        for candidate_text in cls._collect_text_parts(response):
            if candidate_text:
                return candidate_text
        raise APIAdapterError("Gemini response did not contain any text output.")

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        runtime = self._ensure_runtime()
        client = runtime["client"]
        types = runtime["types"]
        user_message = {
            "role": "user",
            "text": user_text,
            "audio_paths": [str(path) for path in audio_paths or []],
        }
        messages = [*session.state.setdefault("messages", []), user_message]
        contents = self._build_contents_from_messages(types, messages)
        try:
            config_kwargs: dict[str, Any] = {
                "system_instruction": session.system_prompt,
                "max_output_tokens": session_output_token_limit(session, self.max_output_tokens),
                "temperature": self.temperature,
                "top_p": self.top_p,
            }
            if self.thinking_level is not None:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_level=self.thinking_level
                )
            config = types.GenerateContentConfig(**config_kwargs)
            response = client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            raise APIAdapterError(
                f"Gemini API error: {exc}",
                status_code=_coerce_status_code(status_code),
            ) from exc

        text = self._extract_text(response)
        session.state["messages"].append(user_message)
        session.state["messages"].append({"role": "model", "text": text})
        return text


class LocalQwenOmniAdapter(BaseAdapter):
    _MIN_AUDIO_SECONDS = 0.25
    _GENERATED_TURN_BOUNDARY_RE = re.compile(
        r"(?:\n|(?<=\s))(?:Human|User|Assistant):",
        re.IGNORECASE,
    )

    def __init__(
        self,
        *,
        model_id: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self._runtime: dict[str, Any] | None = None

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration
        except Exception as exc:
            raise RuntimeError(
                "Missing Qwen dependencies. Install torch and transformers with Qwen2.5-Omni support."
            ) from exc

        if self.torch_dtype in {"", "auto", "none"}:
            dtype = "auto"
        elif hasattr(torch, str(self.torch_dtype)):
            dtype = getattr(torch, str(self.torch_dtype))
        else:
            raise ValueError(f"Unknown torch dtype: {self.torch_dtype}")

        processor = Qwen2_5OmniProcessor.from_pretrained(self.model_id)
        model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            device_map=self.device_map,
            attn_implementation="flash_attention_2",
        )
        self._runtime = {"processor": processor, "model": model}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(
            system_prompt=system_prompt,
            state={
                "messages": [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    }
                ]
            },
        )

    def close_session(self, session: AdapterSession) -> None:
        for path in session.state.pop("temp_audio_paths", []):
            try:
                Path(path).unlink()
            except FileNotFoundError:
                pass

    def _runtime_audio_path(self, session: AdapterSession, audio_path: Path) -> Path:
        path = Path(audio_path)
        with wave.open(str(path), "rb") as source:
            channels = source.getnchannels()
            sample_width = source.getsampwidth()
            sample_rate = source.getframerate()
            frame_count = source.getnframes()
            frames = source.readframes(frame_count)
        min_frames = int(round(float(sample_rate) * self._MIN_AUDIO_SECONDS))
        if frame_count >= min_frames:
            return path
        handle = _temp_wav_handle("qwen_tot_padded_")
        padded_path = Path(handle.name)
        handle.close()
        with wave.open(str(padded_path), "wb") as target:
            target.setnchannels(channels)
            target.setsampwidth(sample_width)
            target.setframerate(sample_rate)
            target.writeframes(frames)
            target.writeframes(b"\x00" * sample_width * channels * max(0, min_frames - frame_count))
        session.state.setdefault("temp_audio_paths", []).append(str(padded_path))
        return padded_path

    @classmethod
    def _truncate_generated_turn(cls, text: str) -> str:
        match = cls._GENERATED_TURN_BOUNDARY_RE.search(str(text))
        return str(text)[: match.start()].rstrip() if match is not None else str(text).strip()

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        runtime = self._ensure_runtime()
        processor = runtime["processor"]
        model = runtime["model"]
        import torch

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for path in audio_paths or []:
            content.append({"type": "audio", "path": str(self._runtime_audio_path(session, Path(path)))})
        session.state["messages"].append({"role": "user", "content": content})

        inputs = processor.apply_chat_template(
            session.state["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": session_output_token_limit(session, self.max_new_tokens),
            "do_sample": self.do_sample,
        }
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            pad_id = getattr(tokenizer, "pad_token_id", None)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            if pad_id is None and eos_id is not None:
                gen_kwargs["pad_token_id"] = eos_id
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        prompt_len = inputs["input_ids"].shape[1]
        try:
            from transformers import StoppingCriteria, StoppingCriteriaList

            class _StopOnLineBoundary(StoppingCriteria):
                def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
                    try:
                        token_ids = input_ids[0, prompt_len:]
                        if hasattr(processor, "batch_decode"):
                            decoded = processor.batch_decode([token_ids], skip_special_tokens=True)[0]
                        else:
                            decoded = processor.tokenizer.decode(
                                token_ids.tolist(),
                                skip_special_tokens=True,
                            )
                    except Exception:
                        return False
                    normalized = decoded.strip()
                    if LocalQwenOmniAdapter._GENERATED_TURN_BOUNDARY_RE.search(decoded):
                        return True
                    return not preserve_multiline and "\n" in normalized

            gen_kwargs["stopping_criteria"] = StoppingCriteriaList([_StopOnLineBoundary()])
        except Exception:
            pass
        with torch.no_grad():
            try:
                output_ids = model.generate(**inputs, **gen_kwargs)
            except TypeError:
                gen_kwargs_no_stop = dict(gen_kwargs)
                gen_kwargs_no_stop.pop("stopping_criteria", None)
                output_ids = model.generate(**inputs, **gen_kwargs_no_stop)
        generated_ids = output_ids[0, prompt_len:]
        if hasattr(processor, "batch_decode"):
            raw_text = processor.batch_decode([generated_ids], skip_special_tokens=True)[0].strip()
        else:
            raw_text = processor.tokenizer.decode(generated_ids.tolist(), skip_special_tokens=True).strip()
        raw_text = self._truncate_generated_turn(raw_text)
        text = raw_text if preserve_multiline else (normalize_first_line(raw_text) or raw_text)
        session.state["messages"].append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )
        return text


def adapter_from_config(config: dict[str, Any]) -> BaseAdapter:
    def configured(adapter: BaseAdapter) -> BaseAdapter:
        adapter.configure_track2_capabilities(config)
        return adapter

    adapter_type = str(config["adapter_type"]).strip().lower()
    if adapter_type == "dummy":
        return configured(DummyAdapter(config.get("outputs", []), model_id=config.get("model_id", "dummy")))
    if adapter_type == "gpt_audio":
        env_path = config.get("api_key_env_path")
        return configured(OpenAIResponsesAudioAdapter(
            model_id=config["model_id"],
            api_key_env_path=Path(env_path) if env_path else None,
            temperature=float(config.get("temperature", 0.0)),
            max_output_tokens=int(config.get("max_output_tokens", 256)),
            request_timeout=float(config.get("request_timeout", 3600.0)),
        ))
    if adapter_type == "gemini":
        env_path = config.get("api_key_env_path")
        key_names = config.get("api_key_names", ["GOOGLE_API_KEY", "GEMINI_API_KEY"])
        if not isinstance(key_names, list) or not key_names:
            raise ValueError("gemini adapter requires a non-empty api_key_names list.")
        return configured(GeminiAudioAdapter(
            model_id=config.get("model_id", "gemini-2.5-flash"),
            api_key_env_path=Path(env_path) if env_path else None,
            api_key_names=[str(key_name) for key_name in key_names],
            temperature=float(config.get("temperature", 0.0)),
            max_output_tokens=int(config.get("max_output_tokens", 256)),
            top_p=float(config.get("top_p", 1.0)),
            thinking_level=config.get("thinking_level"),
        ))
    if adapter_type == "qwen25_omni":
        return configured(LocalQwenOmniAdapter(
            model_id=config["model_id"],
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            do_sample=bool(config.get("do_sample", False)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
        ))
    if adapter_type == "gpt_audio_mini":
        from .audio_model_adapters import GPTAudioMiniAdapter

        env_path = config.get("api_key_env_path")
        key_names = config.get("api_key_names", ["OPENAI_API_KEY"])
        if not isinstance(key_names, list) or not key_names:
            raise ValueError("gpt_audio_mini adapter requires a non-empty api_key_names list.")
        return configured(GPTAudioMiniAdapter(
            model_id=config.get("model_id", "gpt-audio-mini"),
            api_key_env_path=Path(env_path) if env_path else None,
            api_key_names=[str(key_name) for key_name in key_names],
            api_base_url=str(config.get("api_base_url", "")),
            request_timeout=float(config.get("request_timeout", 3600.0)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            max_output_tokens=int(config.get("max_output_tokens", 256)),
        ))
    if adapter_type == "moss_audio":
        from .audio_model_adapters import MOSSAudioAdapter

        return configured(MOSSAudioAdapter(
            model_id=config.get("model_id", "OpenMOSS-Team/MOSS-Audio-8B-Instruct"),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            do_sample=bool(config.get("do_sample", False)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
        ))
    if adapter_type == "mimo_audio":
        from .audio_model_adapters import MiMoAudioAdapter

        return configured(MiMoAudioAdapter(
            model_id=config.get("model_id", "XiaomiMiMo/MiMo-Audio-7B-Instruct"),
            tokenizer_id=config.get("tokenizer_id", "XiaomiMiMo/MiMo-Audio-Tokenizer"),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            do_sample=bool(config.get("do_sample", False)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            thinking=bool(config.get("thinking", False)),
        ))
    if adapter_type == "kimi_audio":
        from .audio_model_adapters import KimiAudioAdapter

        return configured(KimiAudioAdapter(
            model_id=config.get("model_id", "moonshotai/Kimi-Audio-7B-Instruct"),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            do_sample=bool(config.get("do_sample", False)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
        ))
    if adapter_type == "audio_flamingo_next":
        from .audio_model_adapters import AudioFlamingoNextAdapter

        return configured(AudioFlamingoNextAdapter(
            model_id=config.get("model_id", "nvidia/audio-flamingo-next-hf"),
            device_map=str(config.get("device_map", "auto")),
            torch_dtype=str(config.get("torch_dtype", "auto")),
            max_new_tokens=int(config.get("max_new_tokens", 256)),
            do_sample=bool(config.get("do_sample", False)),
            temperature=float(config.get("temperature", 0.0)),
            top_p=float(config.get("top_p", 1.0)),
            repetition_penalty=float(config.get("repetition_penalty", 1.0)),
            strip_thinking=bool(config.get("strip_thinking", False)),
        ))
    raise ValueError(f"Unknown adapter_type: {adapter_type}")


def load_adapter_config(config_path: Path) -> dict[str, Any]:
    """Load settings; relative credential paths are rooted at the repository."""
    resolved_path = Path(config_path).resolve()
    config = json.loads(resolved_path.read_text(encoding="utf-8"))
    env_path = config.get("api_key_env_path")
    if env_path:
        env_path_obj = Path(env_path)
        if not env_path_obj.is_absolute():
            config["api_key_env_path"] = str((Path(__file__).resolve().parents[1] / env_path_obj).resolve())
    return config
