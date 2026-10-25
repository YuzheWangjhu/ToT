from __future__ import annotations

import base64
import io
import json
import math
import os
import re
import tempfile
import wave
from pathlib import Path
from typing import Any, Sequence

import requests

from .adapters import (
    API_ERROR_HARD,
    API_ERROR_SOFT,
    APIAdapterError,
    AdapterSession,
    BaseAdapter,
    load_api_key_from_env_file,
    session_output_token_limit,
    session_requires_strict_text_only,
    wav_bytes_with_min_duration,
)
from .text_protocol import normalize_first_line


def _torch_dtype_arg(torch_module: Any, torch_dtype: str, *, default: Any = "auto") -> Any:
    if torch_dtype in {"", "auto", "none", None}:
        return default
    if hasattr(torch_module, str(torch_dtype)):
        return getattr(torch_module, str(torch_dtype))
    raise ValueError(f"Unknown torch dtype: {torch_dtype}")


def _decode_ids(processor: Any, token_ids: Any) -> str:
    if hasattr(processor, "batch_decode"):
        try:
            return processor.batch_decode([token_ids], skip_special_tokens=True)[0]
        except Exception:
            pass
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids.tolist(), skip_special_tokens=True)
        except Exception:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
    if hasattr(processor, "decode"):
        try:
            return processor.decode(token_ids, skip_special_tokens=True)
        except Exception:
            try:
                return processor.decode(token_ids.tolist(), skip_special_tokens=True)
            except Exception:
                pass
    return str(token_ids)


def _normalized_output(raw_text: Any, *, preserve_multiline: bool = False) -> str:
    raw = str(raw_text or "").strip()
    if preserve_multiline:
        return raw
    return normalize_first_line(raw) or raw


_TRAILING_GENERATION_CONTROL_TOKENS_RE = re.compile(
    r"(?:<\|[A-Za-z0-9_:.\-]+\|>\s*)+$"
)


def _strip_trailing_generation_control_tokens(raw_text: Any) -> str:
    return _TRAILING_GENERATION_CONTROL_TOKENS_RE.sub("", str(raw_text or "")).rstrip()


def _strip_leading_think_blocks(raw_text: Any) -> str:
    raw = str(raw_text or "").strip()
    text = raw
    while text.startswith("<think>"):
        closing_index = text.find("</think>", len("<think>"))
        if closing_index < 0:
            return raw
        text = text[closing_index + len("</think>") :].lstrip()
    if not text:
        return raw
    return text


def _last_text_output(output: Any) -> str:
    if isinstance(output, tuple):
        text_items = [item for item in output if isinstance(item, str)]
        return (text_items[-1] if text_items else str(output[-1] if output else "")).strip()
    if isinstance(output, list):
        text_items = [item for item in output if isinstance(item, str)]
        return (text_items[-1] if text_items else str(output[-1] if output else "")).strip()
    return str(output).strip()


def _read_mono_audio(path: Path, target_sample_rate: int) -> Any:
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
    except Exception as exc:
        raise RuntimeError("Missing audio preprocessing dependencies: numpy, soundfile, and scipy are required.") from exc

    waveform, original_sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    if waveform.shape[1] > 1:
        mono = np.mean(waveform, axis=1)
    else:
        mono = waveform[:, 0]
    original_sample_rate = int(original_sample_rate)
    target_sample_rate = int(target_sample_rate)
    if original_sample_rate != target_sample_rate:
        divisor = math.gcd(original_sample_rate, target_sample_rate)
        mono = resample_poly(
            mono,
            up=target_sample_rate // divisor,
            down=original_sample_rate // divisor,
        ).astype("float32", copy=False)
    return mono


def _temp_wav_handle(prefix: str):
    temp_dir = os.environ.get("SLURM_TMPDIR") or os.environ.get("TMPDIR") or "/tmp"
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(prefix=prefix, suffix=".wav", delete=False, dir=temp_dir)


def _write_concatenated_wav(
    audio_paths: Sequence[Path],
    *,
    target_sample_rate: int = 16000,
    separator_sec: float = 0.25,
) -> Path:
    if not audio_paths:
        raise ValueError("audio_paths must contain at least one path.")
    try:
        import numpy as np
        import soundfile as sf
    except Exception as exc:
        raise RuntimeError("Missing audio concatenation dependencies: numpy and soundfile are required.") from exc

    separator = np.zeros(int(round(float(separator_sec) * target_sample_rate)), dtype="float32")
    chunks = []
    for index, path in enumerate(audio_paths):
        if index:
            chunks.append(separator)
        chunks.append(_read_mono_audio(Path(path), target_sample_rate))
    concatenated = np.concatenate(chunks).astype("float32", copy=False)
    handle = _temp_wav_handle("mimo_tot_")
    tmp_path = Path(handle.name)
    handle.close()
    sf.write(str(tmp_path), concatenated, target_sample_rate, subtype="PCM_16")
    return tmp_path


def _write_silence_wav(*, sample_rate: int = 16000, duration_sec: float = 0.25) -> Path:
    pcm_bytes = b"\x00\x00" * int(round(float(sample_rate) * float(duration_sec)))
    handle = _temp_wav_handle("kimi_tot_silence_")
    tmp_path = Path(handle.name)
    handle.close()
    with wave.open(str(tmp_path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(int(sample_rate))
        wav.writeframes(pcm_bytes)
    return tmp_path


def _session_messages(session: AdapterSession) -> list[dict[str, Any]]:
    return session.state.setdefault("messages", [])


def _append_user_message(session: AdapterSession, user_text: str, audio_paths: Sequence[Path] | None) -> None:
    _session_messages(session).append(
        {
            "role": "user",
            "text": str(user_text),
            "audio_paths": [str(path) for path in audio_paths or []],
        }
    )


def _append_assistant_message(session: AdapterSession, text: str) -> None:
    _session_messages(session).append({"role": "assistant", "text": str(text)})


def _all_audio_paths(session: AdapterSession) -> list[Path]:
    return [
        Path(path)
        for message in _session_messages(session)
        if message.get("role") == "user"
        for path in message.get("audio_paths", [])
    ]


def _text_transcript_for_single_prompt(session: AdapterSession) -> str:
    lines = [session.system_prompt.strip()]
    for message in _session_messages(session):
        role = str(message.get("role", "")).strip().lower()
        text = str(message.get("text", "")).strip()
        audio_paths = message.get("audio_paths", [])
        if role == "user":
            evidence = f" Audio evidence clips attached to this turn: {len(audio_paths)}." if audio_paths else ""
            lines.append(f"User:{evidence}\n{text}")
        elif role == "assistant":
            lines.append(f"Assistant:\n{text}")
        else:
            raise ValueError(f"Unknown message role: {role}")
    return "\n\n".join(line for line in lines if line)


class GPTAudioMiniAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str = "gpt-audio-mini",
        api_key_env_path: Path | None = None,
        api_key_names: Sequence[str] = ("OPENAI_API_KEY",),
        api_base_url: str = "",
        request_timeout: float = 3600.0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_output_tokens: int = 256,
    ) -> None:
        self.model_id = model_id
        self.api_key_env_path = api_key_env_path
        self.api_key_names = [str(name) for name in api_key_names]
        self.api_base_url = str(api_base_url).strip() or "https://api.openai.com/v1"
        self.request_timeout = float(request_timeout)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.max_output_tokens = int(max_output_tokens)
        self._api_key: str | None = None
        self._silence_b64: str | None = None

    def _get_api_key(self) -> str:
        if self._api_key is None:
            self._api_key = load_api_key_from_env_file(
                env_path=self.api_key_env_path,
                key_names=self.api_key_names,
            )
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
            pcm_bytes = b"\x00\x00" * (sample_rate // 4)
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(sample_rate)
                handle.writeframes(pcm_bytes)
            self._silence_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {
            "type": "input_audio",
            "input_audio": {"data": self._silence_b64, "format": "wav"},
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
            text_parts = [
                str(item.get("text", "")).strip()
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            text_parts = [part for part in text_parts if part]
            if text_parts:
                return "\n".join(text_parts)
        raise APIAdapterError(
            "OpenAI response did not contain text output.",
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
        resolved_audio_paths = [Path(path) for path in audio_paths or []]
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
            content.append(self._audio_item(path))

        user_message = {"role": "user", "content": content}
        strict_text_only = session_requires_strict_text_only(session)
        payload = {
            "model": self.model_id,
            "messages": [*session.state["messages"], user_message],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "n": 1,
            "max_completion_tokens": session_output_token_limit(session, self.max_output_tokens),
            "modalities": ["text", "audio"] if strict_text_only else ["text"],
        }
        if strict_text_only:
            payload["audio"] = {"voice": "alloy", "format": "wav"}
        try:
            response = requests.post(
                self.api_base_url.rstrip("/") + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._get_api_key()}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(payload),
                timeout=self.request_timeout,
            )
        except (
            requests.exceptions.MissingSchema,
            requests.exceptions.InvalidSchema,
            requests.exceptions.InvalidURL,
        ) as exc:
            raise APIAdapterError(
                f"OpenAI API endpoint configuration failed: {exc}",
                error_class=API_ERROR_HARD,
            ) from exc
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
            data = response.json()
        except ValueError as exc:
            raise APIAdapterError(
                "OpenAI API response was not valid JSON.",
                error_class=API_ERROR_SOFT,
            ) from exc
        text = (
            self._extract_audio_transcript(data)
            if strict_text_only
            else self._extract_text(data)
        )
        session.state["messages"].append(user_message)
        session.state["messages"].append({"role": "assistant", "content": text})
        return text


class MOSSAudioAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str = "OpenMOSS-Team/MOSS-Audio-8B-Instruct",
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
        except Exception as exc:
            raise RuntimeError("Missing MOSS dependency: torch is required.") from exc
        try:
            from src.modeling_moss_audio import MossAudioModel  # type: ignore
            from src.processing_moss_audio import MossAudioProcessor  # type: ignore
        except Exception:
            try:
                from modeling_moss_audio import MossAudioModel  # type: ignore
                from processing_moss_audio import MossAudioProcessor  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Missing MOSS-Audio runtime. Install OpenMOSS/MOSS-Audio or add it to PYTHONPATH."
                ) from exc

        dtype = _torch_dtype_arg(torch, self.torch_dtype, default="auto")
        processor = MossAudioProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            enable_time_marker=True,
        )
        load_attempts = [
            {
                "device_map": self.device_map,
                "attn_implementation": "flash_attention_2",
                "torch_dtype": dtype,
                "trust_remote_code": True,
            },
            {"device_map": self.device_map, "torch_dtype": dtype, "trust_remote_code": True},
        ]
        model = None
        last_err: Exception | None = None
        for kwargs in load_attempts:
            try:
                model = MossAudioModel.from_pretrained(self.model_id, **kwargs)
                break
            except Exception as exc:
                last_err = exc
        if model is None:
            assert last_err is not None
            raise last_err
        model.eval()
        self._runtime = {"torch": torch, "processor": processor, "model": model}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    def _multi_audio_transcript(self, session: AdapterSession) -> str:
        lines = [
            "The attached audio is a concatenation of all benchmark audio evidence clips in session order, "
            "with 0.25 seconds of silence between clips.",
        ]
        for message in _session_messages(session):
            role = str(message.get("role", "")).strip().lower()
            text = str(message.get("text", "")).strip()
            audio_count = len(message.get("audio_paths", []))
            if role == "user":
                evidence = f" Audio evidence clips attached to this turn: {audio_count}." if audio_count else ""
                lines.append(f"User:{evidence}\n{text}")
            elif role == "assistant":
                lines.append(f"Assistant:\n{text}")
            else:
                raise ValueError(f"Unknown MOSS message role: {role}")
        return "\n\n".join(line for line in lines if line)

    def _build_prompt_and_audio(self, session: AdapterSession) -> tuple[str, list[Any]]:
        runtime = self._ensure_runtime()
        processor = runtime["processor"]
        sample_rate = int(getattr(getattr(processor, "config", None), "mel_sr", 16000))
        audio_history = _all_audio_paths(session)
        if len(audio_history) > 1:
            tmp_path: Path | None = None
            try:
                tmp_path = _write_concatenated_wav(audio_history, target_sample_rate=sample_rate)
                audio_arrays = [_read_mono_audio(tmp_path, sample_rate)]
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass
            prompt = (
                f"<|im_start|>system\n{session.system_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                "<|audio_bos|><|AUDIO|><|audio_eos|>\n"
                f"{self._multi_audio_transcript(session)}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            return prompt, audio_arrays

        prompt_parts = [
            f"<|im_start|>system\n{session.system_prompt}<|im_end|>\n",
        ]
        audio_arrays: list[Any] = []
        for message in _session_messages(session):
            role = str(message["role"])
            text = str(message.get("text", ""))
            if role == "user":
                prompt_parts.append("<|im_start|>user\n")
                for path in message.get("audio_paths", []):
                    prompt_parts.append("<|audio_bos|><|AUDIO|><|audio_eos|>\n")
                    audio_arrays.append(_read_mono_audio(Path(path), sample_rate))
                prompt_parts.append(f"{text}<|im_end|>\n")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{text}<|im_end|>\n")
            else:
                raise ValueError(f"Unknown MOSS message role: {role}")
        prompt_parts.append("<|im_start|>assistant\n")
        return "".join(prompt_parts), audio_arrays

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        _append_user_message(session, user_text, audio_paths)
        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        processor = runtime["processor"]
        model = runtime["model"]
        prompt, audio_arrays = self._build_prompt_and_audio(session)
        inputs = processor(text=prompt, audios=audio_arrays, return_tensors="pt").to(model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
        if "input_ids" in inputs and hasattr(processor, "audio_token_id"):
            inputs["audio_input_mask"] = inputs["input_ids"] == processor.audio_token_id

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": session_output_token_limit(session, self.max_new_tokens),
            "do_sample": self.do_sample,
            "num_beams": 1,
            "use_cache": True,
        }
        tokenizer = getattr(processor, "tokenizer", None) or processor
        pad_id = getattr(tokenizer, "pad_token_id", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if pad_id is None and eos_id is not None:
            gen_kwargs["pad_token_id"] = eos_id
        if self.do_sample:
            gen_kwargs["temperature"] = self.temperature
            gen_kwargs["top_p"] = self.top_p

        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        text = _normalized_output(
            _decode_ids(processor, output_ids[0, prompt_len:]),
            preserve_multiline=preserve_multiline,
        )
        _append_assistant_message(session, text)
        return text


class MiMoAudioAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str = "XiaomiMiMo/MiMo-Audio-7B-Instruct",
        tokenizer_id: str = "XiaomiMiMo/MiMo-Audio-Tokenizer",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.thinking = bool(thinking)
        self._runtime: dict[str, Any] | None = None

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        _ = self.torch_dtype
        try:
            import inspect
            import torch
        except Exception as exc:
            raise RuntimeError("Missing MiMo dependency: torch is required.") from exc
        try:
            from src.mimo_audio.mimo_audio import MimoAudio  # type: ignore
        except Exception:
            try:
                from mimo_audio.mimo_audio import MimoAudio  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    "Missing MiMo-Audio runtime. Install XiaomiMiMo/MiMo-Audio or add it to PYTHONPATH."
                ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = str(self.device_map).strip().lower()
        if device_map in {"cpu", "cuda", "cuda:0"}:
            device = device_map
        try:
            params = inspect.signature(MimoAudio).parameters
        except Exception:
            params = {}
        if "model_path" in params and "mimo_audio_tokenizer_path" in params:
            kwargs: dict[str, Any] = {
                "model_path": self.model_id,
                "mimo_audio_tokenizer_path": self.tokenizer_id,
            }
            if "device" in params:
                kwargs["device"] = device
            model = MimoAudio(**kwargs)
        elif "model_id" in params or "tokenizer_id" in params:
            model = MimoAudio(model_id=self.model_id, tokenizer_id=self.tokenizer_id)
        elif "model_path" in params or "tokenizer_path" in params:
            model = MimoAudio(model_path=self.model_id, tokenizer_path=self.tokenizer_id)
        else:
            model = MimoAudio(self.model_id, self.tokenizer_id)
        try:
            model.to(device)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass
        self._runtime = {"model": model}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    def _text_message_list(self, session: AdapterSession) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        first_user = True
        for message in _session_messages(session):
            role = str(message["role"])
            text = str(message.get("text", ""))
            if role == "user":
                if first_user:
                    text = f"{session.system_prompt}\n\n{text}"
                    first_user = False
                messages.append({"role": "user", "content": text})
            elif role == "assistant":
                messages.append({"role": "assistant", "content": text})
            else:
                raise ValueError(f"Unknown MiMo message role: {role}")
        return messages

    def _mimo_text_stopping_criteria(self, model: Any) -> list[Any]:
        try:
            from src.mimo_audio.modeling_mimo_audio import MiMoStopper  # type: ignore
        except Exception:
            try:
                from mimo_audio.modeling_mimo_audio import MiMoStopper  # type: ignore
            except Exception as exc:
                raise RuntimeError("Missing MiMoStopper in the MiMo-Audio runtime.") from exc
        return [
            MiMoStopper(
                stop_tokens=[model.tokenizer.eos_token_id, model.im_end_idx],
                group_size=model.group_size,
                audio_channels=model.audio_channels,
            )
        ]

    def _audio_understanding_sft(
        self, model: Any, input_speech: str, input_text: str, *, max_new_tokens: int
    ) -> str:
        if not hasattr(model, "get_audio_understanding_sft_prompt"):
            return model.audio_understanding_sft(input_speech, input_text, thinking=self.thinking)
        input_ids = model.get_audio_understanding_sft_prompt(input_speech, input_text, thinking=self.thinking)
        return model.forward(
            input_ids,
            stopping_criteria=self._mimo_text_stopping_criteria(model),
            task_name="audio_understanding",
            max_new_tokens=max_new_tokens,
        )

    def _text_dialogue_sft_multiturn(
        self, model: Any, message_list: list[dict[str, str]], *, max_new_tokens: int
    ) -> str:
        if not hasattr(model, "get_text_dialogue_sft_multiturn_prompt"):
            return model.text_dialogue_sft_multiturn(message_list, thinking=self.thinking)
        input_ids = model.get_text_dialogue_sft_multiturn_prompt(message_list, thinking=self.thinking)
        return model.forward(
            input_ids,
            stopping_criteria=self._mimo_text_stopping_criteria(model),
            task_name="text_chat",
            add_history=False,
            max_new_tokens=max_new_tokens,
        )

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        _append_user_message(session, user_text, audio_paths)
        model = self._ensure_runtime()["model"]
        max_new_tokens = session_output_token_limit(session, self.max_new_tokens)
        audio_history = _all_audio_paths(session)
        if audio_history:
            tmp_path: Path | None = None
            try:
                tmp_path = _write_concatenated_wav(audio_history)
                output = self._audio_understanding_sft(
                    model,
                    str(tmp_path),
                    _text_transcript_for_single_prompt(session),
                    max_new_tokens=max_new_tokens,
                )
            finally:
                if tmp_path is not None:
                    try:
                        tmp_path.unlink()
                    except FileNotFoundError:
                        pass
        else:
            output = self._text_dialogue_sft_multiturn(
                model, self._text_message_list(session), max_new_tokens=max_new_tokens
            )
        text = _normalized_output(_last_text_output(output), preserve_multiline=preserve_multiline)
        _append_assistant_message(session, text)
        return text


class KimiAudioAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str = "moonshotai/Kimi-Audio-7B-Instruct",
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
        _ = self.torch_dtype
        try:
            import inspect
            import torch
            from kimia_infer.api.kimia import KimiAudio  # type: ignore
        except Exception as exc:
            raise RuntimeError("Missing Kimi dependencies. Install the Kimi-Audio inference stack.") from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_map = str(self.device_map).strip().lower()
        if device_map in {"cpu", "cuda", "cuda:0"}:
            device = device_map

        def init_model(load_detokenizer: bool) -> Any:
            kwargs: dict[str, Any] = {"model_path": self.model_id, "load_detokenizer": load_detokenizer}
            try:
                if "device" in inspect.signature(KimiAudio).parameters:
                    kwargs["device"] = device
            except Exception:
                pass
            try:
                return KimiAudio(**kwargs)
            except TypeError:
                return KimiAudio(self.model_id, load_detokenizer=load_detokenizer)

        try:
            model = init_model(load_detokenizer=False)
        except Exception:
            model = init_model(load_detokenizer=True)
        try:
            model.to(device)
        except Exception:
            pass
        try:
            model.eval()
        except Exception:
            pass
        self._runtime = {"torch": torch, "model": model}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    def _kimi_messages(self, session: AdapterSession, *, placeholder_audio_path: Path | None = None) -> list[dict[str, str]]:
        messages = [{"role": "user", "message_type": "text", "content": session.system_prompt}]
        if placeholder_audio_path is not None:
            messages.append(
                {
                    "role": "user",
                    "message_type": "text",
                    "content": (
                        "No benchmark audio evidence is attached yet. "
                        "A silent placeholder audio clip is included only to satisfy the model runtime. "
                        "Do not treat it as episode evidence."
                    ),
                }
            )
            messages.append({"role": "user", "message_type": "audio", "content": str(placeholder_audio_path)})
        for message in _session_messages(session):
            role = str(message["role"])
            text = str(message.get("text", ""))
            if role == "assistant":
                messages.append({"role": "assistant", "message_type": "text", "content": text})
            elif role == "user":
                if text.strip():
                    messages.append({"role": "user", "message_type": "text", "content": text})
                for path in message.get("audio_paths", []):
                    messages.append({"role": "user", "message_type": "audio", "content": str(path)})
            else:
                raise ValueError(f"Unknown Kimi message role: {role}")
        return messages

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        _append_user_message(session, user_text, audio_paths)
        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        model = runtime["model"]
        # Kimi's native text path is unstable for benchmark turns. Track 1 keeps
        # the audio-conditioned generation path active with a silent placeholder;
        # use the same runtime path for Track 2 calls that carry no evidence.
        placeholder_audio_path = (
            None if _all_audio_paths(session) else _write_silence_wav()
        )
        gen_kwargs: dict[str, Any] = {
            "output_type": "text",
            "max_new_tokens": session_output_token_limit(session, self.max_new_tokens),
        }
        if self.do_sample:
            gen_kwargs["text_temperature"] = self.temperature
            gen_kwargs["text_top_k"] = int(max(5, round(50 * max(0.0, min(1.0, self.top_p)))))
        else:
            gen_kwargs["text_temperature"] = 0.0
            gen_kwargs["text_top_k"] = 1
        try:
            import inspect

            sig = inspect.signature(model.generate)
            if not any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values()):
                gen_kwargs = {key: value for key, value in gen_kwargs.items() if key in sig.parameters}
        except Exception:
            pass
        try:
            messages = self._kimi_messages(session, placeholder_audio_path=placeholder_audio_path)
            try:
                with torch.no_grad():
                    output = model.generate(messages, **gen_kwargs)
            except TypeError:
                with torch.no_grad():
                    output = model.generate(messages=messages, **gen_kwargs)
        finally:
            if placeholder_audio_path is not None:
                try:
                    placeholder_audio_path.unlink()
                except FileNotFoundError:
                    pass
        if isinstance(output, tuple) and len(output) == 2:
            raw_text = str(output[1])
        else:
            raw_text = str(output)
        raw_text = _strip_trailing_generation_control_tokens(raw_text)
        text = _normalized_output(raw_text, preserve_multiline=preserve_multiline)
        _append_assistant_message(session, text)
        return text


class AudioFlamingoNextAdapter(BaseAdapter):
    def __init__(
        self,
        *,
        model_id: str = "nvidia/audio-flamingo-next-hf",
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        strip_thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.do_sample = bool(do_sample)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.repetition_penalty = float(repetition_penalty)
        self.strip_thinking = bool(strip_thinking)
        self._runtime: dict[str, Any] | None = None

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        try:
            import torch
            from transformers import AutoModel, AutoModelForCausalLM, AutoProcessor
        except Exception as exc:
            raise RuntimeError("Missing Audio Flamingo Next dependencies: torch and transformers are required.") from exc
        dtype = _torch_dtype_arg(torch, self.torch_dtype, default="auto")
        processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        common_kwargs = {"device_map": self.device_map, "torch_dtype": dtype, "trust_remote_code": True}
        load_attempts = []
        for model_cls in (AutoModel, AutoModelForCausalLM):
            load_attempts.append((model_cls, {**common_kwargs, "attn_implementation": "flash_attention_2"}))
            load_attempts.append((model_cls, common_kwargs))
        model = None
        last_err: Exception | None = None
        for model_cls, kwargs in load_attempts:
            try:
                model = model_cls.from_pretrained(self.model_id, **kwargs)
                break
            except Exception as exc:
                last_err = exc
        if model is None:
            assert last_err is not None
            raise last_err
        model.eval()
        self._runtime = {"torch": torch, "processor": processor, "model": model}
        return self._runtime

    def create_session(self, system_prompt: str) -> AdapterSession:
        return AdapterSession(system_prompt=system_prompt, state={"messages": []})

    @staticmethod
    def _write_mono_16k_temp_wav(audio_path: Path) -> Path:
        try:
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("Missing Audio Flamingo Next dependency: soundfile is required.") from exc
        audio = _read_mono_audio(Path(audio_path), 16000)
        handle = _temp_wav_handle("afnext_tot_")
        tmp_path = Path(handle.name)
        handle.close()
        sf.write(str(tmp_path), audio, 16000, subtype="PCM_16")
        return tmp_path

    def _conversation(self, session: AdapterSession, temp_paths: list[Path]) -> list[list[dict[str, Any]]]:
        audio_history = _all_audio_paths(session)
        if len(audio_history) > 1:
            temp_path = _write_concatenated_wav(audio_history, target_sample_rate=16000)
            temp_paths.append(temp_path)
            return [
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _text_transcript_for_single_prompt(session)},
                            {"type": "audio", "path": str(temp_path)},
                        ],
                    }
                ]
            ]

        turns: list[dict[str, Any]] = []
        first_user = True
        for message in _session_messages(session):
            role = str(message["role"])
            text = str(message.get("text", ""))
            if role == "user":
                if first_user:
                    text = f"{session.system_prompt}\n\n{text}"
                    first_user = False
                content: list[dict[str, str]] = [{"type": "text", "text": text}]
                for path in message.get("audio_paths", []):
                    temp_path = self._write_mono_16k_temp_wav(Path(path))
                    temp_paths.append(temp_path)
                    content.append({"type": "audio", "path": str(temp_path)})
                turns.append({"role": "user", "content": content})
            elif role == "assistant":
                turns.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
            else:
                raise ValueError(f"Unknown Audio Flamingo Next message role: {role}")
        return [turns]

    def send(
        self,
        session: AdapterSession,
        user_text: str,
        audio_paths: Sequence[Path] | None = None,
        preserve_multiline: bool = False,
    ) -> str:
        _append_user_message(session, user_text, audio_paths)
        runtime = self._ensure_runtime()
        torch = runtime["torch"]
        processor = runtime["processor"]
        model = runtime["model"]
        temp_paths: list[Path] = []
        try:
            inputs = processor.apply_chat_template(
                self._conversation(session, temp_paths),
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            if "input_features" in inputs:
                inputs["input_features"] = inputs["input_features"].to(model.dtype)
            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": session_output_token_limit(session, self.max_new_tokens),
                "do_sample": self.do_sample,
            }
            if self.repetition_penalty != 1.0:
                gen_kwargs["repetition_penalty"] = self.repetition_penalty
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
            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)
            raw_text = _decode_ids(processor, output_ids[0, prompt_len:])
            if self.strip_thinking:
                raw_text = _strip_leading_think_blocks(raw_text)
            text = _normalized_output(raw_text, preserve_multiline=preserve_multiline)
        finally:
            for path in temp_paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
        _append_assistant_message(session, text)
        return text
