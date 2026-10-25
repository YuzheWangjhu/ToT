"""Deterministic tones for protocol tests, not benchmark speech assets."""

from pathlib import Path

import numpy as np
import soundfile as sf


def write_audio_fixture(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=False)
    sample_rate = 16000
    for index in range(10):
        label = f"label{index:02d}"
        for variant, duration in (("short", 0.4), ("long", 1.6)):
            time = np.arange(round(sample_rate * duration)) / sample_rate
            audio = 0.1 * np.sin(2 * np.pi * (220 + index * 40) * time)
            sf.write(root / f"{label}__voice-{variant}.wav", audio, sample_rate,
                     subtype="PCM_16")
    return root
