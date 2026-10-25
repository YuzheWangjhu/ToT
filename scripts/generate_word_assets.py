#!/usr/bin/env python3
"""Regenerate controlled-track TTS assets; this does not recover frozen paper audio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from openai_tts import synthesize_speech

VOICES = ("ash", "echo", "coral", "nova")
SPEEDS = ("0.5", "0.75", "1", "1.25")
# Preserve the original generation instruction, including its spelling.
INSTRUCTIONS = "netural narration voice"


def asset_requests(labels_path: Path) -> list[tuple[str, str, str, str]]:
    requests = []
    category = None
    for line in labels_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("Category ") and line.endswith(":"):
            category = line[len("Category "):-1]
            continue
        assert category is not None, "A category must precede the first label"
        assert "/" not in category and "/" not in line
        for voice in VOICES:
            for speed in SPEEDS:
                requests.append((category, line, voice, speed))
    assert len(requests) == 33 * 4 * 4, "Expected the frozen 33-label vocabulary"
    return requests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Describe requests without API calls or writes")
    args = parser.parse_args()
    requests = asset_requests(ROOT / "scripts" / "word_labels.txt")
    if args.dry_run:
        print(f"{len(requests)} TTS calls: 33 labels × 4 voices × 4 speeds")
        print("Generation is stochastic and does not replace the frozen paper audio archive.")
        return
    args.output_root.mkdir(parents=True, exist_ok=False)
    for category, text, voice, speed in requests:
        folder = args.output_root / category
        folder.mkdir(exist_ok=True)
        output = folder / f"{text}__voice-{voice}__speed-{speed.replace('.', '_')}.wav"
        synthesize_speech(text, output, voice=voice, speed=float(speed),
                          instructions=INSTRUCTIONS, env_path=args.env_file)


if __name__ == "__main__":
    main()
