#!/usr/bin/env python3
"""Build audio files for selected Track 2 reels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf

import validate_selected_reels


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SELECTED_REELS = SCRIPT_DIR / "track2_role_ipc_selected_reels_test_v1.csv"
DEFAULT_SELECTED_SCENES = SCRIPT_DIR / "track2_role_ipc_selected_reel_scenes_test_v1.csv"
DEFAULT_SCENE_METADATA = SCRIPT_DIR / "track2_role_ipc_scene_candidates_scene_filter_v1.csv"
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "generated_data" / "track2_role_ipc_test_v1" / "dataset"
DEFAULT_GAP_SECONDS = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selected Track 2 reel WAV files.")
    parser.add_argument("--selected-reels", type=Path, default=DEFAULT_SELECTED_REELS)
    parser.add_argument("--selected-scenes", type=Path, default=DEFAULT_SELECTED_SCENES)
    parser.add_argument("--scene-metadata", type=Path, default=DEFAULT_SCENE_METADATA)
    parser.add_argument("--dataset-root", type=Path, required=True, help="Separately acquired Seamless Interaction datasets directory.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gap-seconds", type=float, default=DEFAULT_GAP_SECONDS)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_mono_wav(path: Path) -> tuple[np.ndarray, int]:
    require_file(path, "source WAV")
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim != 1:
        raise ValueError(f"Expected mono WAV: {path}")
    if len(audio) == 0:
        raise ValueError(f"Empty WAV: {path}")
    return audio.astype(np.float32, copy=False), int(sample_rate)


def assert_same_audio_shape(
    scene_id: str,
    participant1_audio: np.ndarray,
    participant2_audio: np.ndarray,
    participant1_rate: int,
    participant2_rate: int,
) -> None:
    if participant1_rate != participant2_rate:
        raise ValueError(f"{scene_id}: participant sample rates differ: {participant1_rate} vs {participant2_rate}")
    if abs(len(participant1_audio) - len(participant2_audio)) > 1:
        raise ValueError(
            f"{scene_id}: participant frame counts differ by more than one sample: "
            f"{len(participant1_audio)} vs {len(participant2_audio)}"
        )


def mix_scene_audio(scene_id: str, scene_meta: pd.Series, dataset_root: Path) -> tuple[np.ndarray, int, dict[str, str]]:
    participant1_path = dataset_root / str(scene_meta["participant1_wav_relpath"])
    participant2_path = dataset_root / str(scene_meta["participant2_wav_relpath"])
    p1_audio, p1_rate = read_mono_wav(participant1_path)
    p2_audio, p2_rate = read_mono_wav(participant2_path)
    assert_same_audio_shape(scene_id, p1_audio, p2_audio, p1_rate, p2_rate)

    frame_count = min(len(p1_audio), len(p2_audio))
    mixed = 0.5 * p1_audio[:frame_count] + 0.5 * p2_audio[:frame_count]
    if not np.any(np.abs(mixed) > 0):
        raise ValueError(f"{scene_id}: mixed audio is all zeros")
    return mixed.astype(np.float32, copy=False), p1_rate, {
        "participant1_wav_relpath": str(scene_meta["participant1_wav_relpath"]),
        "participant2_wav_relpath": str(scene_meta["participant2_wav_relpath"]),
    }


def build_reel_audio(
    reel: pd.Series,
    reel_scenes: pd.DataFrame,
    scene_lookup: pd.DataFrame,
    dataset_root: Path,
    output_path: Path,
    gap_seconds: float,
    manifest_root: Path = SCRIPT_DIR,
) -> dict[str, Any]:
    scene_chunks: list[np.ndarray] = []
    scene_entries: list[dict[str, Any]] = []
    sample_rate: int | None = None
    cursor = 0
    gap_frames: int | None = None

    for scene_row in reel_scenes.sort_values("scene_rank").itertuples(index=False):
        scene_id = str(scene_row.scene_id)
        scene_meta = scene_lookup.loc[scene_id]
        mixed, scene_rate, source_paths = mix_scene_audio(scene_id, scene_meta, dataset_root)
        if sample_rate is None:
            sample_rate = scene_rate
            gap_frames = int(round(gap_seconds * sample_rate))
        elif scene_rate != sample_rate:
            raise ValueError(f"{reel['reel_id']}: scene sample rate differs: {scene_id}={scene_rate}, expected {sample_rate}")

        start_frame = cursor
        end_frame = start_frame + len(mixed)
        scene_chunks.append(mixed)
        scene_entries.append(
            {
                "scene_rank": int(scene_row.scene_rank),
                "scene_id": scene_id,
                "source_interaction_id": int(scene_row.interaction_id),
                "start_seconds": start_frame / sample_rate,
                "end_seconds": end_frame / sample_rate,
                "gap_after_seconds": float(scene_row.gap_after_seconds),
                **source_paths,
            }
        )
        cursor = end_frame

        if int(scene_row.scene_rank) < 4:
            assert gap_frames is not None
            scene_chunks.append(np.zeros(gap_frames, dtype=np.float32))
            cursor += gap_frames

    if sample_rate is None:
        raise ValueError(f"{reel['reel_id']}: no scenes")

    audio = np.concatenate(scene_chunks)
    peak = float(np.max(np.abs(audio)))
    if peak <= 0:
        raise ValueError(f"{reel['reel_id']}: reel audio is all zeros")
    if peak >= 1.0:
        audio = audio / (peak + 1e-8) * 0.999
        peak = float(np.max(np.abs(audio)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing audio: {output_path}")
    sf.write(str(output_path), audio, sample_rate, subtype="PCM_16")

    info = sf.info(str(output_path))
    expected_duration = float(reel["selected_reel_audio_duration_sec"])
    duration = float(info.frames) / float(info.samplerate)
    if abs(duration - expected_duration) > 1.0 / info.samplerate:
        raise ValueError(
            f"{reel['reel_id']}: written duration {duration} does not match selected duration {expected_duration}"
        )
    if info.channels != 1:
        raise ValueError(f"{reel['reel_id']}: written audio is not mono")

    return {
        "reel_id": str(reel["reel_id"]),
        "candidate_reel_id": str(reel["candidate_reel_id"]),
        "audio_path": str(output_path.relative_to(manifest_root)),
        "sample_rate": int(info.samplerate),
        "num_channels": int(info.channels),
        "num_frames": int(info.frames),
        "duration_seconds": duration,
        "selected_reel_audio_duration_sec": expected_duration,
        "sha256_audio": sha256_file(output_path),
        "peak_abs_before_write": peak,
        "source_split": str(reel["source_split"]),
        "session_id": str(reel["session_id"]),
        "dyad_key": str(reel["dyad_key"]),
        "exact_quote_candidate_count": int(reel["exact_quote_candidate_count"]),
        "partial_quote_candidate_count": int(reel["partial_quote_candidate_count"]),
        "repeated_phrase_5_8_candidate_count": int(reel["repeated_phrase_5_8_candidate_count"]),
        "repeated_phrase_6_8_candidate_count": int(reel["repeated_phrase_6_8_candidate_count"]),
        "structural_relation_candidate_count": int(reel["structural_relation_candidate_count"]),
        "scenes": scene_entries,
    }


def main() -> None:
    args = parse_args()
    validate_selected_reels.validate(
        argparse.Namespace(
            selected_reels=args.selected_reels,
            selected_scenes=args.selected_scenes,
            audited_candidates=validate_selected_reels.DEFAULT_AUDITED_CANDIDATES,
            gap_seconds=args.gap_seconds,
        )
    )
    require_file(args.selected_reels, "selected reels CSV")
    require_file(args.selected_scenes, "selected reel scenes CSV")
    require_file(args.scene_metadata, "scene metadata CSV")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {args.dataset_root}")

    reels = pd.read_csv(args.selected_reels, keep_default_na=False)
    selected_scenes = pd.read_csv(args.selected_scenes, keep_default_na=False)
    scene_metadata = pd.read_csv(args.scene_metadata, keep_default_na=False)
    if scene_metadata["scene_id"].duplicated().any():
        duplicates = sorted(scene_metadata.loc[scene_metadata["scene_id"].duplicated(), "scene_id"].unique())
        raise ValueError(f"Duplicate scene_id values in scene metadata: {duplicates[:10]}")
    scene_lookup = scene_metadata.set_index("scene_id", drop=False)

    episodes_dir = args.output_root / "episodes"
    manifest_entries = []
    for reel in reels.sort_values("reel_id").itertuples(index=False):
        reel_series = pd.Series(reel._asdict())
        reel_id = str(reel_series["reel_id"])
        reel_scenes = selected_scenes[selected_scenes["reel_id"].eq(reel_id)].copy()
        if len(reel_scenes) != 4:
            raise ValueError(f"{reel_id}: expected 4 scenes, got {len(reel_scenes)}")
        output_path = episodes_dir / f"{reel_id}.wav"
        manifest_entries.append(
            build_reel_audio(
                reel_series,
                reel_scenes,
                scene_lookup,
                args.dataset_root,
                output_path,
                args.gap_seconds,
            )
        )

    manifest = {
        "track_id": "2",
        "dataset_id": "track2_role_ipc_test_v1",
        "source_csvs": {
            "selected_reels": str(args.selected_reels.relative_to(SCRIPT_DIR)),
            "selected_scenes": str(args.selected_scenes.relative_to(SCRIPT_DIR)),
            "scene_metadata": str(args.scene_metadata.relative_to(SCRIPT_DIR)),
        },
        "gap_seconds": args.gap_seconds,
        "episode_count": len(manifest_entries),
        "episodes": manifest_entries,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {len(manifest_entries)} reel wavs")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
