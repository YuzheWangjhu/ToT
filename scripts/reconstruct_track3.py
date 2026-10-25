#!/usr/bin/env python3
"""Reconstruct the exact Track 3 recordings and audited tasks from source files."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'track2_dataset'))
import build_selected_reel_audio as audio_builder
import validate_selected_reels
import validate_track2_dataset_v2 as validator
from verify_datasets import check_files, sha256_file, verify

FROZEN = ROOT / 'track2_dataset/generated_data/track2_role_ipc_test_v2/dataset'

def reconstruct(dataset_root: Path, work_dir: Path, output_dir: Path) -> dict:
    verify(ROOT)
    if work_dir.exists():
        raise FileExistsError(f"Work directory already exists: {work_dir}")
    if (output_dir / "episodes").exists():
        raise FileExistsError(f"Output audio directory already exists: {output_dir}")
    source_manifest = json.loads((ROOT / 'provenance/dataset_source_files.json').read_text())
    check_files(dataset_root, source_manifest['files'])
    frozen = json.loads((FROZEN / 'validation_report.json').read_text())
    if not work_dir.resolve().is_relative_to(ROOT):
        raise ValueError('Work directory must be inside this release repository')
    work_dir.mkdir(parents=True, exist_ok=False)
    v1_audio = work_dir / 'v1/episodes'
    v1_audio.mkdir(parents=True)
    validate_selected_reels.validate(argparse.Namespace(
        selected_reels=audio_builder.DEFAULT_SELECTED_REELS,
        selected_scenes=audio_builder.DEFAULT_SELECTED_SCENES,
        audited_candidates=validate_selected_reels.DEFAULT_AUDITED_CANDIDATES,
        gap_seconds=audio_builder.DEFAULT_GAP_SECONDS,
    ))
    reels = pd.read_csv(audio_builder.DEFAULT_SELECTED_REELS, keep_default_na=False)
    scenes = pd.read_csv(audio_builder.DEFAULT_SELECTED_SCENES, keep_default_na=False)
    scene_metadata = pd.read_csv(audio_builder.DEFAULT_SCENE_METADATA, keep_default_na=False)
    scene_lookup = scene_metadata.set_index('scene_id', drop=False)
    manifest = json.loads((validator.V1_DIR / 'manifest.json').read_text())
    expected = {entry['reel_id']: entry for entry in manifest['episodes']}
    for _, reel in reels.sort_values('reel_id').iterrows():
        reel_id = str(reel['reel_id'])
        entry = audio_builder.build_reel_audio(
            reel, scenes[scenes['reel_id'].eq(reel_id)], scene_lookup, dataset_root,
            v1_audio / f'{reel_id}.wav', manifest['gap_seconds'], manifest_root=work_dir,
        )
        entry['audio_path'] = expected[reel_id]['audio_path']
        if entry != expected[reel_id]:
            raise ValueError(f'{reel_id}: reconstructed v1 episode differs from frozen metadata')
        print(f'Reconstructed {reel_id}', flush=True)
    built = work_dir / 'v2'
    subprocess.run([
        sys.executable, '-B', str(validator.BUILDER), '--dataset-root', str(dataset_root),
        '--output-dir', str(built), '--v1-audio-dir', str(v1_audio), '--audio-mode', 'hardlink',
    ], check=True, cwd=ROOT)
    report = validator.validate_dataset(built, dataset_root, v1_audio)
    if report['artifact_sha256'] != frozen['artifact_sha256']:
        raise ValueError('Reconstruction differs from the evaluated dataset')
    if report['deterministic_build_fingerprint'] != frozen['deterministic_build_fingerprint']:
        raise ValueError('Reconstruction fingerprint differs from the evaluated dataset')
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = ['manifest.json', 'task_instances.jsonl', 'task_answers.jsonl', 'candidate_provenance.jsonl', 'validation_report.json']
    for name in paths:
        target = output_dir / name
        if target.exists():
            if sha256_file(target) != sha256_file(FROZEN / name):
                raise ValueError(f'Refusing to replace different metadata: {target}')
        else:
            shutil.copyfile(FROZEN / name, target)
    episode_dir = output_dir / 'episodes'
    episode_dir.mkdir(exist_ok=False)
    for source in sorted((built / 'episodes').glob('*.wav')):
        (episode_dir / source.name).hardlink_to(source)
    return {'status': 'passed', 'output_dir': str(output_dir),
            'episode_count': report['episode_count'], 'task_count': report['task_count'],
            'deterministic_build_fingerprint': report['deterministic_build_fingerprint']}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, default=ROOT / '.scratch/track3_reconstruction')
    parser.add_argument('--output-dir', type=Path, default=FROZEN)
    args = parser.parse_args()
    print(json.dumps(reconstruct(args.dataset_root.resolve(), args.work_dir.resolve(), args.output_dir.resolve()), indent=2))

if __name__ == '__main__':
    main()
