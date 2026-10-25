#!/usr/bin/env python3
"""Verify frozen metadata, or exact evaluation audio when explicitly requested."""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def check_files(root: Path, records: list[dict]) -> None:
    for record in records:
        path = (root / record['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Artifact path escapes root: {record['path']}")
        if sha256_file(path) != record['sha256']:
            raise ValueError(f"SHA-256 mismatch: {record['path']}")

def verify(root: Path, audio: bool = False, group: str = 'all') -> dict:
    metadata = json.loads((root / 'provenance/dataset_files.json').read_text())['files']
    check_files(root, metadata)
    counts = {}
    for track, expected in [('1a', 10), ('1b', 10), ('1c', 240)]:
        manifest = json.loads((root / f'generated_data/theory_of_time_v1/track_{track}/dataset/manifest.json').read_text())
        counts[track] = len(manifest['episodes'])
        if counts[track] != expected:
            raise ValueError(f'Track {track}: expected {expected} episodes')
    track3 = root / 'track2_dataset/generated_data/track2_role_ipc_test_v2/dataset'
    manifest = json.loads((track3 / 'manifest.json').read_text())
    report = json.loads((track3 / 'validation_report.json').read_text())
    if len(manifest['episodes']) != 22 or manifest['task_count'] != 418:
        raise ValueError('Track 3: expected 22 reels and 418 tasks')
    tasks = [json.loads(line) for line in (track3 / 'task_instances.jsonl').read_text().splitlines()]
    expected = {'2A.1': 110, '2A.2': 176, '2A.3': 44, '2B.1': 44, '2B.2': 44}
    if Counter(row['family'] for row in tasks) != Counter(expected):
        raise ValueError('Track 3 task family counts differ')
    for name in ['manifest.json', 'task_instances.jsonl', 'task_answers.jsonl', 'candidate_provenance.jsonl']:
        if sha256_file(track3 / name) != report['artifact_sha256'][name]:
            raise ValueError(f'Track 3 original validation-report mismatch: {name}')
    fingerprint = hashlib.sha256(json.dumps(report['artifact_sha256'], sort_keys=True).encode()).hexdigest()
    if fingerprint != report['deterministic_build_fingerprint']:
        raise ValueError('Track 3 frozen fingerprint is inconsistent')
    audio_records = json.loads((root / 'provenance/dataset_audio.json').read_text())['files']
    selected = [r for r in audio_records if group == 'all' or r['group'] == group]
    if audio:
        check_files(root, selected)
    return {'status': 'passed', 'scope': 'metadata_and_audio' if audio else 'metadata_only',
            'metadata_files': len(metadata), 'audio_files_verified': len(selected) if audio else 0,
            'episode_counts': {**counts, '2': 22}, 'track3_task_count': len(tasks)}

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audio', action='store_true', help='Require every selected frozen WAV and verify its exact hash.')
    parser.add_argument('--group', choices=['all', 'synthetic', 'seamless'], default='all')
    args = parser.parse_args()
    if args.group != 'all' and not args.audio:
        parser.error('--group requires --audio')
    print(json.dumps(verify(args.root.resolve(), args.audio, args.group), indent=2))

if __name__ == '__main__':
    main()
