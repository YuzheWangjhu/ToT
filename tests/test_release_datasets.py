"""Offline regression checks for frozen inputs and source audio mixing."""
from pathlib import Path
import hashlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'track2_dataset'))
from verify_datasets import check_files, verify
import check_release
import build_selected_reel_audio as audio_builder
import numpy as np
import pandas as pd
import soundfile as sf

class ReleaseDatasetTests(unittest.TestCase):
    def test_frozen_metadata_and_expected_task_counts(self):
        report = verify(ROOT)
        self.assertEqual(report['episode_counts'], {'1a': 10, '1b': 10, '1c': 240, '2': 22})
        self.assertEqual(report['track3_task_count'], 418)
        self.assertEqual(report['scope'], 'metadata_only')
        self.assertEqual(report['audio_files_verified'], 0)

    def test_artifact_verifier_rejects_mismatch_and_path_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'record').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                check_files(root, [{'path': 'record', 'sha256': '0' * 64}])
            with self.assertRaisesRegex(ValueError, 'escapes root'):
                check_files(root, [{'path': '../record', 'sha256': '0' * 64}])

    def test_source_mixing_preserves_equal_gain_and_rejects_rate_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sf.write(root / 'one.wav', np.array([0.25, 0.5], dtype=np.float32), 24000, subtype='FLOAT')
            sf.write(root / 'two.wav', np.array([0.5, -0.25], dtype=np.float32), 24000, subtype='FLOAT')
            scene = pd.Series({'participant1_wav_relpath': 'one.wav', 'participant2_wav_relpath': 'two.wav'})
            audio, rate, _ = audio_builder.mix_scene_audio('scene', scene, root)
            np.testing.assert_array_equal(audio, [0.375, 0.125])
            self.assertEqual(rate, 24000)
            with self.assertRaisesRegex(ValueError, 'sample rates differ'):
                audio_builder.assert_same_audio_shape('scene', audio, audio, 24000, 16000)

    def test_frozen_synthetic_audio_is_included(self):
        contract = json.loads((ROOT / 'provenance/dataset_audio.json').read_text())
        self.assertIsNone(contract['synthetic_download_url'])
        self.assertEqual(contract['synthetic_distribution_status'], 'included_in_repository')
        self.assertEqual(len([row for row in contract['files'] if row['group'] == 'synthetic']), 260)
        self.assertEqual(len([row for row in contract['files'] if row['group'] == 'seamless']), 22)
        report = verify(ROOT, audio=True, group='synthetic')
        self.assertEqual(report['audio_files_verified'], 260)
        self.assertEqual(report['scope'], 'metadata_and_audio')

    def test_release_inventory_requires_frozen_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance').mkdir()
            payload = b'frozen synthetic audio'
            expected = {'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}
            contract = {'files': [{'path': 'clip.wav', 'group': 'synthetic', **expected}]}
            (root / 'provenance/dataset_audio.json').write_text(json.dumps(contract))
            with patch.object(check_release, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Missing frozen synthetic audio'):
                    check_release.inventory()
                (root / 'clip.wav').write_bytes(payload)
                self.assertEqual(check_release.inventory()['clip.wav'], expected)
                (root / 'clip.wav').write_bytes(b'changed')
                with self.assertRaisesRegex(ValueError, 'Frozen synthetic audio mismatch'):
                    check_release.inventory()

    def test_release_inventory_rejects_unlisted_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance').mkdir()
            (root / 'provenance/dataset_audio.json').write_text(json.dumps({'files': []}))
            (root / 'source_corpus.wav').write_bytes(b'unlisted recording')
            with patch.object(check_release, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'Unexpected private/generated artifact'):
                    check_release.inventory()

    def test_release_check_needs_only_package_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance').mkdir()
            (root / 'provenance/dataset_audio.json').write_text(json.dumps({'files': []}))
            manifest = root / 'provenance/release_files.json'
            with patch.object(check_release, 'ROOT', root), \
                 patch.object(check_release, 'MANIFEST', manifest), redirect_stdout(io.StringIO()):
                with patch.object(sys, 'argv', ['check_release.py', '--write-manifest']):
                    check_release.main()
                with patch.object(sys, 'argv', ['check_release.py']):
                    check_release.main()
                self.assertEqual(set(json.loads(manifest.read_text())['files']), {'provenance/dataset_audio.json'})
                (root / 'credentials.env').write_text('EXAMPLE_KEY=test-fixture\n')
                with self.assertRaisesRegex(ValueError, 'Unexpected private/generated artifact'):
                    check_release.inventory()

if __name__ == '__main__':
    unittest.main()
