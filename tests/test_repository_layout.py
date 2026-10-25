"""Check configuration paths and standalone maintenance entry points offline."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import check_release
from tot_benchmark import api_error_rerun
from tot_benchmark.adapters import load_adapter_config

ROOT = Path(__file__).resolve().parents[1]


class RepositoryLayoutTests(unittest.TestCase):
    def test_shipped_configs_resolve_credentials_from_repository_root(self):
        paths = sorted((ROOT / 'configs').glob('*/config.json'))
        self.assertEqual(len(paths), 10)
        for path in paths:
            with self.subTest(config=path.parent.name):
                original = path.read_bytes()
                config = load_adapter_config(path)
                if 'api_key_env_path' in config:
                    self.assertEqual(config['api_key_env_path'], str(ROOT / '.env/api_keys.env'))
                self.assertEqual(path.read_bytes(), original)

    def test_custom_config_location_does_not_change_credential_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.json'
            for credential, expected in (
                ('.env/custom.env', str(ROOT / '.env/custom.env')),
                (str(Path(directory) / 'private.env'), str(Path(directory) / 'private.env')),
            ):
                with self.subTest(credential=credential):
                    path.write_text(json.dumps({'api_key_env_path': credential}))
                    self.assertEqual(load_adapter_config(path)['api_key_env_path'], expected)

    def test_maintenance_entrypoints_work_outside_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('archive_api_error_results.py', 'rerun_api_errors.py'):
                with self.subTest(script=name):
                    result = subprocess.run(
                        [sys.executable, '-B', str(ROOT / 'scripts' / name), '--help'],
                        cwd=directory, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('--dry-run', result.stdout)

    def test_retry_command_uses_config_separate_from_results(self):
        config_path = ROOT / 'configs/gpt_audio/config.json'
        model_dir = ROOT / 'runs/arbitrary_experiment'
        spec = SimpleNamespace(track_id='1a', condition='active')
        adapter = object()
        with patch.object(api_error_rerun, 'find_failed_run_specs', side_effect=[[spec], []]), \
             patch.object(api_error_rerun, 'adapter_from_config', return_value=adapter) as create, \
             patch.object(api_error_rerun, '_run_selected_specs') as run, \
             patch.object(api_error_rerun, '_rebuild_track_outputs') as rebuild, \
             redirect_stdout(io.StringIO()):
            status = api_error_rerun.main([
                '--model-dir', str(model_dir), '--adapter-config', str(config_path),
                '--tracks', '1a',
            ])
        self.assertEqual(status, 0)
        create.assert_called_once_with(load_adapter_config(config_path))
        self.assertEqual(run.call_args.kwargs['results_root'], model_dir / 'results')
        self.assertIs(run.call_args.kwargs['adapter'], adapter)
        self.assertEqual(rebuild.call_args.kwargs['results_root'], model_dir / 'results')

    def test_inventory_excludes_original_result_layout_with_nested_configs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'provenance').mkdir()
            (root / 'provenance/dataset_audio.json').write_text(json.dumps({'files': []}))
            config = root / 'configs/model/config.json'
            config.parent.mkdir(parents=True)
            config.write_text('{}')
            result = root / 'model/results/episode.json'
            result.parent.mkdir(parents=True)
            result.write_text('{}')
            (root / 'model/notes.txt').write_text('Included metadata')
            with patch.object(check_release, 'ROOT', root):
                inventory = check_release.inventory()
            self.assertIn('configs/model/config.json', inventory)
            self.assertIn('model/notes.txt', inventory)
            self.assertNotIn('model/results/episode.json', inventory)


if __name__ == '__main__':
    unittest.main()
