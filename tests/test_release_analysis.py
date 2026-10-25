"""Validate release archive confinement and immutable input selection."""
import importlib.util
import io
import hashlib
import json
import tarfile
import tempfile
import unittest
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("frozen_inputs", ROOT / "analysis/frozen_inputs.py")
frozen = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(frozen)


class FrozenInputsTests(unittest.TestCase):
    def test_paths_cannot_escape_release(self):
        for relative in ("../outside.json", "/absolute.json", "a/../../outside.json"):
            with self.subTest(path=relative), self.assertRaises(ValueError):
                frozen._destination(ROOT, relative)

    def test_selected_files_are_explicit_and_unchanged(self):
        scratch = ROOT / ".scratch/analysis/tests"
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as temporary:
            root = Path(temporary)
            (root / "provenance").mkdir()
            archive_path = root / "results.tar.gz"
            relative = "model/results/episode_metrics.csv"
            payload = b"episode_id,score\nepisode_1,0.5\n"
            with tarfile.open(archive_path, "w:gz") as archive:
                member = tarfile.TarInfo(relative)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            specification = {"path": relative, "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            manifest = {
                "archive": {"path": archive_path.name, "size": archive_path.stat().st_size, "sha256": frozen.sha256(archive_path)},
                "members": [specification],
                "track1_metric_files": [relative],
            }
            (root / "provenance/analysis_inputs.json").write_text(json.dumps(manifest))
            frozen.extract_results(root)
            selected = root / relative
            self.assertEqual(selected.read_bytes(), payload)
            (selected.parent / "unrelated.csv").write_text("must not enter paper analysis")
            self.assertEqual(frozen.frozen_input_paths(root, "track1_metric_files"), [selected])
            selected.write_text("conflicting local result")
            with self.assertRaisesRegex(ValueError, "differs"):
                frozen.extract_results(root)

    def test_reproduction_without_manuscript_preserves_reference_checks(self):
        spec = importlib.util.spec_from_file_location("reproduce_paper", ROOT / "scripts/reproduce_paper.py")
        reproduce = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ):
            spec.loader.exec_module(reproduce)
        analysis_stub = ModuleType("analysis")
        analysis_stub.create_v5_plot = ModuleType("analysis.create_v5_plot")
        analysis_stub.create_v5_plot.main = Mock()
        for mismatch in (False, True):
            with self.subTest(reference_mismatch=mismatch), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "provenance").mkdir()
                (root / "provenance/analysis_inputs.json").write_text(json.dumps({"members": []}))
                (root / "analysis/outputs").mkdir(parents=True)
                reference_result = {"source_reproduction_status": "passed", "artifacts": []}
                with patch.object(reproduce, "ROOT", root), \
                     patch.object(reproduce, "verify_rendering_environment"), \
                     patch.object(reproduce, "extract_results"), \
                     patch.object(reproduce, "verify_inputs"), \
                     patch.object(reproduce, "run_notebook"), \
                     patch.object(reproduce, "compare_expected", return_value=reference_result) as compare, \
                     patch.dict(sys.modules, {"analysis": analysis_stub}), \
                     patch("importlib.metadata.version", return_value="test"), \
                     patch("os.chdir"), patch.object(sys, "argv", ["reproduce_paper.py"]), \
                     redirect_stdout(io.StringIO()):
                    if mismatch:
                        compare.side_effect = AssertionError("Figure pixels differ from frozen v5 output")
                        with self.assertRaisesRegex(AssertionError, "Figure pixels differ"):
                            reproduce.main()
                        self.assertFalse((root / "analysis/outputs/reproduction_report.json").exists())
                    else:
                        reproduce.main()
                        report = json.loads((root / "analysis/outputs/reproduction_report.json").read_text())
                        self.assertEqual(report["status"], "passed")
                        self.assertNotIn("manuscript_tables", report)
                    compare.assert_called_once()
                self.assertFalse((root / "paper").exists())


if __name__ == "__main__":
    unittest.main()
