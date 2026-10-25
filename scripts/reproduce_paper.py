#!/usr/bin/env python3
"""Reproduce ToT v5 tables and figures offline from frozen result records."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
for variable, subdirectory in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "cache"), ("MPLCONFIGDIR", "mpl")):
    directory = ROOT / ".scratch" / "analysis" / subdirectory
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(directory)
os.environ["MPLBACKEND"] = "Agg"

from analysis.frozen_inputs import extract_results, sha256, verify_inputs


def verify_rendering_environment() -> None:
    """Reject a different rasterizer before extracting inputs or computing figures."""
    import importlib.metadata
    import matplotlib
    from matplotlib import ft2font

    expected = ("3.10.7", "2.14.1", "system", "1.4.9")
    actual = (matplotlib.__version__, ft2font.__freetype_version__,
              ft2font.__freetype_build_type__, importlib.metadata.version("kiwisolver"))
    if actual != expected:
        raise RuntimeError(
            f"Paper rendering needs Matplotlib/FreeType/build/kiwisolver {expected}, got {actual}. "
            "Create the environment in requirements/rendering-linux.yml; the same-version "
            "pip Matplotlib wheel bundles a different FreeType engine."
        )


def run_notebook() -> None:
    """Execute all cells in source order, including supplementary analyses."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    notebook_path = ROOT / "analysis/analysis_v5.ipynb"
    notebook = json.loads(notebook_path.read_text())
    namespace = {"__file__": str(notebook_path), "__name__": "__analysis_v5_notebook__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        exec(compile(source, f"{notebook_path}:cell_{index}", "exec"), namespace)
        namespace["display"] = lambda *_args, **_kwargs: None
        namespace["ipython_display"] = lambda *_args, **_kwargs: None
        plt.show = lambda *_args, **_kwargs: None
        plt.close("all")
        print(f"Executed notebook cell {index}", flush=True)


def compare_expected() -> dict:
    import pandas as pd
    from pandas.testing import assert_frame_equal
    from PIL import Image, ImageChops

    manifest = json.loads((ROOT / "provenance/analysis_expected.json").read_text())
    expected_root = ROOT / manifest["reference_root"]
    output_root = ROOT / "analysis/outputs/paper_plot_v5"
    rows = []
    for artifact in manifest["artifacts"]:
        expected = expected_root / artifact["path"]
        output = output_root / artifact["path"]
        if sha256(expected) != artifact["sha256"]:
            raise ValueError(f"Frozen reference changed: {expected}")
        if not output.is_file():
            raise FileNotFoundError(f"Analysis did not generate expected artifact: {output}")
        if expected.suffix == ".csv":
            assert_frame_equal(pd.read_csv(expected), pd.read_csv(output), check_dtype=False, rtol=1e-10, atol=1e-12)
            comparison = "numerically_equal"
        else:
            with Image.open(expected) as left, Image.open(output) as right:
                if left.size != right.size or ImageChops.difference(left.convert("RGB"), right.convert("RGB")).getbbox() is not None:
                    raise AssertionError(f"Figure pixels differ from frozen v5 output: {artifact['path']}")
            comparison = "pixel_equal"
        rows.append({"path": artifact["path"], "comparison": comparison, "byte_equal": sha256(expected) == sha256(output)})
    return {"analysis_version": "v5", "source_reproduction_status": "passed", "artifacts": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract-only", action="store_true", help="Extract and verify frozen results and metadata without running analysis")
    args = parser.parse_args()
    os.chdir(ROOT)
    if not args.extract_only:
        verify_rendering_environment()
    print("Extracting and verifying the frozen result archive...", flush=True)
    extract_results(ROOT)
    verify_inputs(ROOT)
    if args.extract_only:
        print("Frozen result records and dataset metadata verified.")
        return
    print("Executing the complete v5 analysis notebook...", flush=True)
    run_notebook()
    print("Generating the manuscript composite figures...", flush=True)
    from analysis import create_v5_plot
    create_v5_plot.main()
    report = compare_expected()
    report["status"] = "passed"
    import importlib.metadata
    report["environment"] = {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "matplotlib", "seaborn", "Pillow", "ipython")}
    report["python"] = sys.version.split()[0]
    inputs = json.loads((ROOT / "provenance/analysis_inputs.json").read_text())
    report["frozen_input_counts"] = {
        kind: sum(member["kind"] == kind for member in inputs["members"])
        for kind in ("track1_metrics", "track1_results", "track2_results")
    }
    report_path = ROOT / "analysis/outputs/reproduction_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified {len(report['artifacts'])} frozen paper artifacts; report: {report_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
