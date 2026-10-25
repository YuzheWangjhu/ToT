#!/usr/bin/env python3
"""Exercise dataset construction and active evaluation with tones and a dummy model."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.audio_fixtures import write_audio_fixture
from tot_benchmark.adapters import DummyAdapter
from tot_benchmark.core import ACTIVE, BenchmarkConfig, build_dataset, run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.output_root is None:
        scratch = ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="smoke-", dir=scratch))
    else:
        output = args.output_root.resolve()
        output.mkdir(parents=True, exist_ok=False)
    assets = write_audio_fixture(output / "tone_assets")
    config = BenchmarkConfig(num_episodes=1, seed=20260413)
    dataset = output / "dataset"
    build_dataset(assets, dataset, config)
    unknown = "\n".join(f"{index}: ??" for index in range(config.num_bins))
    outputs = ["STOP", unknown, "UNEXPLORED", "UNEXPLORED", "UNEXPLORED"] * 3
    results = run_benchmark(
        dataset_root=dataset,
        results_root=output / "results",
        adapter=DummyAdapter(outputs, model_id="offline-tone-smoke"),
        config=config,
        conditions=[ACTIVE],
    )
    assert len(results) == 3, "Expected one result for each prior-information regime"
    assert (output / "results" / "episode_metrics.csv").is_file()
    assert (output / "results" / "summary.json").is_file()
    print(f"Offline smoke passed: three active-condition results in {output}")
    print("These generated tones test the software; they are not paper benchmark audio.")


if __name__ == "__main__":
    main()
