# Reproducing the reported analysis

This workflow uses the archived evaluation outputs and dataset metadata supplied with the package. It requires no model API calls, audio generation, or source-corpus recordings.

## Verify the repository

Run commands from the repository root. The integrity checks use the Python standard library and need no model credentials or downloads:

```bash
python -B scripts/check_release.py
python -B scripts/verify_datasets.py --audio --group synthetic
```

The first checks all inventoried files against `provenance/release_files.json`; the second checks frozen metadata and synthetic audio. See [datasets](datasets.md) for the verification scopes and source-audio reconstruction.

## Environment and command

For exact figure rendering on Linux x86_64, create the pinned conda-forge environment from the repository root:

```bash
PYTHONNOUSERSITE=1 micromamba create --yes --prefix .scratch/render_env --file requirements/rendering-linux.yml
micromamba run --prefix .scratch/render_env python -s -B scripts/reproduce_paper.py
```

The environment pins package builds and native font-rendering libraries. Do not install `requirements/analysis-lock.txt` into this environment: the pip Matplotlib 3.10.7 wheel uses a different FreeType engine. Matching Python package versions alone is insufficient for identical figure pixels. The script checks the required Matplotlib, FreeType, and kiwisolver versions before computing results. See `provenance/rendering_environment.json` for the recorded rendering configuration.

## What is verified

The command:

1. Verifies and extracts `artifacts/paper_results_v5.tar.gz`.
2. Checks the selected model outputs and dataset metadata against their input manifests.
3. Executes the code cells in `analysis/analysis_v5.ipynb` in source order.
4. Runs `analysis/create_v5_plot.py` to assemble the quantitative figures.
5. Compares all 40 reference CSV tables numerically and all 20 reference PNGs pixel by pixel.

The comparison covers the supplied reference outputs in `analysis/expected/paper_plot_v5/`. These include the five main quantitative figures and supporting analyses. No manuscript source files are needed. Scoring, model eligibility, aggregation, confidence intervals, clustered resampling, random seeds, and multiplicity correction are defined by the supplied analysis code and [protocol](protocol.md).

## Outputs and success criteria

Generated tables and figures are saved under `analysis/outputs/paper_plot_v5/`. The report is `analysis/outputs/reproduction_report.json`.

A successful run exits with status zero and reports `status: passed`, `source_reproduction_status: passed`, and 60 artifact comparisons. CSV comparisons report `numerically_equal`; PNG comparisons report `pixel_equal`. The report also records byte equality, environment versions, and the number of frozen inputs used. Numerical or pixel differences cause the command to fail. Compare the regenerated tables with the separately submitted results when reviewing the reported findings.

## Inspect inputs without running the analysis

To extract and verify the supplied results using Python 3.10:

```bash
python -B scripts/reproduce_paper.py --extract-only
```

Existing extracted files must match their recorded hashes. Conflicting files cause an error and are not overwritten. Extracted records retain the original model/result directory layout; the analysis selects inputs through the explicit lists in `provenance/analysis_inputs.json`. Additional result directories cannot silently enter the analysis.

After extraction, the notebook can also be opened from the repository root. Its saved outputs are cleared. Required task metadata and denominator files are checked before analysis.

## Input coverage

The analysis selects 48 metric CSV files across ten model configurations. The archived records include continuous-time diagnostics; the primary comparison uses binned time and the Discovery prior for the controlled tracks. Long-horizon records retain the selected active, full-context, and same-session conditions, including unsuccessful or unsupported runs. The analysis applies the recorded eligibility rules rather than treating missing scores as successes.

Controlled and conversational records contain the identifiers, checkpoint scores, probe metrics, and final rasters used by the notebook. Long-horizon records also retain the relevant query histories and belief ledgers. The archive supports reproduction of the analysis; it is not a replay of complete provider conversations.

`provenance/analysis_inputs.json` records original-source and exported-input hashes. `provenance/analysis_metadata.json` identifies the ground-truth metadata, including 418 long-horizon tasks across 22 recordings. `provenance/analysis_expected.json` records the hashes of the 60 reference outputs. These inventories allow the supplied inputs and references to be checked independently of new model runs.

## Recorded checks

`provenance/analysis_reproduction.json` records the successful full reproduction of 40 numerical tables and 20 pixel-identical figures, including environment versions and input counts. `provenance/release_validation.json` records offline implementation and integrity checks; `provenance/dataset_validation.json` records source reconstruction and the evaluated dataset fingerprint. Each record applies to its stated scope. Historical reference corrections and the manuscript rounding discrepancy are summarized in [reproduction notes](reproduction_notes.md).

Local `.scratch/` files, extracted model result directories, new `runs/`, and regenerated `analysis/outputs/` are excluded from Git and the repository inventory. Archived results retain their original model/result paths; moving configuration files into `configs/` does not change the analysis input selection.

After intentional repository edits, regenerate the inventory with `python -B scripts/check_release.py --write-manifest`, then run the check without that option. Offline validation does not guarantee identical outputs from new model inference.
