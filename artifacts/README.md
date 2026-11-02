# Experiment artifacts

This directory contains archived experiment outputs, run configurations, and result summaries for the CWRU and PU datasets.

## CWRU independent validation

`cwru_independent_validation/` contains 60 task/seed rows from a strict-holdout WK-CDAN validation run. The aggregate is 99.0353% mean accuracy with a 0.8295 percentage-point sample standard deviation. The run configuration, seeds, task list, launch command, and script hash are recorded in `cwru_independent_validation/PROTOCOL_NOTE.md`.

The result CSV contains per-run dataset fingerprints. The repository's CWRU record inventory and preparation instructions are in `cwru/metadata.txt` and `docs/experiments.md`.

## PU historical results

`pu_historical_runs/` contains raw CSV files for the 200-epoch multi-seed run, 200-epoch same-condition run, two 500-epoch OC2 transfer runs, and the manuscript summary. The reconstruction script aggregates the archived task/seed rows and reproduces the manuscript's six-direction cross-condition summary. Run:

```bash
python artifacts/pu_historical_runs/reconstruct_paper_summary.py
```

The script writes `reconstructed_paper_rows.csv` and `reconstructed_summary.csv`, containing 45 rows across five seeds and a cross-condition mean of 95.5403% with a sample standard deviation of 0.3046%. The source CSVs identify the run and epoch group for each archived result.

The PU baseline CSVs and summaries are also included. Their cross-condition means and standard deviations match the manuscript's WDCNN, Vanilla KAN, ResNet-18, DANN + ResNet-18, and CDAN + ResNet-18 entries. Each method's original run output is retained as a separate file; the baseline rows were not combined into a new raw-result file.

The PU run code is `pu_wkcdan_multiseed.py` and `experiments/pu/run_pu_wkcdan_same_200.py`. The repository's PU record mapping and preparation instructions are in `docs/experiments.md`.

## Seeds and data

The released result files retain the recorded seeds (42-46) and data fingerprints. Dataset record inventories and preparation instructions are in `cwru/metadata.txt` and `docs/experiments.md`.
