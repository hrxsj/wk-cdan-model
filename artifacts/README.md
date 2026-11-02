# Experiment artifacts

These files are historical outputs copied from the research workspace. The status below distinguishes results that match a reported value from results whose connection to the exact original paper run is not established.

## CWRU independent validation

`cwru_independent_validation/` contains 60 task/seed rows for the strict-holdout WK-CDAN validation. Their aggregate is 99.0353% mean accuracy with a 0.8295 percentage-point sample standard deviation, matching the manuscript's displayed 99.04 +/- 0.83%. The source run note identifies this as an independent validation batch. The archived metadata does not establish that these are the original runs used to prepare the manuscript table.

The original launch command, seeds, task list, split, epoch policy, batch size, and launch-time runner hash are recorded in `cwru_independent_validation/PROTOCOL_NOTE.md`. The release-tree runner has a different SHA-256 from the recorded launch-time hash, so the command and result are published with that provenance limitation. The result CSV contains per-run dataset fingerprints; no CWRU signal data are included.

## PU historical results

`pu_historical_runs/` contains the available raw CSV files for the 200-epoch multi-seed run, 200-epoch same-condition run, and two 500-epoch OC2 transfer runs, plus the manuscript summary file. The manuscript WK-CDAN value can be reconstructed exactly from these records by starting with the 200-epoch multi-seed rows and replacing all four cross-condition directions involving OC2 with the corresponding 500-epoch rows. Run:

```bash
python artifacts/pu_historical_runs/reconstruct_paper_summary.py
```

The script writes `reconstructed_paper_rows.csv` and `reconstructed_summary.csv`. This reconstruction has 45 rows across five seeds and yields 95.5403 +/- 0.3046% for the six cross-condition directions. It is a mixed-epoch historical reconstruction, not a single uniform-protocol experiment. The separate source CSV files are the raw records; the reconstructed CSV is derived and labeled accordingly.

The PU baseline CSVs and summaries are also included. Their cross-condition means and standard deviations match the manuscript's WDCNN, Vanilla KAN, ResNet-18, DANN + ResNet-18, and CDAN + ResNet-18 entries. Each method's original run output is retained as a separate file; the baseline rows were not combined into a new raw-result file.

The available PU run code is `pu_wkcdan_multiseed.py` and `experiments/pu/run_pu_wkcdan_same_200.py`. The archived outputs do not contain a launch note with the exact commands or the exact source revision for every 500-epoch job. Those details cannot be inferred from the result CSVs alone. PU signal data are not included.

## Seeds and data

The released result files retain the recorded seeds (42-46) and data fingerprints. Neither CWRU nor PU raw signal files are included. Dataset record inventories and preparation instructions are in the repository's `cwru/metadata.txt` and `docs/experiments.md`.
