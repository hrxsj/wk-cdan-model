# Experiment workflows

Run commands from the repository root after installing `requirements.txt`.

## Data and preprocessing

`pipeline.py prepare --dataset cwru --data-root Data/CWRU` downloads the forty selected ten-class, four-load CWRU records. Files are arranged as `Data/CWRU/<NormalBaseline or 12DriveEndFault>/<RPM>/<class>.mat`. The record URLs are in `cwru/metadata.txt`.

For PU, extract downloaded records under `Data/PU/<bearing ID>/`. The class mapping is Normal (`K001`, `K002`), OuterRace (`KA01`, `KA05`), and InnerRace (`KI01`, `KI05`). The benchmark conditions are `N15_M07_F10`, `N15_M01_F10`, and `N15_M07_F04`. The loader reads the vibration field and decimates by five. `pipeline.py prepare --dataset pu --data-root Data/PU` creates the partition inventory.

Both standard workflows split continuous records into chronological 80% training and 20% test segments, then extract windows independently within each segment. CWRU uses 1024-point windows; PU uses 2048-point windows. Overlap is 50%, and each window is standardized using its own mean and standard deviation.

## Main model and six baselines

```bash
python pipeline.py train --dataset cwru --suite all --data-root Data/CWRU --output-dir runs/cwru_comparison
python pipeline.py train --dataset pu --suite all --data-root Data/PU --output-dir runs/pu_comparison
```

The CWRU command runs WK-CDAN and six core baselines. The PU command runs paired Conv-ReLU/WKAN front-end configurations and six core baselines. Each suite stores its results in a separate subdirectory. The combined report is written to `report/`.

The source-only baselines are WDCNN, compact 1D ResNet-18, and cubic B-spline KANConv. Adaptation baselines are DANN, CDAN, and DSAN/LMMD with the compact 1D ResNet-18 backbone. The classifier width follows the dataset class count. Method definitions and optimization parameters are recorded in run manifests.

Modules can also be launched individually:

```bash
python -m experiments.pu.pu_frontend_ablation --data-root Data/PU --output-dir runs/pu_frontend
python -m experiments.pu.pu_core_baselines_strict --data-root Data/PU --output-dir runs/pu_core
python -m experiments.pu.pu_basis_ablation --data-root Data/PU --output-dir runs/pu_basis
python -m experiments.cwru.cwru_source_only_core_strict --data-root Data/CWRU --output-dir runs/cwru_source
python -m experiments.cwru.cwru_uda_core_strict --data-root Data/CWRU --output-dir runs/cwru_uda
python -m experiments.cwru.cwru_noise_baselines_strict --data-root Data/CWRU --output-dir runs/cwru_noisy_baselines
```

## Ablations and robustness

```bash
python pipeline.py train --dataset cwru --suite frontend --data-root Data/CWRU --output-dir runs/cwru_frontend
python pipeline.py train --dataset cwru --suite basis --data-root Data/CWRU --output-dir runs/cwru_basis
python pipeline.py train --dataset cwru --suite structure --data-root Data/CWRU --output-dir runs/cwru_structure
python pipeline.py train --dataset cwru --suite factorial --snr-db=-3 --data-root Data/CWRU --output-dir runs/cwru_factorial_noisy
python pipeline.py train --dataset cwru --suite loss --snr-db=-5 --data-root Data/CWRU --output-dir runs/cwru_loss
python pipeline.py train --dataset cwru --suite wavelet --data-root Data/CWRU --output-dir runs/cwru_wavelets
python pipeline.py train --dataset cwru --suite noise --data-root Data/CWRU --output-dir runs/cwru_snr
python pipeline.py train --dataset pu --suite frontend --data-root Data/PU --output-dir runs/pu_frontend
python pipeline.py train --dataset pu --suite basis --data-root Data/PU --output-dir runs/pu_basis
```

CWRU noise and wavelet suites accept `--snr-levels=clean,-1,-3,-5`. Other CWRU ablations use clean input by default; `--snr-db` selects an injected noise condition. The PU basis study defaults to OC2→OC1 and accepts one alternative task pair through `--tasks`.

Additional CWRU modules provide exact-power AWGN stress and structural studies:

```bash
python -m experiments.cwru.cwru_minus5db_stress_strict --data-root Data/CWRU --output-dir runs/cwru_minus5db
python -m experiments.cwru.cwru_factorial_minus3db_strict --data-root Data/CWRU --output-dir runs/cwru_factorial_minus3db
```

The standard CWRU trainer injects AWGN after per-window normalization. The noise-baseline and stress modules record their exact-power noise injection and normalization settings in their own manifests.

## Runtime configuration

Five initialization values are generated at runtime by default. Users can choose their own values with `--seeds` on the unified CLI, or configure `WKCDAN_RUN_SEEDS` for the process. The unified CLI stores the selected values locally in `session.json` and shares them across methods in that session.

Repeat a unified command with `--resume` to continue from its saved configuration. Direct module commands accept `--resume` together with the initialization settings used for their original run.

`--tasks 1797:1730` selects a CWRU task; `--tasks OC0:OC1` selects a PU task. Comma-separated pairs select multiple directions. `--epochs`, `--batch-size`, and `--lr` configure main-model and PU front-end training. CWRU core baseline optimization settings are defined in the corresponding dataclasses.

CWRU `--protocol strict_721` forms chronological 70%/20%/10% partitions before windowing. `source_val_best` restores the best source-validation checkpoint. `early_stop_last` selects the terminal state reached by the source-validation patience rule. The target-test partition is evaluated after checkpoint selection.

## Evaluation and outputs

Checkpoints contain model states, configuration, task metadata, and metrics. Checkpoint paths are recorded in `raw_results.csv`.

```bash
python pipeline.py evaluate --checkpoint PATH_TO_CHECKPOINT.pt --data-root Data/CWRU --output-dir runs/evaluation
python pipeline.py summarize --input-dir runs/cwru_comparison --output-dir runs/cwru_tables
```

Evaluation exports per-sample predictions, the confusion matrix, accuracy, and macro-F1. Reporting exports task summaries, overall summaries, a LaTeX comparison table, and accuracy plots. Task statistics summarize repeated initializations. Overall statistics first average tasks within each initialization, then compute the mean and sample standard deviation across initializations.
