# WK-CDAN

PyTorch implementation of the Wavelet-KAN Conditional Domain Adversarial Network (WK-CDAN) for cross-domain bearing fault diagnosis, with CWRU and Paderborn University (PU) experiment workflows.

The repository provides data loading and preprocessing, WK-CDAN training, target-domain evaluation, baseline comparisons, ablation studies, checkpoint export, and result aggregation.

## Installation

Use Python 3.9 and install the dependencies:

```bash
pip install -r requirements.txt
```

Training uses CUDA when available. Add `--device cpu` to use the CPU.

## Datasets

### CWRU

The experiment uses ten classes: normal, and ball, inner-race, and outer-race faults at three defect sizes. The four load domains are 0HP/1797 rpm, 1HP/1772 rpm, 2HP/1750 rpm, and 3HP/1730 rpm.

Download metadata is included in `cwru/metadata.txt`. Prepare the selected records with:

```bash
python pipeline.py prepare --dataset cwru --data-root Data/CWRU
```

This downloads missing files, reads the vibration records, and writes a dataset inventory under `runs/cwru_data/`. The source is the [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter).

### PU

Obtain the records from the [Paderborn University Bearing Data Center](https://mb.uni-paderborn.de/en/kat/main-research/datacenter/bearing-datacenter). Extract them into bearing-specific folders:

```text
Data/PU/
  K001/N15_M07_F10_K001_1.mat
  K002/...
  KA01/...
  KA05/...
  KI01/...
  KI05/...
```

The classes are Normal (`K001`, `K002`), OuterRace (`KA01`, `KA05`), and InnerRace (`KI01`, `KI05`). The PU benchmark uses these three operating conditions:

| Alias | Record prefix | Description |
| --- | --- | --- |
| OC0 | `N15_M07_F10` | Reference condition |
| OC1 | `N15_M01_F10` | Lower load torque |
| OC2 | `N15_M07_F04` | Lower radial force |

```bash
python pipeline.py prepare --dataset pu --data-root Data/PU
```

## Train WK-CDAN

Run all twelve directed CWRU cross-load tasks or all six directed PU cross-condition tasks:

```bash
python pipeline.py train --dataset cwru --suite wkcdan --data-root Data/CWRU --output-dir runs/cwru_main
python pipeline.py train --dataset pu --suite wkcdan --data-root Data/PU --output-dir runs/pu_main
```

Select a task with `--tasks`. CWRU uses RPM pairs, and PU accepts operating-condition aliases:

```bash
python pipeline.py train --dataset cwru --suite wkcdan --tasks 1797:1730 --data-root Data/CWRU --output-dir runs/cwru_task
python pipeline.py train --dataset pu --suite wkcdan --tasks OC0:OC1 --data-root Data/PU --output-dir runs/pu_task
```

Each default experiment uses five runtime-generated initializations. Run configuration and initialization settings are saved in the local output directory. Repeat a command with `--resume` to continue unfinished runs:

```bash
python pipeline.py train --dataset cwru --suite wkcdan --data-root Data/CWRU --output-dir runs/cwru_main --resume
```

Users can supply their own initialization values with `--seeds` or `WKCDAN_RUN_SEEDS`. Dataset locations can also be configured with `CWRU_DATA_ROOT` and `PU_DATA_ROOT`.

## Baselines and ablations

The six core baselines are WDCNN, compact 1D ResNet-18, and B-spline KANConv with source-only training, plus DANN, CDAN, and DSAN/LMMD with the compact 1D ResNet-18 backbone.

```bash
python pipeline.py train --dataset cwru --suite baselines --data-root Data/CWRU --output-dir runs/cwru_baselines
python pipeline.py train --dataset pu --suite baselines --data-root Data/PU --output-dir runs/pu_baselines
```

Use `--suite all` to run WK-CDAN and the six core baselines. For PU, this also runs the paired Conv-ReLU versus Mexican Hat front-end comparison.

| Dataset | Suite | Experiment |
| --- | --- | --- |
| CWRU, PU | `frontend` | Conv-ReLU versus Mexican Hat KANConv front end |
| CWRU, PU | `basis` | Gaussian, Morlet, B-spline, Mexican Hat, and Shannon bases |
| CWRU | `structure` | WKAN, SE, and CDAN component ablations |
| CWRU | `factorial` | Conv/WKAN front end crossed with MLP/Wavelet-KAN classifier |
| CWRU | `loss` | Center-loss and target-entropy ablations |
| CWRU | `wavelet` | Mother-wavelet comparison |
| CWRU | `noise` | Main-model and comparison methods across SNR levels |
| CWRU | `noise-baselines` | B-spline, DANN, and CDAN noise baselines |

```bash
python pipeline.py train --dataset cwru --suite factorial --data-root Data/CWRU --output-dir runs/cwru_factorial
python pipeline.py train --dataset cwru --suite noise --data-root Data/CWRU --output-dir runs/cwru_noise
python pipeline.py train --dataset pu --suite basis --data-root Data/PU --output-dir runs/pu_basis
```

The PU basis study defaults to OC2→OC1. CWRU ablations use clean input by default; add `--snr-db=-5` for noise injection. `--snr-levels=clean,-1,-3,-5` controls noise and mother-wavelet comparison conditions.

## Preprocessing and training settings

Each continuous vibration record is split chronologically before extracting windows. The standard protocol uses the first 80% for training and the final 20% for testing, with 50% window overlap and independent per-window z-score normalization.

| Setting | CWRU | PU |
| --- | --- | --- |
| Classes | 10 | 3 |
| Window length | 1024 | 2048 |
| Signal preparation | Drive-end vibration channel | Vibration channel, decimation by 5 |
| Epochs | 100 | 100 |
| Batch size | 32 | 64 |
| Optimizer | Adam | Adam |
| Learning rate | 0.0005 | 0.0001 |
| Weight decay | 0.001 | 0.001 |
| Center-loss coefficient | 0.05 | 0.05 |
| Target-entropy coefficient | `0.05 * alpha` | `0.05 * alpha` |
| Center optimizer | SGD, learning rate 0.5 | SGD, learning rate 0.5 |
| Adaptation warmup | Main model: logistic schedule from the start; UDA baselines: 20 epochs | 20 epochs |
| Checkpoint selection | Final epoch | Final epoch |

Source-only methods train on labeled source data. Adaptation methods additionally use unlabeled target-training signals. Target-test predictions are computed after training.

CWRU also supports chronological 70%/20%/10% partitions and source-domain validation:

```bash
python pipeline.py train --dataset cwru --suite wkcdan --protocol strict_721 --checkpoint-policy source_val_best --data-root Data/CWRU --output-dir runs/cwru_721
```

`early_stop_last` selects the terminal state reached by source-validation early stopping. Source-validation patience is 15. Core baselines use the 80%/20% fixed-epoch protocol.

## Evaluate saved checkpoints

Checkpoint paths are recorded in `raw_results.csv`. Evaluate saved weights against their target partition with:

```bash
python pipeline.py evaluate --checkpoint PATH_TO_CHECKPOINT.pt --data-root Data/CWRU --output-dir runs/cwru_evaluation
python pipeline.py evaluate --checkpoint PATH_TO_CHECKPOINT.pt --data-root Data/PU --output-dir runs/pu_evaluation
```

Evaluation exports `metrics.json`, `predictions.csv`, and `confusion_matrix.csv`.

## Tables, figures, and artifacts

Training exports per-run CSV results, epoch histories, saved checkpoints, configuration records, and method summaries. Rebuild a comparison report with:

```bash
python pipeline.py summarize --input-dir runs/cwru_main --output-dir runs/cwru_report
```

Reports contain `summary.csv`, `task_summary.csv`, `summary.json`, `comparison.tex`, `accuracy.pdf`, and `accuracy.png`. Overall means and sample standard deviations are computed across repeated-run task averages. Runtime data and artifacts are written beneath `runs/`.

## Model interface

The network combines Mexican Hat Wavelet-KAN convolution, CNN feature extraction, SE attention, a Wavelet-KAN classifier, and a conditional domain discriminator. CDAN receives the flattened outer product of features and predicted class probabilities through a gradient reversal layer.

```python
import torch
from model import CDANModel

model = CDANModel(num_classes=10)
model.eval()
with torch.no_grad():
    class_logits, domain_logits, features = model(torch.randn(2, 1, 1024))
```

Outputs have shapes `(batch, num_classes)`, `(batch, 2)`, and `(batch, 128)`. `alpha` controls the reversed adversarial gradient. The main objective is:

```text
L = source classification CE
  + source domain CE + target domain CE
  + 0.05 * source center loss
  + 0.05 * alpha * target prediction entropy
```

Run `python example.py` for a synthetic forward/backward example. Use the dataset workflows above for training on CWRU and PU.

## Repository layout

```text
model.py                         WK-CDAN network and losses
pipeline.py                      Dataset, training, evaluation, and reporting CLI
runtime.py                       Runtime initialization settings
evaluation.py                    Checkpoint loading and target evaluation
reporting.py                     CSV/LaTeX tables and comparison figures
experiments/cwru_ieee_revision.py CWRU model, preprocessing, trainer, and ablations
experiments/cwru/                Source-only, UDA, and noise benchmarks
experiments/pu/                  PU preprocessing, trainer, baselines, and ablations
cwru/metadata.txt                CWRU dataset file URLs
docs/experiments.md              Commands and workflow details
```
