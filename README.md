# WK-CDAN main model

Standalone implementation of the main Wavelet-KAN Conditional Domain Adversarial Network (WK-CDAN), provided for architectural review.

This release contains model source code, loss components, and a synthetic execution example. It does not contain datasets, trained weights, training/evaluation pipelines, experimental results, manuscript materials, or development history. It cannot independently reproduce the paper's numerical results.

## Model

- A Mexican Hat Wavelet-KAN convolution (1 to 32 channels, kernel 31, stride 2, padding 15), followed by batch normalization and pooling.
- Conventional convolutions (32 to 64 to 128 channels), SE channel attention, pooling, dropout, and a 128-dimensional feature representation.
- A Wavelet-KAN classifier with a configurable number of classes (default: 10).
- Conditional domain alignment using the flattened outer product of the feature vector and class probabilities, followed by gradient reversal and a domain discriminator (128 times the class count to 256 to 2).

The alignment is CDAN: the discriminator receives both features and predicted class probabilities through their outer product. The discriminator is used during training; class logits are used for fault classification.

The implementation is extracted from the revised research implementation. The main-model configuration is fixed to the Mexican Hat Wavelet-KAN backbone and classifier, SE enabled, conditional domain alignment enabled, and wavelet translation/response normalization disabled. Baseline architectures and ablation runners are omitted. The default main-model parameter names and tensor shapes are retained.

## Run the synthetic example

Tested with Python 3.9 and PyTorch 2.8.0. Install the dependency in your own environment:

```bash
pip install -r requirements.txt
python example.py
```

The example automatically uses CUDA when available and otherwise uses CPU. It generates random input tensors in memory, checks output shapes, and backpropagates the combined objective. It does not download data or write files. Passing this check verifies execution only.

## Interface

```python
import torch
from model import CDANModel

model = CDANModel(num_classes=10)
model.eval()
with torch.no_grad():
    class_logits, domain_logits, features = model(torch.randn(2, 1, 1024))
```

Inputs have shape `(batch, 1, signal_length)`; the research input window length is 1024. Real input windows are standardized individually as `(window - mean) / (std + 1e-8)` before being passed to the model. Preprocessing is outside this model release.

Outputs are class logits `(batch, num_classes)`, domain logits `(batch, 2)`, and features `(batch, 128)`. `alpha` controls gradient reversal; its default value of zero disables the adversarial gradient flowing back to the feature/classification path. It does not disable discriminator gradients. The research schedule uses `alpha = 2 / (1 + exp(-10 * progress)) - 1`.

## Loss components

The main cross-domain training objective is:

```text
L = source classification CE
  + source domain CE + target domain CE
  + 0.05 * source center loss
  + 0.05 * alpha * target prediction entropy
```

Domain labels are 0 for source and 1 for target. Target class labels do not enter this objective. `CenterLoss` and `target_entropy_loss` are included in `model.py`; cross-entropy uses PyTorch's implementation. For CPU use, construct `CenterLoss(..., use_gpu=False)`.

The synthetic example illustrates the objective and backward pass only. The research trainer uses a separate SGD optimizer for class centers (learning rate 0.5), rescales center gradients by `1 / 0.05` before its step, and optimizes the model with Adam (learning rate 0.0005, weight decay 0.001). Data partitioning, checkpoint selection, full training, and evaluation are outside this release.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | Main model, gradient reversal, center loss, target entropy |
| `example.py` | Synthetic forward/backward execution check |
| `requirements.txt` | Tested PyTorch dependency |
| `.gitignore` | Explicit allowlist for the release files |

