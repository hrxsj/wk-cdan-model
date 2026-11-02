# WK-CDAN

PyTorch implementation of the Wavelet-KAN Conditional Domain Adversarial Network (WK-CDAN) for cross-domain bearing fault diagnosis.

The implementation combines Wavelet-KAN feature extraction and classification, SE channel attention, and conditional domain adversarial learning. The repository includes the model, loss components, and a runnable forward/backward example using synthetic inputs.

## Model

- A Mexican Hat Wavelet-KAN convolution (1 to 32 channels, kernel 31, stride 2, padding 15), followed by batch normalization and pooling.
- Conventional convolutions (32 to 64 to 128 channels), SE channel attention, pooling, dropout, and a 128-dimensional feature representation.
- A Wavelet-KAN classifier with a configurable number of classes (default: 10).
- Conditional domain alignment using the flattened outer product of the feature vector and class probabilities, followed by gradient reversal and a domain discriminator (128 times the class count to 256 to 2).

The alignment is CDAN: the discriminator receives both features and predicted class probabilities through their outer product. The discriminator is used during training; class logits are used for fault classification.

The default configuration uses the Mexican Hat wavelet in both the backbone and classifier, with SE attention and conditional domain alignment enabled. Wavelet translation and response normalization are disabled.

## Quick start

Tested with Python 3.9 and PyTorch 2.8.0:

```bash
pip install -r requirements.txt
python example.py
```

The example automatically uses CUDA when available and otherwise uses CPU. It creates synthetic source and target batches, computes the combined objective, runs backpropagation, and checks output shapes and gradients.

## Interface

```python
import torch
from model import CDANModel

model = CDANModel(num_classes=10)
model.eval()
with torch.no_grad():
    class_logits, domain_logits, features = model(torch.randn(2, 1, 1024))
```

Inputs have shape `(batch, 1, signal_length)`, with a research input window length of 1024. Standardize each vibration window as `(window - mean) / (std + 1e-8)` before passing it to the model.

Outputs are class logits `(batch, num_classes)`, domain logits `(batch, 2)`, and features `(batch, 128)`. `alpha` scales the reversed adversarial gradient flowing back to the feature and classification layers. At the default `alpha=0`, this gradient is zero while the discriminator receives its own gradients. The research schedule uses `alpha = 2 / (1 + exp(-10 * progress)) - 1`.

## Loss components

The main cross-domain training objective is:

```text
L = source classification CE
  + source domain CE + target domain CE
  + 0.05 * source center loss
  + 0.05 * alpha * target prediction entropy
```

Domain labels are 0 for source and 1 for target. Target class labels do not enter this objective. `CenterLoss` and `target_entropy_loss` are included in `model.py`; cross-entropy uses PyTorch's implementation. For CPU use, construct `CenterLoss(..., use_gpu=False)`.

The research training configuration uses Adam for the model (learning rate 0.0005, weight decay 0.001) and a separate SGD optimizer for class centers (learning rate 0.5). Center gradients are rescaled by `1 / 0.05` before the center optimizer step.

## Files

| File | Purpose |
| --- | --- |
| `model.py` | WK-CDAN model, gradient reversal, center loss, and target entropy |
| `example.py` | Forward/backward example with synthetic source and target inputs |
| `requirements.txt` | Python dependency |

