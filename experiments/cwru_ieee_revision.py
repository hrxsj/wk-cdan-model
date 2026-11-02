"""CWRU WK-CDAN models, preprocessing, training, evaluation, and ablations."""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import urllib.request as urllib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class Config:
    data_root: str = os.path.join(ROOT, "Data", "CWRU")
    metadata_path: str = os.path.join(ROOT, "cwru", "metadata.txt")
    output_dir: str = os.path.join(ROOT, "runs", "ieee_revision")
    data_protocol: str = "holdout"
    window_size: int = 1024
    overlap: float = 0.5
    train_ratio: float = 0.8
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-4
    patience: int = 15
    weight_decay: float = 1e-3
    center_weight: float = 0.05
    entropy_weight: float = 0.05
    center_lr: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_windows_per_class: Optional[int] = None
    checkpoint_policy: str = "source_val_best"
    fixed_stop_epoch: Optional[int] = None


CFG = Config()

CLASS_FILES = {
    "Normal": ("NormalBaseline", "Normal.mat"),
    "0.007-Ball": ("12DriveEndFault", "0.007-Ball.mat"),
    "0.014-Ball": ("12DriveEndFault", "0.014-Ball.mat"),
    "0.021-Ball": ("12DriveEndFault", "0.021-Ball.mat"),
    "0.007-InnerRace": ("12DriveEndFault", "0.007-InnerRace.mat"),
    "0.014-InnerRace": ("12DriveEndFault", "0.014-InnerRace.mat"),
    "0.021-InnerRace": ("12DriveEndFault", "0.021-InnerRace.mat"),
    "0.007-OuterRace6": ("12DriveEndFault", "0.007-OuterRace6.mat"),
    "0.014-OuterRace6": ("12DriveEndFault", "0.014-OuterRace6.mat"),
    "0.021-OuterRace6": ("12DriveEndFault", "0.021-OuterRace6.mat"),
}
LABELS = list(CLASS_FILES.keys())
LABEL_TO_IDX = {name: idx for idx, name in enumerate(LABELS)}
NUM_CLASSES = len(LABELS)

ALL_RPMS = ["1797", "1772", "1750", "1730"]
LOAD_MAP = {"1797": "0HP", "1772": "1HP", "1750": "2HP", "1730": "3HP"}
DEFAULT_SEEDS = list(DEFAULT_RUN_SEEDS)
SMOKE_METHODS = [
    "DANN + ResNet-18",
    "CDAN + ResNet-18",
    "CDANN-JMMD",
    "GCDAN-fair",
    "CNN-1D-KAN",
]
DEFAULT_SNR_LEVELS = [None, -1.0, -3.0, -5.0]
DEFAULT_WAVELET_SNR_LEVELS = [None, -5.0]
STRICT_721_RATIOS = (0.7, 0.2, 0.1)
CORE_BASIS_METHOD_NAMES = ["Conv-ReLU Front-end", "Morlet KANConv", "Mexican Hat KANConv"]
CORE_BASIS_TASKS = [("1797", "1730"), ("1730", "1797"), ("1772", "1730")]
CORE_BASIS_OUTPUT_DIR = os.path.join(ROOT, "runs", "cwru_basis_core_45")
CORE_BASIS_PROTOCOL_VERSION = "cwru-basis-core-v1"
CLEAN_BASIS_METHOD_NAMES = [
    "Gaussian KANConv",
    "Morlet KANConv",
    "B-spline KANConv",
    "Mexican Hat KANConv",
    "Shannon KANConv",
]
CLEAN_BASIS_TASKS = [("1797", "1730")]
CLEAN_BASIS_OUTPUT_DIR = os.path.join(ROOT, "runs", "cwru_basis_clean_exploratory_25")
CLEAN_BASIS_PROTOCOL_VERSION = "cwru-basis-clean-exploratory-v1"
FACTORIAL_METHOD_NAMES = [
    "A Conv-MLP-CDAN",
    "B Conv-WaveletKAN-CDAN",
    "C WKAN-MLP-CDAN",
    "D WKAN-WaveletKAN-CDAN",
]
FACTORIAL_AC_METHOD_NAMES = [
    FACTORIAL_METHOD_NAMES[0],
    FACTORIAL_METHOD_NAMES[2],
]
FACTORIAL_A_METHOD_NAMES = [
    FACTORIAL_METHOD_NAMES[0],
]
FACTORIAL_B_METHOD_NAMES = [
    FACTORIAL_METHOD_NAMES[1],
]
FACTORIAL_C_METHOD_NAMES = [
    FACTORIAL_METHOD_NAMES[2],
]
FACTORIAL_TASKS = [("1797", "1730"), ("1730", "1797"), ("1772", "1730")]
FACTORIAL_ALL_CROSS_TASKS = [
    (source_rpm, target_rpm)
    for source_rpm in ALL_RPMS
    for target_rpm in ALL_RPMS
    if source_rpm != target_rpm
]
FACTORIAL_SEEDS = list(DEFAULT_RUN_SEEDS)
FACTORIAL_OUTPUT_DIR = os.path.join(ROOT, "runs", "cwru_factorial_2x2_repeated_runs_100epoch")
FACTORIAL_CLEAN_OUTPUT_DIR = os.path.join(ROOT, "runs", "cwru_factorial_clean_100epoch")
FACTORIAL_CLEAN_B_OUTPUT_DIR = os.path.join(ROOT, "runs", "cwru_factorial_B_clean_100epoch")
FACTORIAL_CLEAN_B_FIXED40_OUTPUT_DIR = os.path.join(
    ROOT, "runs", "cwru_factorial_B_clean_fixed40"
)
FACTORIAL_CLEAN_B_EARLY_STOP_LAST_OUTPUT_DIR = os.path.join(
    ROOT, "runs", "cwru_factorial_B_clean_early_stop_last"
)
FACTORIAL_CLEAN_A_EARLY_STOP_LAST_OUTPUT_DIR = os.path.join(
    ROOT, "runs", "cwru_factorial_A_clean_early_stop_last"
)
FACTORIAL_CLEAN_C_EARLY_STOP_LAST_OUTPUT_DIR = os.path.join(
    ROOT, "runs", "cwru_factorial_C_clean_early_stop_last"
)
FACTORIAL_PROTOCOL_VERSION = "cwru-factorial-2x2-v2-repeated-runs"
FACTORIAL_ALL_CROSS_PROTOCOL_VERSION = "cwru-factorial-all-cross-v1-repeated-runs"
FACTORIAL_ALL_CROSS_STRICT721_PROTOCOL_VERSION = "cwru-factorial-all-cross-strict721-v1"
FACTORIAL_AC_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION = (
    "cwru-factorial-ac-all-cross-strict721-clean-v1"
)
FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION = (
    "cwru-factorial-b-all-cross-strict721-clean-v1"
)
FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_FIXED40_PROTOCOL_VERSION = (
    "cwru-factorial-b-all-cross-strict721-clean-fixed40-v1"
)
FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION = (
    "cwru-factorial-b-all-cross-strict721-clean-early-stop-last-v1"
)
FACTORIAL_A_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION = (
    "cwru-factorial-a-all-cross-strict721-clean-early-stop-last-v1"
)
FACTORIAL_C_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION = (
    "cwru-factorial-c-all-cross-strict721-clean-early-stop-last-v1"
)
FACTORIAL_A_EARLY_STOP_LAST_PARENT_SHA256 = (
    "e7cf9e15ccff862c702ecdd4b79fce5b3e97bf5261432f27ced2538000b40657"
)
FACTORIAL_A_EARLY_STOP_LAST_AMENDMENT_SHA256 = (
    "93d695743f485d7752e7992f4a0b70e779a298aa61e338a216b4dc8214c477f4"
)
FACTORIAL_A_EARLY_STOP_LAST_BASELINE_BINDING_SHA256 = (
    "5859f02260fc502cbecb934a32d8dd59cac178d2636b3a8ff16704b8b00a352d"
)
FACTORIAL_A_EARLY_STOP_LAST_PREREGISTRATION = {
    "parent_protocol": {
        "path": "experiments/protocols/cwru_a_early_stop_last_v1.json",
        "sha256": FACTORIAL_A_EARLY_STOP_LAST_PARENT_SHA256,
    },
    "amendment_01": {
        "path": "experiments/protocols/cwru_a_early_stop_last_v1_amendment_01.json",
        "sha256": FACTORIAL_A_EARLY_STOP_LAST_AMENDMENT_SHA256,
    },
    "amendment_02": {
        "path": "experiments/protocols/cwru_a_early_stop_last_v1_amendment_02.json",
        "sha256": FACTORIAL_A_EARLY_STOP_LAST_BASELINE_BINDING_SHA256,
    },
}
FACTORIAL_B_EARLY_STOP_LAST_PREREGISTRATION = {
    "parent_protocol": {
        "path": "experiments/protocols/cwru_b_early_stop_last_v1.json",
        "sha256": "43e2caea6de0c608a84650e637ca0701eb878bf333c7ba5517733838eaf97e08",
    },
    "amendment_01": {
        "path": "experiments/protocols/cwru_b_early_stop_last_v1_amendment_01.json",
        "sha256": "3b27490efbb0ce483fa3b5d1ef134bd63df4bf8486cff6208f651ce72317ca62",
    },
}
FACTORIAL_C_EARLY_STOP_LAST_PARENT_SHA256 = (
    "2e3ad3555967ca30c882c40630c46d8f5a0a2313ccef246815e1c870510062d6"
)
FACTORIAL_C_EARLY_STOP_LAST_AMENDMENT_SHA256 = (
    "4292e50405088bb80164e7c74917d78b449fb068591a0b46b66e82c664381b1d"
)
FACTORIAL_C_EARLY_STOP_LAST_BASELINE_BINDING_SHA256 = (
    "1bd47ea456806915fa68ce247c6986b2b9b10dcd383d193e787b5a3be83b3e4c"
)
FACTORIAL_C_EARLY_STOP_LAST_PREREGISTRATION = {
    "parent_protocol": {
        "path": "experiments/protocols/cwru_c_early_stop_last_v1.json",
        "sha256": FACTORIAL_C_EARLY_STOP_LAST_PARENT_SHA256,
    },
    "amendment_01": {
        "path": "experiments/protocols/cwru_c_early_stop_last_v1_amendment_01.json",
        "sha256": FACTORIAL_C_EARLY_STOP_LAST_AMENDMENT_SHA256,
    },
    "amendment_02": {
        "path": "experiments/protocols/cwru_c_early_stop_last_v1_amendment_02.json",
        "sha256": FACTORIAL_C_EARLY_STOP_LAST_BASELINE_BINDING_SHA256,
    },
}
FACTORIAL_EARLY_STOP_LAST_PREREGISTRATIONS = {
    FACTORIAL_A_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION: (
        FACTORIAL_A_EARLY_STOP_LAST_PREREGISTRATION
    ),
    FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION: (
        FACTORIAL_B_EARLY_STOP_LAST_PREREGISTRATION
    ),
    FACTORIAL_C_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION: (
        FACTORIAL_C_EARLY_STOP_LAST_PREREGISTRATION
    ),
}
FACTORIAL_DIAGNOSTIC_PROTOCOL_VERSION = "cwru-factorial-task-subset-diagnostic-v1"


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    backbone: str
    use_cdan: bool
    use_center: bool
    use_entropy: bool
    suite: str = "main"
    wavelet: str = "mexhat"
    use_se: bool = True
    use_wavelet_translation: bool = False
    normalize_wavelet: bool = False
    domain_mode: str = "none"
    jmmd_weight: float = 0.0
    pseudo_label: bool = False
    classifier: str = "linear"
    classifier_wavelet: Optional[str] = None


EXPERIMENTS = [
    ExperimentConfig("ResNet-18 Source-only", "resnet", False, False, False),
    ExperimentConfig("CNN-1D-KAN", "cnn", False, False, False, use_se=False, classifier="b_spline_kan"),
    ExperimentConfig("KANConv-based", "bkan", False, False, False, use_se=True, classifier="b_spline_kan"),
    ExperimentConfig("DANN + ResNet-18", "resnet", False, False, False, domain_mode="dann"),
    ExperimentConfig("CDAN + ResNet-18", "resnet", True, False, False, domain_mode="cdan"),
    ExperimentConfig("CDANN-JMMD", "resnet", True, False, False, domain_mode="cdan", jmmd_weight=0.1),
    ExperimentConfig("GCDAN-fair", "resnet", True, False, False, domain_mode="cdan", pseudo_label=True),
    ExperimentConfig("WKAN-Source", "wkan", False, False, False, classifier="wavelet_kan"),
    ExperimentConfig("CDAN + WKAN", "wkan", True, False, False, domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("WK-CDAN Full", "wkan", True, True, True, domain_mode="cdan", classifier="wavelet_kan"),
]

NOISE_ABLATIONS = [
    ExperimentConfig("Full", "wkan", True, True, True, suite="noise_ablation", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("w/o Center Loss", "wkan", True, False, True, suite="noise_ablation", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("w/o Entropy Loss", "wkan", True, True, False, suite="noise_ablation", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("w/o Center + Entropy", "wkan", True, False, False, suite="noise_ablation", domain_mode="cdan", classifier="wavelet_kan"),
]

SNR_COMPARISON = [
    ExperimentConfig("ResNet-18 (1D)", "resnet", False, False, False, suite="snr_comparison"),
    ExperimentConfig("WDCNN", "wdcnn", False, False, False, suite="snr_comparison"),
    ExperimentConfig("Vanilla KAN", "vanilla_kan", False, False, False, suite="snr_comparison", classifier="b_spline_kan"),
    ExperimentConfig(
        "DANN + ResNet-18",
        "resnet",
        False,
        False,
        False,
        suite="snr_comparison",
        domain_mode="dann",
        classifier="linear",
    ),
    ExperimentConfig(
        "CDAN + ResNet-18",
        "resnet",
        True,
        False,
        False,
        suite="snr_comparison",
        domain_mode="cdan",
        classifier="linear",
    ),
    ExperimentConfig(
        "WKAN-Source",
        "wkan",
        False,
        False,
        False,
        suite="snr_comparison",
        domain_mode="none",
        classifier="wavelet_kan",
    ),
    ExperimentConfig(
        "DANN + WKAN",
        "wkan",
        True,
        True,
        True,
        suite="snr_comparison",
        domain_mode="dann",
        classifier="wavelet_kan",
    ),
    ExperimentConfig("WK-CDAN Full", "wkan", True, True, True, suite="snr_comparison", domain_mode="cdan", classifier="wavelet_kan"),
]

WAVELET_ABLATIONS = [
    ExperimentConfig("WK-CDAN-Mexhat", "wkan", True, True, True, suite="wavelet_ablation", wavelet="mexhat", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("WK-CDAN-Morlet", "wkan", True, True, True, suite="wavelet_ablation", wavelet="morlet", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("WK-CDAN-Laplace", "wkan", True, True, True, suite="wavelet_ablation", wavelet="laplace", domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("WK-CDAN-Shannon", "wkan", True, True, True, suite="wavelet_ablation", wavelet="shannon", domain_mode="cdan", classifier="wavelet_kan"),
]

BASIS_ABLATIONS = [
    ExperimentConfig(
        "Conv-ReLU Front-end",
        "conv_relu",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="relu",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "Gaussian KANConv",
        "wkan",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="gaussian",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "Morlet KANConv",
        "wkan",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="morlet_fair",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "B-spline KANConv",
        "bkan",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="b_spline",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "Mexican Hat KANConv",
        "wkan",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="mexhat",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "Shannon KANConv",
        "wkan",
        True,
        True,
        True,
        suite="basis_ablation",
        wavelet="shannon_real_1_1",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
]

STRUCTURE_ABLATIONS = [
    ExperimentConfig("w/o WKAN representation", "cnn", True, True, True, suite="structure_ablation", use_se=True, domain_mode="cdan"),
    ExperimentConfig("WKAN-only", "wkan", False, False, False, suite="structure_ablation", use_se=False, classifier="wavelet_kan"),
    ExperimentConfig("w/o SE", "wkan", True, True, True, suite="structure_ablation", use_se=False, domain_mode="cdan", classifier="wavelet_kan"),
    ExperimentConfig("WKAN-SE w/o CDAN", "wkan", False, True, False, suite="structure_ablation", use_se=True, classifier="wavelet_kan"),
    ExperimentConfig("WK-CDAN Full", "wkan", True, True, True, suite="structure_ablation", use_se=True, domain_mode="cdan", classifier="wavelet_kan"),
]

FACTORIAL_ABLATIONS = [
    ExperimentConfig(
        "A Conv-MLP-CDAN",
        "conv_relu",
        True,
        True,
        True,
        suite="factorial_ablation",
        use_se=True,
        domain_mode="cdan",
        classifier="mlp",
    ),
    ExperimentConfig(
        "B Conv-WaveletKAN-CDAN",
        "conv_relu",
        True,
        True,
        True,
        suite="factorial_ablation",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
    ExperimentConfig(
        "C WKAN-MLP-CDAN",
        "wkan",
        True,
        True,
        True,
        suite="factorial_ablation",
        wavelet="mexhat",
        use_se=True,
        domain_mode="cdan",
        classifier="mlp",
    ),
    ExperimentConfig(
        "D WKAN-WaveletKAN-CDAN",
        "wkan",
        True,
        True,
        True,
        suite="factorial_ablation",
        wavelet="mexhat",
        use_se=True,
        domain_mode="cdan",
        classifier="wavelet_kan",
        classifier_wavelet="mexhat",
    ),
]


class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CenterLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=128, use_gpu=True):
        super().__init__()
        centers = torch.randn(num_classes, feat_dim)
        if use_gpu:
            centers = centers.cuda()
        self.centers = nn.Parameter(centers)
        self.num_classes = num_classes

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        return (distmat * mask.float()).clamp(min=1e-12, max=1e12).sum() / batch_size


def wavelet_basis(x, wavelet="mexhat"):
    wavelet = wavelet.lower()
    if wavelet in {"mexhat", "mexican", "mexican_hat", "mexican-hat"}:
        return (1 - x**2) * torch.exp(-0.5 * x**2)
    if wavelet == "morlet":
        return torch.exp(-0.5 * x**2) * torch.cos(5.0 * x)
    if wavelet == "morlet_fair":
        # A real, admissibility-corrected Morlet with derivative energy close
        # to the Mexican Hat initialization for scales 1, 2, and 3.
        omega0 = 2.0
        correction = math.exp(-0.5 * omega0**2)
        return (torch.cos(omega0 * x) - correction) * torch.exp(-0.5 * x**2)
    if wavelet == "gaussian":
        return torch.exp(-0.5 * x**2)
    if wavelet == "laplace":
        damping = 0.15
        return torch.exp(-damping * torch.abs(x)) * torch.sin(2.0 * math.pi * x)
    if wavelet == "shannon":
        window = torch.exp(-0.5 * (x / 3.0) ** 2)
        return torch.sinc(x / math.pi) * window
    if wavelet == "shannon_real_1_1":
        # Real component of PyWavelets shan1-1:
        # sqrt(B) * sinc(B*x) * exp(j*2*pi*C*x), with B=C=1.
        return torch.sinc(x) * torch.cos(2.0 * math.pi * x)
    raise ValueError(f"Unknown wavelet basis: {wavelet}")


def normalize_wavelet_response(basis, eps=1e-6):
    rms = torch.sqrt(torch.mean(basis.pow(2), dim=-1, keepdim=True) + eps)
    return basis / rms


def pairwise_sq_dists(x, y):
    return torch.cdist(x, y, p=2).pow(2)


def _median_bandwidth(x, y, eps=1e-6):
    with torch.no_grad():
        values = pairwise_sq_dists(torch.cat([x, y], dim=0), torch.cat([x, y], dim=0)).flatten()
        values = values[values > eps]
        if values.numel() == 0:
            return torch.tensor(1.0, device=x.device, dtype=x.dtype)
        return torch.clamp(torch.median(values), min=eps)


def multi_kernel_rbf(x, y, bandwidth, multipliers=(0.5, 1.0, 2.0, 4.0, 8.0)):
    dists = pairwise_sq_dists(x, y)
    kernels = [torch.exp(-dists / (2.0 * bandwidth * scale)) for scale in multipliers]
    return sum(kernels) / len(kernels)


def joint_mmd_loss(source_features, target_features, source_probs, target_probs):
    feature_bw = _median_bandwidth(source_features.detach(), target_features.detach())
    prob_bw = _median_bandwidth(source_probs.detach(), target_probs.detach())

    kff_ss = multi_kernel_rbf(source_features, source_features, feature_bw)
    kff_tt = multi_kernel_rbf(target_features, target_features, feature_bw)
    kff_st = multi_kernel_rbf(source_features, target_features, feature_bw)

    kyy_ss = multi_kernel_rbf(source_probs, source_probs, prob_bw)
    kyy_tt = multi_kernel_rbf(target_probs, target_probs, prob_bw)
    kyy_st = multi_kernel_rbf(source_probs, target_probs, prob_bw)

    k_ss = kff_ss * kyy_ss
    k_tt = kff_tt * kyy_tt
    k_st = kff_st * kyy_st
    return torch.clamp(k_ss.mean() + k_tt.mean() - 2.0 * k_st.mean(), min=0.0)


def sharpen_probabilities(probs, temperature=0.5, eps=1e-8):
    sharpened = probs.clamp_min(eps).pow(1.0 / temperature)
    return sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(eps)


def pseudo_label_weight(epoch, total_epochs, warmup_epochs=20, max_weight=0.1):
    if epoch <= warmup_epochs:
        return 0.0
    denom = max(1, total_epochs - warmup_epochs)
    progress = min(1.0, max(0.0, (epoch - warmup_epochs) / denom))
    return max_weight * progress


def pseudo_label_loss(
    logits,
    epoch,
    total_epochs,
    warmup_epochs=20,
    confidence_threshold=0.9,
    max_weight=0.1,
    temperature=0.5,
):
    weight = pseudo_label_weight(epoch, total_epochs, warmup_epochs, max_weight)
    zero = logits.new_tensor(0.0)
    if weight <= 0.0:
        return zero, 0.0, 0

    probs = F.softmax(logits, dim=1)
    confidence = probs.max(dim=1).values
    mask = confidence > confidence_threshold
    selected = int(mask.sum().item())
    if selected == 0:
        return zero, weight, 0

    targets = sharpen_probabilities(probs[mask], temperature=temperature).detach()
    loss = -(targets * F.log_softmax(logits[mask], dim=1)).sum(dim=1).mean()
    return loss * weight, weight, selected


def count_parameters_m(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad) / 1e6


class MLPClassifier(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=256):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_features, out_features),
        )

    def forward(self, x):
        return self.layers(x)


class KANLinearWavelet(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        scale_base=1.0,
        wavelet="mexhat",
        use_translation=False,
        normalize_response=False,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.use_translation = use_translation
        self.normalize_response = normalize_response
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.wavelet_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        if use_translation:
            scale_init = torch.arange(1, grid_size + 1, dtype=torch.float32).view(1, grid_size).repeat(in_features, 1)
            translation_init = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32).view(1, grid_size).repeat(
                in_features, 1
            )
            self.scales = nn.Parameter(scale_init)
            self.translations = nn.Parameter(translation_init)
        else:
            self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))
        self.silu = nn.SiLU()
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        nn.init.normal_(self.wavelet_weight, mean=0.0, std=0.1)

    def forward(self, x):
        base_output = F.linear(self.silu(x), self.base_weight)
        if self.use_translation:
            x_scaled = (x.unsqueeze(-1) - self.translations.unsqueeze(0)) * self.scales.unsqueeze(0)
        else:
            x_scaled = x.unsqueeze(-1) * self.scales
        basis = wavelet_basis(x_scaled, self.wavelet)
        if self.normalize_response:
            basis = normalize_wavelet_response(basis)
        wavelet_output = torch.einsum("big,oig->bo", basis, self.wavelet_weight)
        return base_output + wavelet_output


class KANConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        grid_size=3,
        wavelet="mexhat",
        use_translation=False,
        normalize_response=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = grid_size
        self.wavelet = wavelet
        self.use_translation = use_translation
        self.normalize_response = normalize_response
        self.wavelet_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, grid_size) * 0.1)
        self.base_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        self.base_activation = nn.SiLU()
        if use_translation:
            scale_init = (
                torch.arange(1, grid_size + 1, dtype=torch.float32)
                .view(1, 1, grid_size)
                .repeat(in_channels, kernel_size, 1)
            )
            translation_init = (
                torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
                .view(1, 1, grid_size)
                .repeat(in_channels, kernel_size, 1)
            )
            self.scales = nn.Parameter(scale_init)
            self.translations = nn.Parameter(translation_init)
        else:
            self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x):
        batch, _, _ = x.shape
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )
        unfolded = unfolded.view(batch, self.in_channels, self.kernel_size, -1).permute(0, 3, 1, 2)
        base_output = torch.einsum("blic,oic->blo", self.base_activation(unfolded), self.base_weights)
        if self.use_translation:
            x_scaled = (
                unfolded.unsqueeze(-1)
                - self.translations.view(1, 1, self.in_channels, self.kernel_size, self.grid_size)
            ) * self.scales.view(1, 1, self.in_channels, self.kernel_size, self.grid_size)
        else:
            x_scaled = unfolded.unsqueeze(-1) * self.scales.view(1, 1, 1, 1, self.grid_size)
        basis = wavelet_basis(x_scaled, self.wavelet)
        if self.normalize_response:
            basis = normalize_wavelet_response(basis)
        wavelet_output = torch.einsum("blicg,oicg->blo", basis, self.wavelet_weights)
        return (base_output + wavelet_output).permute(0, 2, 1)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x)).expand_as(x)


class BasicBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return self.relu(out)


def make_resnet_layer(in_channels, out_channels, blocks, stride):
    layers = [BasicBlock1D(in_channels, out_channels, stride)]
    for _ in range(1, blocks):
        layers.append(BasicBlock1D(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class ResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_resnet_layer(32, 32, 2, stride=1)
        self.layer2 = make_resnet_layer(32, 64, 2, stride=2)
        self.layer3 = make_resnet_layer(64, 128, 2, stride=2)
        self.layer4 = make_resnet_layer(128, 256, 2, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


class Conv1DBackbone(nn.Module):
    def __init__(self, use_se=False):
        super().__init__()
        self.first = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mid_cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1D(128, reduction=8) if use_se else nn.Identity()
        self.tail = nn.Sequential(nn.MaxPool1d(2, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5))

    def forward(self, x):
        x = self.pool1(self.bn1(self.first(x)))
        x = self.mid_cnn(x)
        x = self.se(x)
        return torch.tanh(self.tail(x))


class ConvReLUBackbone(Conv1DBackbone):
    """Kernel-matched conventional Conv1D + ReLU front-end control."""

    def __init__(self, use_se=True):
        super().__init__(use_se=use_se)
        self.first = nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15)
        self.first_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.pool1(self.first_activation(self.bn1(self.first(x))))
        x = self.mid_cnn(x)
        x = self.se(x)
        return torch.tanh(self.tail(x))


class WKANBackbone(Conv1DBackbone):
    def __init__(self, wavelet="mexhat", use_se=True, use_translation=False, normalize_response=False):
        super().__init__(use_se=use_se)
        self.first = KANConv1d(
            1,
            32,
            kernel_size=31,
            stride=2,
            padding=15,
            wavelet=wavelet,
            use_translation=use_translation,
            normalize_response=normalize_response,
        )


class WDCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        return self.features(x)


class KANLinearBSpline(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=(-3.0, 3.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid_min, grid_max = grid_range
        h = (grid_max - grid_min) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_min
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.silu = nn.SiLU()

        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        with torch.no_grad():
            noise = (torch.rand(grid_size + spline_order, in_features, out_features) - 0.5) * scale_noise / grid_size
            self.spline_weight.data.copy_(scale_spline * noise.permute(2, 1, 0))

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[: -(k + 1)]) / (self.grid[k:-1] - self.grid[: -(k + 1)])
            right = (self.grid[k + 1 :] - x) / (self.grid[k + 1 :] - self.grid[1:-k])
            bases = left * bases[:, :, :-1] + right * bases[:, :, 1:]
        return bases

    def forward(self, x):
        base_output = F.linear(self.silu(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class KANConv1dBSpline(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        grid_size=5,
        spline_order=3,
        grid_range=(-3.0, 3.0),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = grid_size
        self.spline_order = spline_order

        grid_min, grid_max = grid_range
        h = (grid_max - grid_min) / grid_size
        grid = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * h + grid_min
        self.register_buffer("grid", grid)

        self.base_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        self.spline_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, grid_size + spline_order) * 0.02
        )
        self.base_activation = nn.SiLU()

    def b_splines(self, x):
        x = x.unsqueeze(-1)
        bases = ((x >= self.grid[:-1]) & (x < self.grid[1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - self.grid[: -(k + 1)]) / (self.grid[k:-1] - self.grid[: -(k + 1)])
            right = (self.grid[k + 1 :] - x) / (self.grid[k + 1 :] - self.grid[1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x):
        batch, _, _ = x.shape
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )
        unfolded = unfolded.view(batch, self.in_channels, self.kernel_size, -1).permute(0, 3, 1, 2)
        base_output = torch.einsum("blic,oic->blo", self.base_activation(unfolded), self.base_weights)
        spline_output = torch.einsum("blicg,oicg->blo", self.b_splines(unfolded), self.spline_weights)
        return (base_output + spline_output).permute(0, 2, 1)


class BKANBackbone(Conv1DBackbone):
    def __init__(self, use_se=False):
        super().__init__(use_se=use_se)
        self.first = KANConv1dBSpline(1, 32, kernel_size=31, stride=2, padding=15, grid_size=5, spline_order=3)


class VanillaKANBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.kan = KANLinearBSpline(1024, 128, grid_size=5, spline_order=3)

    def forward(self, x):
        return self.kan(self.flatten(x))


class CDANModel(nn.Module):
    def __init__(
        self,
        backbone_name,
        num_classes=NUM_CLASSES,
        use_cdan=True,
        wavelet="mexhat",
        use_se=True,
        use_wavelet_translation=False,
        normalize_wavelet=False,
        domain_mode=None,
        classifier=None,
        classifier_wavelet=None,
    ):
        super().__init__()
        if domain_mode is None:
            domain_mode = "cdan" if use_cdan else "none"
        if domain_mode not in {"none", "dann", "cdan"}:
            raise ValueError(f"Unknown domain mode: {domain_mode}")
        self.domain_mode = domain_mode
        self.use_cdan = domain_mode == "cdan"
        self.num_classes = num_classes
        if classifier is None:
            classifier = "wavelet_kan" if backbone_name == "wkan" else "b_spline_kan" if backbone_name in {
                "bkan",
                "vanilla_kan",
            } else "linear"
        if classifier_wavelet is None:
            classifier_wavelet = wavelet

        if backbone_name == "resnet":
            self.backbone = ResNetBackbone()
        elif backbone_name == "cnn":
            self.backbone = Conv1DBackbone(use_se=use_se)
        elif backbone_name == "conv_relu":
            self.backbone = ConvReLUBackbone(use_se=use_se)
        elif backbone_name == "wkan":
            self.backbone = WKANBackbone(
                wavelet=wavelet,
                use_se=use_se,
                use_translation=use_wavelet_translation,
                normalize_response=normalize_wavelet,
            )
        elif backbone_name == "bkan":
            self.backbone = BKANBackbone(use_se=use_se)
        elif backbone_name == "wdcnn":
            self.backbone = WDCNNBackbone()
        elif backbone_name == "vanilla_kan":
            self.backbone = VanillaKANBackbone()
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        if classifier == "linear":
            self.classifier = nn.Linear(128, num_classes)
        elif classifier == "mlp":
            self.classifier = MLPClassifier(128, num_classes)
        elif classifier == "b_spline_kan":
            self.classifier = KANLinearBSpline(128, num_classes, grid_size=5, spline_order=3)
        elif classifier == "wavelet_kan":
            self.classifier = KANLinearWavelet(
                128,
                num_classes,
                wavelet=classifier_wavelet,
                use_translation=use_wavelet_translation,
                normalize_response=normalize_wavelet,
            )
        else:
            raise ValueError(f"Unknown classifier: {classifier}")

        self.domain_classifier = None
        if domain_mode != "none":
            domain_in_dim = 128 * num_classes if domain_mode == "cdan" else 128
            self.domain_classifier = nn.Sequential(
                nn.Linear(domain_in_dim, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            )

    def forward(self, x, alpha=0.0):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        if self.domain_mode == "none":
            return class_logits, None, features
        if self.domain_mode == "cdan":
            softmax_prob = F.softmax(class_logits, dim=1)
            cond_features = torch.bmm(features.unsqueeze(2), softmax_prob.unsqueeze(1)).view(
                -1, 128 * self.num_classes
            )
        else:
            cond_features = features
        reverse_features = GradientReversalLayer.apply(cond_features, alpha)
        domain_logits = self.domain_classifier(reverse_features)
        return class_logits, domain_logits, features


class ClassificationPath(nn.Module):
    """Deployment path; excludes the training-only domain discriminator."""

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.classifier = model.classifier

    def forward(self, x):
        return self.classifier(self.backbone(x))


def count_inference_parameters_m(model):
    path = ClassificationPath(model)
    return sum(param.numel() for param in path.parameters() if param.requires_grad) / 1e6


def _tensor_size(value):
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_size(item) for item in value)
    return 0


def estimate_classification_flops(model, example):
    """Estimate one-sample deployment FLOPs using the paper's operator convention."""

    total = {"value": 0}
    handles = []

    def add(value):
        total["value"] += int(value)

    def hook_conv1d(module, inputs, output):
        batch = inputs[0].shape[0]
        out_len = output.shape[-1]
        kernel_ops = module.kernel_size[0] * module.in_channels // module.groups
        add(2 * batch * module.out_channels * out_len * kernel_ops)

    def hook_linear(module, inputs, output):
        add(2 * _tensor_size(output) * module.in_features)

    def hook_bn(module, inputs, output):
        add(2 * _tensor_size(output))

    def hook_kan_conv(module, inputs, output):
        batch = inputs[0].shape[0]
        out_len = output.shape[-1]
        base = 2 * batch * out_len * module.out_channels * module.in_channels * module.kernel_size
        wavelet = base * module.grid_size
        basis = 6 * batch * out_len * module.in_channels * module.kernel_size * module.grid_size
        add(base + wavelet + basis)

    def hook_wavelet_linear(module, inputs, output):
        batch = inputs[0].shape[0]
        out_features, in_features, grid_size = module.wavelet_weight.shape
        base = 2 * batch * out_features * in_features
        wavelet = base * grid_size
        basis = 6 * batch * in_features * grid_size
        add(base + wavelet + basis)

    def hook_bkan_conv(module, inputs, output):
        batch = inputs[0].shape[0]
        out_len = output.shape[-1]
        basis_count = module.grid_size + module.spline_order
        base = 2 * batch * out_len * module.out_channels * module.in_channels * module.kernel_size
        spline = base * basis_count
        basis = (
            8
            * module.spline_order
            * batch
            * out_len
            * module.in_channels
            * module.kernel_size
            * basis_count
        )
        add(base + spline + basis)

    def hook_bkan_linear(module, inputs, output):
        batch = inputs[0].shape[0]
        basis_count = module.grid_size + module.spline_order
        base = 2 * batch * module.out_features * module.in_features
        spline = base * basis_count
        basis = 8 * module.spline_order * batch * module.in_features * basis_count
        add(base + spline + basis)

    path = ClassificationPath(model).eval()
    for module in path.modules():
        if isinstance(module, KANConv1d):
            handles.append(module.register_forward_hook(hook_kan_conv))
        elif isinstance(module, KANLinearWavelet):
            handles.append(module.register_forward_hook(hook_wavelet_linear))
        elif isinstance(module, KANConv1dBSpline):
            handles.append(module.register_forward_hook(hook_bkan_conv))
        elif isinstance(module, KANLinearBSpline):
            handles.append(module.register_forward_hook(hook_bkan_linear))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(hook_conv1d))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(hook_linear))
        elif isinstance(module, nn.BatchNorm1d):
            handles.append(module.register_forward_hook(hook_bn))

    with torch.no_grad():
        path(example)
    for handle in handles:
        handle.remove()
    return total["value"]


def measure_single_sample_latency(model, example, device, warmup=20, repeats=100):
    path = ClassificationPath(model).eval()
    timings = []
    with torch.no_grad():
        for _ in range(warmup):
            path(example)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(repeats):
                starter.record()
                path(example)
                ender.record()
                torch.cuda.synchronize(device)
                timings.append(float(starter.elapsed_time(ender)))
        else:
            for _ in range(repeats):
                start = time.perf_counter()
                path(example)
                timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.mean(timings)), float(np.std(timings, ddof=1) if len(timings) > 1 else 0.0)


def fingerprint_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def safe_slug(value):
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "run"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_metadata_map(metadata_path):
    meta = {}
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) >= 4:
                meta[(parts[0], parts[1], parts[2])] = parts[3]
    return meta


def ensure_mat_file(data_root, metadata_map, exp_name, rpm, label_name, file_name):
    mat_path = os.path.join(data_root, exp_name, rpm, file_name)
    if not os.path.exists(mat_path):
        ensure_dir(os.path.dirname(mat_path))
        urllib.urlretrieve(metadata_map[(exp_name, rpm, label_name)], mat_path)
    return mat_path


def load_1d_signal(mat_path):
    mat = loadmat(mat_path)
    keys = [key for key in mat.keys() if "DE_time" in key or "FE_time" in key or "BA_time" in key]
    if not keys:
        keys = [key for key, value in mat.items() if isinstance(value, np.ndarray) and value.size > 1000]
    if not keys:
        raise ValueError(f"No vibration signal found in {mat_path}")
    return np.asarray(mat[keys[0]]).reshape(-1).astype(np.float32)


def sliding_windows(signal, window_size, overlap):
    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError("overlap must be lower than 1.0")
    num = 1 + (len(signal) - window_size) // step
    if num <= 0:
        raise ValueError("signal is shorter than window_size")
    windows = np.empty((num, window_size), dtype=np.float32)
    for i in range(num):
        start = i * step
        window = signal[start : start + window_size]
        windows[i] = (window - np.mean(window)) / (np.std(window) + 1e-8)
    return windows


def split_signal_by_time(signal_or_length, train_ratio, test_ratio):
    if not np.isclose(train_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio and test_ratio must sum to 1.0 for train/test time splitting")
    n = int(signal_or_length if isinstance(signal_or_length, (int, np.integer)) else len(signal_or_length))
    n_train = int(n * train_ratio)
    return (0, n_train), (n_train, n)


def split_signal_train_val_test_by_time(signal_or_length, train_ratio, val_ratio, test_ratio):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")
    n = int(signal_or_length if isinstance(signal_or_length, (int, np.integer)) else len(signal_or_length))
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return (0, n_train), (n_train, n_train + n_val), (n_train + n_val, n)


def split_train_side_for_validation(train_range, val_ratio):
    start, end = train_range
    length = end - start
    n_update = int(length * (1.0 - val_ratio))
    return (start, start + n_update), (start + n_update, end)


def window_spans(start, end, window_size, overlap):
    step = int(window_size * (1.0 - overlap))
    if step <= 0:
        raise ValueError("overlap must be lower than 1.0")
    length = end - start
    if length < window_size:
        return []
    num = 1 + (length - window_size) // step
    return [(start + i * step, start + i * step + window_size) for i in range(num)]


def sliding_windows_from_range(signal, start, end, window_size, overlap):
    spans = window_spans(start, end, window_size, overlap)
    if not spans:
        return np.empty((0, window_size), dtype=np.float32)
    windows = np.empty((len(spans), window_size), dtype=np.float32)
    for i, (span_start, span_end) in enumerate(spans):
        window = signal[span_start:span_end]
        windows[i] = (window - np.mean(window)) / (np.std(window) + 1e-8)
    return windows


def add_awgn_noise(windows, snr_db, seed):
    rng = np.random.default_rng(seed)
    signal_power = np.mean(np.square(windows), axis=1, keepdims=True)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=windows.shape)
    noisy = windows + noise.astype(np.float32)
    return noisy.astype(np.float32)


def split_train_val_test_by_time(X, y, train_ratio, val_ratio, test_ratio, seed):
    _ = test_ratio
    X_tr, y_tr, X_va, y_va, X_te, y_te = [], [], [], [], [], []
    for cls in range(NUM_CLASSES):
        idx = np.where(y == cls)[0]
        n = len(idx)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)
        X_tr.append(X[idx[:n_tr]])
        y_tr.append(y[idx[:n_tr]])
        X_va.append(X[idx[n_tr : n_tr + n_va]])
        y_va.append(y[idx[n_tr : n_tr + n_va]])
        X_te.append(X[idx[n_tr + n_va :]])
        y_te.append(y[idx[n_tr + n_va :]])
    rng = np.random.default_rng(seed)
    train_y = np.concatenate(y_tr)
    train_perm = rng.permutation(len(train_y))
    return (
        np.concatenate(X_tr)[train_perm],
        train_y[train_perm],
        np.concatenate(X_va),
        np.concatenate(y_va),
        np.concatenate(X_te),
        np.concatenate(y_te),
    )


def build_domain_dataset(config, metadata_map, rpm, seed=resolve_seed(None), snr_db=None):
    X_list, y_list = [], []
    for label_name in LABELS:
        exp_name, file_name = CLASS_FILES[label_name]
        mat_path = ensure_mat_file(config.data_root, metadata_map, exp_name, rpm, label_name, file_name)
        signal = load_1d_signal(mat_path)
        windows = sliding_windows(signal, config.window_size, config.overlap)
        if config.max_windows_per_class is not None:
            windows = windows[: config.max_windows_per_class]
        if snr_db is not None:
            windows = add_awgn_noise(windows, snr_db=snr_db, seed=seed + LABEL_TO_IDX[label_name])
        labels = np.full((len(windows),), LABEL_TO_IDX[label_name], dtype=np.int64)
        X_list.append(windows)
        y_list.append(labels)
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _append_domain_split(split_store, split_name, windows, label_idx, config, snr_db, noise_seed):
    if config.max_windows_per_class is not None:
        windows = windows[: config.max_windows_per_class]
    if snr_db is not None and len(windows):
        windows = add_awgn_noise(windows, snr_db=snr_db, seed=noise_seed)
    labels = np.full((len(windows),), label_idx, dtype=np.int64)
    split_store[split_name][0].append(windows)
    split_store[split_name][1].append(labels)


def _concat_domain_split(split_store, split_name):
    X_parts, y_parts = split_store[split_name]
    if not X_parts:
        return np.empty((0, CFG.window_size), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def build_domain_dataset_splits(config, metadata_map, rpm, seed=resolve_seed(None), snr_db=None):
    split_store = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    split_offsets = {"train": 0, "val": 10000, "test": 20000}
    for label_name in LABELS:
        exp_name, file_name = CLASS_FILES[label_name]
        label_idx = LABEL_TO_IDX[label_name]
        mat_path = ensure_mat_file(config.data_root, metadata_map, exp_name, rpm, label_name, file_name)
        signal = load_1d_signal(mat_path)
        train_side_range, test_range = split_signal_by_time(signal, config.train_ratio, config.test_ratio)
        update_range, val_range = split_train_side_for_validation(train_side_range, config.val_ratio)
        split_ranges = {"train": update_range, "val": val_range, "test": test_range}
        for split_name, (start, end) in split_ranges.items():
            windows = sliding_windows_from_range(signal, start, end, config.window_size, config.overlap)
            _append_domain_split(
                split_store,
                split_name,
                windows,
                label_idx,
                config,
                snr_db,
                seed + split_offsets[split_name] + label_idx,
            )

    X_train, y_train = _concat_domain_split(split_store, "train")
    X_val, y_val = _concat_domain_split(split_store, "val")
    X_test, y_test = _concat_domain_split(split_store, "test")

    rng = np.random.default_rng(seed)
    if len(y_train):
        train_perm = rng.permutation(len(y_train))
        X_train = X_train[train_perm]
        y_train = y_train[train_perm]

    return X_train, y_train, X_val, y_val, X_test, y_test


def build_domain_dataset_train_test(config, metadata_map, rpm, seed=resolve_seed(None), snr_db=None):
    split_store = {
        "train": ([], []),
        "test": ([], []),
    }
    split_offsets = {"train": 0, "test": 20000}
    for label_name in LABELS:
        exp_name, file_name = CLASS_FILES[label_name]
        label_idx = LABEL_TO_IDX[label_name]
        mat_path = ensure_mat_file(config.data_root, metadata_map, exp_name, rpm, label_name, file_name)
        signal = load_1d_signal(mat_path)
        train_range, test_range = split_signal_by_time(signal, config.train_ratio, config.test_ratio)
        split_ranges = {"train": train_range, "test": test_range}
        for split_name, (start, end) in split_ranges.items():
            windows = sliding_windows_from_range(signal, start, end, config.window_size, config.overlap)
            _append_domain_split(
                split_store,
                split_name,
                windows,
                label_idx,
                config,
                snr_db,
                seed + split_offsets[split_name] + label_idx,
            )

    X_train, y_train = _concat_domain_split(split_store, "train")
    X_test, y_test = _concat_domain_split(split_store, "test")

    rng = np.random.default_rng(seed)
    if len(y_train):
        train_perm = rng.permutation(len(y_train))
        X_train = X_train[train_perm]
        y_train = y_train[train_perm]

    return X_train, y_train, X_test, y_test


def build_domain_dataset_721(config, metadata_map, rpm, seed=resolve_seed(None), snr_db=None):
    split_store = {
        "train": ([], []),
        "val": ([], []),
        "test": ([], []),
    }
    split_offsets = {"train": 0, "val": 10000, "test": 20000}
    train_ratio, val_ratio, test_ratio = STRICT_721_RATIOS
    for label_name in LABELS:
        exp_name, file_name = CLASS_FILES[label_name]
        label_idx = LABEL_TO_IDX[label_name]
        mat_path = ensure_mat_file(config.data_root, metadata_map, exp_name, rpm, label_name, file_name)
        signal = load_1d_signal(mat_path)
        train_range, val_range, test_range = split_signal_train_val_test_by_time(
            signal,
            train_ratio,
            val_ratio,
            test_ratio,
        )
        split_ranges = {"train": train_range, "val": val_range, "test": test_range}
        for split_name, (start, end) in split_ranges.items():
            windows = sliding_windows_from_range(signal, start, end, config.window_size, config.overlap)
            _append_domain_split(
                split_store,
                split_name,
                windows,
                label_idx,
                config,
                snr_db,
                seed + split_offsets[split_name] + label_idx,
            )

    X_train, y_train = _concat_domain_split(split_store, "train")
    X_val, y_val = _concat_domain_split(split_store, "val")
    X_test, y_test = _concat_domain_split(split_store, "test")

    rng = np.random.default_rng(seed)
    if len(y_train):
        train_perm = rng.permutation(len(y_train))
        X_train = X_train[train_perm]
        y_train = y_train[train_perm]

    return X_train, y_train, X_val, y_val, X_test, y_test


def make_loader(X, y, batch_size, shuffle, drop_last=False):
    return DataLoader(
        TensorDataset(torch.from_numpy(X[:, None, :].astype(np.float32)), torch.from_numpy(y.astype(np.int64))),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )


def metrics_from_predictions(preds, targets):
    preds = np.asarray(preds)
    targets = np.asarray(targets)
    accuracy = float(np.mean(preds == targets)) if len(targets) else 0.0
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for target, pred in zip(targets, preds):
        confusion[int(target), int(pred)] += 1
    class_totals = confusion.sum(axis=1)
    class_accuracy = np.divide(
        np.diag(confusion),
        class_totals,
        out=np.zeros(NUM_CLASSES, dtype=np.float64),
        where=class_totals > 0,
    )
    f1s = []
    for cls in range(NUM_CLASSES):
        tp = np.sum((preds == cls) & (targets == cls))
        precision_den = np.sum(preds == cls)
        recall_den = np.sum(targets == cls)
        precision = float(tp / precision_den) if precision_den else 0.0
        recall = float(tp / recall_den) if recall_den else 0.0
        f1s.append(float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0)
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
        "class_accuracy": class_accuracy.tolist(),
        "confusion_matrix": confusion.tolist(),
    }


def evaluate(model, loader, device):
    model.eval()
    preds_all, targets_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits, _, _ = model(xb, alpha=0.0)
            preds_all.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets_all.extend(yb.numpy().tolist())
    metrics = metrics_from_predictions(preds_all, targets_all)
    return metrics


def should_stop_on_source_validation(checkpoint_policy, patience_counter, patience):
    return (
        checkpoint_policy in {"source_val_best", "early_stop_last"}
        and patience_counter >= patience
    )


def train_one_pair(exp_cfg, config, metadata_map, source_rpm, target_rpm, seed, snr_db=None):
    set_seed(seed)
    device = torch.device(config.device)
    same_domain = source_rpm == target_rpm
    if config.checkpoint_policy not in {
        "source_val_best",
        "early_stop_last",
        "final_epoch",
    }:
        raise ValueError(f"Unknown checkpoint policy: {config.checkpoint_policy}")
    if (
        config.checkpoint_policy == "early_stop_last"
        and config.fixed_stop_epoch is not None
    ):
        raise ValueError("early_stop_last uses the configured epoch horizon, not fixed_stop_epoch")
    max_training_epoch = config.fixed_stop_epoch or config.epochs
    if not 1 <= max_training_epoch <= config.epochs:
        raise ValueError(
            "fixed_stop_epoch must be between 1 and the configured schedule horizon"
        )

    if config.data_protocol == "transductive":
        X_all_s, y_all_s = build_domain_dataset(config, metadata_map, source_rpm, seed=seed, snr_db=snr_db)
        X_tr_s, y_tr_s, X_va_s, y_va_s, _, _ = split_train_val_test_by_time(
            X_all_s,
            y_all_s,
            train_ratio=1.0 - config.val_ratio,
            val_ratio=config.val_ratio,
            test_ratio=0.0,
            seed=seed,
        )
        X_all_t, y_all_t = build_domain_dataset(config, metadata_map, target_rpm, seed=seed + 1000, snr_db=snr_db)
        X_tr_t, y_tr_t = X_all_t, y_all_t
        X_te_t, y_te_t = X_all_t, y_all_t
    elif config.data_protocol == "strict_holdout":
        X_tr_s, y_tr_s, _, _ = build_domain_dataset_train_test(
            config, metadata_map, source_rpm, seed=seed, snr_db=snr_db
        )
        X_va_s = np.empty((0, config.window_size), dtype=np.float32)
        y_va_s = np.empty((0,), dtype=np.int64)
        X_tr_t, y_tr_t, X_te_t, y_te_t = build_domain_dataset_train_test(
            config, metadata_map, target_rpm, seed=seed + 1000, snr_db=snr_db
        )
    elif config.data_protocol == "strict_721":
        X_tr_s, y_tr_s, X_va_s, y_va_s, _, _ = build_domain_dataset_721(
            config, metadata_map, source_rpm, seed=seed, snr_db=snr_db
        )
        X_tr_t, y_tr_t, _, _, X_te_t, y_te_t = build_domain_dataset_721(
            config, metadata_map, target_rpm, seed=seed + 1000, snr_db=snr_db
        )
    elif config.data_protocol == "holdout":
        X_tr_s, y_tr_s, X_va_s, y_va_s, _, _ = build_domain_dataset_splits(
            config, metadata_map, source_rpm, seed=seed, snr_db=snr_db
        )
        X_tr_t, y_tr_t, _, _, X_te_t, y_te_t = build_domain_dataset_splits(
            config, metadata_map, target_rpm, seed=seed + 1000, snr_db=snr_db
        )
    else:
        raise ValueError(f"Unknown data protocol: {config.data_protocol}")

    src_loader = make_loader(X_tr_s, y_tr_s, config.batch_size, shuffle=True, drop_last=True)
    tgt_loader = make_loader(X_tr_t, y_tr_t, config.batch_size, shuffle=True, drop_last=True)
    source_val_loader = None
    if len(y_va_s):
        source_val_loader = make_loader(X_va_s, y_va_s, config.batch_size, shuffle=False)
    if config.checkpoint_policy == "early_stop_last" and source_val_loader is None:
        raise ValueError("early_stop_last requires a source validation split")
    test_loader = make_loader(X_te_t, y_te_t, config.batch_size, shuffle=False)
    source_train_hash = fingerprint_arrays(X_tr_s, y_tr_s)
    target_train_hash = fingerprint_arrays(X_tr_t, y_tr_t)
    target_test_hash = fingerprint_arrays(X_te_t, y_te_t)
    dataset_fingerprint = fingerprint_arrays(X_tr_s, y_tr_s, X_tr_t, y_tr_t, X_te_t, y_te_t)

    model = CDANModel(
        exp_cfg.backbone,
        num_classes=NUM_CLASSES,
        use_cdan=exp_cfg.use_cdan,
        wavelet=exp_cfg.wavelet,
        use_se=exp_cfg.use_se,
        use_wavelet_translation=exp_cfg.use_wavelet_translation,
        normalize_wavelet=exp_cfg.normalize_wavelet,
        domain_mode=exp_cfg.domain_mode,
        classifier=exp_cfg.classifier,
        classifier_wavelet=exp_cfg.classifier_wavelet,
    ).to(device)
    params_m = count_parameters_m(model)
    inference_params_m = count_inference_parameters_m(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    center_loss_fn = None
    center_optimizer = None
    if exp_cfg.use_center:
        center_loss_fn = CenterLoss(NUM_CLASSES, feat_dim=128, use_gpu=(device.type == "cuda")).to(device)
        center_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=config.center_lr)

    criterion_class = nn.CrossEntropyLoss()
    criterion_domain = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    early_stopping_triggered = False
    epochs_run = 0
    epoch_history = []
    len_dataloader = min(len(src_loader), len(tgt_loader))
    if len_dataloader == 0:
        raise ValueError("No training batches were produced. Increase max_windows_per_class or lower batch size.")

    training_start = time.perf_counter()
    for epoch in range(1, max_training_epoch + 1):
        model.train()
        epochs_run = epoch
        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)
        epoch_loss_sum = 0.0
        epoch_class_loss_sum = 0.0
        epoch_domain_loss_sum = 0.0
        epoch_center_loss_sum = 0.0
        epoch_entropy_loss_sum = 0.0
        epoch_jmmd_loss_sum = 0.0
        epoch_pseudo_loss_sum = 0.0
        epoch_front_grad_sum = 0.0
        epoch_scale_grad_sum = 0.0
        epoch_grad_steps = 0
        for step in range(len_dataloader):
            src_data, src_label = next(src_iter)
            tgt_data, _ = next(tgt_iter)
            src_data = src_data.to(device)
            src_label = src_label.to(device)
            tgt_data = tgt_data.to(device)

            use_domain = exp_cfg.domain_mode != "none" and not same_domain
            needs_target_logits = use_domain or (exp_cfg.jmmd_weight > 0.0 and not same_domain) or (
                exp_cfg.pseudo_label and not same_domain
            )
            if needs_target_logits:
                progress = float(step + (epoch - 1) * len_dataloader) / (config.epochs * len_dataloader)
                alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
            else:
                alpha = 0.0

            optimizer.zero_grad()
            if center_optimizer is not None:
                center_optimizer.zero_grad()

            src_class_logits, src_domain_logits, src_features = model(src_data, alpha=alpha)
            classification_loss = criterion_class(src_class_logits, src_label)
            loss = classification_loss
            domain_loss = loss.new_tensor(0.0)
            center_loss = loss.new_tensor(0.0)
            entropy_loss = loss.new_tensor(0.0)
            jmmd_loss_value = loss.new_tensor(0.0)
            pseudo_loss_value = loss.new_tensor(0.0)
            tgt_class_logits = None
            tgt_domain_logits = None
            tgt_features = None
            if needs_target_logits:
                tgt_class_logits, tgt_domain_logits, tgt_features = model(tgt_data, alpha=alpha)

            if exp_cfg.use_center and center_loss_fn is not None:
                center_loss = center_loss_fn(src_features, src_label)
                loss = loss + config.center_weight * center_loss

            if use_domain:
                src_domain_labels = torch.zeros(src_data.size(0), dtype=torch.long, device=device)
                tgt_domain_labels = torch.ones(tgt_data.size(0), dtype=torch.long, device=device)
                domain_loss = criterion_domain(src_domain_logits, src_domain_labels)
                domain_loss = domain_loss + criterion_domain(tgt_domain_logits, tgt_domain_labels)
                loss = loss + domain_loss
                if exp_cfg.use_entropy:
                    tgt_softmax = F.softmax(tgt_class_logits, dim=1)
                    entropy_loss = -(tgt_softmax * torch.log(tgt_softmax + 1e-5)).sum(dim=1).mean()
                    loss = loss + (config.entropy_weight * alpha) * entropy_loss

            if exp_cfg.jmmd_weight > 0.0 and not same_domain:
                src_probs = F.softmax(src_class_logits, dim=1)
                tgt_probs = F.softmax(tgt_class_logits, dim=1)
                jmmd_loss_value = joint_mmd_loss(src_features, tgt_features, src_probs, tgt_probs)
                loss = loss + exp_cfg.jmmd_weight * jmmd_loss_value

            if exp_cfg.pseudo_label and not same_domain:
                pseudo_loss_value, _, _ = pseudo_label_loss(
                    tgt_class_logits, epoch=epoch, total_epochs=config.epochs
                )
                loss = loss + pseudo_loss_value

            loss.backward()
            epoch_loss_sum += float(loss.detach().cpu())
            epoch_class_loss_sum += float(classification_loss.detach().cpu())
            epoch_domain_loss_sum += float(domain_loss.detach().cpu())
            epoch_center_loss_sum += float(center_loss.detach().cpu())
            epoch_entropy_loss_sum += float(entropy_loss.detach().cpu())
            epoch_jmmd_loss_sum += float(jmmd_loss_value.detach().cpu())
            epoch_pseudo_loss_sum += float(pseudo_loss_value.detach().cpu())
            front_module = getattr(model.backbone, "first", None)
            if front_module is not None:
                grad_sq = sum(
                    float(param.grad.detach().pow(2).sum().cpu())
                    for param in front_module.parameters()
                    if param.grad is not None
                )
                epoch_front_grad_sum += math.sqrt(max(grad_sq, 0.0))
                scales = getattr(front_module, "scales", None)
                if scales is not None and scales.grad is not None:
                    epoch_scale_grad_sum += float(scales.grad.detach().norm().cpu())
                epoch_grad_steps += 1
            optimizer.step()
            if center_optimizer is not None:
                for param in center_loss_fn.parameters():
                    if param.grad is not None:
                        param.grad.data *= 1.0 / config.center_weight
                center_optimizer.step()

        if source_val_loader is not None:
            val_metrics = evaluate(model, source_val_loader, device)
            val_acc = val_metrics["accuracy"] * 100.0
            val_macro_f1 = val_metrics["macro_f1"] * 100.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            epoch_history.append(
                {
                    "epoch": epoch,
                    "source_val_accuracy": val_acc,
                    "source_val_macro_f1": val_macro_f1,
                    "train_loss": epoch_loss_sum / len_dataloader,
                    "classification_loss": epoch_class_loss_sum / len_dataloader,
                    "domain_loss": epoch_domain_loss_sum / len_dataloader,
                    "center_loss": epoch_center_loss_sum / len_dataloader,
                    "entropy_loss": epoch_entropy_loss_sum / len_dataloader,
                    "jmmd_loss": epoch_jmmd_loss_sum / len_dataloader,
                    "pseudo_loss": epoch_pseudo_loss_sum / len_dataloader,
                    "front_grad_norm": epoch_front_grad_sum / max(epoch_grad_steps, 1),
                    "scale_grad_norm": epoch_scale_grad_sum / max(epoch_grad_steps, 1),
                    "selected": False,
                }
            )
            if should_stop_on_source_validation(
                config.checkpoint_policy,
                patience_counter,
                config.patience,
            ):
                early_stopping_triggered = True
                break
        else:
            epoch_history.append(
                {
                    "epoch": epoch,
                    "source_val_accuracy": "",
                    "source_val_macro_f1": "",
                    "train_loss": epoch_loss_sum / len_dataloader,
                    "classification_loss": epoch_class_loss_sum / len_dataloader,
                    "domain_loss": epoch_domain_loss_sum / len_dataloader,
                    "center_loss": epoch_center_loss_sum / len_dataloader,
                    "entropy_loss": epoch_entropy_loss_sum / len_dataloader,
                    "jmmd_loss": epoch_jmmd_loss_sum / len_dataloader,
                    "pseudo_loss": epoch_pseudo_loss_sum / len_dataloader,
                    "front_grad_norm": epoch_front_grad_sum / max(epoch_grad_steps, 1),
                    "scale_grad_norm": epoch_scale_grad_sum / max(epoch_grad_steps, 1),
                    "selected": False,
                }
            )

    training_seconds = time.perf_counter() - training_start

    source_val_best_epoch = best_epoch
    max_source_val_accuracy = max(best_val_acc, 0.0) if source_val_loader is not None else ""
    if config.checkpoint_policy == "source_val_best" and best_state is not None:
        model.load_state_dict(best_state)
        selected_source_val_accuracy = max_source_val_accuracy
    else:
        best_epoch = epochs_run
        source_val_best_epoch = source_val_best_epoch or epochs_run
        selected_source_val_accuracy = (
            epoch_history[-1]["source_val_accuracy"]
            if source_val_loader is not None
            else ""
        )
    stop_reason = (
        "early_stopping"
        if early_stopping_triggered
        else "fixed_stop_epoch"
        if config.fixed_stop_epoch is not None
        else "epoch_limit"
    )
    for epoch_row in epoch_history:
        epoch_row["selected"] = epoch_row["epoch"] == best_epoch
    test_metrics = evaluate(model, test_loader, device)
    test_acc = test_metrics["accuracy"] * 100.0
    class_accuracy = [value * 100.0 for value in test_metrics["class_accuracy"]]
    confusion_matrix = test_metrics["confusion_matrix"]

    example = torch.from_numpy(X_te_t[:1, None, :].astype(np.float32)).to(device)
    classification_flops = estimate_classification_flops(model, example)
    inference_ms, inference_ms_std = measure_single_sample_latency(model, example, device)

    snr_tag = "clean" if snr_db is None else f"snr_{float(snr_db):g}db".replace("-", "m")
    checkpoint_rel = os.path.join(
        "checkpoints",
        safe_slug(exp_cfg.name),
        f"{LOAD_MAP[source_rpm]}_to_{LOAD_MAP[target_rpm]}",
        f"seed_{seed}_{snr_tag}_final.pt",
    )
    checkpoint_abs = os.path.join(config.output_dir, checkpoint_rel)
    ensure_dir(os.path.dirname(checkpoint_abs))
    model_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    center_state = None
    if center_loss_fn is not None:
        center_state = {key: value.detach().cpu() for key, value in center_loss_fn.state_dict().items()}
    checkpoint_protocol = "general"
    if exp_cfg.suite == "basis_ablation":
        checkpoint_protocol = CORE_BASIS_PROTOCOL_VERSION
    elif exp_cfg.suite == "basis_clean_exploratory":
        checkpoint_protocol = CLEAN_BASIS_PROTOCOL_VERSION
    elif exp_cfg.suite == "factorial_ablation":
        if config.data_protocol == "strict_721" and snr_db is None:
            if (
                exp_cfg.name in FACTORIAL_A_METHOD_NAMES
                and config.checkpoint_policy == "early_stop_last"
                and config.fixed_stop_epoch is None
                and config.epochs == 100
                and config.patience == 15
            ):
                checkpoint_protocol = (
                    FACTORIAL_A_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
                )
            elif (
                exp_cfg.name in FACTORIAL_C_METHOD_NAMES
                and config.checkpoint_policy == "early_stop_last"
                and config.fixed_stop_epoch is None
                and config.epochs == 100
                and config.patience == 15
            ):
                checkpoint_protocol = (
                    FACTORIAL_C_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
                )
            elif (
                exp_cfg.name in FACTORIAL_B_METHOD_NAMES
                and config.checkpoint_policy == "early_stop_last"
                and config.fixed_stop_epoch is None
                and config.epochs == 100
                and config.patience == 15
            ):
                checkpoint_protocol = (
                    FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
                )
            elif (
                exp_cfg.name in FACTORIAL_B_METHOD_NAMES
                and config.checkpoint_policy == "final_epoch"
                and config.fixed_stop_epoch == 40
                and config.epochs == 100
            ):
                checkpoint_protocol = (
                    FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_FIXED40_PROTOCOL_VERSION
                )
            elif exp_cfg.name in FACTORIAL_B_METHOD_NAMES:
                checkpoint_protocol = FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION
            elif exp_cfg.name in FACTORIAL_AC_METHOD_NAMES:
                checkpoint_protocol = FACTORIAL_AC_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION
            else:
                checkpoint_protocol = FACTORIAL_ALL_CROSS_STRICT721_PROTOCOL_VERSION
        elif config.data_protocol == "strict_721":
            checkpoint_protocol = FACTORIAL_ALL_CROSS_STRICT721_PROTOCOL_VERSION
        else:
            checkpoint_protocol = FACTORIAL_PROTOCOL_VERSION
    torch.save(
        {
            "protocol_version": checkpoint_protocol,
            "model_state_dict": model_state,
            "center_loss_state_dict": center_state,
            "experiment_config": asdict(exp_cfg),
            "training_config": asdict(config),
            "source_rpm": source_rpm,
            "target_rpm": target_rpm,
            "seed": seed,
            "snr_db": snr_db,
            "epochs_run": epochs_run,
            "selected_epoch": best_epoch,
            "checkpoint_policy": config.checkpoint_policy,
            "source_val_best_epoch": source_val_best_epoch,
            "max_source_val_accuracy": max_source_val_accuracy,
            "stop_reason": stop_reason,
            "early_stopping_patience": config.patience,
            "patience_counter_at_stop": patience_counter,
            "target_metrics": test_metrics,
            "labels": LABELS,
            "dataset_fingerprint": dataset_fingerprint,
        },
        checkpoint_abs,
    )
    final_history = epoch_history[-1]

    return {
        "suite": exp_cfg.suite,
        "method": exp_cfg.name,
        "data_protocol": config.data_protocol,
        "source_rpm": source_rpm,
        "target_rpm": target_rpm,
        "source_hp": LOAD_MAP[source_rpm],
        "target_hp": LOAD_MAP[target_rpm],
        "seed": seed,
        "snr_db": "" if snr_db is None else snr_db,
        "wavelet": exp_cfg.wavelet,
        "backbone": exp_cfg.backbone,
        "classifier": exp_cfg.classifier,
        "classifier_wavelet": (
            exp_cfg.classifier_wavelet or exp_cfg.wavelet
            if exp_cfg.classifier == "wavelet_kan"
            else ""
        ),
        "factor_frontend": exp_cfg.backbone if exp_cfg.suite == "factorial_ablation" else "",
        "factor_classifier": exp_cfg.classifier if exp_cfg.suite == "factorial_ablation" else "",
        "accuracy": test_acc,
        "macro_f1": test_metrics["macro_f1"] * 100.0,
        "source_val_accuracy": selected_source_val_accuracy,
        "max_source_val_accuracy": max_source_val_accuracy,
        "source_val_best_epoch": source_val_best_epoch,
        "checkpoint_policy": config.checkpoint_policy,
        "fixed_stop_epoch": "" if config.fixed_stop_epoch is None else config.fixed_stop_epoch,
        "schedule_horizon_epochs": config.epochs,
        "stop_reason": stop_reason,
        "early_stopping_patience": config.patience,
        "patience_counter_at_stop": patience_counter,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "params_m": params_m,
        "inference_params_m": inference_params_m,
        "gflops": classification_flops / 1e9,
        "inference_ms": inference_ms,
        "inference_ms_std": inference_ms_std,
        "inference_warmup": 20,
        "inference_repeats": 100,
        "training_seconds": training_seconds,
        "final_train_loss": final_history["train_loss"],
        "final_classification_loss": final_history["classification_loss"],
        "final_domain_loss": final_history["domain_loss"],
        "final_center_loss": final_history["center_loss"],
        "final_entropy_loss": final_history["entropy_loss"],
        "final_front_grad_norm": final_history["front_grad_norm"],
        "final_scale_grad_norm": final_history["scale_grad_norm"],
        "class_accuracy_json": json.dumps(class_accuracy),
        "confusion_matrix_json": json.dumps(confusion_matrix),
        "source_train_hash": source_train_hash,
        "target_train_hash": target_train_hash,
        "target_test_hash": target_test_hash,
        "dataset_fingerprint": dataset_fingerprint,
        "checkpoint_path": checkpoint_rel.replace(os.sep, "/"),
        "_epoch_history": epoch_history,
        "_class_accuracy": class_accuracy,
        "_confusion_matrix": confusion_matrix,
    }


def sample_std(values):
    return stdev(values) if len(values) > 1 else 0.0


def seed_level_accuracy_values(group: Sequence[Dict], predicate):
    by_seed: Dict[str, List[float]] = {}
    for row in group:
        if predicate(row):
            seed_key = str(row.get("seed", "__single_seed__"))
            by_seed.setdefault(seed_key, []).append(float(row["accuracy"]))
    return [mean(values) for values in by_seed.values() if values]


def metric_summary(values: Sequence[float]):
    return {
        "mean": mean(values) if values else 0.0,
        "std": sample_std(list(values)),
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def summarize_results(rows: Sequence[Dict], include_suite=False):
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        if include_suite and row.get("suite") in {
            "snr_comparison",
            "wavelet_ablation",
            "structure_ablation",
            "basis_ablation",
            "factorial_ablation",
        }:
            key = f"{row['suite']}::{row['method']}::{normalize_snr(row.get('snr_db', ''))}"
        else:
            key = f"{row['suite']}::{row['method']}" if include_suite else row["method"]
        grouped.setdefault(key, []).append(row)

    summary = {}
    for key, group in grouped.items():
        same = seed_level_accuracy_values(group, lambda row: row["source_hp"] == row["target_hp"])
        cross = seed_level_accuracy_values(group, lambda row: row["source_hp"] != row["target_hp"])
        overall = seed_level_accuracy_values(group, lambda row: True)
        same_stats = metric_summary(same)
        cross_stats = metric_summary(cross)
        overall_stats = metric_summary(overall)
        params_values = [float(row["params_m"]) for row in group if row.get("params_m") not in {"", None}]
        summary[key] = {
            "same_mean": same_stats["mean"],
            "same_std": same_stats["std"],
            "same_min": same_stats["min"],
            "same_max": same_stats["max"],
            "cross_mean": cross_stats["mean"],
            "cross_std": cross_stats["std"],
            "cross_min": cross_stats["min"],
            "cross_max": cross_stats["max"],
            "overall_mean": overall_stats["mean"],
            "overall_std": overall_stats["std"],
            "overall_min": overall_stats["min"],
            "overall_max": overall_stats["max"],
            "params_m": mean(params_values) if params_values else 0.0,
            "runs": len(group),
        }
        if include_suite:
            parts = key.split("::")
            suite, method = parts[0], parts[1]
            summary[key]["suite"] = suite
            summary[key]["method"] = method
            summary[key]["snr_db"] = parts[2] if len(parts) > 2 else ""
            summary[key]["data_protocol"] = group[0].get("data_protocol", "holdout")
    return summary


def summarize_task_results(rows: Sequence[Dict]):
    grouped: Dict[Tuple[str, str, str, str, str], List[Dict]] = {}
    for row in rows:
        key = (
            row.get("suite", "main"),
            row["method"],
            row["source_hp"],
            row["target_hp"],
            normalize_snr(row.get("snr_db", "")),
        )
        grouped.setdefault(key, []).append(row)

    task_summary = {}
    for key, group in grouped.items():
        values = [float(row["accuracy"]) for row in group]
        macro_f1_values = [float(row["macro_f1"]) for row in group if row.get("macro_f1") not in {"", None}]
        params_values = [float(row["params_m"]) for row in group if row.get("params_m") not in {"", None}]
        stats = metric_summary(values)
        f1_stats = metric_summary(macro_f1_values)
        suite, method, source_hp, target_hp, snr = key
        summary_key = "::".join([suite, method, source_hp, target_hp, snr])
        task_summary[summary_key] = {
            "suite": suite,
            "method": method,
            "source_hp": source_hp,
            "target_hp": target_hp,
            "snr_db": snr,
            "accuracy_mean": stats["mean"],
            "accuracy_std": stats["std"],
            "accuracy_min": stats["min"],
            "accuracy_max": stats["max"],
            "macro_f1_mean": f1_stats["mean"],
            "macro_f1_std": f1_stats["std"],
            "params_m": mean(params_values) if params_values else 0.0,
            "runs": len(group),
        }
    return task_summary


def _paired_core_rows(rows, methods=CORE_BASIS_METHOD_NAMES, suite="basis_ablation"):
    indexed = {}
    for row in rows:
        if row.get("suite") != suite or row.get("method") not in methods:
            continue
        block = (
            str(row.get("source_rpm", "")),
            str(row.get("target_rpm", "")),
            str(row.get("seed", "")),
            normalize_snr(row.get("snr_db", "")),
        )
        indexed[(block, row["method"])] = row
    blocks = sorted({block for block, _ in indexed})
    complete = [block for block in blocks if all((block, method) in indexed for method in methods)]
    return indexed, complete


def rank_biserial_effect(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.abs(differences) > 1e-12]
    if not len(differences):
        return 0.0
    ranks = rankdata(np.abs(differences), method="average")
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    denominator = positive + negative
    return (positive - negative) / denominator if denominator else 0.0


def holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda idx: p_values[idx])
    adjusted = [1.0] * count
    running = 0.0
    for rank_idx, original_idx in enumerate(order):
        candidate = min(1.0, (count - rank_idx) * float(p_values[original_idx]))
        running = max(running, candidate)
        adjusted[original_idx] = running
    return adjusted


def analyze_core_basis_results(
    rows,
    methods=CORE_BASIS_METHOD_NAMES,
    suite="basis_ablation",
    expected_blocks=None,
    protocol_version=CORE_BASIS_PROTOCOL_VERSION,
):
    indexed, blocks = _paired_core_rows(rows, methods=methods, suite=suite)
    if expected_blocks is None:
        expected_blocks = len(CORE_BASIS_TASKS) * len(DEFAULT_SEEDS)
    score_matrix = np.asarray(
        [[float(indexed[(block, method)]["accuracy"]) for method in methods] for block in blocks],
        dtype=np.float64,
    )
    f1_matrix = np.asarray(
        [[float(indexed[(block, method)]["macro_f1"]) for method in methods] for block in blocks],
        dtype=np.float64,
    )
    rank_matrix = np.asarray([rankdata(-row, method="average") for row in score_matrix]) if len(blocks) else None

    def present_mean(method_rows, key):
        values = [float(row[key]) for row in method_rows if row.get(key) not in {"", None}]
        return mean(values) if values else 0.0

    summaries = []
    for method_idx, method in enumerate(methods):
        method_rows = [row for row in rows if row.get("suite") == suite and row.get("method") == method]
        accuracy_values = score_matrix[:, method_idx] if len(blocks) else np.asarray([], dtype=np.float64)
        f1_values = f1_matrix[:, method_idx] if len(blocks) else np.asarray([], dtype=np.float64)
        summaries.append(
            {
                "method": method,
                "paired_blocks": len(blocks),
                "accuracy_mean": float(np.mean(accuracy_values)) if len(accuracy_values) else 0.0,
                "accuracy_std": float(np.std(accuracy_values, ddof=1)) if len(accuracy_values) > 1 else 0.0,
                "accuracy_median": float(np.median(accuracy_values)) if len(accuracy_values) else 0.0,
                "accuracy_min": float(np.min(accuracy_values)) if len(accuracy_values) else 0.0,
                "macro_f1_mean": float(np.mean(f1_values)) if len(f1_values) else 0.0,
                "macro_f1_std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0,
                "average_rank": float(np.mean(rank_matrix[:, method_idx])) if rank_matrix is not None else 0.0,
                "params_m": mean([float(row["params_m"]) for row in method_rows]) if method_rows else 0.0,
                "inference_params_m": present_mean(method_rows, "inference_params_m"),
                "gflops": present_mean(method_rows, "gflops"),
                "inference_ms_mean": present_mean(method_rows, "inference_ms"),
                "training_seconds_mean": present_mean(method_rows, "training_seconds"),
                "runs_available": len(method_rows),
            }
        )

    friedman = {"statistic": None, "p_value": None}
    if len(blocks) >= 2:
        try:
            result = friedmanchisquare(*[score_matrix[:, idx] for idx in range(len(methods))])
            friedman = {"statistic": float(result.statistic), "p_value": float(result.pvalue)}
        except ValueError:
            pass

    pairwise = []
    for left_idx in range(len(methods)):
        for right_idx in range(left_idx + 1, len(methods)):
            differences = score_matrix[:, left_idx] - score_matrix[:, right_idx] if len(blocks) else np.asarray([])
            if not len(differences) or np.allclose(differences, 0.0):
                statistic, p_value = 0.0, 1.0
            else:
                result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            pairwise.append(
                {
                    "method_a": methods[left_idx],
                    "method_b": methods[right_idx],
                    "blocks": len(blocks),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                    "mean_difference_pp": float(np.mean(differences)) if len(differences) else 0.0,
                    "median_difference_pp": float(np.median(differences)) if len(differences) else 0.0,
                    "rank_biserial": rank_biserial_effect(differences),
                    "wins": int(np.sum(differences > 1e-12)),
                    "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                    "losses": int(np.sum(differences < -1e-12)),
                }
            )
    adjusted = holm_adjust([row["p_value"] for row in pairwise]) if pairwise else []
    for row, adjusted_p in zip(pairwise, adjusted):
        row["holm_p_value"] = adjusted_p

    analysis = {
        "protocol_version": protocol_version,
        "methods": list(methods),
        "expected_blocks": expected_blocks,
        "complete_blocks": len(blocks),
        "complete": len(blocks) == expected_blocks,
        "friedman": friedman,
        "pairwise_wilcoxon_holm": pairwise,
        "effect_size": "matched-pairs rank-biserial correlation; positive values favor method_a",
        "ranking": "rank 1 is best within each task-by-seed block",
    }
    return summaries, analysis


def analyze_factorial_results(rows, expected_methods=None):
    relevant = [
        row
        for row in rows
        if row.get("suite") == "factorial_ablation" and row.get("method") in FACTORIAL_METHOD_NAMES
    ]
    observed_methods = [
        method
        for method in FACTORIAL_METHOD_NAMES
        if any(row.get("method") == method for row in relevant)
    ]
    snr_values = {normalize_snr(row.get("snr_db", "")) for row in relevant}
    all_clean = bool(relevant) and snr_values == {""}
    all_minus_5db = bool(relevant) and snr_values == {"-5"}
    snr_condition = "clean" if all_clean else "-5 dB" if all_minus_5db else "mixed"
    if expected_methods is None:
        if all_clean and set(observed_methods).issubset(set(FACTORIAL_AC_METHOD_NAMES)):
            expected_methods = list(FACTORIAL_AC_METHOD_NAMES)
        else:
            expected_methods = list(FACTORIAL_METHOD_NAMES)
    else:
        expected_set = set(expected_methods)
        unknown_methods = expected_set - set(FACTORIAL_METHOD_NAMES)
        if unknown_methods:
            raise ValueError(f"Unknown expected factorial methods: {sorted(unknown_methods)}")
        expected_methods = [
            method for method in FACTORIAL_METHOD_NAMES if method in expected_set
        ]
    indexed = {}
    for row in relevant:
        key = (
            int(row["seed"]),
            str(row["source_rpm"]),
            str(row["target_rpm"]),
            str(row["method"]),
        )
        indexed[key] = row

    observed_tasks = {
        (str(row["source_rpm"]), str(row["target_rpm"]))
        for row in relevant
    }
    if not observed_tasks:
        factorial_tasks = list(FACTORIAL_TASKS)
    elif observed_tasks == set(FACTORIAL_ALL_CROSS_TASKS):
        factorial_tasks = list(FACTORIAL_ALL_CROSS_TASKS)
    elif observed_tasks == set(FACTORIAL_TASKS):
        factorial_tasks = list(FACTORIAL_TASKS)
    else:
        factorial_tasks = [task for task in FACTORIAL_ALL_CROSS_TASKS if task in observed_tasks]
    use_all_cross = factorial_tasks == FACTORIAL_ALL_CROSS_TASKS
    use_diagnostic_subset = (
        factorial_tasks != FACTORIAL_TASKS and factorial_tasks != FACTORIAL_ALL_CROSS_TASKS
    )
    use_strict_721 = bool(relevant) and all(row.get("data_protocol") == "strict_721" for row in relevant)
    use_b_fixed40_protocol = (
        bool(relevant)
        and all_clean
        and expected_methods == FACTORIAL_B_METHOD_NAMES
        and any(row.get("checkpoint_policy") == "final_epoch" for row in relevant)
    )
    use_b_early_stop_last_protocol = (
        bool(relevant)
        and all_clean
        and expected_methods == FACTORIAL_B_METHOD_NAMES
        and any(row.get("checkpoint_policy") == "early_stop_last" for row in relevant)
    )
    use_a_early_stop_last_protocol = (
        bool(relevant)
        and all_clean
        and expected_methods == FACTORIAL_A_METHOD_NAMES
        and any(row.get("checkpoint_policy") == "early_stop_last" for row in relevant)
    )
    use_c_early_stop_last_protocol = (
        bool(relevant)
        and all_clean
        and expected_methods == FACTORIAL_C_METHOD_NAMES
        and any(row.get("checkpoint_policy") == "early_stop_last" for row in relevant)
    )
    use_early_stop_last_protocol = (
        use_a_early_stop_last_protocol
        or use_b_early_stop_last_protocol
        or use_c_early_stop_last_protocol
    )
    all_b_fixed40_final = (
        use_b_fixed40_protocol
        and all(
            row.get("checkpoint_policy") == "final_epoch"
            and int(row.get("epochs_run", 0)) == 40
            and int(row.get("best_epoch", 0)) == 40
            and int(row.get("fixed_stop_epoch", 0)) == 40
            and int(row.get("schedule_horizon_epochs", 0)) == 100
            for row in relevant
        )
    )

    def valid_early_stop_last_row(row):
        try:
            epochs_run = int(row.get("epochs_run", 0))
            selected_epoch = int(row.get("best_epoch", 0))
            source_val_best_epoch = int(row.get("source_val_best_epoch", 0))
            schedule_horizon = int(row.get("schedule_horizon_epochs", 0))
            early_stopping_patience = int(row.get("early_stopping_patience", 0))
            patience_counter_at_stop = int(row.get("patience_counter_at_stop", -1))
            selected_source_val = float(row.get("source_val_accuracy", ""))
            max_source_val = float(row.get("max_source_val_accuracy", ""))
        except (TypeError, ValueError):
            return False
        stop_reason = row.get("stop_reason")
        valid_stop = (
            stop_reason == "early_stopping"
            and patience_counter_at_stop == early_stopping_patience
        ) or (
            stop_reason == "epoch_limit"
            and epochs_run == schedule_horizon
            and patience_counter_at_stop < early_stopping_patience
        )
        return (
            row.get("checkpoint_policy") == "early_stop_last"
            and str(row.get("fixed_stop_epoch", "")).strip() in {"", "None"}
            and schedule_horizon == 100
            and early_stopping_patience == 15
            and 1 <= epochs_run <= schedule_horizon
            and selected_epoch == epochs_run
            and 1 <= source_val_best_epoch <= epochs_run
            and epochs_run - source_val_best_epoch == patience_counter_at_stop
            and max_source_val + 1e-12 >= selected_source_val
            and valid_stop
        )

    all_early_stop_last = (
        use_early_stop_last_protocol
        and all(valid_early_stop_last_row(row) for row in relevant)
    )
    factorial_seeds = DEFAULT_SEEDS if use_strict_721 else FACTORIAL_SEEDS
    if use_diagnostic_subset:
        protocol_version = FACTORIAL_DIAGNOSTIC_PROTOCOL_VERSION
    elif use_strict_721 and use_all_cross:
        if use_a_early_stop_last_protocol:
            protocol_version = (
                FACTORIAL_A_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
            )
        elif use_c_early_stop_last_protocol:
            protocol_version = (
                FACTORIAL_C_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
            )
        elif use_b_early_stop_last_protocol:
            protocol_version = (
                FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_EARLY_STOP_LAST_PROTOCOL_VERSION
            )
        elif use_b_fixed40_protocol:
            protocol_version = (
                FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_FIXED40_PROTOCOL_VERSION
            )
        elif all_clean and expected_methods == FACTORIAL_AC_METHOD_NAMES:
            protocol_version = FACTORIAL_AC_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION
        elif all_clean and expected_methods == FACTORIAL_B_METHOD_NAMES:
            protocol_version = FACTORIAL_B_ALL_CROSS_STRICT721_CLEAN_PROTOCOL_VERSION
        else:
            protocol_version = FACTORIAL_ALL_CROSS_STRICT721_PROTOCOL_VERSION
    else:
        protocol_version = FACTORIAL_ALL_CROSS_PROTOCOL_VERSION if use_all_cross else FACTORIAL_PROTOCOL_VERSION

    def complete_seed_matrix(methods):
        complete = []
        values = []
        for seed in factorial_seeds:
            required = [
                (seed, source, target, method)
                for source, target in factorial_tasks
                for method in methods
            ]
            if all(key in indexed for key in required):
                complete.append(seed)
                values.append(
                    [
                        mean(
                            [
                                float(indexed[(seed, source, target, method)]["accuracy"])
                                for source, target in factorial_tasks
                            ]
                        )
                        for method in methods
                    ]
                )
        return complete, np.asarray(values, dtype=np.float64)

    complete_seeds, _ = complete_seed_matrix(expected_methods)
    observed_complete_seeds, _ = complete_seed_matrix(observed_methods)

    factors = {
        FACTORIAL_METHOD_NAMES[0]: ("Conv1D", "MLP"),
        FACTORIAL_METHOD_NAMES[1]: ("Conv1D", "Wavelet KAN"),
        FACTORIAL_METHOD_NAMES[2]: ("WKAN-Conv1D", "MLP"),
        FACTORIAL_METHOD_NAMES[3]: ("WKAN-Conv1D", "Wavelet KAN"),
    }
    summaries = []
    for method in observed_methods:
        method_complete_seeds = [
            seed
            for seed in factorial_seeds
            if all((seed, source, target, method) in indexed for source, target in factorial_tasks)
        ]
        values = np.asarray(
            [
                mean(
                    [
                        float(indexed[(seed, source, target, method)]["accuracy"])
                        for source, target in factorial_tasks
                    ]
                )
                for seed in method_complete_seeds
            ],
            dtype=np.float64,
        )
        method_rows = [row for row in relevant if row["method"] == method]
        frontend, classifier = factors[method]
        summaries.append(
            {
                "method": method,
                "frontend": frontend,
                "classifier": classifier,
                "complete_seeds": len(method_complete_seeds),
                "tasks_per_seed": len(factorial_tasks),
                "accuracy_mean": float(np.mean(values)) if len(values) else 0.0,
                "accuracy_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "accuracy_median": float(np.median(values)) if len(values) else 0.0,
                "accuracy_min": float(np.min(values)) if len(values) else 0.0,
                "accuracy_max": float(np.max(values)) if len(values) else 0.0,
                "params_m": mean([float(row["params_m"]) for row in method_rows]) if method_rows else 0.0,
                "runs_available": len(method_rows),
            }
        )

    contrasts = [
        ("WKAN minus Conv at MLP", np.asarray([-1.0, 0.0, 1.0, 0.0])),
        ("WKAN minus Conv at Wavelet KAN", np.asarray([0.0, -1.0, 0.0, 1.0])),
        ("Wavelet KAN minus MLP at Conv", np.asarray([-1.0, 1.0, 0.0, 0.0])),
        ("Wavelet KAN minus MLP at WKAN", np.asarray([0.0, 0.0, -1.0, 1.0])),
        ("Front-end main effect: WKAN minus Conv", np.asarray([-0.5, -0.5, 0.5, 0.5])),
        ("Classifier main effect: Wavelet KAN minus MLP", np.asarray([-0.5, 0.5, -0.5, 0.5])),
        ("Interaction: (D-C)-(B-A)", np.asarray([1.0, -1.0, -1.0, 1.0])),
    ]
    effects = []
    for name, coefficients in contrasts:
        required_methods = [
            method
            for method, coefficient in zip(FACTORIAL_METHOD_NAMES, coefficients)
            if not math.isclose(float(coefficient), 0.0)
        ]
        if (
            not (all_clean or all_minus_5db)
            or not set(required_methods).issubset(set(expected_methods))
        ):
            continue
        effect_seeds, effect_matrix = complete_seed_matrix(required_methods)
        if not effect_seeds:
            continue
        active_coefficients = np.asarray(
            [
                coefficient
                for coefficient in coefficients
                if not math.isclose(float(coefficient), 0.0)
            ],
            dtype=np.float64,
        )
        differences = effect_matrix @ active_coefficients
        if not len(differences) or np.allclose(differences, 0.0):
            statistic, p_value = 0.0, 1.0
        else:
            result = wilcoxon(differences, zero_method="wilcox", alternative="two-sided", method="auto")
            statistic, p_value = float(result.statistic), float(result.pvalue)
        delta_mean = float(np.mean(differences)) if len(differences) else 0.0
        delta_std = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
        if len(differences) > 1:
            half_width = float(scipy.stats.t.ppf(0.975, len(differences) - 1)) * delta_std / math.sqrt(
                len(differences)
            )
        else:
            half_width = 0.0
        effects.append(
            {
                "contrast": name,
                "paired_seeds": len(effect_seeds),
                "mean_delta_pp": delta_mean,
                "std_delta_pp": delta_std,
                "median_delta_pp": float(np.median(differences)) if len(differences) else 0.0,
                "ci95_low_pp": delta_mean - half_width,
                "ci95_high_pp": delta_mean + half_width,
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial": rank_biserial_effect(differences),
                "wins": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "losses": int(np.sum(differences < -1e-12)),
            }
        )
    adjusted = holm_adjust([row["p_value"] for row in effects]) if effects else []
    for row, adjusted_p in zip(effects, adjusted):
        row["holm_p_value"] = adjusted_p

    fingerprint_mismatches = []
    for seed in factorial_seeds:
        for source, target in factorial_tasks:
            block_rows = [
                indexed[(seed, source, target, method)]
                for method in expected_methods
                if (seed, source, target, method) in indexed
            ]
            if len(block_rows) != len(expected_methods):
                continue
            for field in ["source_train_hash", "target_train_hash", "target_test_hash", "dataset_fingerprint"]:
                if len({row.get(field, "") for row in block_rows}) != 1:
                    fingerprint_mismatches.append(
                        {"seed": seed, "source_rpm": source, "target_rpm": target, "field": field}
                    )

    expected_runs = len(expected_methods) * len(factorial_tasks) * len(factorial_seeds)
    expected_observed_runs = len(observed_methods) * len(factorial_tasks) * len(factorial_seeds)
    all_epochs_within_limit = all(1 <= int(row.get("epochs_run", 0)) <= 100 for row in relevant)
    all_valid_selected_epoch = all(
        1 <= int(row.get("best_epoch", 0)) <= int(row.get("epochs_run", 0))
        for row in relevant
    )
    protocol_checks = {
        "expected_runs": expected_runs,
        "available_runs": len(relevant),
        "all_rows_present": len(relevant) == expected_runs,
        "complete_seeds": complete_seeds,
        "all_seeds_complete": complete_seeds == factorial_seeds,
        "expected_methods": expected_methods,
        "observed_methods": observed_methods,
        "expected_observed_runs": expected_observed_runs,
        "all_observed_rows_present": len(relevant) == expected_observed_runs,
        "observed_complete_seeds": observed_complete_seeds,
        "all_observed_methods_complete": observed_complete_seeds == factorial_seeds,
        "all_strict_holdout": all(row.get("data_protocol") == "strict_holdout" for row in relevant),
        "all_strict_721": all(row.get("data_protocol") == "strict_721" for row in relevant),
        "snr_condition": snr_condition,
        "all_clean": all_clean,
        "all_minus_5db": all_minus_5db,
        "uniform_supported_snr": all_clean or all_minus_5db,
        "all_100_epochs": all(int(row.get("epochs_run", 0)) == 100 for row in relevant),
        "all_final_epoch_selected": all(int(row.get("best_epoch", 0)) == 100 for row in relevant),
        "all_fixed40_final_epoch_selected": all_b_fixed40_final,
        "all_early_stop_last_selected": all_early_stop_last,
        "all_epochs_within_limit": all_epochs_within_limit,
        "all_valid_selected_epoch": all_valid_selected_epoch,
        "dataset_fingerprints_match": not fingerprint_mismatches,
        "fingerprint_mismatches": fingerprint_mismatches,
    }
    completion_keys = [
        "all_rows_present",
        "all_seeds_complete",
        "uniform_supported_snr",
        "dataset_fingerprints_match",
    ]
    if use_strict_721:
        completion_keys.extend(["all_strict_721", "all_epochs_within_limit", "all_valid_selected_epoch"])
        if use_b_fixed40_protocol:
            completion_keys.append("all_fixed40_final_epoch_selected")
        if use_early_stop_last_protocol:
            completion_keys.append("all_early_stop_last_selected")
    else:
        completion_keys.extend(["all_strict_holdout", "all_100_epochs", "all_final_epoch_selected"])
    protocol_checks["complete"] = all(protocol_checks[key] for key in completion_keys)
    available_completion_keys = [
        "all_observed_rows_present",
        "all_observed_methods_complete",
        "uniform_supported_snr",
        "dataset_fingerprints_match",
    ]
    if use_strict_721:
        available_completion_keys.extend(
            ["all_strict_721", "all_epochs_within_limit", "all_valid_selected_epoch"]
        )
        if use_b_fixed40_protocol:
            available_completion_keys.append("all_fixed40_final_epoch_selected")
        if use_early_stop_last_protocol:
            available_completion_keys.append("all_early_stop_last_selected")
    else:
        available_completion_keys.extend(
            ["all_strict_holdout", "all_100_epochs", "all_final_epoch_selected"]
        )
    protocol_checks["available_methods_complete"] = bool(observed_methods) and all(
        protocol_checks[key] for key in available_completion_keys
    )
    ac_methods = [FACTORIAL_METHOD_NAMES[0], FACTORIAL_METHOD_NAMES[2]]
    ac_complete_seeds, _ = complete_seed_matrix(ac_methods)
    protocol_checks["ac_pair_complete"] = (
        all(method in observed_methods for method in ac_methods)
        and ac_complete_seeds == factorial_seeds
        and protocol_checks["uniform_supported_snr"]
        and protocol_checks["dataset_fingerprints_match"]
        and (
            (
                protocol_checks["all_strict_721"]
                and protocol_checks["all_epochs_within_limit"]
                and protocol_checks["all_valid_selected_epoch"]
            )
            if use_strict_721
            else (
                protocol_checks["all_strict_holdout"]
                and protocol_checks["all_100_epochs"]
                and protocol_checks["all_final_epoch_selected"]
            )
        )
    )
    analysis = {
        "protocol_version": protocol_version,
        "tasks": [list(task) for task in factorial_tasks],
        "seeds": list(factorial_seeds),
        "method_order": list(expected_methods),
        "aggregation": (
            f"For each seed, average the {len(factorial_tasks)} task accuracies; "
            "report mean +/- sample std over seeds."
        ),
        "effect_unit": "accuracy percentage points",
        "effects": effects,
        "protocol_checks": protocol_checks,
    }
    return summaries, effects, analysis


def format_mean_std(values, mean_key, std_key):
    return f"{values[mean_key]:.2f} $\\pm$ {values[std_key]:.2f}"


def format_factorial_latex_tables(summaries, effects, snr_condition="-5 dB", task_count=3):
    condition_caption = "clean-condition" if snr_condition == "clean" else "$-5$~dB"
    method_names = {row.get("method") for row in summaries}
    is_ac_pair = method_names == set(FACTORIAL_AC_METHOD_NAMES)
    comparison_caption = (
        "paired A/C front-end comparison with a fixed MLP-CDAN backend"
        if is_ac_pair
        else "front-end--classifier factorial ablation"
    )
    result_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        (
            f"\\caption{{CWRU {condition_caption} {task_count}-task "
            f"{comparison_caption}}}"
        ),
        "\\begin{tabular}{llcc}",
        "\\toprule",
        "Front end & Classifier & Accuracy (\\%) & Params (M) " + r"\\",
        "\\midrule",
    ]
    for row in summaries:
        result_lines.append(
            f"{row['frontend']} & {row['classifier']} & "
            f"{row['accuracy_mean']:.2f} $\\pm$ {row['accuracy_std']:.2f} & {row['params_m']:.3f} " + r"\\"
        )
    result_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])

    effect_lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        (
            "\\caption{Paired C--A front-end effect with a fixed MLP-CDAN backend}"
            if is_ac_pair
            else "\\caption{Paired effects in the CWRU $2\\times2$ factorial ablation}"
        ),
        "\\footnotesize",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Contrast & Mean $\\pm$ Std (pp) & 95\\% CI & Holm $p$ " + r"\\",
        "\\midrule",
    ]
    for row in effects:
        effect_lines.append(
            f"{row['contrast']} & {row['mean_delta_pp']:.2f} $\\pm$ {row['std_delta_pp']:.2f} & "
            f"[{row['ci95_low_pp']:.2f}, {row['ci95_high_pp']:.2f}] & {row['holm_p_value']:.4f} " + r"\\"
        )
    effect_lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n\n".join(["\n".join(result_lines), "\n".join(effect_lines)])


def format_latex_table(summary: Dict[str, Dict], caption="CWRU CDAN comparison"):
    method_order = [cfg.name for cfg in EXPERIMENTS] + [cfg.name for cfg in NOISE_ABLATIONS]
    ordered_keys = [key for key in method_order if key in summary]
    ordered_keys.extend(key for key in summary.keys() if key not in ordered_keys)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "Method & Same-load (\\%) & Cross-load Avg $\\pm$ Std & Min & Max & Overall (\\%) \\\\",
        "\\midrule",
    ]
    for key in ordered_keys:
        values = summary[key]
        method = values.get("method", key.split("::", 1)[-1])
        same = format_mean_std(values, "same_mean", "same_std")
        cross = format_mean_std(values, "cross_mean", "cross_std")
        cross_min = values.get("cross_min", values["cross_mean"])
        cross_max = values.get("cross_max", values["cross_mean"])
        overall = format_mean_std(values, "overall_mean", "overall_std")
        lines.append(f"{method} & {same} & {cross} & {cross_min:.2f} & {cross_max:.2f} & {overall} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def summarize_snr_results(rows: Sequence[Dict]):
    grouped: Dict[Tuple[str, str], List[Dict]] = {}
    for row in rows:
        grouped.setdefault((row["method"], normalize_snr(row.get("snr_db", ""))), []).append(row)

    summary = {}
    for (method, snr), group in grouped.items():
        cross = seed_level_accuracy_values(group, lambda row: row["source_hp"] != row["target_hp"])
        cross_stats = metric_summary(cross)
        summary[(method, snr)] = {
            "cross_mean": cross_stats["mean"],
            "cross_std": cross_stats["std"],
            "cross_min": cross_stats["min"],
            "cross_max": cross_stats["max"],
        }
    return summary


def format_snr_latex_table(rows: Sequence[Dict], caption="CWRU SNR robustness comparison"):
    summary = summarize_snr_results(rows)
    snr_columns = [("", "No noise"), ("-1", "SNR = $-1$~dB"), ("-3", "SNR = $-3$~dB"), ("-5", "SNR = $-5$~dB")]
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{方法} & " + " & ".join(f"\\textbf{{{label}}}" for _, label in snr_columns) + " \\\\",
        "\\midrule",
    ]
    for cfg in SNR_COMPARISON:
        values = []
        for snr, _ in snr_columns:
            cell = summary.get((cfg.name, snr))
            values.append(format_mean_std(cell, "cross_mean", "cross_std") if cell else "--")
        method = f"\\textbf{{{cfg.name}}}" if cfg.name == "WK-CDAN Full" else cfg.name
        lines.append(f"{method} & " + " & ".join(values) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def wavelet_label(method):
    labels = {
        "WK-CDAN-Mexhat": "Mexhat",
        "WK-CDAN-Morlet": "Morlet",
        "WK-CDAN-Laplace": "Laplace",
        "WK-CDAN-Shannon": "Shannon",
    }
    return labels.get(method, method)


def format_wavelet_ablation_latex_table(rows: Sequence[Dict], caption="CWRU mother wavelet ablation"):
    summary = summarize_snr_results(rows)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "\\textbf{小波基函数} & \\textbf{Clean 跨负载 (\\%)} & "
        "\\textbf{SNR = $-5$~dB 跨负载 (\\%)} & \\textbf{噪声降幅} \\\\",
        "\\midrule",
    ]
    for cfg in WAVELET_ABLATIONS:
        clean = summary.get((cfg.name, ""))
        noisy = summary.get((cfg.name, "-5"))
        clean_cell = format_mean_std(clean, "cross_mean", "cross_std") if clean else "--"
        noisy_cell = format_mean_std(noisy, "cross_mean", "cross_std") if noisy else "--"
        if clean and noisy:
            drop = clean["cross_mean"] - noisy["cross_mean"]
            drop_cell = f"{drop:.2f}"
        else:
            drop_cell = "--"
        method = f"\\textbf{{{wavelet_label(cfg.name)}}}" if cfg.wavelet == "mexhat" else wavelet_label(cfg.name)
        lines.append(f"{method} & {clean_cell} & {noisy_cell} & {drop_cell} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def format_basis_ablation_latex_table(rows: Sequence[Dict], caption="CWRU front-end basis ablation"):
    summary = summarize_results(rows)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\footnotesize",
        "\\renewcommand{\\arraystretch}{1.2}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "\\textbf{Front-end variant} & \\textbf{Same-load (\\%)} & "
        "\\textbf{Cross-load (\\%)} & \\textbf{Overall (\\%)} & \\textbf{Params (M)} \\\\",
        "\\midrule",
    ]
    for cfg in BASIS_ABLATIONS:
        values = summary.get(cfg.name)
        if not values:
            continue
        label = f"\\textbf{{{cfg.name}}}" if cfg.name == "Mexican Hat KANConv" else cfg.name
        lines.append(
            f"{label} & {format_mean_std(values, 'same_mean', 'same_std')} & "
            f"{format_mean_std(values, 'cross_mean', 'cross_std')} & "
            f"{format_mean_std(values, 'overall_mean', 'overall_std')} & "
            f"{values['params_m']:.3f} \\\\" 
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def _row_json_value(row, internal_key, csv_key, default):
    if internal_key in row:
        internal_value = row[internal_key]
        if internal_value is not None and internal_value != "":
            return internal_value
    value = row.get(csv_key, "")
    if value in {None, ""}:
        return default
    return json.loads(value)


def write_prediction_artifacts(rows, output_dir):
    class_accuracy_path = os.path.join(output_dir, "class_accuracy.csv")
    class_rows = []
    confusion_root = os.path.join(output_dir, "confusion_matrices")
    for row in rows:
        class_accuracy = _row_json_value(row, "_class_accuracy", "class_accuracy_json", [])
        confusion = _row_json_value(row, "_confusion_matrix", "confusion_matrix_json", [])
        if class_accuracy:
            for class_idx, value in enumerate(class_accuracy):
                class_rows.append(
                    {
                        "suite": row.get("suite", ""),
                        "method": row.get("method", ""),
                        "source_hp": row.get("source_hp", ""),
                        "target_hp": row.get("target_hp", ""),
                        "seed": row.get("seed", ""),
                        "snr_db": row.get("snr_db", ""),
                        "class_index": class_idx,
                        "class_name": LABELS[class_idx],
                        "class_accuracy": value,
                    }
                )
        if confusion:
            matrix_dir = os.path.join(confusion_root, safe_slug(row.get("method", "method")))
            ensure_dir(matrix_dir)
            snr_tag = "clean" if row.get("snr_db", "") in {"", None} else f"snr_{row['snr_db']}"
            matrix_name = (
                f"{row.get('source_hp', '')}_to_{row.get('target_hp', '')}_"
                f"seed_{row.get('seed', '')}_{snr_tag}.csv"
            )
            matrix_path = os.path.join(matrix_dir, matrix_name)
            with open(matrix_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["true\\pred", *LABELS])
                for label, values in zip(LABELS, confusion):
                    writer.writerow([label, *values])

    with open(class_accuracy_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "suite",
            "method",
            "source_hp",
            "target_hp",
            "seed",
            "snr_db",
            "class_index",
            "class_name",
            "class_accuracy",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(class_rows)
    return class_accuracy_path, confusion_root


def write_outputs(rows, output_dir, expected_factorial_methods=None):
    ensure_dir(output_dir)
    raw_path = os.path.join(output_dir, "raw_results.csv")
    epoch_history_path = os.path.join(output_dir, "epoch_history.csv")
    fieldnames = [
        "suite",
        "method",
        "data_protocol",
        "source_rpm",
        "target_rpm",
        "source_hp",
        "target_hp",
        "seed",
        "snr_db",
        "wavelet",
        "backbone",
        "classifier",
        "classifier_wavelet",
        "factor_frontend",
        "factor_classifier",
        "accuracy",
        "macro_f1",
        "source_val_accuracy",
        "max_source_val_accuracy",
        "source_val_best_epoch",
        "checkpoint_policy",
        "fixed_stop_epoch",
        "schedule_horizon_epochs",
        "stop_reason",
        "early_stopping_patience",
        "patience_counter_at_stop",
        "best_epoch",
        "epochs_run",
        "params_m",
        "inference_params_m",
        "gflops",
        "inference_ms",
        "inference_ms_std",
        "inference_warmup",
        "inference_repeats",
        "training_seconds",
        "final_train_loss",
        "final_classification_loss",
        "final_domain_loss",
        "final_center_loss",
        "final_entropy_loss",
        "final_front_grad_norm",
        "final_scale_grad_norm",
        "class_accuracy_json",
        "confusion_matrix_json",
        "source_train_hash",
        "target_train_hash",
        "target_test_hash",
        "dataset_fingerprint",
        "checkpoint_path",
    ]
    with open(raw_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    history_fieldnames = [
        "suite",
        "method",
        "data_protocol",
        "source_rpm",
        "target_rpm",
        "source_hp",
        "target_hp",
        "seed",
        "snr_db",
        "wavelet",
        "train_loss",
        "classification_loss",
        "domain_loss",
        "center_loss",
        "entropy_loss",
        "jmmd_loss",
        "pseudo_loss",
        "front_grad_norm",
        "scale_grad_norm",
        "epoch",
        "source_val_accuracy",
        "source_val_macro_f1",
        "selected",
    ]
    history_rows = []
    for row in rows:
        for epoch_row in row.get("_epoch_history", []) or []:
            history_rows.append(
                {
                    "suite": row.get("suite", ""),
                    "method": row.get("method", ""),
                    "data_protocol": row.get("data_protocol", ""),
                    "source_rpm": row.get("source_rpm", ""),
                    "target_rpm": row.get("target_rpm", ""),
                    "source_hp": row.get("source_hp", ""),
                    "target_hp": row.get("target_hp", ""),
                    "seed": row.get("seed", ""),
                    "snr_db": row.get("snr_db", ""),
                    "wavelet": row.get("wavelet", ""),
                    **epoch_row,
                }
            )
    with open(epoch_history_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history_fieldnames)
        writer.writeheader()
        writer.writerows(history_rows)

    summary = summarize_results(rows, include_suite=True)
    summary_path = os.path.join(output_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "suite",
            "method",
            "data_protocol",
            "snr_db",
            "same_mean",
            "same_std",
            "same_min",
            "same_max",
            "cross_mean",
            "cross_std",
            "cross_min",
            "cross_max",
            "overall_mean",
            "overall_std",
            "overall_min",
            "overall_max",
            "params_m",
            "runs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key, values in summary.items():
            writer.writerow(
                {
                    "suite": values["suite"],
                    "method": values["method"],
                    "data_protocol": values.get("data_protocol", ""),
                    "snr_db": values.get("snr_db", ""),
                    **values,
                }
            )

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    task_summary = summarize_task_results(rows)
    task_summary_path = os.path.join(output_dir, "task_summary.csv")
    with open(task_summary_path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "suite",
            "method",
            "source_hp",
            "target_hp",
            "snr_db",
            "accuracy_mean",
            "accuracy_std",
            "accuracy_min",
            "accuracy_max",
            "macro_f1_mean",
            "macro_f1_std",
            "params_m",
            "runs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for values in task_summary.values():
            writer.writerow(values)

    task_summary_json_path = os.path.join(output_dir, "task_summary.json")
    with open(task_summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(task_summary, handle, ensure_ascii=False, indent=2)

    class_accuracy_path, confusion_root = write_prediction_artifacts(rows, output_dir)

    core_summary_path = ""
    core_summary_json_path = ""
    pairwise_stats_path = ""
    statistical_analysis_path = ""
    if any(
        row.get("suite") == "basis_ablation" and row.get("method") in CORE_BASIS_METHOD_NAMES
        for row in rows
    ):
        core_summary, core_analysis = analyze_core_basis_results(rows)
        core_summary_path = os.path.join(output_dir, "core_summary.csv")
        with open(core_summary_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(core_summary[0].keys()) if core_summary else ["method"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(core_summary)
        core_summary_json_path = os.path.join(output_dir, "core_summary.json")
        with open(core_summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(core_summary, handle, ensure_ascii=False, indent=2)
        pairwise_stats_path = os.path.join(output_dir, "pairwise_statistics.csv")
        pairwise_rows = core_analysis["pairwise_wilcoxon_holm"]
        with open(pairwise_stats_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(pairwise_rows[0].keys()) if pairwise_rows else ["method_a", "method_b"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pairwise_rows)
        statistical_analysis_path = os.path.join(output_dir, "statistical_analysis.json")
        with open(statistical_analysis_path, "w", encoding="utf-8") as handle:
            json.dump(core_analysis, handle, ensure_ascii=False, indent=2)

    clean_summary_path = ""
    clean_summary_json_path = ""
    clean_pairwise_stats_path = ""
    clean_statistical_analysis_path = ""
    if any(row.get("suite") == "basis_clean_exploratory" for row in rows):
        clean_summary, clean_analysis = analyze_core_basis_results(
            rows,
            methods=CLEAN_BASIS_METHOD_NAMES,
            suite="basis_clean_exploratory",
            expected_blocks=len(CLEAN_BASIS_TASKS) * len(DEFAULT_SEEDS),
            protocol_version=CLEAN_BASIS_PROTOCOL_VERSION,
        )
        clean_summary_path = os.path.join(output_dir, "clean_basis_summary.csv")
        with open(clean_summary_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(clean_summary[0].keys()) if clean_summary else ["method"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_summary)
        clean_summary_json_path = os.path.join(output_dir, "clean_basis_summary.json")
        with open(clean_summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(clean_summary, handle, ensure_ascii=False, indent=2)
        clean_pairwise_stats_path = os.path.join(output_dir, "clean_pairwise_statistics.csv")
        clean_pairwise_rows = clean_analysis["pairwise_wilcoxon_holm"]
        with open(clean_pairwise_stats_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(clean_pairwise_rows[0].keys()) if clean_pairwise_rows else ["method_a", "method_b"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(clean_pairwise_rows)
        clean_statistical_analysis_path = os.path.join(output_dir, "clean_statistical_analysis.json")
        with open(clean_statistical_analysis_path, "w", encoding="utf-8") as handle:
            json.dump(clean_analysis, handle, ensure_ascii=False, indent=2)

    factorial_summary_path = ""
    factorial_summary_json_path = ""
    factorial_effects_path = ""
    factorial_effects_json_path = ""
    factorial_analysis_path = ""
    factorial_summary = []
    factorial_effects = []
    if any(row.get("suite") == "factorial_ablation" for row in rows):
        factorial_summary, factorial_effects, factorial_analysis = analyze_factorial_results(
            rows,
            expected_methods=expected_factorial_methods,
        )
        factorial_summary_path = os.path.join(output_dir, "factorial_summary.csv")
        with open(factorial_summary_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(factorial_summary[0].keys()) if factorial_summary else ["method"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(factorial_summary)
        factorial_summary_json_path = os.path.join(output_dir, "factorial_summary.json")
        with open(factorial_summary_json_path, "w", encoding="utf-8") as handle:
            json.dump(factorial_summary, handle, ensure_ascii=False, indent=2)
        factorial_effects_path = os.path.join(output_dir, "factorial_effects.csv")
        with open(factorial_effects_path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = list(factorial_effects[0].keys()) if factorial_effects else ["contrast"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(factorial_effects)
        factorial_effects_json_path = os.path.join(output_dir, "factorial_effects.json")
        with open(factorial_effects_json_path, "w", encoding="utf-8") as handle:
            json.dump(factorial_effects, handle, ensure_ascii=False, indent=2)
        factorial_analysis_path = os.path.join(output_dir, "factorial_analysis.json")
        with open(factorial_analysis_path, "w", encoding="utf-8") as handle:
            json.dump(factorial_analysis, handle, ensure_ascii=False, indent=2)

    latex_path = os.path.join(output_dir, "latex_tables.tex")
    main_rows = [row for row in rows if row["suite"] == "main"]
    noise_rows = [row for row in rows if row["suite"] == "noise_ablation"]
    snr_rows = [row for row in rows if row["suite"] == "snr_comparison"]
    wavelet_rows = [row for row in rows if row["suite"] == "wavelet_ablation"]
    structure_rows = [row for row in rows if row["suite"] == "structure_ablation"]
    basis_rows = [row for row in rows if row["suite"] == "basis_ablation"]
    clean_basis_rows = [row for row in rows if row["suite"] == "basis_clean_exploratory"]
    factorial_rows = [row for row in rows if row["suite"] == "factorial_ablation"]
    latex_chunks = []
    if main_rows:
        latex_chunks.append(format_latex_table(summarize_results(main_rows), "CWRU CDAN-focused comparison"))
    if snr_rows:
        latex_chunks.append(format_snr_latex_table(snr_rows, "CWRU SNR robustness comparison"))
    if wavelet_rows:
        latex_chunks.append(format_wavelet_ablation_latex_table(wavelet_rows, "CWRU mother wavelet ablation"))
    if structure_rows:
        latex_chunks.append(format_latex_table(summarize_results(structure_rows), "CWRU -5 dB structural ablation"))
    if basis_rows:
        latex_chunks.append(
            format_basis_ablation_latex_table(basis_rows, "CWRU -5 dB front-end basis and operator ablation")
        )
    if clean_basis_rows:
        latex_chunks.append(
            format_basis_ablation_latex_table(
                clean_basis_rows,
                "CWRU clean-condition front-end basis and operator comparison",
            )
        )
    if noise_rows:
        latex_chunks.append(format_latex_table(summarize_results(noise_rows), "CWRU -5 dB loss ablation"))
    if factorial_rows:
        latex_chunks.append(
            format_factorial_latex_tables(
                factorial_summary,
                factorial_effects,
                snr_condition=factorial_analysis["protocol_checks"]["snr_condition"],
                task_count=len(factorial_analysis["tasks"]),
            )
        )
    with open(latex_path, "w", encoding="utf-8") as handle:
        handle.write("\n\n".join(latex_chunks))

    return {
        "raw": raw_path,
        "epoch_history": epoch_history_path,
        "summary_csv": summary_path,
        "summary_json": json_path,
        "task_summary_csv": task_summary_path,
        "task_summary_json": task_summary_json_path,
        "class_accuracy": class_accuracy_path,
        "confusion_matrices": confusion_root,
        "core_summary_csv": core_summary_path,
        "core_summary_json": core_summary_json_path,
        "pairwise_statistics": pairwise_stats_path,
        "statistical_analysis": statistical_analysis_path,
        "clean_basis_summary_csv": clean_summary_path,
        "clean_basis_summary_json": clean_summary_json_path,
        "clean_pairwise_statistics": clean_pairwise_stats_path,
        "clean_statistical_analysis": clean_statistical_analysis_path,
        "factorial_summary_csv": factorial_summary_path,
        "factorial_summary_json": factorial_summary_json_path,
        "factorial_effects_csv": factorial_effects_path,
        "factorial_effects_json": factorial_effects_json_path,
        "factorial_analysis": factorial_analysis_path,
        "latex": latex_path,
    }


def load_existing_rows(output_dir):
    raw_path = os.path.join(output_dir, "raw_results.csv")
    if not os.path.exists(raw_path):
        return []
    with open(raw_path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    history_path = os.path.join(output_dir, "epoch_history.csv")
    if not os.path.exists(history_path):
        return rows
    with open(history_path, "r", newline="", encoding="utf-8") as handle:
        history_rows = list(csv.DictReader(handle))

    history_metadata_fields = {
        "suite",
        "method",
        "data_protocol",
        "source_rpm",
        "target_rpm",
        "source_hp",
        "target_hp",
        "seed",
        "snr_db",
        "wavelet",
    }
    histories = {}
    for history_row in history_rows:
        key = result_key(history_row)
        epoch_row = {
            name: value
            for name, value in history_row.items()
            if name not in history_metadata_fields
        }
        histories.setdefault(key, []).append(epoch_row)
    for history in histories.values():
        history.sort(key=lambda item: int(item["epoch"]))
    for row in rows:
        row["_epoch_history"] = histories.get(result_key(row), [])
    return rows


def normalize_snr(value):
    if value is None or value == "":
        return ""
    return f"{float(value):g}"


def result_key(row):
    return (
        str(row.get("data_protocol", "holdout")),
        str(row["suite"]),
        str(row["method"]),
        str(row["source_rpm"]),
        str(row["target_rpm"]),
        str(row["seed"]),
        normalize_snr(row.get("snr_db", "")),
    )


def expected_key(exp_cfg, source_rpm, target_rpm, seed, current_snr, data_protocol="holdout"):
    return (
        data_protocol,
        exp_cfg.suite,
        exp_cfg.name,
        source_rpm,
        target_rpm,
        str(seed),
        normalize_snr(current_snr),
    )


def suite_snr_values(exp_cfg, snr_db, snr_levels, wavelet_snr_levels=None):
    if exp_cfg.suite in {"noise_ablation", "structure_ablation", "basis_ablation", "factorial_ablation"}:
        return [snr_db]
    if exp_cfg.suite == "snr_comparison":
        return snr_levels
    if exp_cfg.suite == "wavelet_ablation":
        return wavelet_snr_levels or DEFAULT_WAVELET_SNR_LEVELS
    return [None]


def filter_pending_runs(
    experiments,
    sources,
    targets,
    seeds,
    snr_db,
    existing_rows,
    snr_levels=None,
    wavelet_snr_levels=None,
    data_protocol="holdout",
    task_pairs=None,
):
    existing_keys = {result_key(row) for row in existing_rows}
    pending = []
    snr_levels = snr_levels or DEFAULT_SNR_LEVELS
    pairs = list(task_pairs) if task_pairs is not None else [(source, target) for source in sources for target in targets]
    for exp_cfg in experiments:
        for seed in seeds:
            for source_rpm, target_rpm in pairs:
                for current_snr in suite_snr_values(
                    exp_cfg, snr_db, snr_levels, wavelet_snr_levels=wavelet_snr_levels
                ):
                    key = expected_key(
                        exp_cfg,
                        source_rpm,
                        target_rpm,
                        seed,
                        current_snr,
                        data_protocol=data_protocol,
                    )
                    if key not in existing_keys:
                        pending.append((exp_cfg, source_rpm, target_rpm, seed, current_snr))
    return pending


def parse_csv_list(value):
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_task_pairs(value):
    pairs = []
    for item in parse_csv_list(value):
        separator = ":" if ":" in item else "->" if "->" in item else None
        if separator is None:
            raise ValueError(f"Invalid task pair {item!r}; use SOURCE:TARGET")
        source, target = [part.strip() for part in item.split(separator, 1)]
        if source not in ALL_RPMS or target not in ALL_RPMS:
            raise ValueError(f"Unknown RPM pair: {source}:{target}")
        pairs.append((source, target))
    return pairs


def parse_snr_levels(value):
    if not value:
        return DEFAULT_SNR_LEVELS
    levels = []
    for item in parse_csv_list(value):
        if item.lower() in {"clean", "none", "no_noise", "no-noise"}:
            levels.append(None)
        else:
            levels.append(float(item))
    return levels


def select_experiments(
    names,
    include_noise,
    include_snr=False,
    include_wavelet=False,
    include_structure=False,
    include_basis=False,
):
    experiments = list(EXPERIMENTS)
    if include_noise:
        experiments.extend(NOISE_ABLATIONS)
    if include_snr:
        experiments.extend(SNR_COMPARISON)
    if include_wavelet:
        experiments.extend(WAVELET_ABLATIONS)
    if include_structure:
        experiments.extend(STRUCTURE_ABLATIONS)
    if include_basis:
        experiments.extend(BASIS_ABLATIONS)
    if not names:
        return experiments
    wanted = set(names)
    selected = [cfg for cfg in experiments if cfg.name in wanted]
    missing = wanted - {cfg.name for cfg in selected}
    if missing:
        raise ValueError(f"Unknown method names: {sorted(missing)}")
    return selected


def run_experiments(
    config,
    experiments,
    sources,
    targets,
    seeds,
    snr_db,
    existing_rows=None,
    snr_levels=None,
    wavelet_snr_levels=None,
    task_pairs=None,
):
    metadata_map = load_metadata_map(config.metadata_path)
    rows = list(existing_rows or [])
    pending = filter_pending_runs(
        experiments,
        sources,
        targets,
        seeds,
        snr_db,
        rows,
        snr_levels=snr_levels,
        wavelet_snr_levels=wavelet_snr_levels,
        data_protocol=config.data_protocol,
        task_pairs=task_pairs,
    )
    total = len(pending)
    if rows:
        print(f"Loaded {len(rows)} existing result rows; {total} runs remain.", flush=True)
    for counter, (exp_cfg, source_rpm, target_rpm, seed, current_snr) in enumerate(pending, start=1):
        print(
            f"[{counter}/{total}] {exp_cfg.name} | seed={seed} | "
            f"{LOAD_MAP[source_rpm]}->{LOAD_MAP[target_rpm]} | snr={current_snr}",
            flush=True,
        )
        row = train_one_pair(exp_cfg, config, metadata_map, source_rpm, target_rpm, seed, current_snr)
        rows.append(row)
        write_outputs(
            rows,
            config.output_dir,
            expected_factorial_methods=[
                cfg.name for cfg in experiments if cfg.suite == "factorial_ablation"
            ],
        )
        print(f"  accuracy={row['accuracy']:.2f}% macro_f1={row['macro_f1']:.2f}%", flush=True)
    return rows


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def runtime_environment():
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
    }


def write_experiment_manifest(
    config,
    experiments,
    task_pairs,
    seeds,
    snr_db,
    argv,
    protocol_version=CORE_BASIS_PROTOCOL_VERSION,
    noise_protocol="window z-score, then deterministic AWGN per split/class",
):
    ensure_dir(config.output_dir)
    script_path = os.path.abspath(__file__)
    signature = {
        "protocol_version": protocol_version,
        "git_commit": current_git_commit(),
        "script_sha256": file_sha256(script_path),
        "config": asdict(config),
        "experiments": [asdict(cfg) for cfg in experiments],
        "task_pairs": [list(pair) for pair in task_pairs],
        "seeds": list(seeds),
        "snr_db": snr_db,
        "labels": LABELS,
        "noise_protocol": noise_protocol,
        "checkpoint_selection": (
            (
                f"source-validation early stopping with patience {config.patience}; "
                "select the last epoch reached without restoring the source-validation-best "
                f"state; GRL schedule horizon {config.epochs}; target test evaluated once"
            )
            if config.checkpoint_policy == "early_stop_last"
            else
            (
                f"final epoch {config.fixed_stop_epoch}; source validation monitored only; "
                f"early stopping disabled; GRL schedule horizon {config.epochs}; "
                "target test evaluated once"
            )
            if config.checkpoint_policy == "final_epoch"
            else "best source-domain validation accuracy; target test evaluated once after model selection"
            if config.data_protocol in {"holdout", "strict_721"}
            else "final epoch; no validation or early stopping"
        ),
    }
    if config.checkpoint_policy == "early_stop_last":
        preregistration_records = FACTORIAL_EARLY_STOP_LAST_PREREGISTRATIONS.get(
            protocol_version
        )
        if preregistration_records is None:
            raise ValueError(
                "early_stop_last requires a protocol-specific frozen preregistration binding"
            )
        preregistration = {}
        for label, record in preregistration_records.items():
            absolute_path = os.path.join(ROOT, *record["path"].split("/"))
            actual_sha256 = file_sha256(absolute_path)
            if actual_sha256 != record["sha256"]:
                raise ValueError(
                    f"Frozen preregistration hash mismatch for {record['path']}: "
                    f"expected {record['sha256']}, got {actual_sha256}"
                )
            preregistration[label] = dict(record)
        signature["preregistration"] = preregistration
    path = os.path.join(config.output_dir, "experiment_manifest.json")
    command = " ".join([sys.executable, script_path, *argv])
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing.get("frozen_signature") != signature:
            raise ValueError(
                "The output directory contains a different frozen experiment signature; "
                "use a new output directory instead of mixing protocols."
            )
        commands = existing.get("commands", [])
        if command not in commands:
            commands.append(command)
        existing["commands"] = commands
        existing["last_resumed_utc"] = datetime.now(timezone.utc).isoformat()
        manifest = existing
    else:
        manifest = {
            "frozen_signature": signature,
            "environment": runtime_environment(),
            "commands": [command],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "preflight_test_command": (
                f"{sys.executable} {os.path.join(ROOT, 'tests', 'test_cwru_ieee_revision.py')}"
            ),
        }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return path


