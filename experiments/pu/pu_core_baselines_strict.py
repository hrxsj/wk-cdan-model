"""PU source-only, DANN, CDAN, and DSAN/LMMD baseline experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
STRICT_DATA_SCRIPT = Path(__file__).with_name("pu_frontend_ablation.py")


def _load_strict_data_protocol():
    """Load the adjacent strict split implementation without mutating it."""

    module_name = "_pu_frontend_ablation_strict_dependency"
    spec = importlib.util.spec_from_file_location(module_name, STRICT_DATA_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load strict PU protocol: {STRICT_DATA_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


strict_data = _load_strict_data_protocol()

PROTOCOL_VERSION = "pu-core-baselines-strict-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "pu_core_baselines_strict_100epoch"
SMOKE_OUTPUT_DIR = REPO_ROOT / "runs" / "pu_core_baselines_strict_smoke"
DEFAULT_SEEDS = tuple(DEFAULT_RUN_SEEDS)
DEFAULT_TASKS = tuple(strict_data.DEFAULT_TASKS)
CONDITIONS = strict_data.CONDITIONS
LABELS = strict_data.LABELS
NUM_CLASSES = len(LABELS)

METHOD_WDCNN = "WDCNN (source-only)"
METHOD_RESNET = "ResNet-18 (1D, source-only)"
METHOD_BKAN = "B-spline KANConv (source-only)"
METHOD_DANN = "DANN + ResNet-18 (1D)"
METHOD_CDAN = "CDAN + ResNet-18 (1D)"
METHOD_DSAN = "DSAN/LMMD + ResNet-18 (1D)"
METHODS = (
    METHOD_WDCNN,
    METHOD_RESNET,
    METHOD_BKAN,
    METHOD_DANN,
    METHOD_CDAN,
    METHOD_DSAN,
)
DEFAULT_METHOD_ALIASES = "wdcnn,resnet,bkan,dann,cdan,dsan"
SOURCE_ONLY_METHODS = {METHOD_WDCNN, METHOD_RESNET, METHOD_BKAN}
ADVERSARIAL_METHODS = {METHOD_DANN, METHOD_CDAN}
UDA_METHODS = ADVERSARIAL_METHODS | {METHOD_DSAN}

METHOD_DEFINITIONS: Mapping[str, Mapping[str, object]] = {
    METHOD_WDCNN: {
        "family": "source_only",
        "feature_extractor": (
            "WDCNN wide-kernel 1D CNN: channels 16-32-64-128, first kernel 64, "
            "stride 8, global average pooling"
        ),
        "classifier": "linear 128->3",
        "adaptation": "none",
    },
    METHOD_RESNET: {
        "family": "source_only",
        "feature_extractor": (
            "1D ResNet-18 [2,2,2,2], widths 32-64-128-256, 128-D projection"
        ),
        "classifier": "linear 128->3",
        "adaptation": "none",
    },
    METHOD_BKAN: {
        "family": "source_only",
        "feature_extractor": (
            "cubic B-spline KANConv1D first layer (kernel 31, stride 2, "
            "grid_size 5, order 3) + CNN trunk + SE, 128-D output"
        ),
        "classifier": "cubic B-spline KANLinear 128->3, grid_size 5, order 3",
        "adaptation": "none",
    },
    METHOD_DANN: {
        "family": "unsupervised_domain_adaptation",
        "feature_extractor": "same 1D ResNet-18 and classifier as ResNet source-only",
        "classifier": "linear 128->3",
        "adaptation": "DANN gradient reversal on 128-D features; domain CE",
    },
    METHOD_CDAN: {
        "family": "unsupervised_domain_adaptation",
        "feature_extractor": "same 1D ResNet-18 and classifier as ResNet source-only",
        "classifier": "linear 128->3",
        "adaptation": (
            "CDAN gradient reversal on vec(f outer softmax(z)); domain CE"
        ),
    },
    METHOD_DSAN: {
        "family": "unsupervised_domain_adaptation",
        "feature_extractor": "same 1D ResNet-18 and classifier as ResNet source-only",
        "classifier": "linear 128->3",
        "adaptation": (
            "class-conditional multi-kernel Gaussian LMMD; common classes are the "
            "intersection of source labels and target argmax pseudo-labels, while "
            "detached target soft probabilities provide within-class weights"
        ),
    },
}


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: str = os.environ.get("PU_DATA_ROOT", str(REPO_ROOT / "Data" / "PU"))
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    window_size: int = 2048
    overlap: float = 0.5
    train_ratio: float = 0.8
    batch_size: int = 64
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 20
    adaptation_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_records_per_bearing: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    smoke_windows_per_class: Optional[int] = None
    log_every: int = 10


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class SEBlock1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.excitation(self.squeeze(x))


class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


def make_resnet_layer(
    in_channels: int, out_channels: int, blocks: int, stride: int
):
    layers: List[nn.Module] = [BasicBlock1D(in_channels, out_channels, stride)]
    layers.extend(BasicBlock1D(out_channels, out_channels) for _ in range(1, blocks))
    return nn.Sequential(*layers)


class ResNet18Backbone1D(nn.Module):
    """The same compact 1D ResNet-18 definition used by the CWRU benchmark."""

    output_dim = 128

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
            nn.Linear(256, self.output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.head(x)


class WDCNNBackbone(nn.Module):
    """Wide-kernel 1D CNN aligned with the CWRU comparison implementation."""

    output_dim = 128

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor):
        return self.features(x)


class KANLinearBSpline(nn.Module):
    """Cubic B-spline KAN linear map with an explicit fixed knot grid."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-3.0, 3.0),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        grid_min, grid_max = grid_range
        spacing = (grid_max - grid_min) / grid_size
        grid = (
            torch.arange(
                -spline_order,
                grid_size + spline_order + 1,
                dtype=torch.float32,
            )
            * spacing
            + grid_min
        )
        self.register_buffer("grid", grid)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = nn.SiLU()
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5.0))
        nn.init.normal_(self.spline_weight, mean=0.0, std=0.02)

    def b_splines(self, x: torch.Tensor):
        expanded = x.unsqueeze(-1)
        bases = (
            (expanded >= self.grid[:-1]) & (expanded < self.grid[1:])
        ).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left_denominator = self.grid[order:-1] - self.grid[: -(order + 1)]
            right_denominator = (
                self.grid[order + 1 :] - self.grid[1:-order]
            )
            left = (expanded - self.grid[: -(order + 1)]) / left_denominator
            right = (self.grid[order + 1 :] - expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x: torch.Tensor):
        base = F.linear(self.base_activation(x), self.base_weight)
        spline = F.linear(
            self.b_splines(x).reshape(x.size(0), -1),
            self.spline_weight.reshape(self.out_features, -1),
        )
        return base + spline


class KANConv1dBSpline(nn.Module):
    """Cubic B-spline KAN convolution used as a temporal first layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Tuple[float, float] = (-3.0, 3.0),
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
        spacing = (grid_max - grid_min) / grid_size
        grid = (
            torch.arange(
                -spline_order,
                grid_size + spline_order + 1,
                dtype=torch.float32,
            )
            * spacing
            + grid_min
        )
        self.register_buffer("grid", grid)
        self.base_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size) * 0.1
        )
        self.spline_weights = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                kernel_size,
                grid_size + spline_order,
            )
            * 0.02
        )
        self.base_activation = nn.SiLU()

    def b_splines(self, x: torch.Tensor):
        expanded = x.unsqueeze(-1)
        bases = (
            (expanded >= self.grid[:-1]) & (expanded < self.grid[1:])
        ).to(x.dtype)
        for order in range(1, self.spline_order + 1):
            left_denominator = self.grid[order:-1] - self.grid[: -(order + 1)]
            right_denominator = (
                self.grid[order + 1 :] - self.grid[1:-order]
            )
            left = (expanded - self.grid[: -(order + 1)]) / left_denominator
            right = (self.grid[order + 1 :] - expanded) / right_denominator
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )
        unfolded = unfolded.view(
            batch_size, self.in_channels, self.kernel_size, -1
        ).permute(0, 3, 1, 2)
        base = torch.einsum(
            "blic,oic->blo", self.base_activation(unfolded), self.base_weights
        )
        spline = torch.einsum(
            "blicg,oicg->blo", self.b_splines(unfolded), self.spline_weights
        )
        return (base + spline).permute(0, 2, 1)


class BSplineKANConvBackbone(nn.Module):
    output_dim = 128

    def __init__(self):
        super().__init__()
        self.first = KANConv1dBSpline(
            1,
            32,
            kernel_size=31,
            stride=2,
            padding=15,
            grid_size=5,
            spline_order=3,
        )
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2, 2)
        self.mid_cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1D(128, reduction=8)
        self.tail = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.bn1(self.first(x)))
        x = self.mid_cnn(x)
        x = self.se(x)
        return torch.tanh(self.tail(x))


class CoreBaselineModel(nn.Module):
    """One explicit model wrapper for all source-only and UDA baselines."""

    def __init__(
        self, method: str, num_classes: int = NUM_CLASSES, init_seed: int = resolve_seed(None)
    ):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.num_classes = num_classes
        self.domain_mode = (
            "dann"
            if method == METHOD_DANN
            else "cdan"
            if method == METHOD_CDAN
            else "lmmd"
            if method == METHOD_DSAN
            else "none"
        )

        if method == METHOD_WDCNN:
            backbone_factory = WDCNNBackbone
        elif method == METHOD_BKAN:
            backbone_factory = BSplineKANConvBackbone
        else:
            backbone_factory = ResNet18Backbone1D

        self.backbone = strict_data.initialized(backbone_factory, init_seed + 1)
        feature_dim = self.backbone.output_dim
        self.feature_dim = feature_dim
        if method == METHOD_BKAN:
            classifier_factory = lambda: KANLinearBSpline(
                feature_dim, num_classes, grid_size=5, spline_order=3
            )
        else:
            classifier_factory = lambda: nn.Linear(feature_dim, num_classes)
        self.classifier = strict_data.initialized(classifier_factory, init_seed + 2)

        if self.domain_mode in {"dann", "cdan"}:
            domain_input = (
                feature_dim * num_classes
                if self.domain_mode == "cdan"
                else feature_dim
            )
            self.domain_classifier = strict_data.initialized(
                lambda: nn.Sequential(
                    nn.Linear(domain_input, 128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(128, 2),
                ),
                init_seed + 3,
            )
        else:
            self.domain_classifier = None

    def extract_features(self, x: torch.Tensor):
        return self.backbone(x)

    def classify_features(self, features: torch.Tensor):
        return self.classifier(features)

    def classify(self, x: torch.Tensor):
        return self.classify_features(self.extract_features(x))

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        features = self.extract_features(x)
        logits = self.classify_features(features)
        domain_logits = None
        if self.domain_mode in {"dann", "cdan"}:
            domain_input = features
            if self.domain_mode == "cdan":
                probabilities = F.softmax(logits, dim=1)
                domain_input = torch.bmm(
                    features.unsqueeze(2), probabilities.unsqueeze(1)
                ).reshape(features.size(0), -1)
            reversed_features = GradientReversal.apply(domain_input, alpha)
            domain_logits = self.domain_classifier(reversed_features)
        return logits, domain_logits, features


def gaussian_multi_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fixed_sigma: Optional[float] = None,
):
    """Return the standard multi-bandwidth Gaussian kernel matrix."""

    combined = torch.cat((source, target), dim=0)
    squared_distance = torch.cdist(combined, combined, p=2).pow(2)
    if fixed_sigma is None:
        count = combined.size(0)
        denominator = max(1, count * count - count)
        bandwidth = squared_distance.detach().sum() / denominator
    else:
        bandwidth = squared_distance.new_tensor(float(fixed_sigma))
    bandwidth = bandwidth.clamp_min(1e-6) / (kernel_mul ** (kernel_num // 2))
    return sum(
        torch.exp(-squared_distance / (bandwidth * (kernel_mul**index)))
        for index in range(kernel_num)
    )


def lmmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_labels: torch.Tensor,
    target_probabilities: torch.Tensor,
):
    """Class-conditional LMMD without accessing target labels.

    Source class membership is hard. Following the DSAN reference implementation,
    the shared-class set is the intersection of source labels and target argmax
    pseudo-labels. Target membership weights inside those shared classes remain
    the detached softmax probabilities. Per-class weights are normalized
    separately before computing weighted multi-kernel MMD and averaging over the
    shared classes.
    """

    kernels = gaussian_multi_kernel(source_features, target_features)
    source_count = source_features.size(0)
    source_kernel = kernels[:source_count, :source_count]
    target_kernel = kernels[source_count:, source_count:]
    cross_kernel = kernels[:source_count, source_count:]
    target_weights_all = target_probabilities.detach()
    shared_classes = lmmd_shared_classes(source_labels, target_weights_all)
    losses: List[torch.Tensor] = []

    for class_index in shared_classes:
        source_weights = (source_labels == class_index).to(source_features.dtype)
        target_weights = target_weights_all[:, class_index]
        source_weights = source_weights / source_weights.sum()
        target_weights = target_weights / target_weights.sum()
        source_source = torch.outer(source_weights, source_weights)
        target_target = torch.outer(target_weights, target_weights)
        source_target = torch.outer(source_weights, target_weights)
        losses.append(
            (source_source * source_kernel).sum()
            + (target_target * target_kernel).sum()
            - 2.0 * (source_target * cross_kernel).sum()
        )

    if not losses:
        return source_features.sum() * 0.0
    return torch.stack(losses).mean()


def lmmd_shared_classes(
    source_labels: torch.Tensor, target_probabilities: torch.Tensor
):
    """Return source/target pseudo-label intersection used by DSAN LMMD."""

    source_classes = set(int(value) for value in source_labels.detach().cpu().tolist())
    target_pseudo_labels = target_probabilities.detach().argmax(dim=1)
    target_classes = set(
        int(value) for value in target_pseudo_labels.cpu().tolist()
    )
    return tuple(sorted(source_classes & target_classes))


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parameter_count(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def inference_parameter_count(model: CoreBaselineModel):
    excluded = (
        {id(parameter) for parameter in model.domain_classifier.parameters()}
        if model.domain_classifier is not None
        else set()
    )
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    )


def safe_slug(value: str):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def evaluate_once(
    model: CoreBaselineModel, loader: DataLoader, device: torch.device
):
    """The sole target-test access point used by ``train_one``."""

    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model.classify(x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return strict_data.confusion_metrics(
        np.concatenate(targets), np.concatenate(predictions)
    )


def make_unlabeled_loader(
    x: np.ndarray,
    batch_size: int,
    generator_seed: int,
):
    """Build a target-train loader whose dataset contains no label tensor."""

    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    dataset = TensorDataset(
        torch.from_numpy(x[:, None, :].astype(np.float32, copy=False))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


def _limit_array_by_class(
    x: np.ndarray, y: np.ndarray, limit_per_class: Optional[int]
):
    if limit_per_class is None:
        return x, y
    indices = np.concatenate(
        [
            np.flatnonzero(y == class_index)[:limit_per_class]
            for class_index in range(NUM_CLASSES)
        ]
    )
    return x[indices], y[indices]


def limit_domain_for_smoke(
    split: "strict_data.DomainSplit", limit_per_class: Optional[int]
):
    """Create a clearly marked, balanced small split only for smoke execution."""

    if limit_per_class is None:
        return split
    train_x, train_y = _limit_array_by_class(
        split.train_x, split.train_y, limit_per_class
    )
    test_x, test_y = _limit_array_by_class(
        split.test_x, split.test_y, limit_per_class
    )
    return strict_data.DomainSplit(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        record_count=split.record_count,
        train_windows_by_class=tuple(
            int(np.sum(train_y == class_index)) for class_index in range(NUM_CLASSES)
        ),
        test_windows_by_class=tuple(
            int(np.sum(test_y == class_index)) for class_index in range(NUM_CLASSES)
        ),
        fingerprint=strict_data.fingerprint_arrays(train_x, train_y, test_x, test_y),
    )


def write_epoch_history(path: Path, rows: Sequence[Mapping[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def train_one(
    config: ExperimentConfig,
    method: str,
    seed: int,
    source_condition: str,
    target_condition: str,
    domains: Mapping[str, "strict_data.DomainSplit"],
):
    """Train one fixed-epoch block, then evaluate target-test exactly once."""

    set_seed(seed)
    device = torch.device(config.device)
    source = domains[source_condition]
    target = domains[target_condition]
    source_loader = strict_data.make_loader(
        source.train_x,
        source.train_y,
        config.batch_size,
        shuffle=True,
        drop_last=True,
        generator_seed=seed * 1000 + 11,
    )
    target_loader = None
    if method in UDA_METHODS:
        target_loader = make_unlabeled_loader(
            target.train_x,
            config.batch_size,
            generator_seed=seed * 1000 + 29,
        )
    target_test_loader = strict_data.make_loader(
        target.test_x,
        target.test_y,
        config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    steps_per_epoch = len(source_loader)
    if target_loader is not None:
        steps_per_epoch = min(steps_per_epoch, len(target_loader))
    if config.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No complete training batch was produced")

    model = CoreBaselineModel(method, init_seed=seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    training_start = time.perf_counter()

    # Reset stochastic layers after deterministic parameter construction. Loader
    # generators preserve the same source/target batch order across methods.
    set_seed(seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader) if target_loader is not None else None
        sums = {"total": 0.0, "classification": 0.0, "adaptation": 0.0}
        alpha = 0.0

        for step in range(steps_per_epoch):
            source_x, source_y = next(source_iterator)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            alpha = (
                strict_data.alpha_schedule(
                    epoch,
                    step,
                    steps_per_epoch,
                    config.epochs,
                    config.warmup_epochs,
                )
                if method in UDA_METHODS
                else 0.0
            )

            optimizer.zero_grad(set_to_none=True)
            source_logits, source_domain, source_features = model(source_x, alpha)
            classification = criterion(source_logits, source_y)
            adaptation = classification.new_zeros(())
            total = classification

            if method in UDA_METHODS:
                assert target_iterator is not None
                (target_x,) = next(target_iterator)
                target_x = target_x.to(device)
                target_logits, target_domain, target_features = model(target_x, alpha)

                if method in ADVERSARIAL_METHODS:
                    assert source_domain is not None and target_domain is not None
                    source_domain_labels = torch.zeros(
                        source_x.size(0), dtype=torch.long, device=device
                    )
                    target_domain_labels = torch.ones(
                        target_x.size(0), dtype=torch.long, device=device
                    )
                    adaptation = criterion(
                        source_domain, source_domain_labels
                    ) + criterion(target_domain, target_domain_labels)
                    total = classification + config.adaptation_weight * adaptation
                elif method == METHOD_DSAN:
                    target_probabilities = F.softmax(target_logits, dim=1)
                    adaptation = lmmd_loss(
                        source_features,
                        target_features,
                        source_y,
                        target_probabilities,
                    )
                    total = (
                        classification
                        + config.adaptation_weight * alpha * adaptation
                    )

            total.backward()
            optimizer.step()
            sums["total"] += float(total.detach().cpu())
            sums["classification"] += float(classification.detach().cpu())
            sums["adaptation"] += float(adaptation.detach().cpu())

        history_row = {
            "epoch": epoch,
            "train_loss": sums["total"] / steps_per_epoch,
            "classification_loss": sums["classification"] / steps_per_epoch,
            "adaptation_loss": sums["adaptation"] / steps_per_epoch,
            "alpha_final_step": alpha,
        }
        history.append(history_row)
        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            print(
                f"    epoch {epoch:3d}/{config.epochs} "
                f"loss={history_row['train_loss']:.4f} "
                f"cls={history_row['classification_loss']:.4f} "
                f"adapt={history_row['adaptation_loss']:.4f} alpha={alpha:.3f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - training_start

    # Sole target-test pass. No checkpoint or hyperparameter choice follows it.
    target_metrics = evaluate_once(model, target_test_loader, device)
    output_dir = Path(config.output_dir)
    task_slug = f"{CONDITIONS[source_condition]}_to_{CONDITIONS[target_condition]}"
    method_slug = safe_slug(method)
    history_path = output_dir / "history" / method_slug / task_slug / f"seed_{seed}.csv"
    write_epoch_history(history_path, history)
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / method_slug
        / task_slug
        / f"seed_{seed}_final.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "method_definition": METHOD_DEFINITIONS[method],
            "seed": seed,
            "source_condition": source_condition,
            "target_condition": target_condition,
            "epochs_run": config.epochs,
            "selected_epoch": config.epochs,
            "selection_rule": "fixed_final_epoch",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "target_metrics": {
                "accuracy": target_metrics["accuracy"],
                "macro_f1": target_metrics["macro_f1"],
                "class_accuracy": target_metrics["class_accuracy"],
                "confusion_matrix": target_metrics["confusion_matrix"].tolist(),
            },
            "source_fingerprint": source.fingerprint,
            "target_fingerprint": target.fingerprint,
            "config": asdict(config),
        },
        checkpoint_path,
    )

    return {
        "protocol_version": PROTOCOL_VERSION,
        "method": method,
        "seed": seed,
        "source_condition": source_condition,
        "target_condition": target_condition,
        "source_name": CONDITIONS[source_condition],
        "target_name": CONDITIONS[target_condition],
        "accuracy": 100.0 * target_metrics["accuracy"],
        "macro_f1": 100.0 * target_metrics["macro_f1"],
        "normal_accuracy": 100.0 * target_metrics["class_accuracy"][0],
        "outerrace_accuracy": 100.0 * target_metrics["class_accuracy"][1],
        "innerrace_accuracy": 100.0 * target_metrics["class_accuracy"][2],
        "confusion_matrix": json.dumps(target_metrics["confusion_matrix"].tolist()),
        "epochs_run": config.epochs,
        "selected_epoch": config.epochs,
        "selection_rule": "fixed_final_epoch",
        "params_m": parameter_count(model) / 1e6,
        "inference_params_m": inference_parameter_count(model) / 1e6,
        "training_seconds": training_seconds,
        "final_train_loss": history[-1]["train_loss"],
        "final_classification_loss": history[-1]["classification_loss"],
        "final_adaptation_loss": history[-1]["adaptation_loss"],
        "source_train_windows": len(source.train_y),
        "target_train_windows": len(target.train_y) if method in UDA_METHODS else 0,
        "target_test_windows": len(target.test_y),
        "source_fingerprint": source.fingerprint,
        "target_fingerprint": target.fingerprint,
        "history_path": str(history_path.relative_to(output_dir)),
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
    }


RAW_FIELDS = (
    "protocol_version",
    "method",
    "seed",
    "source_condition",
    "target_condition",
    "source_name",
    "target_name",
    "accuracy",
    "macro_f1",
    "normal_accuracy",
    "outerrace_accuracy",
    "innerrace_accuracy",
    "confusion_matrix",
    "epochs_run",
    "selected_epoch",
    "selection_rule",
    "params_m",
    "inference_params_m",
    "training_seconds",
    "final_train_loss",
    "final_classification_loss",
    "final_adaptation_loss",
    "source_train_windows",
    "target_train_windows",
    "target_test_windows",
    "source_fingerprint",
    "target_fingerprint",
    "history_path",
    "checkpoint_path",
)


def sample_std(values: Sequence[float]):
    return stdev(values) if len(values) > 1 else 0.0


def metric_stats(values: Sequence[float]):
    return {
        "mean": mean(values) if values else 0.0,
        "std": sample_std(values),
        "median": float(np.median(values)) if values else 0.0,
        "min": min(values) if values else 0.0,
        "max": max(values) if values else 0.0,
    }


def task_summary(rows: Sequence[Mapping[str, object]]):
    grouped: MutableMapping[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    for row in rows:
        key = (str(row["method"]), str(row["source_name"]), str(row["target_name"]))
        grouped.setdefault(key, []).append(row)
    output = []
    for (method, source_name, target_name), group in sorted(grouped.items()):
        accuracy = metric_stats([float(row["accuracy"]) for row in group])
        macro_f1 = metric_stats([float(row["macro_f1"]) for row in group])
        output.append(
            {
                "method": method,
                "source_name": source_name,
                "target_name": target_name,
                "accuracy_mean": accuracy["mean"],
                "accuracy_std": accuracy["std"],
                "accuracy_median": accuracy["median"],
                "accuracy_min": accuracy["min"],
                "accuracy_max": accuracy["max"],
                "macro_f1_mean": macro_f1["mean"],
                "macro_f1_std": macro_f1["std"],
                "runs": len(group),
            }
        )
    return output


def method_summary(
    rows: Sequence[Mapping[str, object]], methods: Sequence[str]
):
    output = []
    for method in methods:
        method_rows = [row for row in rows if str(row["method"]) == method]
        by_seed: MutableMapping[int, List[Mapping[str, object]]] = {}
        for row in method_rows:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        seed_accuracy = [
            mean(float(row["accuracy"]) for row in group)
            for _, group in sorted(by_seed.items())
        ]
        seed_f1 = [
            mean(float(row["macro_f1"]) for row in group)
            for _, group in sorted(by_seed.items())
        ]
        accuracy = metric_stats(seed_accuracy)
        macro_f1 = metric_stats(seed_f1)
        output.append(
            {
                "method": method,
                "cross_accuracy_mean": accuracy["mean"],
                "cross_accuracy_std": accuracy["std"],
                "cross_accuracy_median": accuracy["median"],
                "cross_accuracy_min_seed_mean": accuracy["min"],
                "cross_accuracy_max_seed_mean": accuracy["max"],
                "cross_macro_f1_mean": macro_f1["mean"],
                "cross_macro_f1_std": macro_f1["std"],
                "runs": len(method_rows),
                "seeds": len(by_seed),
                "tasks_per_seed_min": min(
                    (len(group) for group in by_seed.values()), default=0
                ),
                "params_m": mean(float(row["params_m"]) for row in method_rows)
                if method_rows
                else 0.0,
                "inference_params_m": mean(
                    float(row["inference_params_m"]) for row in method_rows
                )
                if method_rows
                else 0.0,
                "training_seconds_mean": mean(
                    float(row["training_seconds"]) for row in method_rows
                )
                if method_rows
                else 0.0,
            }
        )
    return output


def write_validation_report(
    path: Path,
    methods_summary: Sequence[Mapping[str, object]],
    tasks_summary: Sequence[Mapping[str, object]],
):
    lines = [
        "## Material Passport",
        "",
        f"- Experiment: `{PROTOCOL_VERSION}`",
        "- Type: strict PU core-baseline benchmark",
        "- Verification status: ANALYZED",
        "- Selection rule: fixed final epoch",
        "- Target-test policy: exactly one evaluation after training",
        "- Aggregation: task means are across seeds; overall mean/std are across "
        "seed-level six-task means",
        "",
        "## Method Summary",
        "",
        "| Method | Accuracy (%) | Macro-F1 (%) | Params (M) | Inference params (M) | Runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in methods_summary:
        lines.append(
            f"| {row['method']} | {row['cross_accuracy_mean']:.2f} +/- "
            f"{row['cross_accuracy_std']:.2f} | {row['cross_macro_f1_mean']:.2f} +/- "
            f"{row['cross_macro_f1_std']:.2f} | {row['params_m']:.3f} | "
            f"{row['inference_params_m']:.3f} | {row['runs']} |"
        )
    lines.extend(
        [
            "",
            "## Direction Summary",
            "",
            "| Method | Direction | Accuracy (%) | Macro-F1 (%) | Seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in tasks_summary:
        lines.append(
            f"| {row['method']} | {row['source_name']}->{row['target_name']} | "
            f"{row['accuracy_mean']:.2f} +/- {row['accuracy_std']:.2f} | "
            f"{row['macro_f1_mean']:.2f} +/- {row['macro_f1_std']:.2f} | "
            f"{row['runs']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This report records final-epoch numerical evidence under one frozen "
            "protocol. It does not select runs by target performance and does not "
            "claim significance without a separately pre-specified paired analysis.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(
    rows: Sequence[Mapping[str, object]], output_dir: Path, methods: Sequence[str]
):
    tasks = task_summary(rows)
    methods_output = method_summary(rows, methods)
    strict_data.write_csv(
        output_dir / "task_summary.csv",
        tasks,
        list(tasks[0]) if tasks else ["method"],
    )
    strict_data.write_csv(
        output_dir / "summary.csv",
        methods_output,
        list(methods_output[0]) if methods_output else ["method"],
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"methods": methods_output, "tasks": tasks},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_validation_report(
        output_dir / "validation_report.md", methods_output, tasks
    )
    return methods_output, tasks


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        ).strip()
        return {"commit": commit, "dirty": bool(status), "status": status.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None, "status": []}


def parameter_inventory(methods: Sequence[str]):
    inventory = {}
    for method in methods:
        model = CoreBaselineModel(method, init_seed=resolve_seed(None))
        inventory[method] = {
            "trainable": parameter_count(model),
            "trainable_m": parameter_count(model) / 1e6,
            "inference": inference_parameter_count(model),
            "inference_m": inference_parameter_count(model) / 1e6,
        }
    return inventory


def build_manifest(
    config: ExperimentConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    tasks: Sequence[Tuple[str, str]],
    domains: Mapping[str, "strict_data.DomainSplit"],
    status: str,
    completed_runs: int,
):
    script_path = Path(__file__).resolve()
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "created_or_updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv],
        "script": str(script_path),
        "script_sha256": sha256_file(script_path),
        "dependencies": [
            {
                "path": str(STRICT_DATA_SCRIPT.resolve()),
                "sha256": sha256_file(STRICT_DATA_SCRIPT),
                "purpose": "PU loading and per-record split primitives",
            }
        ],
        "config": asdict(config),
        "methods": list(methods),
        "method_definitions": {
            method: METHOD_DEFINITIONS[method] for method in methods
        },
        "parameter_inventory": parameter_inventory(methods),
        "seeds": list(seeds),
        "tasks": [
            {
                "source_condition": source,
                "target_condition": target,
                "source_name": CONDITIONS[source],
                "target_name": CONDITIONS[target],
            }
            for source, target in tasks
        ],
        "expected_runs": len(methods) * len(seeds) * len(tasks),
        "completed_runs": completed_runs,
        "selection_rule": "fixed_final_epoch",
        "target_test_policy": (
            "Target-test labels are consumed only by evaluate_once after the final "
            f"epoch ({config.epochs}); no decision follows from the result."
        ),
        "split_policy": (
            "Each original continuous PU measurement is split chronologically into "
            "the first 80% train segment and final 20% test segment before windowing."
        ),
        "domain_data": {
            condition: {
                "name": CONDITIONS[condition],
                "record_count": split.record_count,
                "train_windows": len(split.train_y),
                "test_windows": len(split.test_y),
                "train_windows_by_class": list(split.train_windows_by_class),
                "test_windows_by_class": list(split.test_windows_by_class),
                "fingerprint": split.fingerprint,
            }
            for condition, split in domains.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "scipy": scipy.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(),
        },
        "git": git_metadata(),
    }


def write_manifest(path: Path, manifest: Mapping[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def validate_resume_manifest(path: Path, expected: Mapping[str, object]):
    if not path.exists():
        raise FileNotFoundError(f"Cannot resume without manifest: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "protocol_version",
        "script_sha256",
        "dependencies",
        "config",
        "methods",
        "method_definitions",
        "parameter_inventory",
        "seeds",
        "tasks",
        "domain_data",
        "selection_rule",
        "target_test_policy",
        "split_policy",
    )
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "Refusing to resume a different strict PU core benchmark: "
            + ", ".join(mismatches)
        )


def resolve_methods(value: str):
    aliases = {
        "wdcnn": METHOD_WDCNN,
        "resnet": METHOD_RESNET,
        "resnet18": METHOD_RESNET,
        "bkan": METHOD_BKAN,
        "bspline": METHOD_BKAN,
        "kanconv": METHOD_BKAN,
        "dann": METHOD_DANN,
        "cdan": METHOD_CDAN,
        "dsan": METHOD_DSAN,
        "lmmd": METHOD_DSAN,
    }
    methods = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        method = aliases.get(stripped.lower(), stripped)
        if method not in METHODS:
            raise ValueError(f"Unknown method '{stripped}'. Expected one of {METHODS}")
        if method not in methods:
            methods.append(method)
    if not methods:
        raise ValueError("At least one method is required")
    return methods


def resolve_seeds(value: str):
    seeds = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not seeds:
        raise ValueError("At least one seed is required")
    return seeds


def resolve_condition(value: str):
    stripped = value.strip()
    condition = strict_data.ALIASES.get(stripped.upper(), stripped)
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown PU condition '{value}'")
    return condition


def resolve_tasks(value: str):
    if not value:
        return list(DEFAULT_TASKS)
    tasks = []
    for token in value.split(","):
        parts = re.split(r"\s*(?:->|:)\s*", token.strip())
        if len(parts) != 2:
            raise ValueError(f"Task must be SOURCE->TARGET, got '{token}'")
        task = (resolve_condition(parts[0]), resolve_condition(parts[1]))
        if task[0] == task[1]:
            raise ValueError("Core benchmark contains cross-condition tasks only")
        if task not in tasks:
            tasks.append(task)
    return tasks


def row_key(row: Mapping[str, object]):
    return (
        str(row["method"]),
        int(row["seed"]),
        str(row["source_condition"]),
        str(row["target_condition"]),
    )


def protect_output_directory(output_dir: Path, resume: bool):
    """Refuse accidental overwrite or reuse of any non-matching experiment."""

    manifest_path = output_dir / "manifest.json"
    raw_path = output_dir / "raw_results.csv"
    if resume:
        return
    if manifest_path.exists() or raw_path.exists():
        raise FileExistsError(
            f"Output already contains experiment state; use --resume or a new path: {output_dir}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run strict PU core baselines with fixed final epoch and one target-test pass."
        )
    )
    parser.add_argument("--data-root", default=os.environ.get("PU_DATA_ROOT", str(REPO_ROOT / "Data" / "PU")))
    parser.add_argument(
        "--output-dir",
        default=None,
        help="New output directory; defaults differ for full and smoke execution.",
    )
    parser.add_argument(
        "--methods",
        default=DEFAULT_METHOD_ALIASES,
        help="Comma-separated aliases: wdcnn,resnet,bkan,dann,cdan,dsan.",
    )
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated SOURCE->TARGET tasks; default is all six cross tasks.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adaptation-weight", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-records-per-bearing", type=int, default=None)
    parser.add_argument("--max-steps-per-epoch", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "One seed/task/epoch/step with one record per bearing and balanced "
            "small windows; never suitable for reporting."
        ),
    )
    parser.add_argument("--smoke-windows-per-class", type=int, default=8)
    parser.add_argument(
        "--list-parameters",
        action="store_true",
        help="Print architecture parameter counts and exit without loading data.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    methods = resolve_methods(args.methods)
    if args.list_parameters:
        print(json.dumps(parameter_inventory(methods), ensure_ascii=False, indent=2))
        return 0

    seeds = resolve_seeds(args.seeds)
    tasks = resolve_tasks(args.tasks)
    output_dir = Path(
        os.path.abspath(
            args.output_dir
            if args.output_dir is not None
            else str(SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
        )
    )
    config = ExperimentConfig(
        data_root=os.path.abspath(args.data_root),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        adaptation_weight=args.adaptation_weight,
        device=args.device,
        max_records_per_bearing=args.max_records_per_bearing,
        max_steps_per_epoch=args.max_steps_per_epoch,
        log_every=args.log_every,
    )
    if args.smoke:
        seeds = seeds[:1]
        tasks = tasks[:1]
        config = replace(
            config,
            batch_size=min(config.batch_size, 8),
            epochs=1,
            warmup_epochs=0,
            max_records_per_bearing=1,
            max_steps_per_epoch=1,
            smoke_windows_per_class=args.smoke_windows_per_class,
        )

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"PU data root does not exist: {config.data_root}")
    protect_output_directory(output_dir, args.resume)

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Mode: {'SMOKE (not reportable)' if args.smoke else 'FULL'}")
    print(f"Device: {config.device}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Tasks: {[f'{CONDITIONS[s]}->{CONDITIONS[t]}' for s, t in tasks]}")
    print("Building strict 80/20 per-record PU splits...", flush=True)
    needed_conditions = sorted({condition for task in tasks for condition in task})
    domains = {}
    for condition in needed_conditions:
        domain = strict_data.build_domain_split(
            config.data_root,
            condition,
            config.window_size,
            config.overlap,
            config.train_ratio,
            config.max_records_per_bearing,
        )
        domains[condition] = limit_domain_for_smoke(
            domain, config.smoke_windows_per_class
        )
    for condition, split in domains.items():
        print(
            f"  {CONDITIONS[condition]}: records={split.record_count}, "
            f"train={len(split.train_y)}, test={len(split.test_y)}, "
            f"hash={split.fingerprint[:12]}",
            flush=True,
        )

    raw_path = output_dir / "raw_results.csv"
    initial_manifest = build_manifest(
        config, methods, seeds, tasks, domains, "running", 0
    )
    if args.resume:
        validate_resume_manifest(output_dir / "manifest.json", initial_manifest)
    rows = strict_data.read_rows(raw_path) if args.resume else []
    completed = {row_key(row) for row in rows}
    pending = [
        (method, seed, source, target)
        for method in methods
        for seed in seeds
        for source, target in tasks
        if (method, seed, source, target) not in completed
    ]
    write_manifest(
        output_dir / "manifest.json",
        build_manifest(config, methods, seeds, tasks, domains, "running", len(rows)),
    )
    print(f"Pending fits: {len(pending)}", flush=True)

    for index, (method, seed, source, target) in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {method} | seed={seed} | "
            f"{CONDITIONS[source]}->{CONDITIONS[target]}",
            flush=True,
        )
        result = train_one(config, method, seed, source, target, domains)
        rows.append(result)
        strict_data.write_csv(raw_path, rows, RAW_FIELDS)
        write_summaries(rows, output_dir, methods)
        write_manifest(
            output_dir / "manifest.json",
            build_manifest(
                config, methods, seeds, tasks, domains, "running", len(rows)
            ),
        )
        print(
            f"  final target accuracy={result['accuracy']:.2f}% "
            f"macro-F1={result['macro_f1']:.2f}% "
            f"time={result['training_seconds']:.1f}s",
            flush=True,
        )

    methods_summary, _ = write_summaries(rows, output_dir, methods)
    expected = len(methods) * len(seeds) * len(tasks)
    unique_completed = len({row_key(row) for row in rows})
    status = "complete" if unique_completed == expected else "partial"
    write_manifest(
        output_dir / "manifest.json",
        build_manifest(
            config, methods, seeds, tasks, domains, status, unique_completed
        ),
    )
    print("Method summary:")
    for row in methods_summary:
        print(
            f"  {row['method']}: {row['cross_accuracy_mean']:.2f} +/- "
            f"{row['cross_accuracy_std']:.2f}% | params={row['params_m']:.3f}M"
        )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
