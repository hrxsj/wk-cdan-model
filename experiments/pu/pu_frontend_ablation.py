"""PU WK-CDAN training and paired Conv-ReLU/Wavelet-KAN front-end experiments."""

from __future__ import annotations

import argparse
import csv
import glob
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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import decimate
from scipy.stats import rankdata, t as student_t, wilcoxon
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.data import DataLoader, TensorDataset


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "pu-frontend-ablation-strict-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "pu_frontend_ablation_strict_100epoch"
DEFAULT_SEEDS = tuple(DEFAULT_RUN_SEEDS)
METHOD_CONV = "Conv-ReLU"
METHOD_WKAN = "Mexican Hat WKAN-Conv1D"
METHODS = (METHOD_CONV, METHOD_WKAN)

BEARING_MAP: Mapping[str, Sequence[str]] = {
    "Normal": ("K001", "K002"),
    "OuterRace": ("KA01", "KA05"),
    "InnerRace": ("KI01", "KI05"),
}
LABELS = tuple(BEARING_MAP)
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
CONDITIONS: Mapping[str, str] = {
    "N15_M07_F10": "OC0",
    "N15_M01_F10": "OC1",
    "N15_M07_F04": "OC2",
}
ALIASES = {alias: condition for condition, alias in CONDITIONS.items()}
DEFAULT_TASKS = tuple(
    (source, target)
    for source in CONDITIONS
    for target in CONDITIONS
    if source != target
)


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
    center_weight: float = 0.05
    entropy_weight: float = 0.05
    center_lr: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_records_per_bearing: Optional[int] = None
    log_every: int = 10


@dataclass(frozen=True)
class DomainSplit:
    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    record_count: int
    train_windows_by_class: Tuple[int, ...]
    test_windows_by_class: Tuple[int, ...]
    fingerprint: str


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class CenterLoss(nn.Module):
    def __init__(self, num_classes: int, feature_dim: int):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))

    def forward(self, features: torch.Tensor, labels: torch.Tensor):
        centers = self.centers.index_select(0, labels)
        return (features - centers).pow(2).sum(dim=1).mean()


class MexicanHatKANConv1d(nn.Module):
    """The existing PU WKAN-Conv1D parameterization, frozen for this study."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        grid_size: int = 3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = grid_size
        self.wavelet_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, grid_size) * 0.1
        )
        self.base_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size) * 0.1
        )
        self.base_activation = nn.SiLU()
        self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
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
        scaled = unfolded.unsqueeze(-1) * self.scales.view(1, 1, 1, 1, -1)
        basis = (1.0 - scaled.pow(2)) * torch.exp(-0.5 * scaled.pow(2))
        wavelet = torch.einsum("blicg,oicg->blo", basis, self.wavelet_weights)
        return (base + wavelet).permute(0, 2, 1)


class WaveletKANClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int, grid_size: int = 3):
        super().__init__()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.wavelet_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size)
        )
        self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))
        self.silu = nn.SiLU()
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5.0))
        nn.init.normal_(self.wavelet_weight, mean=0.0, std=0.1)

    def forward(self, x: torch.Tensor):
        base = F.linear(self.silu(x), self.base_weight)
        scaled = x.unsqueeze(-1) * self.scales
        basis = (1.0 - scaled.pow(2)) * torch.exp(-0.5 * scaled.pow(2))
        wavelet = torch.einsum("big,oig->bo", basis, self.wavelet_weight)
        return base + wavelet


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


@contextmanager
def seeded_initialization(seed: int):
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield


def initialized(factory, seed: int):
    with seeded_initialization(seed):
        return factory()


class PUFrontEndCDAN(nn.Module):
    """PU WK-CDAN with a single controlled front-end switch."""

    def __init__(self, method: str, num_classes: int = len(LABELS), init_seed: int = resolve_seed(None)):
        super().__init__()
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.num_classes = num_classes

        if method == METHOD_CONV:
            self.frontend = initialized(
                lambda: nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
                init_seed + 1,
            )
            self.front_activation = nn.ReLU(inplace=True)
        else:
            self.frontend = initialized(
                lambda: MexicanHatKANConv1d(1, 32, kernel_size=31, stride=2, padding=15),
                init_seed + 1,
            )
            self.front_activation = nn.Identity()

        self.bn1 = initialized(lambda: nn.BatchNorm1d(32), init_seed + 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mid_cnn = initialized(
            lambda: nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
            ),
            init_seed + 3,
        )
        self.se = initialized(lambda: SEBlock1D(128, reduction=8), init_seed + 4)
        self.tail = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
        )
        self.classifier = initialized(
            lambda: WaveletKANClassifier(128, num_classes), init_seed + 5
        )
        self.domain_classifier = initialized(
            lambda: nn.Sequential(
                nn.Linear(128 * num_classes, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 2),
            ),
            init_seed + 6,
        )

    def extract_features(self, x: torch.Tensor):
        x = self.frontend(x)
        x = self.front_activation(self.bn1(x))
        x = self.pool1(x)
        x = self.mid_cnn(x)
        x = self.se(x)
        return torch.tanh(self.tail(x))

    def classify(self, x: torch.Tensor):
        features = self.extract_features(x)
        return self.classifier(features)

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        features = self.extract_features(x)
        class_logits = self.classifier(features)
        probabilities = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1))
        conditional = conditional.view(-1, 128 * self.num_classes)
        reversed_features = GradientReversal.apply(conditional, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits, features


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_arrays(*arrays: np.ndarray):
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def load_pu_vibration(path: str):
    mat = loadmat(path)
    keys = [key for key in mat if not key.startswith("__")]
    if len(keys) != 1:
        raise ValueError(f"Expected one PU data key in {path}, found {keys}")
    struct = mat[keys[0]][0, 0]
    vibration = struct["Y"][0, 6]["Data"].flatten().astype(np.float64)
    return decimate(vibration, 5).astype(np.float32)


def sliding_windows(signal: np.ndarray, window_size: int, overlap: float):
    step = int(round(window_size * (1.0 - overlap)))
    if step <= 0:
        raise ValueError("overlap must produce a positive window step")
    if signal.size < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    count = 1 + (signal.size - window_size) // step
    windows = np.empty((count, window_size), dtype=np.float32)
    for index in range(count):
        start = index * step
        window = signal[start : start + window_size]
        windows[index] = (window - window.mean()) / (window.std() + 1e-8)
    return windows


def split_record_windows(
    signal: np.ndarray, window_size: int, overlap: float, train_ratio: float
):
    split_index = int(signal.size * train_ratio)
    train_signal = signal[:split_index]
    test_signal = signal[split_index:]
    return (
        sliding_windows(train_signal, window_size, overlap),
        sliding_windows(test_signal, window_size, overlap),
        split_index,
    )


def build_domain_split(
    data_root: str,
    condition: str,
    window_size: int,
    overlap: float,
    train_ratio: float,
    max_records_per_bearing: Optional[int] = None,
):
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    train_labels: List[np.ndarray] = []
    test_labels: List[np.ndarray] = []
    train_counts = [0] * len(LABELS)
    test_counts = [0] * len(LABELS)
    record_count = 0

    for label, bearing_ids in BEARING_MAP.items():
        label_index = LABEL_TO_INDEX[label]
        for bearing_id in bearing_ids:
            pattern = os.path.join(
                data_root, bearing_id, f"{condition}_{bearing_id}_*.mat"
            )
            paths = sorted(glob.glob(pattern))
            if max_records_per_bearing is not None:
                paths = paths[:max_records_per_bearing]
            if not paths:
                raise FileNotFoundError(f"No PU records matched {pattern}")
            for path in paths:
                signal = load_pu_vibration(path)
                train_windows, test_windows, _ = split_record_windows(
                    signal, window_size, overlap, train_ratio
                )
                if not len(train_windows) or not len(test_windows):
                    raise ValueError(f"Record too short after strict split: {path}")
                train_parts.append(train_windows)
                test_parts.append(test_windows)
                train_labels.append(
                    np.full(len(train_windows), label_index, dtype=np.int64)
                )
                test_labels.append(
                    np.full(len(test_windows), label_index, dtype=np.int64)
                )
                train_counts[label_index] += len(train_windows)
                test_counts[label_index] += len(test_windows)
                record_count += 1

    train_x = np.concatenate(train_parts)
    train_y = np.concatenate(train_labels)
    test_x = np.concatenate(test_parts)
    test_y = np.concatenate(test_labels)
    fingerprint = fingerprint_arrays(train_x, train_y, test_x, test_y)
    return DomainSplit(
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        record_count=record_count,
        train_windows_by_class=tuple(train_counts),
        test_windows_by_class=tuple(test_counts),
        fingerprint=fingerprint,
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator_seed: Optional[int] = None,
):
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(generator_seed))
    dataset = TensorDataset(
        torch.from_numpy(x[:, None, :].astype(np.float32, copy=False)),
        torch.from_numpy(y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )


def confusion_metrics(targets: np.ndarray, predictions: np.ndarray):
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for target, prediction in zip(targets, predictions):
        matrix[int(target), int(prediction)] += 1
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0
    class_accuracy = []
    f1_values = []
    for class_index in range(len(LABELS)):
        true_positive = float(matrix[class_index, class_index])
        support = float(matrix[class_index].sum())
        predicted = float(matrix[:, class_index].sum())
        class_accuracy.append(true_positive / support if support else 0.0)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1_values.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1_values)),
        "class_accuracy": class_accuracy,
        "confusion_matrix": matrix,
    }


def evaluate(model: PUFrontEndCDAN, loader: DataLoader, device: torch.device):
    model.eval()
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            logits = model.classify(x.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            targets.append(y.numpy())
    return confusion_metrics(np.concatenate(targets), np.concatenate(predictions))


def parameter_count_m(model: nn.Module):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) / 1e6


def inference_parameter_count_m(model: PUFrontEndCDAN):
    excluded = {id(parameter) for parameter in model.domain_classifier.parameters()}
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in excluded
    ) / 1e6


def alpha_schedule(
    epoch: int, step: int, steps_per_epoch: int, epochs: int, warmup_epochs: int
):
    if epoch <= warmup_epochs:
        return 0.0
    effective_epochs = max(1, epochs - warmup_epochs)
    progress = (step + (epoch - warmup_epochs - 1) * steps_per_epoch) / (
        effective_epochs * steps_per_epoch
    )
    return float(2.0 / (1.0 + math.exp(-5.0 * progress)) - 1.0)


def safe_slug(value: str):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def write_epoch_history(path: Path, rows: Sequence[Mapping[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    config: ExperimentConfig,
    method: str,
    seed: int,
    source_condition: str,
    target_condition: str,
    domains: Mapping[str, DomainSplit],
):
    set_seed(seed)
    device = torch.device(config.device)
    source = domains[source_condition]
    target = domains[target_condition]
    source_loader = make_loader(
        source.train_x,
        source.train_y,
        config.batch_size,
        shuffle=True,
        drop_last=True,
        generator_seed=seed * 1000 + 11,
    )
    target_loader = make_loader(
        target.train_x,
        target.train_y,
        config.batch_size,
        shuffle=True,
        drop_last=True,
        generator_seed=seed * 1000 + 29,
    )
    test_loader = make_loader(
        target.test_x,
        target.test_y,
        config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    steps_per_epoch = min(len(source_loader), len(target_loader))
    if steps_per_epoch == 0:
        raise ValueError("No complete training batch was produced")

    model = PUFrontEndCDAN(method, init_seed=seed).to(device)
    center_loss = initialized(
        lambda: CenterLoss(len(LABELS), 128), seed + 7
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    center_optimizer = torch.optim.SGD(center_loss.parameters(), lr=config.center_lr)
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    training_start = time.perf_counter()

    # Reset stochastic layers after parameter initialization. Dedicated loader
    # generators keep paired methods on identical batch orders.
    set_seed(seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)
        sums = {
            "total": 0.0,
            "classification": 0.0,
            "domain": 0.0,
            "center": 0.0,
            "entropy": 0.0,
            "front_grad": 0.0,
            "scale_grad": 0.0,
        }

        for step in range(steps_per_epoch):
            source_x, source_y = next(source_iterator)
            target_x, _ = next(target_iterator)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)
            alpha = alpha_schedule(
                epoch,
                step,
                steps_per_epoch,
                config.epochs,
                config.warmup_epochs,
            )

            optimizer.zero_grad(set_to_none=True)
            center_optimizer.zero_grad(set_to_none=True)
            source_logits, source_domain, source_features = model(source_x, alpha=alpha)
            target_logits, target_domain, _ = model(target_x, alpha=alpha)

            classification = criterion(source_logits, source_y)
            source_domain_labels = torch.zeros(
                source_x.size(0), dtype=torch.long, device=device
            )
            target_domain_labels = torch.ones(
                target_x.size(0), dtype=torch.long, device=device
            )
            domain = criterion(source_domain, source_domain_labels) + criterion(
                target_domain, target_domain_labels
            )
            center = center_loss(source_features, source_y)
            target_probabilities = F.softmax(target_logits, dim=1)
            entropy = -(
                target_probabilities * torch.log(target_probabilities + 1e-5)
            ).sum(dim=1).mean()
            total = (
                classification
                + domain
                + config.center_weight * center
                + config.entropy_weight * alpha * entropy
            )
            total.backward()

            front_grad_squared = sum(
                float(parameter.grad.detach().pow(2).sum().cpu())
                for parameter in model.frontend.parameters()
                if parameter.grad is not None
            )
            sums["front_grad"] += math.sqrt(max(front_grad_squared, 0.0))
            scales = getattr(model.frontend, "scales", None)
            if scales is not None and scales.grad is not None:
                sums["scale_grad"] += float(scales.grad.detach().norm().cpu())

            optimizer.step()
            for parameter in center_loss.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(1.0 / config.center_weight)
            center_optimizer.step()

            sums["total"] += float(total.detach().cpu())
            sums["classification"] += float(classification.detach().cpu())
            sums["domain"] += float(domain.detach().cpu())
            sums["center"] += float(center.detach().cpu())
            sums["entropy"] += float(entropy.detach().cpu())

        row = {
            "epoch": epoch,
            "train_loss": sums["total"] / steps_per_epoch,
            "classification_loss": sums["classification"] / steps_per_epoch,
            "domain_loss": sums["domain"] / steps_per_epoch,
            "center_loss": sums["center"] / steps_per_epoch,
            "target_entropy": sums["entropy"] / steps_per_epoch,
            "front_grad_norm": sums["front_grad"] / steps_per_epoch,
            "scale_grad_norm": sums["scale_grad"] / steps_per_epoch,
            "alpha_final_step": alpha,
        }
        history.append(row)
        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            print(
                f"    epoch {epoch:3d}/{config.epochs} "
                f"loss={row['train_loss']:.4f} cls={row['classification_loss']:.4f} "
                f"domain={row['domain_loss']:.4f} alpha={alpha:.3f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - training_start

    # This is the only target-test evaluation in the run. No checkpoint or
    # hyperparameter decision follows from this result.
    target_metrics = evaluate(model, test_loader, device)
    output_dir = Path(config.output_dir)
    task_name = f"{CONDITIONS[source_condition]}_to_{CONDITIONS[target_condition]}"
    run_name = f"seed_{seed}"
    history_path = (
        output_dir / "history" / safe_slug(method) / task_name / f"{run_name}.csv"
    )
    write_epoch_history(history_path, history)
    checkpoint_path = (
        output_dir
        / "checkpoints"
        / safe_slug(method)
        / task_name
        / f"{run_name}_final.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "method": method,
            "seed": seed,
            "source_condition": source_condition,
            "target_condition": target_condition,
            "epochs_run": config.epochs,
            "selection_rule": "fixed_final_epoch",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "center_loss_state_dict": {
                key: value.detach().cpu() for key, value in center_loss.state_dict().items()
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
        "params_m": parameter_count_m(model),
        "inference_params_m": inference_parameter_count_m(model),
        "training_seconds": training_seconds,
        "final_train_loss": history[-1]["train_loss"],
        "final_classification_loss": history[-1]["classification_loss"],
        "final_domain_loss": history[-1]["domain_loss"],
        "final_center_loss": history[-1]["center_loss"],
        "final_target_entropy": history[-1]["target_entropy"],
        "final_front_grad_norm": history[-1]["front_grad_norm"],
        "final_scale_grad_norm": history[-1]["scale_grad_norm"],
        "source_train_windows": len(source.train_y),
        "target_train_windows": len(target.train_y),
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
    "final_domain_loss",
    "final_center_loss",
    "final_target_entropy",
    "final_front_grad_norm",
    "final_scale_grad_norm",
    "source_train_windows",
    "target_train_windows",
    "target_test_windows",
    "source_fingerprint",
    "target_fingerprint",
    "history_path",
    "checkpoint_path",
)


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


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
        accuracy = [float(row["accuracy"]) for row in group]
        macro_f1 = [float(row["macro_f1"]) for row in group]
        accuracy_stats = metric_stats(accuracy)
        f1_stats = metric_stats(macro_f1)
        output.append(
            {
                "method": method,
                "source_name": source_name,
                "target_name": target_name,
                "accuracy_mean": accuracy_stats["mean"],
                "accuracy_std": accuracy_stats["std"],
                "accuracy_median": accuracy_stats["median"],
                "accuracy_min": accuracy_stats["min"],
                "accuracy_max": accuracy_stats["max"],
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "runs": len(group),
            }
        )
    return output


def method_summary(rows: Sequence[Mapping[str, object]]):
    output = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        by_seed: MutableMapping[int, List[Mapping[str, object]]] = {}
        for row in method_rows:
            by_seed.setdefault(int(row["seed"]), []).append(row)
        seed_accuracy = [
            mean([float(row["accuracy"]) for row in group])
            for _, group in sorted(by_seed.items())
        ]
        seed_f1 = [
            mean([float(row["macro_f1"]) for row in group])
            for _, group in sorted(by_seed.items())
        ]
        accuracy_stats = metric_stats(seed_accuracy)
        f1_stats = metric_stats(seed_f1)
        output.append(
            {
                "method": method,
                "cross_accuracy_mean": accuracy_stats["mean"],
                "cross_accuracy_std": accuracy_stats["std"],
                "cross_accuracy_median": accuracy_stats["median"],
                "cross_accuracy_min_seed_mean": accuracy_stats["min"],
                "cross_accuracy_max_seed_mean": accuracy_stats["max"],
                "cross_macro_f1_mean": f1_stats["mean"],
                "cross_macro_f1_std": f1_stats["std"],
                "paired_blocks": len(method_rows),
                "seeds": len(by_seed),
                "tasks_per_seed_min": min((len(group) for group in by_seed.values()), default=0),
                "params_m": mean([float(row["params_m"]) for row in method_rows])
                if method_rows
                else 0.0,
                "inference_params_m": mean(
                    [float(row["inference_params_m"]) for row in method_rows]
                )
                if method_rows
                else 0.0,
                "training_seconds_mean": mean(
                    [float(row["training_seconds"]) for row in method_rows]
                )
                if method_rows
                else 0.0,
            }
        )
    return output


def paired_frontend_comparison(rows: Sequence[Mapping[str, object]]):
    grouped: MutableMapping[Tuple[int, str, str], Dict[str, float]] = {}
    for row in rows:
        key = (int(row["seed"]), str(row["source_name"]), str(row["target_name"]))
        grouped.setdefault(key, {})[str(row["method"])] = float(row["accuracy"])
    paired = [
        values
        for _, values in sorted(grouped.items())
        if METHOD_CONV in values and METHOD_WKAN in values
    ]
    differences = np.asarray(
        [values[METHOD_WKAN] - values[METHOD_CONV] for values in paired], dtype=float
    )
    if not len(differences):
        return {"paired_blocks": 0}
    nonzero = differences[np.abs(differences) > 1e-12]
    if len(nonzero):
        statistic, p_value = wilcoxon(nonzero, alternative="two-sided", method="auto")
        ranks = rankdata(np.abs(nonzero))
        positive = float(ranks[nonzero > 0].sum())
        negative = float(ranks[nonzero < 0].sum())
        rank_biserial = (positive - negative) / (positive + negative)
    else:
        statistic, p_value, rank_biserial = 0.0, 1.0, 0.0
    difference_mean = float(differences.mean())
    difference_std = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    if len(differences) > 1:
        margin = float(
            student_t.ppf(0.975, len(differences) - 1)
            * difference_std
            / math.sqrt(len(differences))
        )
    else:
        margin = 0.0
    return {
        "contrast": f"{METHOD_WKAN} minus {METHOD_CONV}",
        "paired_blocks": len(differences),
        "mean_difference_pp": difference_mean,
        "std_difference_pp": difference_std,
        "ci95_low_pp": difference_mean - margin,
        "ci95_high_pp": difference_mean + margin,
        "median_difference_pp": float(np.median(differences)),
        "wins": int(np.sum(differences > 1e-12)),
        "ties": int(np.sum(np.abs(differences) <= 1e-12)),
        "losses": int(np.sum(differences < -1e-12)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p": float(p_value),
        "rank_biserial": float(rank_biserial),
    }


def write_summaries(rows: Sequence[Mapping[str, object]], output_dir: Path):
    tasks = task_summary(rows)
    methods = method_summary(rows)
    paired = paired_frontend_comparison(rows)
    task_fields = list(tasks[0]) if tasks else ["method"]
    method_fields = list(methods[0]) if methods else ["method"]
    paired_fields = list(paired) if paired else ["paired_blocks"]
    write_csv(output_dir / "task_summary.csv", tasks, task_fields)
    write_csv(output_dir / "summary.csv", methods, method_fields)
    write_csv(output_dir / "paired_comparison.csv", [paired], paired_fields)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"methods": methods, "tasks": tasks, "paired_comparison": paired},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_validation_report(output_dir / "validation_report.md", methods, tasks, paired)
    return methods, tasks, paired


def write_validation_report(
    path: Path,
    methods: Sequence[Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]],
    paired: Mapping[str, object],
):
    lines = [
        "## Material Passport",
        "",
        f"- Experiment: `{PROTOCOL_VERSION}`",
        "- Type: paired code experiment / front-end ablation",
        "- Verification status: ANALYZED",
        "- Selection rule: fixed final epoch",
        "- Target-test policy: evaluated once after training; never used for selection",
        "",
        "## Method Summary",
        "",
        "| Method | Cross-condition accuracy (%) | Macro-F1 (%) | Params (M) | Runs |",
        "|---|---:|---:|---:|---:|",
    ]
    for summary in methods:
        lines.append(
            f"| {summary['method']} | {summary['cross_accuracy_mean']:.2f} +/- "
            f"{summary['cross_accuracy_std']:.2f} | "
            f"{summary['cross_macro_f1_mean']:.2f} +/- "
            f"{summary['cross_macro_f1_std']:.2f} | "
            f"{summary['params_m']:.3f} | {summary['paired_blocks']} |"
        )
    lines.extend(
        [
            "",
            "## Task Summary",
            "",
            "| Method | Task | Accuracy (%) | Macro-F1 (%) | Seeds |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for summary in tasks:
        lines.append(
            f"| {summary['method']} | {summary['source_name']}->{summary['target_name']} | "
            f"{summary['accuracy_mean']:.2f} +/- {summary['accuracy_std']:.2f} | "
            f"{summary['macro_f1_mean']:.2f} +/- {summary['macro_f1_std']:.2f} | "
            f"{summary['runs']} |"
        )
    lines.extend(["", "## Paired Comparison", ""])
    if int(paired.get("paired_blocks", 0)):
        lines.extend(
            [
                f"- Contrast: {paired['contrast']}",
                f"- Paired task x seed blocks: {paired['paired_blocks']}",
                f"- Mean difference: {paired['mean_difference_pp']:.3f} percentage points",
                f"- 95% t interval: [{paired['ci95_low_pp']:.3f}, "
                f"{paired['ci95_high_pp']:.3f}] percentage points",
                f"- Wins/ties/losses: {paired['wins']}/{paired['ties']}/{paired['losses']}",
                f"- Paired Wilcoxon p: {paired['wilcoxon_p']:.6g}",
                f"- Rank-biserial effect: {paired['rank_biserial']:.3f}",
            ]
        )
    else:
        lines.append("No complete method pairs are available yet.")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "This report describes the paired numerical evidence. It does not infer "
            "that a front-end is universally superior outside the tested PU tasks, "
            "training protocol, and random seeds.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def build_manifest(
    config: ExperimentConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    tasks: Sequence[Tuple[str, str]],
    domains: Mapping[str, DomainSplit],
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
        "config": asdict(config),
        "methods": list(methods),
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
            "Labels are inaccessible to the training loop and are read exactly once "
            f"by the final evaluator after epoch {config.epochs}."
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
                "train_windows_by_class": split.train_windows_by_class,
                "test_windows_by_class": split.test_windows_by_class,
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


def write_manifest(output_dir: Path, manifest: Mapping[str, object]):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def validate_resume_manifest(
    path: Path,
    config: ExperimentConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    tasks: Sequence[Tuple[str, str]],
    domains: Mapping[str, DomainSplit],
):
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    expected_tasks = [
        {
            "source_condition": source,
            "target_condition": target,
            "source_name": CONDITIONS[source],
            "target_name": CONDITIONS[target],
        }
        for source, target in tasks
    ]
    checks = {
        "protocol_version": (existing.get("protocol_version"), PROTOCOL_VERSION),
        "config": (existing.get("config"), asdict(config)),
        "methods": (existing.get("methods"), list(methods)),
        "seeds": (existing.get("seeds"), list(seeds)),
        "tasks": (existing.get("tasks"), expected_tasks),
        "script_sha256": (
            existing.get("script_sha256"),
            sha256_file(Path(__file__).resolve()),
        ),
    }
    current_hashes = {
        condition: split.fingerprint for condition, split in domains.items()
    }
    existing_hashes = {
        condition: values.get("fingerprint")
        for condition, values in existing.get("domain_data", {}).items()
    }
    checks["domain_fingerprints"] = (existing_hashes, current_hashes)
    mismatches = [name for name, (old, new) in checks.items() if old != new]
    if mismatches:
        raise RuntimeError(
            "Refusing to resume into a different experiment configuration: "
            + ", ".join(mismatches)
        )


def resolve_methods(value: str):
    if not value:
        return list(METHODS)
    aliases = {
        "conv": METHOD_CONV,
        "conv-relu": METHOD_CONV,
        "mexhat": METHOD_WKAN,
        "mexican": METHOD_WKAN,
        "wkan": METHOD_WKAN,
    }
    methods = []
    for token in value.split(","):
        stripped = token.strip()
        method = aliases.get(stripped.lower(), stripped)
        if method not in METHODS:
            raise ValueError(f"Unknown method '{stripped}'. Expected one of {METHODS}")
        if method not in methods:
            methods.append(method)
    return methods


def resolve_condition(value: str):
    stripped = value.strip()
    condition = ALIASES.get(stripped.upper(), stripped)
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
        if task not in tasks:
            tasks.append(task)
    return tasks


def resolve_seeds(value: str):
    return [int(token.strip()) for token in value.split(",") if token.strip()]


def row_key(row: Mapping[str, object]):
    return (
        str(row["method"]),
        int(row["seed"]),
        str(row["source_condition"]),
        str(row["target_condition"]),
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the paired PU Conv-ReLU vs Mexican Hat WKAN front-end study."
    )
    parser.add_argument("--data-root", default=os.environ.get("PU_DATA_ROOT", str(REPO_ROOT / "Data" / "PU")))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated SOURCE->TARGET tasks using OC aliases; default is all 6 cross tasks.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-records-per-bearing", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run both methods on one task/seed for one epoch using one record per bearing.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    methods = resolve_methods(args.methods)
    seeds = resolve_seeds(args.seeds)
    tasks = resolve_tasks(args.tasks)
    config = ExperimentConfig(
        data_root=os.path.abspath(args.data_root),
        output_dir=os.path.abspath(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        device=args.device,
        max_records_per_bearing=args.max_records_per_bearing,
        log_every=args.log_every,
    )
    if args.smoke:
        seeds = seeds[:1]
        tasks = tasks[:1]
        config = replace(config, epochs=1, warmup_epochs=0, max_records_per_bearing=1)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"PU data root does not exist: {config.data_root}")

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Device: {config.device}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Tasks: {[f'{CONDITIONS[s]}->{CONDITIONS[t]}' for s, t in tasks]}")
    print("Building strict 80/20 per-record PU splits...", flush=True)
    needed_conditions = sorted({condition for task in tasks for condition in task})
    domains = {
        condition: build_domain_split(
            config.data_root,
            condition,
            config.window_size,
            config.overlap,
            config.train_ratio,
            config.max_records_per_bearing,
        )
        for condition in needed_conditions
    }
    for condition, split in domains.items():
        print(
            f"  {CONDITIONS[condition]}: records={split.record_count}, "
            f"train={len(split.train_y)}, test={len(split.test_y)}, "
            f"hash={split.fingerprint[:12]}",
            flush=True,
        )

    output_dir = Path(config.output_dir)
    raw_path = output_dir / "raw_results.csv"
    if args.resume:
        validate_resume_manifest(
            output_dir / "manifest.json", config, methods, seeds, tasks, domains
        )
    rows = read_rows(raw_path) if args.resume else []
    completed = {row_key(row) for row in rows}
    pending = [
        (method, seed, source, target)
        for method in methods
        for seed in seeds
        for source, target in tasks
        if (method, seed, source, target) not in completed
    ]
    write_manifest(
        output_dir,
        build_manifest(config, methods, seeds, tasks, domains, "running", len(rows)),
    )
    print(f"Pending pair fits: {len(pending)}", flush=True)

    for index, (method, seed, source, target) in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {method} | seed={seed} | "
            f"{CONDITIONS[source]}->{CONDITIONS[target]}",
            flush=True,
        )
        result = train_one(config, method, seed, source, target, domains)
        rows.append(result)
        write_csv(raw_path, rows, RAW_FIELDS)
        write_summaries(rows, output_dir)
        write_manifest(
            output_dir,
            build_manifest(config, methods, seeds, tasks, domains, "running", len(rows)),
        )
        print(
            f"  final target accuracy={result['accuracy']:.2f}% "
            f"macro-F1={result['macro_f1']:.2f}% "
            f"time={result['training_seconds'] / 60.0:.1f} min",
            flush=True,
        )

    methods_summary, _, paired = write_summaries(rows, output_dir)
    expected = len(methods) * len(seeds) * len(tasks)
    status = "complete" if len({row_key(row) for row in rows}) >= expected else "partial"
    write_manifest(
        output_dir,
        build_manifest(config, methods, seeds, tasks, domains, status, len(rows)),
    )
    print("Method summary:")
    for summary in methods_summary:
        print(
            f"  {summary['method']}: "
            f"{summary['cross_accuracy_mean']:.2f} +/- "
            f"{summary['cross_accuracy_std']:.2f}%"
        )
    print(f"Paired comparison: {json.dumps(paired, ensure_ascii=False)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
