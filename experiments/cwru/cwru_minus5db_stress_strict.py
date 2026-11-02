"""CWRU WK-CDAN and baseline stress experiments with exact-power AWGN."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cwru_ieee_revision as base  # noqa: E402
from experiments.pu import pu_core_baselines_strict as pu_core  # noqa: E402


PROTOCOL_VERSION = "cwru-minus5db-two-anchor-strict-v2"
NOISE_PROTOCOL_VERSION = "per-window-power-exact-awgn-v1"
SNR_DB = -5.0
DEFAULT_SEEDS = tuple(DEFAULT_RUN_SEEDS)
DEFAULT_TASKS = tuple(
    (source, target)
    for source in base.ALL_RPMS
    for target in base.ALL_RPMS
    if source != target
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_minus5db_two_anchor_strict_100epoch"
SMOKE_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_minus5db_two_anchor_strict_smoke"

METHOD_RESNET_SOURCE = pu_core.METHOD_RESNET
METHOD_WK_CDAN_FULL = "WK-CDAN Full"


@dataclass(frozen=True)
class MethodSpec:
    name: str
    backbone: str
    classifier: str
    domain_mode: str
    use_center: bool
    use_entropy: bool
    use_se: bool = True
    wavelet: str = "mexhat"
    use_wavelet_translation: bool = False
    normalize_wavelet: bool = False

    @property
    def uses_target_training(self):
        return self.domain_mode != "none"


METHODS = (
    MethodSpec(
        name=METHOD_RESNET_SOURCE,
        backbone="resnet",
        classifier="linear",
        domain_mode="none",
        use_center=False,
        use_entropy=False,
    ),
    MethodSpec(
        name=METHOD_WK_CDAN_FULL,
        backbone="wkan",
        classifier="wavelet_kan",
        domain_mode="cdan",
        use_center=True,
        use_entropy=True,
    ),
)
METHOD_BY_NAME = {method.name: method for method in METHODS}


@dataclass(frozen=True)
class StressConfig:
    data_root: str = base.CFG.data_root
    metadata_path: str = base.CFG.metadata_path
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    data_protocol: str = "strict_holdout"
    checkpoint_policy: str = "final_epoch"
    snr_db: float = SNR_DB
    window_size: int = 1024
    overlap: float = 0.5
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-3
    center_weight: float = 0.05
    entropy_weight: float = 0.05
    center_lr: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_windows_per_class: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    log_every: int = 10
    reportable: bool = True


@dataclass(frozen=True)
class NoisyPartition:
    rpm: str
    role: str
    x: np.ndarray
    y: np.ndarray
    windows_by_class: Tuple[int, ...]
    noise_seeds_by_class: Tuple[int, ...]
    realized_snr_mean_db: float
    realized_snr_min_db: float
    realized_snr_max_db: float
    clean_fingerprint: str
    noisy_fingerprint: str

    @property
    def split(self):
        return "test" if self.role == "target_test" else "train"


def method_definition(spec: MethodSpec):
    common = {
        "noise_condition": "-5 dB deterministic per-window AWGN on every consumed partition",
        "target_test_policy": "one final evaluation after fixed epoch 100",
    }
    if spec.name == METHOD_RESNET_SOURCE:
        return {
            **common,
            "family": "source_only",
            "feature_extractor": (
                "PU-core-aligned compact 1D ResNet-18 [2,2,2,2], 128-D output"
            ),
            "classifier": "linear 128-to-10",
            "objective": "source cross-entropy",
            "target_training_partition": "not constructed",
            "target_training_loader": "not constructed",
            "domain_discriminator": "none",
        }
    return {
        **common,
        "family": "unsupervised_domain_adaptation",
        "feature_extractor": "Mexican-Hat WKAN Conv1D + CNN trunk + SE, 128-D output",
        "classifier": "Mexican-Hat Wavelet-KAN 128-to-10",
        "objective": (
            "source cross-entropy + 0.05 source center loss + CDAN domain "
            "cross-entropy through scheduled GRL + 0.05*alpha target entropy"
        ),
        "target_training_partition": "first 80% of target records, noisy and unlabeled",
        "target_training_loader": "signals only; target labels excluded",
        "domain_discriminator": "CDAN on vec(feature outer softmax(class logits))",
    }


NOISE_POLICY: Mapping[str, object] = {
    "version": NOISE_PROTOCOL_VERSION,
    "snr_db": SNR_DB,
    "injection_stage": (
        "after chronological record split and window extraction; each clean window "
        "is normalized by the established CWRU windowing primitive before AWGN"
    ),
    "power_definition": "mean(square(window)) computed separately for every window",
    "noise_generation": (
        "independent standard-normal samples are zero-centered per window and "
        "rescaled so realized noise power equals signal_power / 10^(SNR/10)"
    ),
    "rng_isolation": (
        "a fresh numpy Generator is created for every experiment-seed/RPM/class/role; "
        "roles source_train, target_train, and target_test never share a Generator"
    ),
    "source_train": "first 80% of source record; noise role source_train",
    "target_train": (
        "first 80% of target record; constructed only for WK-CDAN; noise role target_train"
    ),
    "target_test": "final 20% of target record; noise role target_test",
}


def derive_noise_seed(
    experiment_seed: int,
    rpm: str,
    class_index: int,
    role: str,
):
    if role not in {"source_train", "target_train", "target_test"}:
        raise ValueError(f"Unknown noise role: {role}")
    payload = (
        f"{NOISE_PROTOCOL_VERSION}|{experiment_seed}|{rpm}|"
        f"{class_index}|{role}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def add_awgn_per_window(
    clean_windows: np.ndarray,
    snr_db: float,
    seed: int,
):
    """Add deterministic Gaussian noise at exact per-window realized SNR."""

    clean = np.asarray(clean_windows, dtype=np.float32)
    if clean.ndim != 2:
        raise ValueError("clean_windows must have shape [windows, samples]")
    if len(clean) == 0:
        return clean.copy(), np.empty((0,), dtype=np.float64)
    signal_power = np.mean(np.square(clean, dtype=np.float64), axis=1, keepdims=True)
    if np.any(signal_power <= 1e-12):
        raise ValueError("Per-window SNR is undefined for a zero-power window")

    rng = np.random.default_rng(seed)
    raw_noise = rng.standard_normal(clean.shape)
    raw_noise = raw_noise - np.mean(raw_noise, axis=1, keepdims=True)
    raw_noise_power = np.mean(np.square(raw_noise), axis=1, keepdims=True)
    if np.any(raw_noise_power <= 1e-12):
        raise RuntimeError("Generated a degenerate Gaussian noise window")
    desired_noise_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = raw_noise * np.sqrt(desired_noise_power / raw_noise_power)
    noisy = (clean.astype(np.float64) + noise).astype(np.float32)
    realized_noise = noisy.astype(np.float64) - clean.astype(np.float64)
    realized_noise_power = np.mean(np.square(realized_noise), axis=1)
    realized_snr = 10.0 * np.log10(signal_power[:, 0] / realized_noise_power)
    return noisy, realized_snr


def build_noisy_partition(
    config: StressConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
    role: str,
    experiment_seed: int,
):
    """Split each record first, window second, then apply role-isolated AWGN."""

    if role not in {"source_train", "target_train", "target_test"}:
        raise ValueError(f"Unknown partition role: {role}")
    use_test_side = role == "target_test"
    clean_parts: List[np.ndarray] = []
    noisy_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    snr_parts: List[np.ndarray] = []
    windows_by_class: List[int] = []
    noise_seeds: List[int] = []

    for label_name in base.LABELS:
        experiment_name, file_name = base.CLASS_FILES[label_name]
        class_index = base.LABEL_TO_IDX[label_name]
        path = base.ensure_mat_file(
            config.data_root,
            metadata_map,
            experiment_name,
            rpm,
            label_name,
            file_name,
        )
        signal = base.load_1d_signal(path)
        train_range, test_range = base.split_signal_by_time(
            signal, config.train_ratio, config.test_ratio
        )
        start, end = test_range if use_test_side else train_range
        clean_windows = base.sliding_windows_from_range(
            signal, start, end, config.window_size, config.overlap
        )
        if config.max_windows_per_class is not None:
            clean_windows = clean_windows[: config.max_windows_per_class]
        noise_seed = derive_noise_seed(
            experiment_seed, rpm, class_index, role
        )
        noisy_windows, realized_snr = add_awgn_per_window(
            clean_windows, config.snr_db, noise_seed
        )
        labels = np.full((len(clean_windows),), class_index, dtype=np.int64)
        clean_parts.append(clean_windows)
        noisy_parts.append(noisy_windows)
        label_parts.append(labels)
        snr_parts.append(realized_snr)
        windows_by_class.append(len(clean_windows))
        noise_seeds.append(noise_seed)

    clean = np.concatenate(clean_parts, axis=0)
    noisy = np.concatenate(noisy_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    realized_snr = np.concatenate(snr_parts, axis=0)
    if not use_test_side:
        shuffle_role = "source" if role == "source_train" else "target"
        shuffle_payload = (
            f"{PROTOCOL_VERSION}|shuffle|{experiment_seed}|{rpm}|{shuffle_role}"
        ).encode("utf-8")
        shuffle_seed = int.from_bytes(
            hashlib.sha256(shuffle_payload).digest()[:8], "big", signed=False
        )
        permutation = np.random.default_rng(shuffle_seed).permutation(len(labels))
        clean = clean[permutation]
        noisy = noisy[permutation]
        labels = labels[permutation]
        realized_snr = realized_snr[permutation]

    return NoisyPartition(
        rpm=rpm,
        role=role,
        x=noisy,
        y=labels,
        windows_by_class=tuple(windows_by_class),
        noise_seeds_by_class=tuple(noise_seeds),
        realized_snr_mean_db=float(np.mean(realized_snr)),
        realized_snr_min_db=float(np.min(realized_snr)),
        realized_snr_max_db=float(np.max(realized_snr)),
        clean_fingerprint=base.fingerprint_arrays(clean, labels),
        noisy_fingerprint=base.fingerprint_arrays(noisy, labels),
    )


def build_model(spec: MethodSpec, init_seed: int = resolve_seed(None)):
    if spec.name == METHOD_RESNET_SOURCE:
        model = pu_core.CoreBaselineModel(
            pu_core.METHOD_RESNET,
            num_classes=base.NUM_CLASSES,
            init_seed=init_seed,
        )
        if model.domain_classifier is not None or model.domain_mode != "none":
            raise RuntimeError("Source-only model must not contain a discriminator")
        return model

    model = base.CDANModel(
        spec.backbone,
        num_classes=base.NUM_CLASSES,
        use_cdan=spec.domain_mode == "cdan",
        wavelet=spec.wavelet,
        use_se=spec.use_se,
        use_wavelet_translation=spec.use_wavelet_translation,
        normalize_wavelet=spec.normalize_wavelet,
        domain_mode=spec.domain_mode,
        classifier=spec.classifier,
        classifier_wavelet=spec.wavelet if spec.classifier == "wavelet_kan" else None,
    )
    if spec.uses_target_training and model.domain_classifier is None:
        raise RuntimeError("WK-CDAN Full requires a domain discriminator")
    return model


def model_state_sha256(model: nn.Module):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def source_loader_seed(experiment_seed: int, source_rpm: str):
    """Source ordering is independent of every target-side quantity."""

    payload = (
        f"{PROTOCOL_VERSION}|loader|{experiment_seed}|{source_rpm}|source_train"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def target_loader_seed(
    experiment_seed: int, source_rpm: str, target_rpm: str
):
    payload = (
        f"{PROTOCOL_VERSION}|loader|{experiment_seed}|{source_rpm}|"
        f"{target_rpm}|target_train"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def make_labeled_loader(
    partition: NoisyPartition,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    generator_seed: Optional[int] = None,
):
    dataset = TensorDataset(
        torch.from_numpy(partition.x[:, None, :].astype(np.float32, copy=False)),
        torch.from_numpy(partition.y.astype(np.int64, copy=False)),
    )
    generator = None
    if generator_seed is not None:
        generator = torch.Generator().manual_seed(generator_seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        generator=generator,
    )


def make_unlabeled_loader(
    partition: NoisyPartition,
    batch_size: int,
    generator_seed: int,
):
    if partition.role != "target_train":
        raise ValueError("Unlabeled loader is reserved for target_train")
    dataset = TensorDataset(
        torch.from_numpy(partition.x[:, None, :].astype(np.float32, copy=False))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(generator_seed),
    )


def adaptation_schedule(epoch: int, step: int, steps_per_epoch: int, epochs: int):
    if not 1 <= epoch <= epochs:
        raise ValueError("epoch lies outside the training horizon")
    if not 0 <= step < steps_per_epoch:
        raise ValueError("step lies outside the epoch")
    progress = float(step + (epoch - 1) * steps_per_epoch) / (
        epochs * steps_per_epoch
    )
    return float(2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def evaluate_once_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
):
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []
    rows: List[Dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for x, y in loader:
            logits, _, _ = model(x.to(device), alpha=0.0)
            probabilities = F.softmax(logits, dim=1).cpu()
            predicted = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            for local_index in range(len(y)):
                true_index = int(y[local_index])
                predicted_index = int(predicted[local_index])
                predictions.append(predicted_index)
                targets.append(true_index)
                rows.append(
                    {
                        "method": spec.name,
                        "seed": seed,
                        "source_rpm": source_rpm,
                        "target_rpm": target_rpm,
                        "source_hp": base.LOAD_MAP[source_rpm],
                        "target_hp": base.LOAD_MAP[target_rpm],
                        "snr_db": SNR_DB,
                        "sample_index": sample_index,
                        "true_index": true_index,
                        "true_label": base.LABELS[true_index],
                        "predicted_index": predicted_index,
                        "predicted_label": base.LABELS[predicted_index],
                        "confidence": float(confidence[local_index]),
                        "probabilities_json": json.dumps(
                            probabilities[local_index].tolist()
                        ),
                    }
                )
                sample_index += 1
    return base.metrics_from_predictions(predictions, targets), rows


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fieldnames: Sequence[str],
):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_confusion_matrix(path: Path, confusion: Sequence[Sequence[int]]):
    rows = [
        {"true_label": label, **dict(zip(base.LABELS, values))}
        for label, values in zip(base.LABELS, confusion)
    ]
    _write_csv(path, rows, ("true_label", *base.LABELS))


def partition_metadata(partition: Optional[NoisyPartition]):
    if partition is None:
        return None
    return {
        "rpm": partition.rpm,
        "role": partition.role,
        "windows": len(partition.y),
        "windows_by_class": list(partition.windows_by_class),
        "noise_seeds_by_class": list(partition.noise_seeds_by_class),
        "realized_snr_mean_db": partition.realized_snr_mean_db,
        "realized_snr_min_db": partition.realized_snr_min_db,
        "realized_snr_max_db": partition.realized_snr_max_db,
        "clean_fingerprint": partition.clean_fingerprint,
        "noisy_fingerprint": partition.noisy_fingerprint,
    }


def train_one(
    config: StressConfig,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
    source_train: NoisyPartition,
    target_test_factory: Callable[[], NoisyPartition],
    target_train: Optional[NoisyPartition] = None,
):
    """Train one final-epoch block and perform one target-test evaluation."""

    if source_rpm == target_rpm:
        raise ValueError("Stress test contains cross-load tasks only")
    if config.data_protocol != "strict_holdout" or config.checkpoint_policy != "final_epoch":
        raise ValueError("Stress test requires strict_holdout and final_epoch")
    if not np.isclose(config.snr_db, SNR_DB):
        raise ValueError("Stress test is frozen to -5 dB")
    if source_train.role != "source_train" or source_train.rpm != source_rpm:
        raise ValueError("Incorrect source-training partition")
    if spec.uses_target_training:
        if target_train is None or target_train.role != "target_train":
            raise ValueError("WK-CDAN Full requires a target_train partition")
    elif target_train is not None:
        raise ValueError("Source-only must not receive a target_train partition")

    base.set_seed(seed)
    device = torch.device(config.device)
    source_loader = make_labeled_loader(
        source_train,
        config.batch_size,
        shuffle=True,
        drop_last=True,
        generator_seed=source_loader_seed(seed, source_rpm),
    )
    target_loader = None
    if spec.uses_target_training:
        target_loader = make_unlabeled_loader(
            target_train,
            config.batch_size,
            generator_seed=target_loader_seed(seed, source_rpm, target_rpm),
        )

    steps_per_epoch = len(source_loader)
    if target_loader is not None:
        steps_per_epoch = min(steps_per_epoch, len(target_loader))
    if config.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No complete training batch was produced")

    model = build_model(spec, init_seed=seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    center_loss_fn = None
    center_optimizer = None
    if spec.use_center:
        center_loss_fn = base.CenterLoss(
            base.NUM_CLASSES,
            feat_dim=128,
            use_gpu=(device.type == "cuda"),
        ).to(device)
        center_optimizer = torch.optim.SGD(
            center_loss_fn.parameters(), lr=config.center_lr
        )
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    training_start = time.perf_counter()

    base.set_seed(seed)
    for epoch in range(1, config.epochs + 1):
        model.train()
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader) if target_loader is not None else None
        sums = {
            "total": 0.0,
            "classification": 0.0,
            "center": 0.0,
            "domain": 0.0,
            "entropy": 0.0,
        }
        alpha = 0.0
        for step in range(steps_per_epoch):
            source_x, source_y = next(source_iterator)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            alpha = (
                adaptation_schedule(epoch, step, steps_per_epoch, config.epochs)
                if spec.uses_target_training
                else 0.0
            )
            optimizer.zero_grad()
            if center_optimizer is not None:
                center_optimizer.zero_grad()

            source_logits, source_domain_logits, source_features = model(
                source_x, alpha=alpha
            )
            classification = criterion(source_logits, source_y)
            center = classification.new_zeros(())
            domain = classification.new_zeros(())
            entropy = classification.new_zeros(())
            total = classification
            if center_loss_fn is not None:
                center = center_loss_fn(source_features, source_y)
                total = total + config.center_weight * center

            if spec.uses_target_training:
                assert target_iterator is not None
                (target_x,) = next(target_iterator)
                target_x = target_x.to(device)
                target_logits, target_domain_logits, _ = model(
                    target_x, alpha=alpha
                )
                if source_domain_logits is None or target_domain_logits is None:
                    raise RuntimeError("WK-CDAN discriminator logits are missing")
                source_domain_labels = torch.zeros(
                    source_x.size(0), dtype=torch.long, device=device
                )
                target_domain_labels = torch.ones(
                    target_x.size(0), dtype=torch.long, device=device
                )
                domain = criterion(source_domain_logits, source_domain_labels)
                domain = domain + criterion(
                    target_domain_logits, target_domain_labels
                )
                target_probabilities = F.softmax(target_logits, dim=1)
                entropy = -(
                    target_probabilities
                    * torch.log(target_probabilities + 1e-5)
                ).sum(dim=1).mean()
                total = total + domain + config.entropy_weight * alpha * entropy
            elif source_domain_logits is not None:
                raise RuntimeError("Source-only unexpectedly produced domain logits")

            total.backward()
            optimizer.step()
            if center_optimizer is not None:
                for parameter in center_loss_fn.parameters():
                    if parameter.grad is not None:
                        parameter.grad.data *= 1.0 / config.center_weight
                center_optimizer.step()

            sums["total"] += float(total.detach().cpu())
            sums["classification"] += float(classification.detach().cpu())
            sums["center"] += float(center.detach().cpu())
            sums["domain"] += float(domain.detach().cpu())
            sums["entropy"] += float(entropy.detach().cpu())

        history_row = {
            "epoch": epoch,
            "train_loss": sums["total"] / steps_per_epoch,
            "classification_loss": sums["classification"] / steps_per_epoch,
            "center_loss": sums["center"] / steps_per_epoch,
            "domain_loss": sums["domain"] / steps_per_epoch,
            "target_entropy_loss": sums["entropy"] / steps_per_epoch,
            "alpha_final_step": alpha,
            "target_training_batches": (
                steps_per_epoch if spec.uses_target_training else 0
            ),
        }
        history.append(history_row)
        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            print(
                f"    epoch {epoch:3d}/{config.epochs} "
                f"loss={history_row['train_loss']:.4f} "
                f"cls={history_row['classification_loss']:.4f} "
                f"domain={history_row['domain_loss']:.4f} "
                f"entropy={history_row['target_entropy_loss']:.4f}",
                flush=True,
            )
    training_seconds = time.perf_counter() - training_start

    post_train_state_sha256 = model_state_sha256(model)
    target_test = target_test_factory()
    if target_test.role != "target_test" or target_test.rpm != target_rpm:
        raise ValueError("Incorrect target-test partition")
    target_test_loader = make_labeled_loader(
        target_test,
        config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Sole target-test iteration. Complexity uses zeros later, never test data.
    target_metrics, prediction_rows = evaluate_once_with_predictions(
        model,
        target_test_loader,
        device,
        spec,
        source_rpm,
        target_rpm,
        seed,
    )
    post_evaluation_state_sha256 = model_state_sha256(model)
    if post_evaluation_state_sha256 != post_train_state_sha256:
        raise RuntimeError("Final target-test evaluation mutated the trained model")
    output_dir = Path(config.output_dir)
    task_slug = f"{base.LOAD_MAP[source_rpm]}_to_{base.LOAD_MAP[target_rpm]}"
    artifact_dir = (
        output_dir
        / "artifacts"
        / base.safe_slug(spec.name)
        / task_slug
        / f"seed_{seed}"
    )
    history_path = artifact_dir / "epoch_history.csv"
    prediction_path = artifact_dir / "target_predictions.csv"
    confusion_path = artifact_dir / "confusion_matrix.csv"
    checkpoint_path = artifact_dir / "final_checkpoint.pt"
    _write_csv(history_path, history, tuple(history[0]))
    prediction_fields = tuple(prediction_rows[0]) if prediction_rows else (
        "method",
        "sample_index",
    )
    _write_csv(prediction_path, prediction_rows, prediction_fields)
    write_confusion_matrix(confusion_path, target_metrics["confusion_matrix"])

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
            "noise_protocol_version": NOISE_PROTOCOL_VERSION,
            "noise_policy": dict(NOISE_POLICY),
            "method": spec.name,
            "method_spec": asdict(spec),
            "method_definition": method_definition(spec),
            "config": asdict(config),
            "source_rpm": source_rpm,
            "target_rpm": target_rpm,
            "seed": seed,
            "epochs_run": config.epochs,
            "selected_epoch": config.epochs,
            "selection_rule": "fixed_final_epoch",
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "center_loss_state_dict": (
                {
                    key: value.detach().cpu()
                    for key, value in center_loss_fn.state_dict().items()
                }
                if center_loss_fn is not None
                else None
            ),
            "target_metrics": target_metrics,
            "partitions": {
                "source_train": partition_metadata(source_train),
                "target_train": partition_metadata(target_train),
                "target_test": partition_metadata(target_test),
            },
            "target_training_loader_built": target_loader is not None,
            "target_training_labels_consumed": False,
            "target_test_partition_built_after_training": True,
            "target_test_partition_builds": 1,
            "target_test_loader_builds": 1,
            "target_test_evaluations": 1,
            "target_test_labels_consumed_only_in_final_evaluation": True,
            "post_train_state_sha256": post_train_state_sha256,
            "post_evaluation_state_sha256": post_evaluation_state_sha256,
        },
        checkpoint_path,
    )

    zero_example = torch.zeros(
        (1, 1, config.window_size), dtype=torch.float32, device=device
    )
    classification_flops = base.estimate_classification_flops(model, zero_example)
    model_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    center_params = (
        sum(
            parameter.numel()
            for parameter in center_loss_fn.parameters()
            if parameter.requires_grad
        )
        if center_loss_fn is not None
        else 0
    )
    final_history = history[-1]
    artifact_hashes = {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_sha256": sha256_file(prediction_path),
        "confusion_matrix_sha256": sha256_file(confusion_path),
        "history_sha256": sha256_file(history_path),
    }
    return {
        "protocol_version": PROTOCOL_VERSION,
        "noise_protocol_version": NOISE_PROTOCOL_VERSION,
        "method": spec.name,
        "data_protocol": config.data_protocol,
        "checkpoint_policy": config.checkpoint_policy,
        "source_rpm": source_rpm,
        "target_rpm": target_rpm,
        "source_hp": base.LOAD_MAP[source_rpm],
        "target_hp": base.LOAD_MAP[target_rpm],
        "seed": seed,
        "snr_db": config.snr_db,
        "accuracy": 100.0 * target_metrics["accuracy"],
        "macro_f1": 100.0 * target_metrics["macro_f1"],
        "epochs_run": config.epochs,
        "selected_epoch": config.epochs,
        "selection_rule": "fixed_final_epoch",
        "model_params_m": model_params / 1e6,
        "center_aux_params_m": center_params / 1e6,
        "inference_params_m": base.count_inference_parameters_m(model),
        "gflops": classification_flops / 1e9,
        "training_seconds": training_seconds,
        "final_train_loss": final_history["train_loss"],
        "final_classification_loss": final_history["classification_loss"],
        "final_center_loss": final_history["center_loss"],
        "final_domain_loss": final_history["domain_loss"],
        "final_target_entropy_loss": final_history["target_entropy_loss"],
        "final_alpha": final_history["alpha_final_step"],
        "target_training_loader_built": target_loader is not None,
        "target_training_labels_consumed": False,
        "target_test_partition_built_after_training": True,
        "target_test_partition_builds": 1,
        "target_test_loader_builds": 1,
        "target_test_evaluations": 1,
        "target_test_labels_consumed_only_in_final_evaluation": True,
        "post_train_state_sha256": post_train_state_sha256,
        "post_evaluation_state_sha256": post_evaluation_state_sha256,
        "source_train_windows": len(source_train.y),
        "target_train_windows": len(target_train.y) if target_train is not None else 0,
        "target_test_windows": len(target_test.y),
        "source_train_snr_mean_db": source_train.realized_snr_mean_db,
        "target_train_snr_mean_db": (
            target_train.realized_snr_mean_db if target_train is not None else ""
        ),
        "target_test_snr_mean_db": target_test.realized_snr_mean_db,
        "source_train_clean_fingerprint": source_train.clean_fingerprint,
        "source_train_noisy_fingerprint": source_train.noisy_fingerprint,
        "target_train_clean_fingerprint": (
            target_train.clean_fingerprint if target_train is not None else ""
        ),
        "target_train_noisy_fingerprint": (
            target_train.noisy_fingerprint if target_train is not None else ""
        ),
        "target_test_clean_fingerprint": target_test.clean_fingerprint,
        "target_test_noisy_fingerprint": target_test.noisy_fingerprint,
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
        "prediction_path": str(prediction_path.relative_to(output_dir)),
        "confusion_matrix_path": str(confusion_path.relative_to(output_dir)),
        "history_path": str(history_path.relative_to(output_dir)),
        **artifact_hashes,
    }


RAW_FIELDS = (
    "protocol_version",
    "noise_protocol_version",
    "method",
    "data_protocol",
    "checkpoint_policy",
    "source_rpm",
    "target_rpm",
    "source_hp",
    "target_hp",
    "seed",
    "snr_db",
    "accuracy",
    "macro_f1",
    "epochs_run",
    "selected_epoch",
    "selection_rule",
    "model_params_m",
    "center_aux_params_m",
    "inference_params_m",
    "gflops",
    "training_seconds",
    "final_train_loss",
    "final_classification_loss",
    "final_center_loss",
    "final_domain_loss",
    "final_target_entropy_loss",
    "final_alpha",
    "target_training_loader_built",
    "target_training_labels_consumed",
    "target_test_partition_built_after_training",
    "target_test_partition_builds",
    "target_test_loader_builds",
    "target_test_evaluations",
    "target_test_labels_consumed_only_in_final_evaluation",
    "post_train_state_sha256",
    "post_evaluation_state_sha256",
    "source_train_windows",
    "target_train_windows",
    "target_test_windows",
    "source_train_snr_mean_db",
    "target_train_snr_mean_db",
    "target_test_snr_mean_db",
    "source_train_clean_fingerprint",
    "source_train_noisy_fingerprint",
    "target_train_clean_fingerprint",
    "target_train_noisy_fingerprint",
    "target_test_clean_fingerprint",
    "target_test_noisy_fingerprint",
    "checkpoint_path",
    "prediction_path",
    "confusion_matrix_path",
    "history_path",
    "checkpoint_sha256",
    "prediction_sha256",
    "confusion_matrix_sha256",
    "history_sha256",
)


def sample_std(values: Sequence[float]):
    return stdev(values) if len(values) > 1 else 0.0


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    expected_tasks: Sequence[Tuple[str, str]] = DEFAULT_TASKS,
):
    expected_task_set = set(expected_tasks)
    task_groups: MutableMapping[Tuple[str, str, str], List[Mapping[str, object]]] = {}
    seed_groups: MutableMapping[Tuple[str, int], List[Mapping[str, object]]] = {}
    for row in rows:
        method = str(row["method"])
        source = str(row["source_rpm"])
        target = str(row["target_rpm"])
        seed = int(row["seed"])
        task_groups.setdefault((method, source, target), []).append(row)
        seed_groups.setdefault((method, seed), []).append(row)

    task_summary = []
    for (method, source, target), group in sorted(task_groups.items()):
        accuracies = [float(row["accuracy"]) for row in group]
        macro_f1 = [float(row["macro_f1"]) for row in group]
        task_summary.append(
            {
                "method": method,
                "source_rpm": source,
                "target_rpm": target,
                "source_hp": base.LOAD_MAP[source],
                "target_hp": base.LOAD_MAP[target],
                "accuracy_mean": mean(accuracies),
                "accuracy_std": sample_std(accuracies),
                "macro_f1_mean": mean(macro_f1),
                "macro_f1_std": sample_std(macro_f1),
                "seeds_completed": len({int(row["seed"]) for row in group}),
            }
        )

    seed_summary = []
    for (method, seed), group in sorted(seed_groups.items()):
        observed_tasks = {
            (str(row["source_rpm"]), str(row["target_rpm"])) for row in group
        }
        accuracies = [float(row["accuracy"]) for row in group]
        macro_f1 = [float(row["macro_f1"]) for row in group]
        seed_summary.append(
            {
                "method": method,
                "seed": seed,
                "cross_accuracy_mean": mean(accuracies),
                "cross_accuracy_std_across_tasks": sample_std(accuracies),
                "cross_macro_f1_mean": mean(macro_f1),
                "cross_macro_f1_std_across_tasks": sample_std(macro_f1),
                "tasks_completed": len(observed_tasks),
                "complete_seed_block": observed_tasks == expected_task_set,
            }
        )

    method_summary = []
    for method in [spec.name for spec in METHODS]:
        complete = [
            row
            for row in seed_summary
            if row["method"] == method and row["complete_seed_block"]
        ]
        accuracy = [float(row["cross_accuracy_mean"]) for row in complete]
        macro_f1 = [float(row["cross_macro_f1_mean"]) for row in complete]
        method_rows = [row for row in rows if str(row["method"]) == method]
        method_summary.append(
            {
                "method": method,
                "cross_accuracy_mean_across_seeds": mean(accuracy) if accuracy else "",
                "cross_accuracy_std_across_seeds": sample_std(accuracy) if accuracy else "",
                "cross_macro_f1_mean_across_seeds": mean(macro_f1) if macro_f1 else "",
                "cross_macro_f1_std_across_seeds": sample_std(macro_f1) if macro_f1 else "",
                "complete_seed_blocks": len(complete),
                "runs_completed": len(method_rows),
                "model_params_m": (
                    float(method_rows[0]["model_params_m"]) if method_rows else ""
                ),
                "gflops": float(method_rows[0]["gflops"]) if method_rows else "",
            }
        )
    return task_summary, seed_summary, method_summary


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    expected_tasks: Sequence[Tuple[str, str]],
):
    _write_csv(output_dir / "raw_results.csv", rows, RAW_FIELDS)
    task_summary, seed_summary, method_summary = summarize_rows(
        rows, expected_tasks=expected_tasks
    )
    _write_csv(
        output_dir / "task_summary.csv",
        task_summary,
        tuple(task_summary[0]) if task_summary else ("method", "source_rpm", "target_rpm"),
    )
    _write_csv(
        output_dir / "seed_level_summary.csv",
        seed_summary,
        tuple(seed_summary[0]) if seed_summary else ("method", "seed"),
    )
    _write_csv(
        output_dir / "method_summary.csv",
        method_summary,
        tuple(method_summary[0]) if method_summary else ("method",),
    )
    (output_dir / "method_summary.json").write_text(
        json.dumps(method_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return task_summary, seed_summary, method_summary


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_inventory(
    config: StressConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
):
    inventory = []
    for rpm in base.ALL_RPMS:
        for label_name in base.LABELS:
            experiment_name, file_name = base.CLASS_FILES[label_name]
            path = Path(
                base.ensure_mat_file(
                    config.data_root,
                    metadata_map,
                    experiment_name,
                    rpm,
                    label_name,
                    file_name,
                )
            ).resolve()
            inventory.append(
                {
                    "rpm": rpm,
                    "label": label_name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def parameter_inventory():
    inventory = {}
    for spec in METHODS:
        base.set_seed(resolve_seed(None))
        model = build_model(spec, init_seed=resolve_seed(None))
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        inventory[spec.name] = {
            "trainable": trainable,
            "trainable_m": trainable / 1e6,
            "inference": int(round(base.count_inference_parameters_m(model) * 1e6)),
            "inference_m": base.count_inference_parameters_m(model),
            "center_auxiliary": base.NUM_CLASSES * 128 if spec.use_center else 0,
        }
    return inventory


def build_manifest(
    config: StressConfig,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    dataset_inventory: Sequence[Mapping[str, object]],
    status: str,
    completed_keys: Sequence[Tuple[str, int, str, str]],
):
    script_path = Path(__file__).resolve()
    dependency_path = Path(base.__file__).resolve()
    pu_dependency_path = Path(pu_core.__file__).resolve()
    pu_initializer_path = Path(pu_core.STRICT_DATA_SCRIPT).resolve()
    metadata_path = Path(config.metadata_path).resolve()
    frozen_signature = {
        "protocol_version": PROTOCOL_VERSION,
        "script_sha256": sha256_file(script_path),
        "dependency": {
            "path": str(dependency_path),
            "sha256": sha256_file(dependency_path),
            "purpose": "CWRU split/model/loss/metric/complexity primitives",
        },
        "pu_model_dependency": {
            "path": str(pu_dependency_path),
            "sha256": sha256_file(pu_dependency_path),
            "purpose": "exact PU-aligned ResNet-18 source-only model identity",
        },
        "pu_initializer_dependency": {
            "path": str(pu_initializer_path),
            "sha256": sha256_file(pu_initializer_path),
            "purpose": "isolated deterministic PU model initializer",
        },
        "metadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "dataset_inventory": list(dataset_inventory),
        "config": asdict(config),
        "methods": [spec.name for spec in METHODS],
        "method_specs": {spec.name: asdict(spec) for spec in METHODS},
        "method_definitions": {
            spec.name: method_definition(spec) for spec in METHODS
        },
        "noise_policy": dict(NOISE_POLICY),
        "tasks": [list(task) for task in tasks],
        "seeds": list(seeds),
        "expected_runs": len(METHODS) * len(tasks) * len(seeds),
        "parameter_inventory": parameter_inventory(),
        "selection_rule": "fixed_final_epoch",
        "split_policy": (
            "Every original continuous record is split chronologically 80/20 "
            "before sliding windows are extracted."
        ),
        "source_only_target_policy": (
            "ResNet source-only never constructs target_train data or a target "
            "loader; target identity is absent from source noise, shuffling, model "
            "initialization, optimization steps, and stopping."
        ),
        "wk_target_policy": (
            "WK-CDAN constructs target_train but its DataLoader contains signals only."
        ),
        "target_test_policy": (
            "Each target-test partition and loader is constructed only after the "
            "fixed final epoch and iterated exactly once; model state must be "
            "unchanged by evaluation and complexity uses a zero tensor."
        ),
        "statistics_policy": (
            "First average the 12 directed cross-load tasks within each seed; "
            "then report mean and sample standard deviation across the five seed-level means."
        ),
        "reportable": config.reportable,
    }
    return {
        "frozen_signature": frozen_signature,
        "status": status,
        "completed_keys": [list(key) for key in completed_keys],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv],
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "platform": platform.platform(),
        },
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
    if existing.get("frozen_signature") != expected.get("frozen_signature"):
        raise RuntimeError(
            "Refusing resume: code, data, noise, methods, tasks, seeds, or config differ"
        )


def row_key(row: Mapping[str, object]):
    return (
        str(row["method"]),
        int(row["seed"]),
        str(row["source_rpm"]),
        str(row["target_rpm"]),
    )


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _csv_bool(value: object):
    return str(value).strip().lower() == "true"


def validate_existing_rows(
    rows: Sequence[Mapping[str, object]],
    config: StressConfig,
    output_dir: Path,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
):
    allowed_keys = {
        (spec.name, seed, source, target)
        for spec in METHODS
        for seed in seeds
        for source, target in tasks
    }
    seen = set()
    source_only_optimization_blocks: MutableMapping[
        Tuple[str, int, str], set
    ] = {}
    artifact_fields = {
        "checkpoint_path": "checkpoint_sha256",
        "prediction_path": "prediction_sha256",
        "confusion_matrix_path": "confusion_matrix_sha256",
        "history_path": "history_sha256",
    }
    for row in rows:
        key = row_key(row)
        if key not in allowed_keys:
            raise RuntimeError(f"Unexpected result row during resume: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate result row during resume: {key}")
        seen.add(key)
        spec = METHOD_BY_NAME[key[0]]
        if str(row.get("protocol_version")) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol mismatch in row: {key}")
        if str(row.get("noise_protocol_version")) != NOISE_PROTOCOL_VERSION:
            raise RuntimeError(f"Noise protocol mismatch in row: {key}")
        if not np.isclose(float(row.get("snr_db", "nan")), config.snr_db):
            raise RuntimeError(f"SNR mismatch in row: {key}")
        if int(row.get("epochs_run", 0)) != config.epochs:
            raise RuntimeError(f"Epoch-count mismatch in row: {key}")
        if int(row.get("selected_epoch", 0)) != config.epochs:
            raise RuntimeError(f"Selected-epoch mismatch in row: {key}")
        if str(row.get("selection_rule")) != "fixed_final_epoch":
            raise RuntimeError(f"Selection-rule mismatch in row: {key}")
        if str(row.get("data_protocol")) != "strict_holdout":
            raise RuntimeError(f"Data-protocol mismatch in row: {key}")
        if str(row.get("checkpoint_policy")) != "final_epoch":
            raise RuntimeError(f"Checkpoint-policy mismatch in row: {key}")
        if int(row.get("target_test_evaluations", 0)) != 1:
            raise RuntimeError(f"Target-test policy mismatch in row: {key}")
        if not _csv_bool(row.get("target_test_partition_built_after_training", False)):
            raise RuntimeError(f"Target-test was not built after training: {key}")
        if int(row.get("target_test_partition_builds", 0)) != 1:
            raise RuntimeError(f"Target-test partition build mismatch: {key}")
        if int(row.get("target_test_loader_builds", 0)) != 1:
            raise RuntimeError(f"Target-test loader build mismatch: {key}")
        if not _csv_bool(
            row.get("target_test_labels_consumed_only_in_final_evaluation", False)
        ):
            raise RuntimeError(f"Target-test label policy mismatch: {key}")
        if str(row.get("post_train_state_sha256", "")) != str(
            row.get("post_evaluation_state_sha256", "")
        ):
            raise RuntimeError(f"Target evaluation changed model state: {key}")
        loader_built = _csv_bool(row.get("target_training_loader_built", False))
        if loader_built != spec.uses_target_training:
            raise RuntimeError(f"Target-loader policy mismatch in row: {key}")
        if _csv_bool(row.get("target_training_labels_consumed", False)):
            raise RuntimeError(f"Target training labels were consumed: {key}")
        for field in ("source_train_snr_mean_db", "target_test_snr_mean_db"):
            if not np.isclose(float(row.get(field, "nan")), config.snr_db, atol=1e-4):
                raise RuntimeError(f"Realized-SNR mismatch ({field}) in row: {key}")
        if spec.uses_target_training:
            if int(row.get("target_train_windows", 0)) <= 0:
                raise RuntimeError(f"Missing target training windows in row: {key}")
            if not np.isclose(
                float(row.get("target_train_snr_mean_db", "nan")),
                config.snr_db,
                atol=1e-4,
            ):
                raise RuntimeError(f"Target-train SNR mismatch in row: {key}")
        elif int(row.get("target_train_windows", -1)) != 0:
            raise RuntimeError(f"Source-only target-window mismatch in row: {key}")
        if not spec.uses_target_training:
            block_key = (spec.name, int(row["seed"]), str(row["source_rpm"]))
            signature = (
                str(row.get("source_train_clean_fingerprint", "")),
                str(row.get("source_train_noisy_fingerprint", "")),
                int(row.get("source_train_windows", 0)),
                str(row.get("history_sha256", "")),
                str(row.get("post_train_state_sha256", "")),
            )
            source_only_optimization_blocks.setdefault(block_key, set()).add(signature)
        for path_field, hash_field in artifact_fields.items():
            artifact = output_dir / str(row.get(path_field, ""))
            if not artifact.is_file():
                raise RuntimeError(f"Missing resume artifact {path_field}: {artifact}")
            if sha256_file(artifact) != str(row.get(hash_field, "")):
                raise RuntimeError(f"Artifact hash mismatch {path_field}: {artifact}")
    for block_key, signatures in source_only_optimization_blocks.items():
        if len(signatures) != 1:
            raise RuntimeError(
                "Source-only optimization changed with target identity: "
                f"{block_key}"
            )
    return seen


def protect_output_directory(output_dir: Path, resume: bool):
    if resume:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume or a new path: {output_dir}"
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run the strict CWRU -5 dB two-anchor stress test."
    )
    parser.add_argument("--data-root", default=base.CFG.data_root)
    parser.add_argument("--metadata-path", default=base.CFG.metadata_path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Non-reportable one-task/seed/epoch/step test of both methods.",
    )
    parser.add_argument("--smoke-windows-per-class", type=int, default=2)
    parser.add_argument("--list-parameters", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    if args.list_parameters:
        print(json.dumps(parameter_inventory(), ensure_ascii=False, indent=2))
        return 0
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("epochs, batch size, and log_every must be positive")
    if not args.smoke:
        required = {
            "epochs": (args.epochs, 100),
            "batch_size": (args.batch_size, 32),
            "lr": (args.lr, 5e-4),
            "weight_decay": (args.weight_decay, 1e-3),
        }
        mismatched = [
            name
            for name, (actual, expected) in required.items()
            if not np.isclose(actual, expected)
        ]
        if mismatched:
            raise ValueError(
                "Reportable protocol has frozen hyperparameters: "
                + ", ".join(mismatched)
            )

    output_dir = Path(
        os.path.abspath(
            args.output_dir
            if args.output_dir is not None
            else str(SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
        )
    )
    config = StressConfig(
        data_root=os.path.abspath(args.data_root),
        metadata_path=os.path.abspath(args.metadata_path),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        log_every=args.log_every,
    )
    tasks = list(DEFAULT_TASKS)
    seeds = list(DEFAULT_SEEDS)
    if args.smoke:
        tasks = tasks[:1]
        seeds = seeds[:1]
        config = replace(
            config,
            epochs=1,
            batch_size=min(config.batch_size, 10),
            max_windows_per_class=args.smoke_windows_per_class,
            max_steps_per_epoch=1,
            log_every=1,
            reportable=False,
        )

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"CWRU data root does not exist: {config.data_root}")
    if not Path(config.metadata_path).is_file():
        raise FileNotFoundError(f"CWRU metadata file does not exist: {config.metadata_path}")
    protect_output_directory(output_dir, args.resume)

    metadata_map = base.load_metadata_map(config.metadata_path)
    dataset_inventory = build_dataset_inventory(config, metadata_map)
    expected_manifest = build_manifest(
        config, tasks, seeds, dataset_inventory, "running", []
    )
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        validate_resume_manifest(manifest_path, expected_manifest)

    raw_path = output_dir / "raw_results.csv"
    rows = read_rows(raw_path) if args.resume else []
    completed = validate_existing_rows(
        rows, config, output_dir, tasks, seeds
    ) if rows else set()
    pending = [
        (spec, seed, source, target)
        for spec in METHODS
        for seed in seeds
        for source, target in tasks
        if (spec.name, seed, source, target) not in completed
    ]
    write_manifest(
        manifest_path,
        build_manifest(
            config, tasks, seeds, dataset_inventory, "running", sorted(completed)
        ),
    )

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Mode: {'SMOKE (not reportable)' if args.smoke else 'FULL'}")
    print(f"Noise: {config.snr_db:g} dB | deterministic per-window exact power")
    print(f"Device: {config.device}")
    print(f"Pending fits: {len(pending)}", flush=True)
    for index, (spec, seed, source_rpm, target_rpm) in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {spec.name} | seed={seed} | "
            f"{base.LOAD_MAP[source_rpm]}->{base.LOAD_MAP[target_rpm]}",
            flush=True,
        )
        source_train = build_noisy_partition(
            config, metadata_map, source_rpm, "source_train", seed
        )
        target_train = None
        if spec.uses_target_training:
            target_train = build_noisy_partition(
                config, metadata_map, target_rpm, "target_train", seed
            )

        def target_test_factory():
            return build_noisy_partition(
                config, metadata_map, target_rpm, "target_test", seed
            )

        result = train_one(
            config,
            spec,
            source_rpm,
            target_rpm,
            seed,
            source_train,
            target_test_factory,
            target_train=target_train,
        )
        rows.append(result)
        write_outputs(rows, output_dir, tasks)
        completed.add(row_key(result))
        write_manifest(
            manifest_path,
            build_manifest(
                config,
                tasks,
                seeds,
                dataset_inventory,
                "running",
                sorted(completed),
            ),
        )
        print(
            f"  target accuracy={result['accuracy']:.2f}% "
            f"macro-F1={result['macro_f1']:.2f}%",
            flush=True,
        )

    expected_runs = len(METHODS) * len(tasks) * len(seeds)
    status = "complete" if len(completed) == expected_runs else "partial"
    write_manifest(
        manifest_path,
        build_manifest(
            config,
            tasks,
            seeds,
            dataset_inventory,
            status,
            sorted(completed),
        ),
    )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
