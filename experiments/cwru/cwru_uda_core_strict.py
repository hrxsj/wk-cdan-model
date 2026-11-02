"""CWRU DANN, CDAN, and DSAN/LMMD experiments with chronological partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
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


PROTOCOL_VERSION = "cwru-uda-core-strict-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_uda_core_strict_100epoch_20260721"
SMOKE_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_uda_core_strict_smoke"
DEFAULT_SEEDS = tuple(DEFAULT_RUN_SEEDS)
DEFAULT_TASKS = tuple(
    (source, target)
    for source in base.ALL_RPMS
    for target in base.ALL_RPMS
    if source != target
)

METHOD_DANN = pu_core.METHOD_DANN
METHOD_CDAN = pu_core.METHOD_CDAN
METHOD_DSAN = pu_core.METHOD_DSAN


@dataclass(frozen=True)
class MethodSpec:
    name: str
    pu_method: str
    domain_mode: str
    adaptation: str


METHODS = (
    MethodSpec(
        METHOD_DANN,
        pu_core.METHOD_DANN,
        "dann",
        "GRL on 128-D features with summed source/target domain CE",
    ),
    MethodSpec(
        METHOD_CDAN,
        pu_core.METHOD_CDAN,
        "cdan",
        (
            "GRL on vec(feature outer softmax(class logits)) with summed "
            "source/target domain CE"
        ),
    ),
    MethodSpec(
        METHOD_DSAN,
        pu_core.METHOD_DSAN,
        "lmmd",
        (
            "PU strict class-conditional multi-kernel LMMD using source labels "
            "and detached target soft pseudo-label probabilities"
        ),
    ),
)
METHOD_BY_NAME = {spec.name: spec for spec in METHODS}


@dataclass(frozen=True)
class UDAConfig:
    data_root: str = base.CFG.data_root
    metadata_path: str = base.CFG.metadata_path
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    data_protocol: str = "strict_holdout"
    checkpoint_policy: str = "final_epoch"
    window_size: int = 1024
    overlap: float = 0.5
    train_ratio: float = 0.8
    test_ratio: float = 0.2
    batch_size: int = 32
    epochs: int = 100
    lr: float = 5e-4
    weight_decay: float = 1e-3
    warmup_epochs: int = 20
    adaptation_weight: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_windows_per_class: Optional[int] = None
    max_steps_per_epoch: Optional[int] = None
    log_every: int = 10
    reportable: bool = True


@dataclass(frozen=True)
class LabeledPartition:
    rpm: str
    role: str
    x: np.ndarray
    y: np.ndarray
    windows_by_class: Tuple[int, ...]
    fingerprint: str


@dataclass(frozen=True)
class UnlabeledPartition:
    rpm: str
    role: str
    x: np.ndarray
    windows_by_class: Tuple[int, ...]
    fingerprint: str


@dataclass
class TrainingOutcome:
    model: nn.Module
    history: List[Dict[str, object]]
    training_seconds: float
    source_batches_consumed: int
    target_batches_consumed: int
    target_signals_consumed: int
    training_finished_ns: int
    post_train_state_sha256: str


def method_definition(spec: MethodSpec):
    definition = {
        "family": "unsupervised_domain_adaptation",
        "architecture_source": str(Path(pu_core.__file__).resolve()),
        "pu_method_identity": spec.pu_method,
        "feature_extractor": (
            "PU-aligned compact 1D ResNet-18 [2,2,2,2], widths "
            "32-64-128-256, 128-D projection"
        ),
        "classifier": "PU-aligned linear 128-to-10",
        "domain_mode": spec.domain_mode,
        "adaptation": spec.adaptation,
        "objective": (
            "source classification CE plus adaptation_weight=1.0 times the "
            "PU strict method-specific adaptation objective"
        ),
        "alpha_schedule": (
            "PU strict logistic GRL/LMMD schedule after 20 frozen warm-up epochs"
        ),
        "target_training_loader": "one signal tensor only; no class-label tensor",
        "target_test_usage": "one final evaluation after fixed epoch 100",
    }
    if spec.domain_mode == "lmmd":
        definition["domain_discriminator"] = "absent"
    else:
        domain_input = "128-D features" if spec.domain_mode == "dann" else "1280-D conditional features"
        definition["domain_discriminator"] = (
            f"PU-aligned {domain_input} -> 128 -> 2 discriminator"
        )
    return definition


def stable_seed(*parts: object):
    payload = "|".join(str(part) for part in (PROTOCOL_VERSION, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def partition_seed(seed: int, rpm: str, role: str):
    if role not in {"source_train", "target_train"}:
        raise ValueError("Only train partitions have RNG seeds")
    return stable_seed("partition", int(seed), rpm, role)


def loader_seed(seed: int, rpm: str, role: str):
    if role not in {"source_train", "target_train"}:
        raise ValueError("Only train loaders have RNG seeds")
    return stable_seed("loader", int(seed), rpm, role)


def _partition_windows(
    config: UDAConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
    side: str,
):
    if side not in {"train", "test"}:
        raise ValueError("side must be train or test")
    x_parts: List[np.ndarray] = []
    windows_by_class: List[int] = []
    for label_name in base.LABELS:
        experiment_name, file_name = base.CLASS_FILES[label_name]
        mat_path = base.ensure_mat_file(
            config.data_root,
            metadata_map,
            experiment_name,
            rpm,
            label_name,
            file_name,
        )
        signal = base.load_1d_signal(mat_path)
        train_range, test_range = base.split_signal_by_time(
            signal, config.train_ratio, config.test_ratio
        )
        start, end = train_range if side == "train" else test_range
        windows = base.sliding_windows_from_range(
            signal, start, end, config.window_size, config.overlap
        )
        if config.max_windows_per_class is not None:
            windows = windows[: config.max_windows_per_class]
        x_parts.append(windows)
        windows_by_class.append(len(windows))
    return x_parts, tuple(windows_by_class)


def build_source_train_partition(
    config: UDAConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
    seed: int,
):
    x_parts, counts = _partition_windows(config, metadata_map, rpm, "train")
    y_parts = [
        np.full(count, class_index, dtype=np.int64)
        for class_index, count in enumerate(counts)
    ]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    permutation = np.random.default_rng(
        partition_seed(seed, rpm, "source_train")
    ).permutation(len(y))
    x = x[permutation]
    y = y[permutation]
    return LabeledPartition(
        rpm=rpm,
        role="source_train",
        x=x,
        y=y,
        windows_by_class=counts,
        fingerprint=base.fingerprint_arrays(x, y),
    )


def build_target_train_partition(
    config: UDAConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
    seed: int,
):
    """Build first-80% target signals without creating a class-label array."""

    x_parts, counts = _partition_windows(config, metadata_map, rpm, "train")
    x = np.concatenate(x_parts, axis=0)
    permutation = np.random.default_rng(
        partition_seed(seed, rpm, "target_train")
    ).permutation(len(x))
    x = x[permutation]
    return UnlabeledPartition(
        rpm=rpm,
        role="target_train",
        x=x,
        windows_by_class=counts,
        fingerprint=base.fingerprint_arrays(x),
    )


def build_target_test_partition(
    config: UDAConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
):
    """Build the held-out final-20% partition with no RNG operation."""

    x_parts, counts = _partition_windows(config, metadata_map, rpm, "test")
    y_parts = [
        np.full(count, class_index, dtype=np.int64)
        for class_index, count in enumerate(counts)
    ]
    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return LabeledPartition(
        rpm=rpm,
        role="target_test",
        x=x,
        y=y,
        windows_by_class=counts,
        fingerprint=base.fingerprint_arrays(x, y),
    )


def make_source_loader(
    partition: LabeledPartition, batch_size: int, generator_seed: int
):
    if partition.role != "source_train":
        raise ValueError("source loader requires source_train")
    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    dataset = TensorDataset(
        torch.from_numpy(partition.x[:, None, :].astype(np.float32, copy=False)),
        torch.from_numpy(partition.y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


def make_unlabeled_target_loader(
    partition: UnlabeledPartition, batch_size: int, generator_seed: int
):
    if partition.role != "target_train":
        raise ValueError("target loader requires target_train")
    generator = torch.Generator()
    generator.manual_seed(int(generator_seed))
    dataset = TensorDataset(
        torch.from_numpy(partition.x[:, None, :].astype(np.float32, copy=False))
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )


def make_target_test_loader(partition: LabeledPartition, batch_size: int):
    if partition.role != "target_test":
        raise ValueError("target test loader requires target_test")
    dataset = TensorDataset(
        torch.from_numpy(partition.x[:, None, :].astype(np.float32, copy=False)),
        torch.from_numpy(partition.y.astype(np.int64, copy=False)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def build_model(spec: MethodSpec, init_seed: int):
    model = pu_core.CoreBaselineModel(
        spec.pu_method, num_classes=base.NUM_CLASSES, init_seed=init_seed
    )
    if model.domain_mode != spec.domain_mode:
        raise RuntimeError("PU method domain mode does not match the frozen spec")
    if spec.domain_mode in {"dann", "cdan"} and model.domain_classifier is None:
        raise RuntimeError("Adversarial method is missing its discriminator")
    if spec.domain_mode == "lmmd" and model.domain_classifier is not None:
        raise RuntimeError("DSAN/LMMD must not contain a discriminator")
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


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optimization_signature(
    config: UDAConfig,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
    source_fingerprint: str,
    target_train_fingerprint: str,
):
    """Hash training inputs and policy; held-out target-test data is absent."""

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "method_spec": asdict(spec),
        "architecture_dependency_sha256": sha256_file(Path(pu_core.__file__)),
        "source_rpm": source_rpm,
        "target_train_rpm": target_rpm,
        "source_fingerprint": source_fingerprint,
        "target_train_fingerprint": target_train_fingerprint,
        "experiment_seed": int(seed),
        "model_init_seed": int(seed),
        "source_partition_seed": partition_seed(seed, source_rpm, "source_train"),
        "target_partition_seed": partition_seed(seed, target_rpm, "target_train"),
        "source_loader_seed": loader_seed(seed, source_rpm, "source_train"),
        "target_loader_seed": loader_seed(seed, target_rpm, "target_train"),
        "window_size": config.window_size,
        "overlap": config.overlap,
        "train_ratio": config.train_ratio,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "adaptation_weight": config.adaptation_weight,
        "max_windows_per_class": config.max_windows_per_class,
        "max_steps_per_epoch": config.max_steps_per_epoch,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def train_uda(
    config: UDAConfig,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
    source_train: LabeledPartition,
    target_train: UnlabeledPartition,
):
    """Train from labeled source and signal-only target-train batches."""

    if config.data_protocol != "strict_holdout":
        raise ValueError("UDA core requires strict_holdout")
    if config.checkpoint_policy != "final_epoch":
        raise ValueError("UDA core requires final_epoch")
    if source_train.role != "source_train" or source_train.rpm != source_rpm:
        raise ValueError("Source partition identity mismatch")
    if target_train.role != "target_train" or target_train.rpm != target_rpm:
        raise ValueError("Target-train partition identity mismatch")

    base.set_seed(seed)
    device = torch.device(config.device)
    source_loader = make_source_loader(
        source_train,
        config.batch_size,
        loader_seed(seed, source_rpm, "source_train"),
    )
    target_loader = make_unlabeled_target_loader(
        target_train,
        config.batch_size,
        loader_seed(seed, target_rpm, "target_train"),
    )
    if len(target_loader.dataset.tensors) != 1:
        raise RuntimeError("Target-train loader must contain signals only")
    steps_per_epoch = min(len(source_loader), len(target_loader))
    if config.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, config.max_steps_per_epoch)
    if steps_per_epoch <= 0:
        raise ValueError("No complete source/target training batch was produced")

    model = build_model(spec, seed).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    history: List[Dict[str, object]] = []
    source_batches_consumed = 0
    target_batches_consumed = 0
    target_signals_consumed = 0

    base.set_seed(seed)
    training_start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        source_iterator = iter(source_loader)
        target_iterator = iter(target_loader)
        sums = {"total": 0.0, "classification": 0.0, "adaptation": 0.0}
        alpha = 0.0
        for step in range(steps_per_epoch):
            source_x, source_y = next(source_iterator)
            (target_x,) = next(target_iterator)
            source_x = source_x.to(device)
            source_y = source_y.to(device)
            target_x = target_x.to(device)
            alpha = pu_core.strict_data.alpha_schedule(
                epoch,
                step,
                steps_per_epoch,
                config.epochs,
                config.warmup_epochs,
            )

            optimizer.zero_grad(set_to_none=True)
            source_logits, source_domain, source_features = model(source_x, alpha)
            target_logits, target_domain, target_features = model(target_x, alpha)
            classification = criterion(source_logits, source_y)
            if spec.domain_mode in {"dann", "cdan"}:
                if source_domain is None or target_domain is None:
                    raise RuntimeError("DANN/CDAN discriminator output is missing")
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
            else:
                if source_domain is not None or target_domain is not None:
                    raise RuntimeError("DSAN/LMMD unexpectedly produced domain logits")
                target_probabilities = F.softmax(target_logits, dim=1)
                adaptation = pu_core.lmmd_loss(
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
            source_batches_consumed += 1
            target_batches_consumed += 1
            target_signals_consumed += int(target_x.size(0))

        history_row = {
            "epoch": epoch,
            "train_loss": sums["total"] / steps_per_epoch,
            "classification_loss": sums["classification"] / steps_per_epoch,
            "adaptation_loss": sums["adaptation"] / steps_per_epoch,
            "alpha_final_step": alpha,
            "source_training_batches": steps_per_epoch,
            "target_training_batches": steps_per_epoch,
            "target_training_labels_consumed": False,
        }
        history.append(history_row)
        if epoch == 1 or epoch == config.epochs or epoch % config.log_every == 0:
            print(
                f"    epoch {epoch:3d}/{config.epochs} "
                f"loss={history_row['train_loss']:.4f} "
                f"cls={history_row['classification_loss']:.4f} "
                f"adapt={history_row['adaptation_loss']:.4f} "
                f"alpha={alpha:.3f}",
                flush=True,
            )

    training_seconds = time.perf_counter() - training_start
    training_finished_ns = time.perf_counter_ns()
    return TrainingOutcome(
        model=model,
        history=history,
        training_seconds=training_seconds,
        source_batches_consumed=source_batches_consumed,
        target_batches_consumed=target_batches_consumed,
        target_signals_consumed=target_signals_consumed,
        training_finished_ns=training_finished_ns,
        post_train_state_sha256=model_state_sha256(model),
    )


def evaluate_once_with_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
):
    """The sole held-out target-test model pass."""

    model.eval()
    predictions: List[int] = []
    targets: List[int] = []
    prediction_rows: List[Dict[str, object]] = []
    sample_index = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model.classify(x.to(device))
            probabilities = F.softmax(logits, dim=1).cpu()
            predicted = probabilities.argmax(dim=1)
            confidence = probabilities.max(dim=1).values
            for local_index in range(len(y)):
                true_index = int(y[local_index])
                predicted_index = int(predicted[local_index])
                predictions.append(predicted_index)
                targets.append(true_index)
                prediction_rows.append(
                    {
                        "method": spec.name,
                        "seed": seed,
                        "source_rpm": source_rpm,
                        "target_rpm": target_rpm,
                        "source_hp": base.LOAD_MAP[source_rpm],
                        "target_hp": base.LOAD_MAP[target_rpm],
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
    return base.metrics_from_predictions(predictions, targets), prediction_rows


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


def write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]):
    rows = [
        {"true_label": label, **dict(zip(base.LABELS, values))}
        for label, values in zip(base.LABELS, matrix)
    ]
    _write_csv(path, rows, ("true_label", *base.LABELS))


def run_one(
    config: UDAConfig,
    spec: MethodSpec,
    source_rpm: str,
    target_rpm: str,
    seed: int,
    source_train: LabeledPartition,
    target_train: UnlabeledPartition,
    target_test_factory: Callable[[], LabeledPartition],
):
    outcome = train_uda(
        config,
        spec,
        source_rpm,
        target_rpm,
        seed,
        source_train,
        target_train,
    )

    target_build_started_ns = time.perf_counter_ns()
    if target_build_started_ns < outcome.training_finished_ns:
        raise RuntimeError("Target-test construction started before training finished")
    target_test = target_test_factory()
    if target_test.role != "target_test" or target_test.rpm != target_rpm:
        raise ValueError("Target-test partition identity mismatch")
    test_loader = make_target_test_loader(target_test, config.batch_size)
    target_metrics, prediction_rows = evaluate_once_with_predictions(
        outcome.model,
        test_loader,
        torch.device(config.device),
        spec,
        source_rpm,
        target_rpm,
        seed,
    )
    post_evaluation_state_sha256 = model_state_sha256(outcome.model)
    if post_evaluation_state_sha256 != outcome.post_train_state_sha256:
        raise RuntimeError("Final target evaluation mutated model state")

    output_dir = Path(config.output_dir)
    artifact_dir = (
        output_dir
        / "artifacts"
        / base.safe_slug(spec.name)
        / f"{base.LOAD_MAP[source_rpm]}_to_{base.LOAD_MAP[target_rpm]}"
        / f"seed_{seed}"
    )
    history_path = artifact_dir / "epoch_history.csv"
    prediction_path = artifact_dir / "target_predictions.csv"
    confusion_path = artifact_dir / "confusion_matrix.csv"
    checkpoint_path = artifact_dir / "final_checkpoint.pt"
    _write_csv(history_path, outcome.history, tuple(outcome.history[0]))
    _write_csv(
        prediction_path,
        prediction_rows,
        tuple(prediction_rows[0]) if prediction_rows else ("method", "sample_index"),
    )
    write_confusion_matrix(confusion_path, target_metrics["confusion_matrix"])

    optimization_hash = optimization_signature(
        config,
        spec,
        source_rpm,
        target_rpm,
        seed,
        source_train.fingerprint,
        target_train.fingerprint,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "protocol_version": PROTOCOL_VERSION,
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
            "optimization_signature": optimization_hash,
            "post_train_state_sha256": outcome.post_train_state_sha256,
            "post_evaluation_state_sha256": post_evaluation_state_sha256,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in outcome.model.state_dict().items()
            },
            "source_train_fingerprint": source_train.fingerprint,
            "target_train_fingerprint": target_train.fingerprint,
            "target_test_fingerprint": target_test.fingerprint,
            "target_training_loader_tensor_count": 1,
            "target_training_labels_consumed": False,
            "target_test_partition_builds": 1,
            "target_test_evaluations": 1,
            "target_metrics": target_metrics,
        },
        checkpoint_path,
    )

    model_params = sum(
        parameter.numel()
        for parameter in outcome.model.parameters()
        if parameter.requires_grad
    )
    inference_params = pu_core.inference_parameter_count(outcome.model)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "method": spec.name,
        "domain_mode": spec.domain_mode,
        "data_protocol": config.data_protocol,
        "checkpoint_policy": config.checkpoint_policy,
        "source_rpm": source_rpm,
        "target_rpm": target_rpm,
        "source_hp": base.LOAD_MAP[source_rpm],
        "target_hp": base.LOAD_MAP[target_rpm],
        "seed": seed,
        "accuracy": 100.0 * target_metrics["accuracy"],
        "macro_f1": 100.0 * target_metrics["macro_f1"],
        "class_accuracy_json": json.dumps(target_metrics["class_accuracy"]),
        "confusion_matrix_json": json.dumps(target_metrics["confusion_matrix"]),
        "epochs_run": config.epochs,
        "selected_epoch": config.epochs,
        "selection_rule": "fixed_final_epoch",
        "model_params_m": model_params / 1e6,
        "inference_params_m": inference_params / 1e6,
        "training_seconds": outcome.training_seconds,
        "final_train_loss": outcome.history[-1]["train_loss"],
        "final_classification_loss": outcome.history[-1]["classification_loss"],
        "final_adaptation_loss": outcome.history[-1]["adaptation_loss"],
        "final_alpha": outcome.history[-1]["alpha_final_step"],
        "source_training_loader_built": True,
        "source_training_loader_tensor_count": 2,
        "source_training_batches_consumed": outcome.source_batches_consumed,
        "target_training_partition_built": True,
        "target_training_loader_built": True,
        "target_training_loader_tensor_count": 1,
        "target_training_labels_available_to_loader": False,
        "target_training_labels_consumed": False,
        "target_training_batches_consumed": outcome.target_batches_consumed,
        "target_training_samples_consumed": outcome.target_signals_consumed,
        "target_test_partition_built_after_training": True,
        "target_test_partition_builds": 1,
        "target_test_loader_builds": 1,
        "target_test_evaluations": 1,
        "target_test_labels_consumed_only_in_final_evaluation": True,
        "source_train_windows": len(source_train.y),
        "target_train_windows": len(target_train.x),
        "target_test_windows": len(target_test.y),
        "source_train_fingerprint": source_train.fingerprint,
        "target_train_fingerprint": target_train.fingerprint,
        "target_test_fingerprint": target_test.fingerprint,
        "optimization_signature": optimization_hash,
        "post_train_state_sha256": outcome.post_train_state_sha256,
        "post_evaluation_state_sha256": post_evaluation_state_sha256,
        "history_path": str(history_path.relative_to(output_dir)),
        "history_sha256": sha256_file(history_path),
        "prediction_path": str(prediction_path.relative_to(output_dir)),
        "prediction_sha256": sha256_file(prediction_path),
        "confusion_matrix_path": str(confusion_path.relative_to(output_dir)),
        "confusion_matrix_sha256": sha256_file(confusion_path),
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }


def execute_run(
    config: UDAConfig,
    spec: MethodSpec,
    metadata_map: Mapping[Tuple[str, str, str], str],
    source_rpm: str,
    target_rpm: str,
    seed: int,
):
    source_train = build_source_train_partition(
        config, metadata_map, source_rpm, seed
    )
    target_train = build_target_train_partition(
        config, metadata_map, target_rpm, seed
    )

    def target_test_factory():
        return build_target_test_partition(config, metadata_map, target_rpm)

    return run_one(
        config,
        spec,
        source_rpm,
        target_rpm,
        seed,
        source_train,
        target_train,
        target_test_factory,
    )


RAW_FIELDS = (
    "protocol_version",
    "method",
    "domain_mode",
    "data_protocol",
    "checkpoint_policy",
    "source_rpm",
    "target_rpm",
    "source_hp",
    "target_hp",
    "seed",
    "accuracy",
    "macro_f1",
    "class_accuracy_json",
    "confusion_matrix_json",
    "epochs_run",
    "selected_epoch",
    "selection_rule",
    "model_params_m",
    "inference_params_m",
    "training_seconds",
    "final_train_loss",
    "final_classification_loss",
    "final_adaptation_loss",
    "final_alpha",
    "source_training_loader_built",
    "source_training_loader_tensor_count",
    "source_training_batches_consumed",
    "target_training_partition_built",
    "target_training_loader_built",
    "target_training_loader_tensor_count",
    "target_training_labels_available_to_loader",
    "target_training_labels_consumed",
    "target_training_batches_consumed",
    "target_training_samples_consumed",
    "target_test_partition_built_after_training",
    "target_test_partition_builds",
    "target_test_loader_builds",
    "target_test_evaluations",
    "target_test_labels_consumed_only_in_final_evaluation",
    "source_train_windows",
    "target_train_windows",
    "target_test_windows",
    "source_train_fingerprint",
    "target_train_fingerprint",
    "target_test_fingerprint",
    "optimization_signature",
    "post_train_state_sha256",
    "post_evaluation_state_sha256",
    "history_path",
    "history_sha256",
    "prediction_path",
    "prediction_sha256",
    "confusion_matrix_path",
    "confusion_matrix_sha256",
    "checkpoint_path",
    "checkpoint_sha256",
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


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    expected_tasks: Sequence[Tuple[str, str]] = DEFAULT_TASKS,
):
    task_groups: MutableMapping[
        Tuple[str, str, str], List[Mapping[str, object]]
    ] = {}
    for row in rows:
        key = (str(row["method"]), str(row["source_rpm"]), str(row["target_rpm"]))
        task_groups.setdefault(key, []).append(row)
    task_output = []
    for (method, source, target), group in sorted(task_groups.items()):
        accuracy = metric_stats([float(row["accuracy"]) for row in group])
        macro_f1 = metric_stats([float(row["macro_f1"]) for row in group])
        task_output.append(
            {
                "method": method,
                "source_rpm": source,
                "target_rpm": target,
                "source_hp": base.LOAD_MAP[source],
                "target_hp": base.LOAD_MAP[target],
                "accuracy_mean": accuracy["mean"],
                "accuracy_std": accuracy["std"],
                "macro_f1_mean": macro_f1["mean"],
                "macro_f1_std": macro_f1["std"],
                "runs": len(group),
            }
        )

    expected_task_set = set(expected_tasks)
    seed_groups: MutableMapping[
        Tuple[str, int], List[Mapping[str, object]]
    ] = {}
    for row in rows:
        seed_groups.setdefault((str(row["method"]), int(row["seed"])), []).append(row)
    seed_output = []
    for (method, seed), group in sorted(seed_groups.items()):
        observed = {
            (str(row["source_rpm"]), str(row["target_rpm"])) for row in group
        }
        seed_output.append(
            {
                "method": method,
                "seed": seed,
                "cross_accuracy_mean": mean(float(row["accuracy"]) for row in group),
                "cross_macro_f1_mean": mean(float(row["macro_f1"]) for row in group),
                "tasks": len(group),
                "complete_seed_block": observed == expected_task_set,
            }
        )

    method_output = []
    for spec in METHODS:
        complete = [
            row
            for row in seed_output
            if row["method"] == spec.name and row["complete_seed_block"]
        ]
        accuracy = metric_stats(
            [float(row["cross_accuracy_mean"]) for row in complete]
        )
        macro_f1 = metric_stats(
            [float(row["cross_macro_f1_mean"]) for row in complete]
        )
        method_rows = [row for row in rows if str(row["method"]) == spec.name]
        method_output.append(
            {
                "method": spec.name,
                "cross_accuracy_mean_across_seeds": accuracy["mean"],
                "cross_accuracy_std_across_seeds": accuracy["std"],
                "cross_macro_f1_mean_across_seeds": macro_f1["mean"],
                "cross_macro_f1_std_across_seeds": macro_f1["std"],
                "complete_seed_blocks": len(complete),
                "runs": len(method_rows),
                "model_params_m": mean(
                    float(row["model_params_m"]) for row in method_rows
                )
                if method_rows
                else 0.0,
                "inference_params_m": mean(
                    float(row["inference_params_m"]) for row in method_rows
                )
                if method_rows
                else 0.0,
            }
        )
    return task_output, seed_output, method_output


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    tasks: Sequence[Tuple[str, str]],
):
    _write_csv(output_dir / "raw_results.csv", rows, RAW_FIELDS)
    task_rows, seed_rows, method_rows = summarize_rows(rows, tasks)
    _write_csv(
        output_dir / "task_summary.csv",
        task_rows,
        tuple(task_rows[0]) if task_rows else ("method",),
    )
    _write_csv(
        output_dir / "seed_level_summary.csv",
        seed_rows,
        tuple(seed_rows[0]) if seed_rows else ("method",),
    )
    _write_csv(
        output_dir / "method_summary.csv",
        method_rows,
        tuple(method_rows[0]) if method_rows else ("method",),
    )
    path = output_dir / "method_summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {"methods": method_rows, "seeds": seed_rows, "tasks": task_rows},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
    return task_rows, seed_rows, method_rows


def build_dataset_inventory(
    config: UDAConfig,
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
                    "hp": base.LOAD_MAP[rpm],
                    "label": label_name,
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return inventory


def parameter_inventory(methods: Sequence[MethodSpec] = METHODS):
    inventory = {}
    for spec in methods:
        model = build_model(spec, 0)
        trainable = pu_core.parameter_count(model)
        inference = pu_core.inference_parameter_count(model)
        domain = trainable - inference
        inventory[spec.name] = {
            "trainable": trainable,
            "trainable_m": trainable / 1e6,
            "inference": inference,
            "inference_m": inference / 1e6,
            "domain_discriminator": domain,
            "domain_discriminator_m": domain / 1e6,
        }
    return inventory


def _environment_signature():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "platform": platform.platform(),
    }


def build_manifest(
    config: UDAConfig,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    dataset_inventory: Sequence[Mapping[str, object]],
    status: str,
    completed_keys: Sequence[Tuple[str, int, str, str]],
):
    script_path = Path(__file__).resolve()
    base_path = Path(base.__file__).resolve()
    pu_path = Path(pu_core.__file__).resolve()
    pu_data_path = Path(pu_core.STRICT_DATA_SCRIPT).resolve()
    metadata_path = Path(config.metadata_path).resolve()
    frozen_signature = {
        "protocol_version": PROTOCOL_VERSION,
        "runner": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "dependencies": [
            {
                "path": str(base_path),
                "sha256": sha256_file(base_path),
                "purpose": "CWRU record loading, split/window primitives, metrics",
            },
            {
                "path": str(pu_path),
                "sha256": sha256_file(pu_path),
                "purpose": "exact PU DANN/CDAN/DSAN models and LMMD",
            },
            {
                "path": str(pu_data_path),
                "sha256": sha256_file(pu_data_path),
                "purpose": "PU model initializer and alpha schedule",
            },
        ],
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
        "parameter_inventory": parameter_inventory(),
        "tasks": [
            {
                "source_rpm": source,
                "target_rpm": target,
                "source_hp": base.LOAD_MAP[source],
                "target_hp": base.LOAD_MAP[target],
            }
            for source, target in tasks
        ],
        "seeds": list(seeds),
        "expected_runs": len(METHODS) * len(tasks) * len(seeds),
        "split_policy": (
            "Each continuous record is split chronologically: first 80% train, "
            "final 20% held-out test, before sliding windows; no window crosses "
            "the boundary."
        ),
        "training_data_policy": (
            "Each update uses source signals plus source labels and target-train "
            "signals only. The target loader contains exactly one tensor and no "
            "target class-label array is created for target training."
        ),
        "target_test_policy": (
            "The target-test partition/loader is constructed only after fixed "
            "epoch 100, evaluated exactly once, cannot mutate model state, and "
            "cannot influence selection or tuning."
        ),
        "rng_policy": (
            "Source and target-train partition/loader RNGs are independently "
            "derived from protocol, experiment seed, RPM, and role. Target-test "
            "construction is deterministic and has no RNG."
        ),
        "statistics_policy": (
            "Task mean/sample std are across five seeds. Overall mean/sample std "
            "are across five seed-level averages, each averaging all 12 tasks."
        ),
        "raw_hash_policy": (
            "All 40 raw MAT files are byte-hashed for integrity without using "
            "held-out target windows in optimization."
        ),
        "environment": _environment_signature(),
        "reportable": config.reportable,
    }
    return {
        "frozen_signature": frozen_signature,
        "status": status,
        "completed_runs": len(completed_keys),
        "completed_keys": [list(key) for key in sorted(completed_keys)],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv],
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
            "Refusing resume: code, data hashes, environment, methods, or "
            "configuration differ from the frozen UDA manifest"
        )


def _as_bool(value: object):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


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


def validate_existing_rows(
    rows: Sequence[Mapping[str, object]],
    config: UDAConfig,
    output_dir: Path,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
):
    expected = {
        (spec.name, int(seed), source, target)
        for spec in METHODS
        for seed in seeds
        for source, target in tasks
    }
    seen = set()
    for row in rows:
        key = row_key(row)
        if key not in expected:
            raise RuntimeError(f"Unexpected raw-result key: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate raw-result key: {key}")
        seen.add(key)
        spec = METHOD_BY_NAME[key[0]]
        if str(row.get("protocol_version")) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol mismatch in row: {key}")
        if str(row.get("domain_mode")) != spec.domain_mode:
            raise RuntimeError(f"Domain-mode mismatch in row: {key}")
        if str(row.get("data_protocol")) != "strict_holdout":
            raise RuntimeError(f"Split protocol mismatch in row: {key}")
        if str(row.get("checkpoint_policy")) != "final_epoch":
            raise RuntimeError(f"Checkpoint policy mismatch in row: {key}")
        if int(row.get("epochs_run", -1)) != config.epochs:
            raise RuntimeError(f"Epoch-count mismatch in row: {key}")
        if int(row.get("selected_epoch", -1)) != config.epochs:
            raise RuntimeError(f"Selected-epoch mismatch in row: {key}")
        if str(row.get("selection_rule")) != "fixed_final_epoch":
            raise RuntimeError(f"Selection-rule mismatch in row: {key}")
        if not _as_bool(row.get("source_training_loader_built")):
            raise RuntimeError(f"Missing source loader in row: {key}")
        if int(row.get("source_training_loader_tensor_count", -1)) != 2:
            raise RuntimeError(f"Source loader schema mismatch in row: {key}")
        if int(row.get("source_training_batches_consumed", 0)) <= 0:
            raise RuntimeError(f"No source batches consumed in row: {key}")
        if not _as_bool(row.get("target_training_partition_built")):
            raise RuntimeError(f"Missing target-train partition in row: {key}")
        if not _as_bool(row.get("target_training_loader_built")):
            raise RuntimeError(f"Missing target-train loader in row: {key}")
        if int(row.get("target_training_loader_tensor_count", -1)) != 1:
            raise RuntimeError(f"Target loader is not signal-only in row: {key}")
        if _as_bool(row.get("target_training_labels_available_to_loader")):
            raise RuntimeError(f"Target labels were exposed to loader in row: {key}")
        if _as_bool(row.get("target_training_labels_consumed")):
            raise RuntimeError(f"Target labels entered training in row: {key}")
        if int(row.get("target_training_batches_consumed", 0)) <= 0:
            raise RuntimeError(f"No target batches consumed in row: {key}")
        if int(row.get("target_training_samples_consumed", 0)) <= 0:
            raise RuntimeError(f"No target signals consumed in row: {key}")
        if int(row.get("target_train_windows", 0)) <= 0:
            raise RuntimeError(f"No target-train windows in row: {key}")
        if not _as_bool(row.get("target_test_partition_built_after_training")):
            raise RuntimeError(f"Target test was built too early in row: {key}")
        if int(row.get("target_test_partition_builds", -1)) != 1:
            raise RuntimeError(f"Target-test build count mismatch in row: {key}")
        if int(row.get("target_test_loader_builds", -1)) != 1:
            raise RuntimeError(f"Target-test loader count mismatch in row: {key}")
        if int(row.get("target_test_evaluations", -1)) != 1:
            raise RuntimeError(f"Target-test evaluation count mismatch in row: {key}")
        if not _as_bool(
            row.get("target_test_labels_consumed_only_in_final_evaluation")
        ):
            raise RuntimeError(f"Target-test label policy mismatch in row: {key}")
        if row.get("post_train_state_sha256") != row.get(
            "post_evaluation_state_sha256"
        ):
            raise RuntimeError(f"Target evaluation changed model state in row: {key}")
        expected_optimization = optimization_signature(
            config,
            spec,
            key[2],
            key[3],
            key[1],
            str(row["source_train_fingerprint"]),
            str(row["target_train_fingerprint"]),
        )
        if str(row.get("optimization_signature")) != expected_optimization:
            raise RuntimeError(f"Optimization signature mismatch in row: {key}")
        for path_field, hash_field in (
            ("history_path", "history_sha256"),
            ("prediction_path", "prediction_sha256"),
            ("confusion_matrix_path", "confusion_matrix_sha256"),
            ("checkpoint_path", "checkpoint_sha256"),
        ):
            artifact = output_dir / str(row[path_field])
            if not artifact.is_file():
                raise RuntimeError(f"Missing artifact for {key}: {artifact}")
            if sha256_file(artifact) != str(row[hash_field]):
                raise RuntimeError(f"Artifact hash mismatch for {key}: {artifact}")
    return seen


def protect_output_directory(output_dir: Path, resume: bool):
    if resume:
        return
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume or a new path: {output_dir}"
        )


def validate_reportable_config(config: UDAConfig):
    if not config.reportable:
        return
    requirements = {
        "data_protocol": (config.data_protocol, "strict_holdout"),
        "checkpoint_policy": (config.checkpoint_policy, "final_epoch"),
        "window_size": (config.window_size, 1024),
        "overlap": (config.overlap, 0.5),
        "train_ratio": (config.train_ratio, 0.8),
        "test_ratio": (config.test_ratio, 0.2),
        "batch_size": (config.batch_size, 32),
        "epochs": (config.epochs, 100),
        "lr": (config.lr, 5e-4),
        "weight_decay": (config.weight_decay, 1e-3),
        "warmup_epochs": (config.warmup_epochs, 20),
        "adaptation_weight": (config.adaptation_weight, 1.0),
        "max_windows_per_class": (config.max_windows_per_class, None),
        "max_steps_per_epoch": (config.max_steps_per_epoch, None),
    }
    mismatches = [
        f"{name}={actual!r} (required {required!r})"
        for name, (actual, required) in requirements.items()
        if actual != required
    ]
    if mismatches:
        raise ValueError("Reportable UDA protocol is frozen: " + ", ".join(mismatches))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PU-aligned strict CWRU DANN/CDAN/DSAN baselines."
    )
    parser.add_argument("--data-root", default=base.CFG.data_root)
    parser.add_argument("--metadata-path", default=base.CFG.metadata_path)
    parser.add_argument("--tasks", default="", help="Comma-separated SOURCE:TARGET RPM pairs")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--adaptation-weight", type=float, default=1.0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run three one-step fits for engineering validation only.",
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
    if args.lr <= 0 or args.weight_decay < 0 or args.adaptation_weight <= 0:
        raise ValueError("lr/adaptation weight must be positive; weight decay non-negative")
    if args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs:
        raise ValueError("warmup epochs must be non-negative and lower than epochs")

    output_dir = Path(
        os.path.abspath(
            args.output_dir
            if args.output_dir is not None
            else str(SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
        )
    )
    config = UDAConfig(
        data_root=os.path.abspath(args.data_root),
        metadata_path=os.path.abspath(args.metadata_path),
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        adaptation_weight=args.adaptation_weight,
        device=args.device,
        log_every=args.log_every,
    )
    tasks = base.parse_task_pairs(args.tasks) if args.tasks else list(DEFAULT_TASKS)
    if not tasks or any(source == target for source, target in tasks):
        raise ValueError("Choose at least one cross-load task")
    seeds = list(DEFAULT_SEEDS)
    if args.smoke:
        tasks = tasks[:1]
        seeds = seeds[:1]
        config = replace(
            config,
            epochs=1,
            batch_size=min(config.batch_size, base.NUM_CLASSES),
            warmup_epochs=0,
            max_windows_per_class=args.smoke_windows_per_class,
            max_steps_per_epoch=1,
            log_every=1,
            reportable=False,
        )
    validate_reportable_config(config)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"CWRU data root does not exist: {config.data_root}")
    if not Path(config.metadata_path).is_file():
        raise FileNotFoundError(
            f"CWRU metadata file does not exist: {config.metadata_path}"
        )
    protect_output_directory(output_dir, args.resume)

    metadata_map = base.load_metadata_map(config.metadata_path)
    print("Hashing the 40 raw CWRU records for the frozen manifest...", flush=True)
    dataset_inventory = build_dataset_inventory(config, metadata_map)
    initial_manifest = build_manifest(
        config, tasks, seeds, dataset_inventory, "running", []
    )
    manifest_path = output_dir / "manifest.json"
    if args.resume:
        validate_resume_manifest(manifest_path, initial_manifest)
    raw_path = output_dir / "raw_results.csv"
    rows = read_rows(raw_path) if args.resume else []
    completed = validate_existing_rows(rows, config, output_dir, tasks, seeds)
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
    print(f"Device: {config.device}")
    print(f"Pending fits: {len(pending)}", flush=True)
    for index, (spec, seed, source, target) in enumerate(pending, start=1):
        print(
            f"[{index}/{len(pending)}] {spec.name} | seed={seed} | "
            f"{base.LOAD_MAP[source]}->{base.LOAD_MAP[target]}",
            flush=True,
        )
        result = execute_run(
            config, spec, metadata_map, source, target, seed
        )
        rows.append(result)
        write_outputs(rows, output_dir, tasks)
        completed = validate_existing_rows(rows, config, output_dir, tasks, seeds)
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
            f"  final target accuracy={result['accuracy']:.2f}% "
            f"macro-F1={result['macro_f1']:.2f}% "
            f"time={result['training_seconds']:.1f}s",
            flush=True,
        )

    _, _, method_rows = write_outputs(rows, output_dir, tasks)
    expected_runs = len(METHODS) * len(tasks) * len(seeds)
    completed = validate_existing_rows(rows, config, output_dir, tasks, seeds)
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
    print("Method summary:")
    for row in method_rows:
        print(
            f"  {row['method']}: "
            f"{row['cross_accuracy_mean_across_seeds']:.2f} +/- "
            f"{row['cross_accuracy_std_across_seeds']:.2f}%"
        )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
