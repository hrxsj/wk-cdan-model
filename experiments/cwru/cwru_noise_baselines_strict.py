"""CWRU B-spline KANConv, DANN, and CDAN comparisons with AWGN."""

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
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cwru_ieee_revision as base  # noqa: E402
from experiments.cwru import cwru_source_only_core_strict as source_core  # noqa: E402
from experiments.cwru import cwru_uda_core_strict as uda_core  # noqa: E402
from experiments.pu import pu_core_baselines_strict as pu_core  # noqa: E402


PROTOCOL_VERSION = "cwru-noise-baselines-strict-v1"
NOISE_PROTOCOL_VERSION = "per-window-power-exact-awgn-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_noise_baselines_strict_20260723"
SMOKE_OUTPUT_DIR = REPO_ROOT / "runs" / "cwru_noise_baselines_strict_smoke"
DEFAULT_SEEDS = tuple(DEFAULT_RUN_SEEDS)
DEFAULT_SNRS = (-1.0, -3.0, -5.0)
DEFAULT_TASKS = tuple(
    (source, target)
    for source in base.ALL_RPMS
    for target in base.ALL_RPMS
    if source != target
)


@dataclass(frozen=True)
class MethodSpec:
    token: str
    name: str
    family: str
    domain_mode: str

    @property
    def uses_target_training(self) -> bool:
        return self.family == "uda"


METHODS = (
    MethodSpec("bkan", source_core.METHOD_BKAN, "source_only", "none"),
    MethodSpec("dann", uda_core.METHOD_DANN, "uda", "dann"),
    MethodSpec("cdan", uda_core.METHOD_CDAN, "uda", "cdan"),
)
METHOD_BY_TOKEN = {spec.token: spec for spec in METHODS}
METHOD_BY_NAME = {spec.name: spec for spec in METHODS}


@dataclass(frozen=True)
class NoiseBaselineConfig:
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
class NoisyPartition:
    rpm: str
    role: str
    x: np.ndarray
    y: Optional[np.ndarray]
    windows_by_class: Tuple[int, ...]
    noise_seeds_by_class: Tuple[int, ...]
    realized_snr_mean_db: float
    realized_snr_min_db: float
    realized_snr_max_db: float
    clean_fingerprint: str
    noisy_fingerprint: str


NOISE_POLICY: Mapping[str, object] = {
    "version": NOISE_PROTOCOL_VERSION,
    "snr_levels_db": list(DEFAULT_SNRS),
    "injection_stage": (
        "after chronological 80/20 record split and normalized window extraction"
    ),
    "power_definition": "mean(square(window)) separately for every window",
    "noise_generation": (
        "zero-mean Gaussian samples rescaled per window to exact requested SNR"
    ),
    "fairness": (
        "noise seed depends only on experiment seed, RPM/load, class, and data role; "
        "it excludes method and SNR"
    ),
    "role_isolation": "source_train, target_train, and target_test use distinct seeds",
}


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in (PROTOCOL_VERSION, *parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def derive_noise_seed(
    experiment_seed: int, rpm: str, class_index: int, role: str
) -> int:
    """Derive a method- and SNR-independent seed for one data-role block."""

    if role not in {"source_train", "target_train", "target_test"}:
        raise ValueError(f"Unknown noise role: {role}")
    payload = (
        f"{NOISE_PROTOCOL_VERSION}|{int(experiment_seed)}|{rpm}|"
        f"{int(class_index)}|{role}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def add_awgn_per_window(
    clean_windows: np.ndarray, snr_db: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Add deterministic Gaussian noise at exact per-window realized SNR."""

    clean = np.asarray(clean_windows, dtype=np.float32)
    if clean.ndim != 2:
        raise ValueError("clean_windows must have shape [windows, samples]")
    if len(clean) == 0:
        return clean.copy(), np.empty((0,), dtype=np.float64)
    signal_power = np.mean(np.square(clean, dtype=np.float64), axis=1, keepdims=True)
    if np.any(signal_power <= 1e-12):
        raise ValueError("Per-window SNR is undefined for a zero-power window")

    rng = np.random.default_rng(int(seed))
    raw_noise = rng.standard_normal(clean.shape)
    raw_noise -= np.mean(raw_noise, axis=1, keepdims=True)
    raw_noise_power = np.mean(np.square(raw_noise), axis=1, keepdims=True)
    if np.any(raw_noise_power <= 1e-12):
        raise RuntimeError("Generated a degenerate Gaussian noise window")
    wanted_power = signal_power / (10.0 ** (float(snr_db) / 10.0))
    noise = raw_noise * np.sqrt(wanted_power / raw_noise_power)
    noisy = (clean.astype(np.float64) + noise).astype(np.float32)
    realized_noise = noisy.astype(np.float64) - clean.astype(np.float64)
    realized_power = np.mean(np.square(realized_noise), axis=1)
    realized_snr = 10.0 * np.log10(signal_power[:, 0] / realized_power)
    return noisy, realized_snr


def build_noisy_partition(
    config: NoiseBaselineConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
    rpm: str,
    role: str,
    experiment_seed: int,
    snr_db: float,
) -> NoisyPartition:
    """Split records first, window second, then add role-isolated AWGN.

    A target-training label vector is deliberately never created.
    """

    if role not in {"source_train", "target_train", "target_test"}:
        raise ValueError(f"Unknown partition role: {role}")
    test_side = role == "target_test"
    clean_parts: List[np.ndarray] = []
    noisy_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    snr_parts: List[np.ndarray] = []
    counts: List[int] = []
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
        start, end = test_range if test_side else train_range
        clean_windows = base.sliding_windows_from_range(
            signal, start, end, config.window_size, config.overlap
        )
        if config.max_windows_per_class is not None:
            clean_windows = clean_windows[: config.max_windows_per_class]
        noise_seed = derive_noise_seed(experiment_seed, rpm, class_index, role)
        noisy_windows, realized_snr = add_awgn_per_window(
            clean_windows, snr_db, noise_seed
        )
        clean_parts.append(clean_windows)
        noisy_parts.append(noisy_windows)
        if role != "target_train":
            label_parts.append(
                np.full(len(clean_windows), class_index, dtype=np.int64)
            )
        snr_parts.append(realized_snr)
        counts.append(len(clean_windows))
        noise_seeds.append(noise_seed)

    clean = np.concatenate(clean_parts, axis=0)
    noisy = np.concatenate(noisy_parts, axis=0)
    realized = np.concatenate(snr_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0) if label_parts else None
    if role != "target_test":
        permutation = np.random.default_rng(
            stable_seed("partition_shuffle", experiment_seed, rpm, role)
        ).permutation(len(noisy))
        clean = clean[permutation]
        noisy = noisy[permutation]
        realized = realized[permutation]
        if labels is not None:
            labels = labels[permutation]

    clean_fingerprint = (
        base.fingerprint_arrays(clean, labels)
        if labels is not None
        else base.fingerprint_arrays(clean)
    )
    noisy_fingerprint = (
        base.fingerprint_arrays(noisy, labels)
        if labels is not None
        else base.fingerprint_arrays(noisy)
    )
    return NoisyPartition(
        rpm=rpm,
        role=role,
        x=noisy,
        y=labels,
        windows_by_class=tuple(counts),
        noise_seeds_by_class=tuple(noise_seeds),
        realized_snr_mean_db=float(np.mean(realized)),
        realized_snr_min_db=float(np.min(realized)),
        realized_snr_max_db=float(np.max(realized)),
        clean_fingerprint=clean_fingerprint,
        noisy_fingerprint=noisy_fingerprint,
    )


def source_spec() -> source_core.MethodSpec:
    return source_core.METHOD_BY_NAME[source_core.METHOD_BKAN]


def uda_spec(spec: MethodSpec) -> uda_core.MethodSpec:
    if spec.family != "uda":
        raise ValueError("A UDA method spec is required")
    return uda_core.METHOD_BY_NAME[spec.name]


def build_model(spec: MethodSpec, init_seed: int = resolve_seed(None)):
    """Build the exact clean-runner model identity for structural auditing."""

    if spec.family == "source_only":
        return source_core.build_model(source_spec(), init_seed)
    return uda_core.build_model(uda_spec(spec), init_seed)


def source_config(config: NoiseBaselineConfig) -> source_core.SourceOnlyConfig:
    return source_core.SourceOnlyConfig(
        data_root=config.data_root,
        metadata_path=config.metadata_path,
        output_dir=config.output_dir,
        data_protocol=config.data_protocol,
        checkpoint_policy=config.checkpoint_policy,
        window_size=config.window_size,
        overlap=config.overlap,
        train_ratio=config.train_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        device=config.device,
        max_windows_per_class=config.max_windows_per_class,
        max_steps_per_epoch=config.max_steps_per_epoch,
        log_every=config.log_every,
        reportable=config.reportable,
    )


def uda_config(config: NoiseBaselineConfig) -> uda_core.UDAConfig:
    return uda_core.UDAConfig(
        data_root=config.data_root,
        metadata_path=config.metadata_path,
        output_dir=config.output_dir,
        data_protocol=config.data_protocol,
        checkpoint_policy=config.checkpoint_policy,
        window_size=config.window_size,
        overlap=config.overlap,
        train_ratio=config.train_ratio,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        epochs=config.epochs,
        lr=config.lr,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
        adaptation_weight=config.adaptation_weight,
        device=config.device,
        max_windows_per_class=config.max_windows_per_class,
        max_steps_per_epoch=config.max_steps_per_epoch,
        log_every=config.log_every,
        reportable=config.reportable,
    )


def to_source_labeled(partition: NoisyPartition) -> source_core.DataPartition:
    if partition.y is None:
        raise ValueError("A labeled source/test partition is required")
    return source_core.DataPartition(
        rpm=partition.rpm,
        role=partition.role,
        x=partition.x,
        y=partition.y,
        windows_by_class=partition.windows_by_class,
        fingerprint=partition.noisy_fingerprint,
    )


def to_uda_labeled(partition: NoisyPartition) -> uda_core.LabeledPartition:
    if partition.y is None:
        raise ValueError("A labeled source/test partition is required")
    return uda_core.LabeledPartition(
        rpm=partition.rpm,
        role=partition.role,
        x=partition.x,
        y=partition.y,
        windows_by_class=partition.windows_by_class,
        fingerprint=partition.noisy_fingerprint,
    )


def to_uda_unlabeled(partition: NoisyPartition) -> uda_core.UnlabeledPartition:
    if partition.role != "target_train" or partition.y is not None:
        raise ValueError("Target-training partition must be signal-only")
    return uda_core.UnlabeledPartition(
        rpm=partition.rpm,
        role=partition.role,
        x=partition.x,
        windows_by_class=partition.windows_by_class,
        fingerprint=partition.noisy_fingerprint,
    )


def method_definition(spec: MethodSpec) -> Mapping[str, object]:
    if spec.family == "source_only":
        definition = dict(source_core.method_definition(source_spec()))
    else:
        definition = dict(uda_core.method_definition(uda_spec(spec)))
    definition.update(
        {
            "noise_protocol": NOISE_PROTOCOL_VERSION,
            "noise_levels_db": list(DEFAULT_SNRS),
            "target_test_policy": "one final evaluation after fixed epoch 100",
        }
    )
    return definition


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_csv(
    path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_checkpoint(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def write_confusion_matrix(path: Path, matrix: Sequence[Sequence[int]]) -> None:
    fields = ("true_class",) + tuple(f"pred_{label}" for label in base.LABELS)
    rows = []
    for class_index, values in enumerate(matrix):
        rows.append(
            {
                "true_class": base.LABELS[class_index],
                **{f"pred_{label}": int(values[i]) for i, label in enumerate(base.LABELS)},
            }
        )
    _atomic_csv(path, rows, fields)


def snr_slug(snr_db: float) -> str:
    sign = "m" if snr_db < 0 else "p"
    magnitude = f"{abs(float(snr_db)):g}".replace(".", "p")
    return f"snr_{sign}{magnitude}db"


def partition_metadata(partition: Optional[NoisyPartition]) -> Optional[Mapping[str, object]]:
    if partition is None:
        return None
    return {
        "rpm": partition.rpm,
        "role": partition.role,
        "windows": len(partition.x),
        "windows_by_class": list(partition.windows_by_class),
        "noise_seeds_by_class": list(partition.noise_seeds_by_class),
        "realized_snr_mean_db": partition.realized_snr_mean_db,
        "realized_snr_min_db": partition.realized_snr_min_db,
        "realized_snr_max_db": partition.realized_snr_max_db,
        "clean_fingerprint": partition.clean_fingerprint,
        "noisy_fingerprint": partition.noisy_fingerprint,
        "labels_present": partition.y is not None,
    }


def optimization_signature(
    config: NoiseBaselineConfig,
    spec: MethodSpec,
    snr_db: float,
    source_rpm: str,
    target_rpm: str,
    seed: int,
    source_train: NoisyPartition,
    target_train: Optional[NoisyPartition],
) -> str:
    payload: Dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "method": asdict(spec),
        "snr_db": float(snr_db),
        "source_rpm": source_rpm,
        "experiment_seed": int(seed),
        "source_train_noisy_fingerprint": source_train.noisy_fingerprint,
        "config": asdict(config),
    }
    if target_train is not None:
        payload.update(
            {
                "target_rpm": target_rpm,
                "target_train_noisy_fingerprint": target_train.noisy_fingerprint,
            }
        )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def run_one(
    config: NoiseBaselineConfig,
    spec: MethodSpec,
    metadata_map: Mapping[Tuple[str, str, str], str],
    snr_db: float,
    source_rpm: str,
    target_rpm: str,
    seed: int,
) -> Mapping[str, object]:
    """Train one fit, then build and evaluate target-test exactly once."""

    source_train = build_noisy_partition(
        config, metadata_map, source_rpm, "source_train", seed, snr_db
    )
    target_train: Optional[NoisyPartition] = None
    if spec.family == "source_only":
        outcome = source_core.train_source_only(
            source_config(config),
            source_spec(),
            source_rpm,
            seed,
            to_source_labeled(source_train),
        )
    else:
        target_train = build_noisy_partition(
            config, metadata_map, target_rpm, "target_train", seed, snr_db
        )
        if target_train.y is not None:
            raise RuntimeError("Target-training labels must not be constructed")
        outcome = uda_core.train_uda(
            uda_config(config),
            uda_spec(spec),
            source_rpm,
            target_rpm,
            seed,
            to_uda_labeled(source_train),
            to_uda_unlabeled(target_train),
        )

    target_build_started_ns = time.perf_counter_ns()
    if target_build_started_ns < outcome.training_finished_ns:
        raise RuntimeError("Target-test construction started before training finished")
    target_test = build_noisy_partition(
        config, metadata_map, target_rpm, "target_test", seed, snr_db
    )
    if spec.family == "source_only":
        test_loader = source_core.make_target_test_loader(
            to_source_labeled(target_test), config.batch_size
        )
        metrics, prediction_rows = source_core.evaluate_once_with_predictions(
            outcome.model,
            test_loader,
            torch.device(config.device),
            source_spec(),
            source_rpm,
            target_rpm,
            seed,
        )
        target_batches = 0
        target_signals = 0
    else:
        test_loader = uda_core.make_target_test_loader(
            to_uda_labeled(target_test), config.batch_size
        )
        metrics, prediction_rows = uda_core.evaluate_once_with_predictions(
            outcome.model,
            test_loader,
            torch.device(config.device),
            uda_spec(spec),
            source_rpm,
            target_rpm,
            seed,
        )
        target_batches = outcome.target_batches_consumed
        target_signals = outcome.target_signals_consumed
    post_evaluation_hash = model_state_sha256(outcome.model)
    if post_evaluation_hash != outcome.post_train_state_sha256:
        raise RuntimeError("Final target evaluation mutated model state")

    output_dir = Path(config.output_dir)
    artifact_dir = (
        output_dir
        / "artifacts"
        / base.safe_slug(spec.name)
        / snr_slug(snr_db)
        / f"{base.LOAD_MAP[source_rpm]}_to_{base.LOAD_MAP[target_rpm]}"
        / f"seed_{seed}"
    )
    history_path = artifact_dir / "epoch_history.csv"
    prediction_path = artifact_dir / "target_predictions.csv"
    confusion_path = artifact_dir / "confusion_matrix.csv"
    checkpoint_path = artifact_dir / "final_checkpoint.pt"

    history_rows = [
        {
            "protocol_version": PROTOCOL_VERSION,
            "method": spec.name,
            "snr_db": float(snr_db),
            "source_rpm": source_rpm,
            "target_rpm": target_rpm,
            "seed": seed,
            **row,
        }
        for row in outcome.history
    ]
    for row in prediction_rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["snr_db"] = float(snr_db)
    _atomic_csv(history_path, history_rows, tuple(history_rows[0]))
    prediction_fields = (
        tuple(prediction_rows[0])
        if prediction_rows
        else ("protocol_version", "method", "snr_db", "sample_index")
    )
    _atomic_csv(prediction_path, prediction_rows, prediction_fields)
    write_confusion_matrix(confusion_path, metrics["confusion_matrix"])

    optimization_hash = optimization_signature(
        config,
        spec,
        snr_db,
        source_rpm,
        target_rpm,
        seed,
        source_train,
        target_train,
    )
    checkpoint = {
        "protocol_version": PROTOCOL_VERSION,
        "noise_protocol_version": NOISE_PROTOCOL_VERSION,
        "method": spec.name,
        "method_spec": asdict(spec),
        "method_definition": method_definition(spec),
        "config": asdict(config),
        "snr_db": float(snr_db),
        "source_rpm": source_rpm,
        "target_rpm": target_rpm,
        "seed": int(seed),
        "epochs_run": config.epochs,
        "selected_epoch": config.epochs,
        "selection_rule": "fixed_final_epoch",
        "optimization_signature": optimization_hash,
        "model_state_dict": {
            key: value.detach().cpu() for key, value in outcome.model.state_dict().items()
        },
        "partitions": {
            "source_train": partition_metadata(source_train),
            "target_train": partition_metadata(target_train),
            "target_test": partition_metadata(target_test),
        },
        "target_training_partition_built": target_train is not None,
        "target_training_loader_built": target_train is not None,
        "target_training_loader_tensor_count": 1 if target_train is not None else 0,
        "target_training_labels_consumed": False,
        "target_test_partition_built_after_training": True,
        "target_test_partition_builds": 1,
        "target_test_loader_builds": 1,
        "target_test_evaluations": 1,
        "target_test_labels_consumed_only_in_final_evaluation": True,
        "post_train_state_sha256": outcome.post_train_state_sha256,
        "post_evaluation_state_sha256": post_evaluation_hash,
        "target_metrics": metrics,
    }
    _atomic_checkpoint(checkpoint_path, checkpoint)

    device = torch.device(config.device)
    zero = torch.zeros((1, 1, config.window_size), dtype=torch.float32, device=device)
    classification_flops = base.estimate_classification_flops(outcome.model, zero)
    model_params = sum(
        parameter.numel()
        for parameter in outcome.model.parameters()
        if parameter.requires_grad
    )
    inference_params = pu_core.inference_parameter_count(outcome.model)
    final_history = outcome.history[-1]
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "noise_protocol_version": NOISE_PROTOCOL_VERSION,
        "method": spec.name,
        "method_token": spec.token,
        "family": spec.family,
        "domain_mode": spec.domain_mode,
        "data_protocol": config.data_protocol,
        "checkpoint_policy": config.checkpoint_policy,
        "source_rpm": source_rpm,
        "target_rpm": target_rpm,
        "source_hp": base.LOAD_MAP[source_rpm],
        "target_hp": base.LOAD_MAP[target_rpm],
        "seed": int(seed),
        "snr_db": float(snr_db),
        "accuracy": 100.0 * float(metrics["accuracy"]),
        "macro_f1": 100.0 * float(metrics["macro_f1"]),
        "class_accuracy_json": json.dumps(metrics["class_accuracy"]),
        "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
        "epochs_run": config.epochs,
        "selected_epoch": config.epochs,
        "selection_rule": "fixed_final_epoch",
        "model_params_m": model_params / 1e6,
        "inference_params_m": inference_params / 1e6,
        "gflops": classification_flops / 1e9,
        "training_seconds": outcome.training_seconds,
        "final_train_loss": final_history["train_loss"],
        "final_classification_loss": final_history["classification_loss"],
        "final_adaptation_loss": final_history.get("adaptation_loss", 0.0),
        "final_alpha": final_history.get("alpha_final_step", 0.0),
        "source_training_loader_built": True,
        "source_training_batches_consumed": outcome.source_batches_consumed,
        "target_training_partition_built": target_train is not None,
        "target_training_loader_built": target_train is not None,
        "target_training_loader_tensor_count": 1 if target_train is not None else 0,
        "target_training_labels_available_to_loader": False,
        "target_training_labels_consumed": False,
        "target_training_batches_consumed": target_batches,
        "target_training_samples_consumed": target_signals,
        "target_test_partition_built_after_training": True,
        "target_test_partition_builds": 1,
        "target_test_loader_builds": 1,
        "target_test_evaluations": 1,
        "target_test_labels_consumed_only_in_final_evaluation": True,
        "source_train_windows": len(source_train.x),
        "target_train_windows": len(target_train.x) if target_train is not None else 0,
        "target_test_windows": len(target_test.x),
        "source_train_snr_mean_db": source_train.realized_snr_mean_db,
        "source_train_snr_min_db": source_train.realized_snr_min_db,
        "source_train_snr_max_db": source_train.realized_snr_max_db,
        "target_train_snr_mean_db": (
            target_train.realized_snr_mean_db if target_train is not None else ""
        ),
        "target_train_snr_min_db": (
            target_train.realized_snr_min_db if target_train is not None else ""
        ),
        "target_train_snr_max_db": (
            target_train.realized_snr_max_db if target_train is not None else ""
        ),
        "target_test_snr_mean_db": target_test.realized_snr_mean_db,
        "target_test_snr_min_db": target_test.realized_snr_min_db,
        "target_test_snr_max_db": target_test.realized_snr_max_db,
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
        "optimization_signature": optimization_hash,
        "post_train_state_sha256": outcome.post_train_state_sha256,
        "post_evaluation_state_sha256": post_evaluation_hash,
        "history_path": str(history_path.relative_to(output_dir)),
        "prediction_path": str(prediction_path.relative_to(output_dir)),
        "confusion_matrix_path": str(confusion_path.relative_to(output_dir)),
        "checkpoint_path": str(checkpoint_path.relative_to(output_dir)),
    }
    for field in ("history", "prediction", "confusion_matrix", "checkpoint"):
        result[f"{field}_sha256"] = sha256_file(output_dir / result[f"{field}_path"])
    return result


RAW_FIELDS = (
    "protocol_version",
    "noise_protocol_version",
    "method",
    "method_token",
    "family",
    "domain_mode",
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
    "class_accuracy_json",
    "confusion_matrix_json",
    "epochs_run",
    "selected_epoch",
    "selection_rule",
    "model_params_m",
    "inference_params_m",
    "gflops",
    "training_seconds",
    "final_train_loss",
    "final_classification_loss",
    "final_adaptation_loss",
    "final_alpha",
    "source_training_loader_built",
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
    "source_train_snr_mean_db",
    "source_train_snr_min_db",
    "source_train_snr_max_db",
    "target_train_snr_mean_db",
    "target_train_snr_min_db",
    "target_train_snr_max_db",
    "target_test_snr_mean_db",
    "target_test_snr_min_db",
    "target_test_snr_max_db",
    "source_train_clean_fingerprint",
    "source_train_noisy_fingerprint",
    "target_train_clean_fingerprint",
    "target_train_noisy_fingerprint",
    "target_test_clean_fingerprint",
    "target_test_noisy_fingerprint",
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


def sample_std(values: Sequence[float]) -> float:
    return stdev(values) if len(values) > 1 else 0.0


def summarize_rows(
    rows: Sequence[Mapping[str, object]],
    expected_tasks: Sequence[Tuple[str, str]] = DEFAULT_TASKS,
    methods: Sequence[MethodSpec] = METHODS,
    snrs: Sequence[float] = DEFAULT_SNRS,
):
    """Summarize with task means, then seed means, then sample SD over seeds."""

    task_groups: MutableMapping[
        Tuple[str, float, str, str], List[Mapping[str, object]]
    ] = {}
    seed_groups: MutableMapping[
        Tuple[str, float, int], List[Mapping[str, object]]
    ] = {}
    for row in rows:
        method = str(row["method"])
        snr = float(row["snr_db"])
        source = str(row["source_rpm"])
        target = str(row["target_rpm"])
        seed = int(row["seed"])
        task_groups.setdefault((method, snr, source, target), []).append(row)
        seed_groups.setdefault((method, snr, seed), []).append(row)

    task_rows = []
    for (method, snr, source, target), group in sorted(task_groups.items()):
        accuracy = [float(row["accuracy"]) for row in group]
        macro_f1 = [float(row["macro_f1"]) for row in group]
        task_rows.append(
            {
                "method": method,
                "snr_db": snr,
                "source_rpm": source,
                "target_rpm": target,
                "source_hp": base.LOAD_MAP[source],
                "target_hp": base.LOAD_MAP[target],
                "accuracy_mean": mean(accuracy),
                "accuracy_std": sample_std(accuracy),
                "macro_f1_mean": mean(macro_f1),
                "macro_f1_std": sample_std(macro_f1),
                "seeds_completed": len({int(row["seed"]) for row in group}),
            }
        )

    expected_set = set(expected_tasks)
    seed_rows = []
    for (method, snr, seed), group in sorted(seed_groups.items()):
        observed = {
            (str(row["source_rpm"]), str(row["target_rpm"])) for row in group
        }
        seed_rows.append(
            {
                "method": method,
                "snr_db": snr,
                "seed": seed,
                "cross_accuracy_mean": mean(float(row["accuracy"]) for row in group),
                "cross_macro_f1_mean": mean(float(row["macro_f1"]) for row in group),
                "tasks_completed": len(observed),
                "complete_seed_block": observed == expected_set,
            }
        )

    method_rows = []
    for spec in methods:
        for snr in snrs:
            complete = [
                row
                for row in seed_rows
                if row["method"] == spec.name
                and np.isclose(float(row["snr_db"]), float(snr))
                and row["complete_seed_block"]
            ]
            accuracy = [float(row["cross_accuracy_mean"]) for row in complete]
            macro_f1 = [float(row["cross_macro_f1_mean"]) for row in complete]
            matching = [
                row
                for row in rows
                if str(row["method"]) == spec.name
                and np.isclose(float(row["snr_db"]), float(snr))
            ]
            method_rows.append(
                {
                    "method": spec.name,
                    "snr_db": float(snr),
                    "cross_accuracy_mean_across_seeds": mean(accuracy) if accuracy else "",
                    "cross_accuracy_std_across_seeds": sample_std(accuracy) if accuracy else "",
                    "cross_macro_f1_mean_across_seeds": mean(macro_f1) if macro_f1 else "",
                    "cross_macro_f1_std_across_seeds": sample_std(macro_f1) if macro_f1 else "",
                    "complete_seed_blocks": len(complete),
                    "runs_completed": len(matching),
                    "model_params_m": matching[0]["model_params_m"] if matching else "",
                    "inference_params_m": matching[0]["inference_params_m"] if matching else "",
                    "gflops": matching[0]["gflops"] if matching else "",
                }
            )
    return task_rows, seed_rows, method_rows


def write_outputs(
    rows: Sequence[Mapping[str, object]],
    output_dir: Path,
    tasks: Sequence[Tuple[str, str]],
    methods: Sequence[MethodSpec],
    snrs: Sequence[float],
) -> None:
    _atomic_csv(output_dir / "raw_results.csv", rows, RAW_FIELDS)
    task_rows, seed_rows, method_rows = summarize_rows(rows, tasks, methods, snrs)
    _atomic_csv(
        output_dir / "task_summary.csv",
        task_rows,
        tuple(task_rows[0]) if task_rows else ("method", "snr_db"),
    )
    _atomic_csv(
        output_dir / "seed_level_summary.csv",
        seed_rows,
        tuple(seed_rows[0]) if seed_rows else ("method", "snr_db", "seed"),
    )
    _atomic_csv(
        output_dir / "method_summary.csv",
        method_rows,
        tuple(method_rows[0]) if method_rows else ("method", "snr_db"),
    )
    _atomic_json(
        output_dir / "method_summary.json",
        {"methods": method_rows, "seeds": seed_rows, "tasks": task_rows},
    )


def build_dataset_inventory(
    config: NoiseBaselineConfig,
    metadata_map: Mapping[Tuple[str, str, str], str],
) -> List[Mapping[str, object]]:
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


def parameter_inventory(methods: Sequence[MethodSpec] = METHODS):
    inventory = {}
    for spec in methods:
        base.set_seed(resolve_seed(None))
        model = build_model(spec, 0)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inference = pu_core.inference_parameter_count(model)
        inventory[spec.name] = {
            "trainable": trainable,
            "trainable_m": trainable / 1e6,
            "inference": inference,
            "inference_m": inference / 1e6,
            "clean_runner_model_state_sha256": model_state_sha256(model),
        }
    return inventory


def build_manifest(
    config: NoiseBaselineConfig,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    snrs: Sequence[float],
    methods: Sequence[MethodSpec],
    dataset_inventory: Sequence[Mapping[str, object]],
    status: str,
    completed_keys: Sequence[Tuple[str, float, str, str, int]],
):
    script_path = Path(__file__).resolve()
    dependencies = [
        Path(base.__file__).resolve(),
        Path(source_core.__file__).resolve(),
        Path(uda_core.__file__).resolve(),
        Path(pu_core.__file__).resolve(),
        Path(pu_core.STRICT_DATA_SCRIPT).resolve(),
    ]
    frozen_signature = {
        "protocol_version": PROTOCOL_VERSION,
        "script": {"path": str(script_path), "sha256": sha256_file(script_path)},
        "dependencies": [
            {"path": str(path), "sha256": sha256_file(path)} for path in dependencies
        ],
        "metadata": {
            "path": str(Path(config.metadata_path).resolve()),
            "sha256": sha256_file(Path(config.metadata_path).resolve()),
        },
        "dataset_inventory": list(dataset_inventory),
        "config": asdict(config),
        "methods": [asdict(spec) for spec in methods],
        "method_definitions": {
            spec.name: method_definition(spec) for spec in methods
        },
        "noise_policy": dict(NOISE_POLICY),
        "snrs_db": [float(value) for value in snrs],
        "tasks": [list(task) for task in tasks],
        "seeds": [int(seed) for seed in seeds],
        "expected_runs": len(methods) * len(snrs) * len(tasks) * len(seeds),
        "parameter_inventory": parameter_inventory(methods),
        "selection_rule": "fixed_final_epoch",
        "statistics_policy": (
            "mean over 12 directed tasks within each seed, then mean and sample "
            "standard deviation over the five seed-level means"
        ),
        "target_label_policy": (
            "source-only never constructs target-train; DANN/CDAN target loader has "
            "one signal tensor; target-test is built after training and evaluated once"
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


def manifest_with_progress(
    frozen_manifest: Mapping[str, object],
    status: str,
    completed_keys: Sequence[Tuple[str, float, str, str, int]],
):
    """Update mutable progress without recomputing model/data/code inventories."""

    return {
        **frozen_manifest,
        "status": status,
        "completed_keys": [list(key) for key in completed_keys],
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def validate_resume_manifest(path: Path, expected: Mapping[str, object]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume without manifest: {path}")
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("frozen_signature") != expected.get("frozen_signature"):
        raise RuntimeError(
            "Refusing resume: code, data, noise, methods, SNRs, tasks, seeds, or config differ"
        )


def row_key(row: Mapping[str, object]) -> Tuple[str, float, str, str, int]:
    return (
        str(row["method"]),
        float(row["snr_db"]),
        str(row["source_rpm"]),
        str(row["target_rpm"]),
        int(row["seed"]),
    )


def read_rows(path: Path) -> List[Mapping[str, object]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: object) -> bool:
    return str(value).strip().lower() == "true"


def validate_existing_rows(
    rows: Sequence[Mapping[str, object]],
    config: NoiseBaselineConfig,
    output_dir: Path,
    tasks: Sequence[Tuple[str, str]],
    seeds: Sequence[int],
    snrs: Sequence[float],
    methods: Sequence[MethodSpec],
):
    allowed = {
        (spec.name, float(snr), source, target, int(seed))
        for spec in methods
        for snr in snrs
        for source, target in tasks
        for seed in seeds
    }
    seen = set()
    artifact_fields = {
        "history_path": "history_sha256",
        "prediction_path": "prediction_sha256",
        "confusion_matrix_path": "confusion_matrix_sha256",
        "checkpoint_path": "checkpoint_sha256",
    }
    for row in rows:
        key = row_key(row)
        if key not in allowed:
            raise RuntimeError(f"Unexpected result row during resume: {key}")
        if key in seen:
            raise RuntimeError(f"Duplicate result row during resume: {key}")
        seen.add(key)
        spec = METHOD_BY_NAME[key[0]]
        if str(row.get("protocol_version")) != PROTOCOL_VERSION:
            raise RuntimeError(f"Protocol mismatch in row: {key}")
        if str(row.get("noise_protocol_version")) != NOISE_PROTOCOL_VERSION:
            raise RuntimeError(f"Noise protocol mismatch in row: {key}")
        if int(row.get("epochs_run", 0)) != config.epochs:
            raise RuntimeError(f"Epoch-count mismatch in row: {key}")
        if int(row.get("selected_epoch", 0)) != config.epochs:
            raise RuntimeError(f"Selected-epoch mismatch in row: {key}")
        if str(row.get("selection_rule")) != "fixed_final_epoch":
            raise RuntimeError(f"Selection-rule mismatch in row: {key}")
        if int(row.get("target_test_evaluations", 0)) != 1:
            raise RuntimeError(f"Target-test evaluation mismatch in row: {key}")
        if not _as_bool(row.get("target_test_partition_built_after_training")):
            raise RuntimeError(f"Target-test timing mismatch in row: {key}")
        if str(row.get("post_train_state_sha256")) != str(
            row.get("post_evaluation_state_sha256")
        ):
            raise RuntimeError(f"Evaluation changed model state: {key}")
        built_target = _as_bool(row.get("target_training_partition_built"))
        if built_target != spec.uses_target_training:
            raise RuntimeError(f"Target-training policy mismatch in row: {key}")
        if _as_bool(row.get("target_training_labels_available_to_loader")) or _as_bool(
            row.get("target_training_labels_consumed")
        ):
            raise RuntimeError(f"Target-training labels leaked in row: {key}")
        for field in ("source_train_snr_mean_db", "target_test_snr_mean_db"):
            if not np.isclose(float(row[field]), key[1], atol=1e-4):
                raise RuntimeError(f"Realized-SNR mismatch for {field}: {key}")
        if spec.uses_target_training and not np.isclose(
            float(row["target_train_snr_mean_db"]), key[1], atol=1e-4
        ):
            raise RuntimeError(f"Target-train realized-SNR mismatch: {key}")
        for path_field, hash_field in artifact_fields.items():
            path = output_dir / str(row.get(path_field, ""))
            if not path.is_file():
                raise RuntimeError(f"Missing resume artifact {path_field}: {path}")
            if sha256_file(path) != str(row.get(hash_field, "")):
                raise RuntimeError(f"Artifact hash mismatch for {path_field}: {path}")
    return seen


def protect_output_directory(output_dir: Path, resume: bool) -> None:
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; use --resume or a new path: {output_dir}"
        )


def validate_reportable_config(
    config: NoiseBaselineConfig,
    methods: Sequence[MethodSpec],
    snrs: Sequence[float],
    seeds: Sequence[int],
    tasks: Sequence[Tuple[str, str]],
) -> None:
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
        f"{name}={actual!r} (required {wanted!r})"
        for name, (actual, wanted) in requirements.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError("Reportable protocol is frozen: " + ", ".join(mismatches))
    if tuple(spec.token for spec in methods) != tuple(spec.token for spec in METHODS):
        raise ValueError("Reportable run requires methods bkan,dann,cdan in that order")
    if tuple(float(value) for value in snrs) != DEFAULT_SNRS:
        raise ValueError("Reportable run requires SNRs -1,-3,-5 in that order")
    if tuple(int(seed) for seed in seeds) != DEFAULT_SEEDS:
        raise ValueError("Use the initialization settings selected for this runtime")
    if tuple(tasks) != DEFAULT_TASKS:
        raise ValueError("Reportable run requires all 12 directed cross-load tasks")


def parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> List[MethodSpec]:
    tokens = parse_csv(value.lower())
    unknown = [token for token in tokens if token not in METHOD_BY_TOKEN]
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; choose bkan,dann,cdan")
    if len(tokens) != len(set(tokens)):
        raise ValueError("Duplicate method tokens are not allowed")
    return [METHOD_BY_TOKEN[token] for token in tokens]


def parse_snrs(value: str) -> List[float]:
    snrs = [float(item) for item in parse_csv(value)]
    if len(snrs) != len(set(snrs)):
        raise ValueError("Duplicate SNR levels are not allowed")
    allowed = set(DEFAULT_SNRS)
    if any(snr not in allowed for snr in snrs):
        raise ValueError("Only -1, -3, and -5 dB are permitted")
    return snrs


def parse_seeds(value: str) -> List[int]:
    seeds = [int(item) for item in parse_csv(value)]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Duplicate seeds are not allowed")
    if any(seed not in DEFAULT_SEEDS for seed in seeds):
        raise ValueError("Only frozen five runtime-selected initializations are permitted")
    return seeds


def parse_tasks(value: Optional[str]) -> List[Tuple[str, str]]:
    if not value:
        return list(DEFAULT_TASKS)
    tasks = []
    for item in parse_csv(value):
        separator = "->" if "->" in item else ":" if ":" in item else None
        if separator is None:
            raise ValueError(f"Invalid task {item!r}; use SOURCE->TARGET")
        source, target = (part.strip() for part in item.split(separator, 1))
        if source not in base.ALL_RPMS or target not in base.ALL_RPMS or source == target:
            raise ValueError(f"Invalid directed cross-load task: {item}")
        tasks.append((source, target))
    if len(tasks) != len(set(tasks)):
        raise ValueError("Duplicate tasks are not allowed")
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run strict CWRU noisy B-KAN/DANN/CDAN Table-4 baselines."
    )
    parser.add_argument("--data-root", default=base.CFG.data_root)
    parser.add_argument("--metadata-path", default=base.CFG.metadata_path)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--methods", default="bkan,dann,cdan")
    parser.add_argument("--snrs", default="-1,-3,-5")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--tasks", default=None)
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-windows-per-class", type=int, default=2)
    parser.add_argument("--list-parameters", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    snrs = parse_snrs(args.snrs)
    seeds = parse_seeds(args.seeds)
    tasks = parse_tasks(args.tasks)
    if not methods or not snrs or not seeds or not tasks:
        raise ValueError("Methods, SNRs, seeds, and tasks must all be non-empty")
    if args.list_parameters:
        print(json.dumps(parameter_inventory(methods), ensure_ascii=False, indent=2))
        return 0
    if args.epochs <= 0 or args.batch_size <= 0 or args.log_every <= 0:
        raise ValueError("epochs, batch size, and log_every must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.adaptation_weight <= 0:
        raise ValueError("lr/adaptation weight must be positive; weight decay non-negative")
    if not args.smoke and (
        args.warmup_epochs < 0 or args.warmup_epochs >= args.epochs
    ):
        raise ValueError("warmup epochs must be non-negative and lower than epochs")

    output_dir = Path(
        os.path.abspath(
            args.output_dir
            if args.output_dir is not None
            else str(SMOKE_OUTPUT_DIR if args.smoke else DEFAULT_OUTPUT_DIR)
        )
    )
    config = NoiseBaselineConfig(
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
    if args.smoke:
        tasks = tasks[:1]
        seeds = seeds[:1]
        snrs = snrs[:1]
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
    validate_reportable_config(config, methods, snrs, seeds, tasks)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    if not Path(config.data_root).is_dir():
        raise FileNotFoundError(f"CWRU data root does not exist: {config.data_root}")
    if not Path(config.metadata_path).is_file():
        raise FileNotFoundError(f"CWRU metadata file does not exist: {config.metadata_path}")
    protect_output_directory(output_dir, args.resume)

    metadata_map = base.load_metadata_map(config.metadata_path)
    print("Hashing the 40 raw CWRU records for the frozen manifest...", flush=True)
    dataset_inventory = build_dataset_inventory(config, metadata_map)
    manifest_path = output_dir / "manifest.json"
    initial_manifest = build_manifest(
        config,
        tasks,
        seeds,
        snrs,
        methods,
        dataset_inventory,
        "running",
        [],
    )
    if args.resume:
        validate_resume_manifest(manifest_path, initial_manifest)
    raw_path = output_dir / "raw_results.csv"
    rows = read_rows(raw_path) if args.resume else []
    completed = validate_existing_rows(
        rows, config, output_dir, tasks, seeds, snrs, methods
    )
    _atomic_json(
        manifest_path,
        manifest_with_progress(initial_manifest, "running", sorted(completed)),
    )

    pending = [
        (spec, snr, source, target, seed)
        for snr in snrs
        for spec in methods
        for seed in seeds
        for source, target in tasks
        if (spec.name, float(snr), source, target, int(seed)) not in completed
    ]
    total = len(methods) * len(snrs) * len(tasks) * len(seeds)
    print(f"Pending fits: {len(pending)} / {total}", flush=True)
    for index, (spec, snr, source, target, seed) in enumerate(pending, 1):
        print(
            f"[{index}/{len(pending)}] {spec.token} SNR={snr:g} dB "
            f"{base.LOAD_MAP[source]}->{base.LOAD_MAP[target]} seed={seed}",
            flush=True,
        )
        row = run_one(config, spec, metadata_map, snr, source, target, seed)
        rows.append(row)
        completed.add(row_key(row))
        write_outputs(rows, output_dir, tasks, methods, snrs)
        _atomic_json(
            manifest_path,
            manifest_with_progress(initial_manifest, "running", sorted(completed)),
        )

    expected = {
        (spec.name, float(snr), source, target, int(seed))
        for spec in methods
        for snr in snrs
        for source, target in tasks
        for seed in seeds
    }
    if completed != expected:
        missing = sorted(expected - completed)
        raise RuntimeError(f"Run ended with missing result keys: {missing[:5]}")
    validate_existing_rows(rows, config, output_dir, tasks, seeds, snrs, methods)
    _atomic_json(
        manifest_path,
        manifest_with_progress(initial_manifest, "complete", sorted(completed)),
    )
    print(f"Completed {len(completed)}/{total} strict noisy fits.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
