"""Evaluate saved experiment checkpoints against their target-test partition."""
import csv
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent


def cwru_checkpoint(checkpoint, folder, device):
    from experiments import cwru_ieee_revision as base
    metadata = base.load_metadata_map(str(ROOT / "cwru/metadata.txt"))
    target = checkpoint["target_rpm"]
    if "experiment_config" in checkpoint:
        experiment = base.ExperimentConfig(**checkpoint["experiment_config"])
        config = replace(base.Config(**checkpoint["training_config"]), data_root=folder, device=device)
        model = base.CDANModel(experiment.backbone, num_classes=base.NUM_CLASSES,
                              use_cdan=experiment.use_cdan, wavelet=experiment.wavelet, use_se=experiment.use_se,
                              use_wavelet_translation=experiment.use_wavelet_translation,
                              normalize_wavelet=experiment.normalize_wavelet, domain_mode=experiment.domain_mode,
                              classifier=experiment.classifier, classifier_wavelet=experiment.classifier_wavelet)
        seed = checkpoint["seed"] + 1000
        kwargs = {"seed": seed, "snr_db": checkpoint.get("snr_db")}
        if config.data_protocol == "strict_holdout":
            _, _, x, y = base.build_domain_dataset_train_test(config, metadata, target, **kwargs)
        elif config.data_protocol == "strict_721":
            _, _, _, _, x, y = base.build_domain_dataset_721(config, metadata, target, **kwargs)
        else:
            raise ValueError(f"Unsupported checkpoint data protocol: {config.data_protocol}")
    else:
        from experiments.pu import pu_core_baselines_strict as core
        spec = checkpoint["method_spec"]
        if "pu_method" in spec:
            model = core.CoreBaselineModel(spec["pu_method"], num_classes=base.NUM_CLASSES, init_seed=checkpoint["seed"])
            if checkpoint["protocol_version"].startswith("cwru-source-only"):
                from experiments.cwru import cwru_source_only_core_strict as runner
                config = replace(runner.SourceOnlyConfig(**checkpoint["config"]), data_root=folder, device=device)
                partition = runner.build_time_partition(config, metadata, target, "target_test")
            else:
                from experiments.cwru import cwru_uda_core_strict as runner
                config = replace(runner.UDAConfig(**checkpoint["config"]), data_root=folder, device=device)
                partition = runner.build_target_test_partition(config, metadata, target)
        else:
            if "family" in spec:
                from experiments.cwru import cwru_noise_baselines_strict as runner
                config = replace(runner.NoiseBaselineConfig(**checkpoint["config"]), data_root=folder, device=device)
                partition = runner.build_noisy_partition(config, metadata, target, "target_test",
                                                          checkpoint["seed"], checkpoint["snr_db"])
            else:
                from experiments.cwru import cwru_minus5db_stress_strict as runner
                config = replace(runner.StressConfig(**checkpoint["config"]), data_root=folder, device=device)
                partition = runner.build_noisy_partition(config, metadata, target, "target_test", checkpoint["seed"])
            model = runner.build_model(runner.MethodSpec(**spec), init_seed=checkpoint["seed"])
        x, y = partition.x, partition.y
    return model, x, y, config.batch_size, base.LABELS


def pu_checkpoint(checkpoint, folder, device):
    from experiments.pu import pu_frontend_ablation as pu
    from experiments.pu import pu_core_baselines_strict as core
    config = checkpoint["config"]
    split = pu.build_domain_split(folder, checkpoint["target_condition"], config["window_size"],
                                  config["overlap"], config["train_ratio"], config["max_records_per_bearing"])
    method = checkpoint["method"]
    if method in core.METHODS:
        model = core.CoreBaselineModel(method, init_seed=checkpoint["seed"])
        split = core.limit_domain_for_smoke(split, config.get("smoke_windows_per_class"))
    elif method in pu.METHODS:
        model = pu.PUFrontEndCDAN(method, init_seed=checkpoint["seed"])
    else:
        from experiments.pu import pu_basis_ablation as basis
        model = basis.PUBasisCDAN(method, init_seed=checkpoint["seed"])
    return model, split.test_x, split.test_y, config["batch_size"], pu.LABELS


def evaluate_checkpoint(args):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(args.device)
    folder = str(Path(args.data_root).resolve())
    loader_fn = cwru_checkpoint if "target_rpm" in checkpoint else pu_checkpoint
    model, x, y, batch_size, labels = loader_fn(checkpoint, folder, args.device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32).unsqueeze(1),
                                      torch.tensor(y, dtype=torch.long)), batch_size=batch_size, shuffle=False)
    targets, predictions = [], []
    with torch.no_grad():
        for samples, truth in loader:
            logits = model.classify(samples.to(device)) if hasattr(model, "classify") else model(samples.to(device))[0]
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(truth.tolist())
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for truth, prediction in zip(targets, predictions):
        matrix[truth, prediction] += 1
    tp = np.diag(matrix).astype(float)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    metrics = {"accuracy": float(tp.sum() / matrix.sum()), "macro_f1": float(f1.mean()),
               "samples": len(targets), "labels": list(labels), "confusion_matrix": matrix.tolist()}
    original = checkpoint.get("target_metrics", {})
    metrics["saved_accuracy"] = float(original["accuracy"]) if "accuracy" in original else None
    metrics["accuracy_matches_saved"] = bool(np.isclose(metrics["accuracy"], metrics["saved_accuracy"])) if metrics["saved_accuracy"] is not None else None
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (destination / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_index", "target", "prediction"))
        writer.writerows((index, truth, pred) for index, (truth, pred) in enumerate(zip(targets, predictions)))
    np.savetxt(destination / "confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    print(f"Target accuracy: {100 * metrics['accuracy']:.2f}%; macro-F1: {100 * metrics['macro_f1']:.2f}%")
    print(f"Metrics and predictions saved to {destination}")
    return metrics
