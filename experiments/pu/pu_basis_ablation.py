"""PU Wavelet-KAN basis comparisons with chronological per-record partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
import torch
import torch.nn as nn
import torch.nn.functional as F


from runtime import DEFAULT_RUN_SEEDS, resolve_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments import cwru_ieee_revision as cwru  # noqa: E402
from experiments.pu import pu_frontend_ablation as base  # noqa: E402


PROTOCOL_VERSION = "pu-basis-hard-task-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "pu_basis_oc2_to_oc1_25"
DEFAULT_SOURCE = "N15_M07_F04"  # OC2: low radial force
DEFAULT_TARGET = "N15_M01_F10"  # OC1: low load torque
METHOD_GAUSSIAN = "Gaussian KANConv"
METHOD_MORLET = "Morlet KANConv"
METHOD_BSPLINE = "B-spline KANConv"
METHOD_MEXICAN = "Mexican Hat KANConv"
METHOD_SHANNON = "Shannon KANConv"
BASIS_METHODS = (
    METHOD_GAUSSIAN,
    METHOD_MORLET,
    METHOD_BSPLINE,
    METHOD_MEXICAN,
    METHOD_SHANNON,
)
METHOD_TO_BASIS = {
    METHOD_GAUSSIAN: "gaussian",
    METHOD_MORLET: "morlet_fair",
    METHOD_BSPLINE: "b_spline",
    METHOD_MEXICAN: "mexhat",
    METHOD_SHANNON: "shannon_real_1_1",
}


def build_frontend(method: str):
    basis = METHOD_TO_BASIS[method]
    if basis == "b_spline":
        return cwru.KANConv1dBSpline(
            1,
            32,
            kernel_size=31,
            stride=2,
            padding=15,
            grid_size=5,
            spline_order=3,
        )
    return cwru.KANConv1d(
        1,
        32,
        kernel_size=31,
        stride=2,
        padding=15,
        grid_size=3,
        wavelet=basis,
    )


class PUBasisCDAN(nn.Module):
    """The verified PU model with only its first function basis switched."""

    def __init__(self, method: str, num_classes: int = len(base.LABELS), init_seed: int = resolve_seed(None)):
        super().__init__()
        if method not in BASIS_METHODS:
            raise ValueError(f"Unknown PU basis method: {method}")
        self.method = method
        self.num_classes = num_classes
        self.frontend = base.initialized(lambda: build_frontend(method), init_seed + 1)
        self.front_activation = nn.Identity()
        self.bn1 = base.initialized(lambda: nn.BatchNorm1d(32), init_seed + 2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mid_cnn = base.initialized(
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
        self.se = base.initialized(lambda: base.SEBlock1D(128, reduction=8), init_seed + 4)
        self.tail = nn.Sequential(
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
        )
        self.classifier = base.initialized(
            lambda: base.WaveletKANClassifier(128, num_classes), init_seed + 5
        )
        self.domain_classifier = base.initialized(
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
        return self.classifier(self.extract_features(x))

    def forward(self, x: torch.Tensor, alpha: float = 0.0):
        features = self.extract_features(x)
        class_logits = self.classifier(features)
        probabilities = F.softmax(class_logits, dim=1)
        conditional = torch.bmm(features.unsqueeze(2), probabilities.unsqueeze(1))
        conditional = conditional.view(-1, 128 * self.num_classes)
        reversed_features = base.GradientReversal.apply(conditional, alpha)
        domain_logits = self.domain_classifier(reversed_features)
        return class_logits, domain_logits, features


def install_training_adapter():
    """Route the verified shared training loop through the five-basis model."""

    base.PROTOCOL_VERSION = PROTOCOL_VERSION
    base.METHODS = BASIS_METHODS
    base.PUFrontEndCDAN = PUBasisCDAN


def resolve_methods(value: str):
    aliases = {
        "gaussian": METHOD_GAUSSIAN,
        "morlet": METHOD_MORLET,
        "b-spline": METHOD_BSPLINE,
        "bspline": METHOD_BSPLINE,
        "mexhat": METHOD_MEXICAN,
        "mexican": METHOD_MEXICAN,
        "shannon": METHOD_SHANNON,
    }
    if not value:
        return list(BASIS_METHODS)
    methods = []
    for token in value.split(","):
        stripped = token.strip()
        method = aliases.get(stripped.lower(), stripped)
        if method not in BASIS_METHODS:
            raise ValueError(f"Unknown basis '{stripped}'")
        if method not in methods:
            methods.append(method)
    return methods


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def common_seed_matrix(rows: Sequence[Mapping[str, object]], methods: Sequence[str]):
    by_method: MutableMapping[str, Dict[int, float]] = {method: {} for method in methods}
    for row in rows:
        method = str(row["method"])
        if method in by_method:
            by_method[method][int(row["seed"])] = float(row["accuracy"])
    common_seeds = sorted(set.intersection(*(set(values) for values in by_method.values())))
    if common_seeds:
        matrix = np.asarray(
            [[by_method[method][seed] for method in methods] for seed in common_seeds],
            dtype=float,
        )
    else:
        matrix = np.empty((0, len(methods)), dtype=float)
    return common_seeds, matrix


def average_ranks(rows: Sequence[Mapping[str, object]], methods: Sequence[str]):
    common_seeds, matrix = common_seed_matrix(rows, methods)
    if not len(common_seeds):
        return {method: 0.0 for method in methods}
    ranks = np.vstack([rankdata(-block, method="average") for block in matrix])
    return {method: float(ranks[:, index].mean()) for index, method in enumerate(methods)}


def summarize_methods(rows: Sequence[Mapping[str, object]], methods: Sequence[str]):
    ranks = average_ranks(rows, methods)
    output = []
    for method in methods:
        group = [row for row in rows if row["method"] == method]
        accuracy = [float(row["accuracy"]) for row in group]
        macro_f1 = [float(row["macro_f1"]) for row in group]
        normal = [float(row["normal_accuracy"]) for row in group]
        outer = [float(row["outerrace_accuracy"]) for row in group]
        inner = [float(row["innerrace_accuracy"]) for row in group]
        acc_stats = metric_stats(accuracy)
        f1_stats = metric_stats(macro_f1)
        output.append(
            {
                "method": method,
                "basis": METHOD_TO_BASIS[method],
                "accuracy_mean": acc_stats["mean"],
                "accuracy_std": acc_stats["std"],
                "accuracy_median": acc_stats["median"],
                "accuracy_min": acc_stats["min"],
                "accuracy_max": acc_stats["max"],
                "macro_f1_mean": f1_stats["mean"],
                "macro_f1_std": f1_stats["std"],
                "normal_accuracy_mean": mean(normal) if normal else 0.0,
                "outerrace_accuracy_mean": mean(outer) if outer else 0.0,
                "innerrace_accuracy_mean": mean(inner) if inner else 0.0,
                "average_rank": ranks[method],
                "params_m": mean([float(row["params_m"]) for row in group])
                if group
                else 0.0,
                "inference_params_m": mean(
                    [float(row["inference_params_m"]) for row in group]
                )
                if group
                else 0.0,
                "training_seconds_mean": mean(
                    [float(row["training_seconds"]) for row in group]
                )
                if group
                else 0.0,
                "runs": len(group),
            }
        )
    return output


def rank_biserial(differences: np.ndarray):
    nonzero = differences[np.abs(differences) > 1e-12]
    if not len(nonzero):
        return 0.0
    ranks = rankdata(np.abs(nonzero))
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: Sequence[float]):
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def statistical_analysis(rows: Sequence[Mapping[str, object]], methods: Sequence[str]):
    common_seeds, matrix = common_seed_matrix(rows, methods)
    friedman = {
        "blocks": len(common_seeds),
        "statistic": None,
        "p": None,
    }
    if len(common_seeds) >= 2:
        statistic, p_value = friedmanchisquare(
            *[matrix[:, index] for index in range(matrix.shape[1])]
        )
        friedman.update({"statistic": float(statistic), "p": float(p_value)})

    if METHOD_MEXICAN not in methods:
        return {"friedman": friedman, "mexican_pairwise": []}
    mexican_index = methods.index(METHOD_MEXICAN)
    comparisons = []
    raw_p = []
    for index, method in enumerate(methods):
        if method == METHOD_MEXICAN:
            continue
        differences = matrix[:, mexican_index] - matrix[:, index]
        nonzero = differences[np.abs(differences) > 1e-12]
        if len(nonzero):
            statistic, p_value = wilcoxon(
                nonzero, alternative="two-sided", method="auto"
            )
        else:
            statistic, p_value = 0.0, 1.0
        raw_p.append(float(p_value))
        comparisons.append(
            {
                "contrast": f"{METHOD_MEXICAN} minus {method}",
                "blocks": len(differences),
                "mean_difference_pp": float(differences.mean())
                if len(differences)
                else 0.0,
                "median_difference_pp": float(np.median(differences))
                if len(differences)
                else 0.0,
                "wins": int(np.sum(differences > 1e-12)),
                "ties": int(np.sum(np.abs(differences) <= 1e-12)),
                "losses": int(np.sum(differences < -1e-12)),
                "wilcoxon_statistic": float(statistic),
                "wilcoxon_p": float(p_value),
                "rank_biserial": float(rank_biserial(differences)),
            }
        )
    adjusted = holm_adjust(raw_p) if raw_p else []
    for row, adjusted_p in zip(comparisons, adjusted):
        row["holm_p"] = adjusted_p
    return {"friedman": friedman, "mexican_pairwise": comparisons}


def write_report(
    path: Path,
    summaries: Sequence[Mapping[str, object]],
    statistics: Mapping[str, object],
):
    lines = [
        "## Material Passport",
        "",
        f"- Experiment: `{PROTOCOL_VERSION}`",
        "- Task: PU OC2->OC1",
        "- Type: paired five-basis ablation",
        "- Verification status: ANALYZED",
        "- Target-test policy: one final evaluation after fixed epoch 100",
        "",
        "## Results",
        "",
        "| Method | Accuracy (%) | Macro-F1 (%) | IR accuracy (%) | Avg. rank | Params (M) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['method']} | {row['accuracy_mean']:.2f} +/- "
            f"{row['accuracy_std']:.2f} | {row['macro_f1_mean']:.2f} +/- "
            f"{row['macro_f1_std']:.2f} | {row['innerrace_accuracy_mean']:.2f} | "
            f"{row['average_rank']:.2f} | {row['params_m']:.3f} |"
        )
    friedman = statistics["friedman"]
    lines.extend(
        [
            "",
            "## Statistics",
            "",
            f"- Friedman blocks: {friedman['blocks']}",
            f"- Friedman statistic: {friedman['statistic']}",
            f"- Friedman p: {friedman['p']}",
            "",
            "| Contrast | Mean difference (pp) | W/T/L | Raw p | Holm p | Rank-biserial |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in statistics["mexican_pairwise"]:
        lines.append(
            f"| {row['contrast']} | {row['mean_difference_pp']:+.3f} | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['wilcoxon_p']:.6g} | {row['holm_p']:.6g} | "
            f"{row['rank_biserial']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Scope Guardrail",
            "",
            "These functions are amplitude-axis activation bases inside WKAN-Conv1D, "
            "not standalone temporal CWT filters. A single pre-specified difficult task "
            "can diagnose basis sensitivity but cannot establish universal superiority.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_summaries(
    rows: Sequence[Mapping[str, object]], output_dir: Path, methods: Sequence[str]
):
    summaries = summarize_methods(rows, methods)
    statistics = statistical_analysis(rows, methods)
    base.write_csv(output_dir / "summary.csv", summaries, list(summaries[0]))
    pairwise = statistics["mexican_pairwise"]
    pair_fields = list(pairwise[0]) if pairwise else ["contrast"]
    base.write_csv(output_dir / "mexican_pairwise.csv", pairwise, pair_fields)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {"methods": summaries, "statistics": statistics},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    write_report(output_dir / "validation_report.md", summaries, statistics)
    return summaries, statistics


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


def manifest_payload(
    config: base.ExperimentConfig,
    methods: Sequence[str],
    seeds: Sequence[int],
    source: str,
    target: str,
    domains: Mapping[str, base.DomainSplit],
    status: str,
    completed: int,
):
    script = Path(__file__).resolve()
    dependencies = [
        Path(base.__file__).resolve(),
        Path(cwru.__file__).resolve(),
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "status": status,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "command": [sys.executable, *sys.argv],
        "script": str(script),
        "script_sha256": sha256_file(script),
        "dependencies": [
            {"path": str(path), "sha256": sha256_file(path)} for path in dependencies
        ],
        "config": asdict(config),
        "methods": list(methods),
        "basis_definitions": {
            METHOD_GAUSSIAN: "exp(-x^2/2)",
            METHOD_MORLET: "(cos(2x)-exp(-2))*exp(-x^2/2)",
            METHOD_BSPLINE: "cubic B-spline, grid_size=5, grid_range=[-3,3]",
            METHOD_MEXICAN: "(1-x^2)*exp(-x^2/2)",
            METHOD_SHANNON: "sinc(x)*cos(2*pi*x), real shan1-1",
        },
        "classifier_basis": "Mexican Hat for every method",
        "seeds": list(seeds),
        "task": {
            "source_condition": source,
            "target_condition": target,
            "source_name": base.CONDITIONS[source],
            "target_name": base.CONDITIONS[target],
            "selection_reason": (
                "Pre-specified from the complete six-direction PU front-end study as "
                "the least saturated and most variable transfer direction."
            ),
        },
        "expected_runs": len(methods) * len(seeds),
        "completed_runs": completed,
        "selection_rule": "fixed_final_epoch",
        "target_test_policy": "evaluated once after training; never used for selection",
        "domain_data": {
            condition: {
                "name": base.CONDITIONS[condition],
                "record_count": split.record_count,
                "train_windows": len(split.train_y),
                "test_windows": len(split.test_y),
                "fingerprint": split.fingerprint,
            }
            for condition, split in domains.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "git": git_metadata(),
    }


def write_manifest(path: Path, payload: Mapping[str, object]):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_resume(path: Path, expected: Mapping[str, object]):
    if not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = (
        "protocol_version",
        "script_sha256",
        "dependencies",
        "config",
        "methods",
        "seeds",
        "task",
        "domain_data",
    )
    mismatches = [key for key in keys if existing.get(key) != expected.get(key)]
    if mismatches:
        raise RuntimeError(
            "Refusing to resume a different PU basis experiment: "
            + ", ".join(mismatches)
        )


def row_key(row: Mapping[str, object]):
    return str(row["method"]), int(row["seed"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run five WKAN-Conv1D bases on the PU OC2->OC1 hard task."
    )
    parser.add_argument("--data-root", default=os.environ.get("PU_DATA_ROOT", str(REPO_ROOT / "Data" / "PU")))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--methods", default=",".join(BASIS_METHODS))
    parser.add_argument("--seeds", default=",".join(map(str, base.DEFAULT_SEEDS)))
    parser.add_argument("--source", default="OC2")
    parser.add_argument("--target", default="OC1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-records-per-bearing", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    install_training_adapter()
    methods = resolve_methods(args.methods)
    seeds = base.resolve_seeds(args.seeds)
    source = base.resolve_condition(args.source)
    target = base.resolve_condition(args.target)
    if source == target:
        raise ValueError("The PU basis diagnostic requires a cross-condition task")
    config = base.ExperimentConfig(
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
        config = replace(config, epochs=1, warmup_epochs=0, max_records_per_bearing=1)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    print(f"Protocol: {PROTOCOL_VERSION}")
    print(f"Task: {base.CONDITIONS[source]}->{base.CONDITIONS[target]}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Device: {config.device}")
    domains = {
        condition: base.build_domain_split(
            config.data_root,
            condition,
            config.window_size,
            config.overlap,
            config.train_ratio,
            config.max_records_per_bearing,
        )
        for condition in (source, target)
    }
    for condition, split in domains.items():
        print(
            f"  {base.CONDITIONS[condition]} records={split.record_count} "
            f"train={len(split.train_y)} test={len(split.test_y)} "
            f"hash={split.fingerprint[:12]}",
            flush=True,
        )

    output_dir = Path(config.output_dir)
    raw_path = output_dir / "raw_results.csv"
    initial_manifest = manifest_payload(
        config, methods, seeds, source, target, domains, "running", 0
    )
    if args.resume:
        validate_resume(output_dir / "manifest.json", initial_manifest)
    rows = base.read_rows(raw_path) if args.resume else []
    completed = {row_key(row) for row in rows}
    pending = [
        (method, seed)
        for method in methods
        for seed in seeds
        if (method, seed) not in completed
    ]
    write_manifest(
        output_dir / "manifest.json",
        manifest_payload(
            config, methods, seeds, source, target, domains, "running", len(rows)
        ),
    )
    print(f"Pending runs: {len(pending)}", flush=True)

    for index, (method, seed) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {method} | seed={seed}", flush=True)
        result = base.train_one(
            config, method, seed, source, target, domains
        )
        rows.append(result)
        base.write_csv(raw_path, rows, base.RAW_FIELDS)
        write_summaries(rows, output_dir, methods)
        write_manifest(
            output_dir / "manifest.json",
            manifest_payload(
                config,
                methods,
                seeds,
                source,
                target,
                domains,
                "running",
                len(rows),
            ),
        )
        print(
            f"  accuracy={result['accuracy']:.2f}% "
            f"macro-F1={result['macro_f1']:.2f}% "
            f"time={result['training_seconds'] / 60.0:.1f} min",
            flush=True,
        )

    summaries, statistics = write_summaries(rows, output_dir, methods)
    expected = len(methods) * len(seeds)
    status = "complete" if len({row_key(row) for row in rows}) >= expected else "partial"
    write_manifest(
        output_dir / "manifest.json",
        manifest_payload(
            config, methods, seeds, source, target, domains, status, len(rows)
        ),
    )
    print("Summary:")
    for row in summaries:
        print(
            f"  {row['method']}: {row['accuracy_mean']:.2f} +/- "
            f"{row['accuracy_std']:.2f}% rank={row['average_rank']:.2f}"
        )
    print(f"Statistics: {json.dumps(statistics, ensure_ascii=False)}")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
