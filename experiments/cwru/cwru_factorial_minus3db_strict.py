"""CWRU front-end and classifier factorial experiments at minus three dB."""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import numpy as np
import torch

from runtime import DEFAULT_RUN_SEEDS, resolve_seed

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROTOCOL_VERSION = "cwru-factorial-minus3db-strict80-final100-v1"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "cwru_factorial_minus3db_strict_100epoch_20261010"
ENGINE_PATH = Path(__file__).with_name("cwru_minus5db_stress_strict.py")


def load_engine():
    name = "_cwru_factorial_minus3db_engine"
    spec = importlib.util.spec_from_file_location(name, ENGINE_PATH)
    engine = importlib.util.module_from_spec(spec)
    sys.modules[name] = engine
    spec.loader.exec_module(engine)
    engine.PROTOCOL_VERSION = PROTOCOL_VERSION
    engine.SNR_DB = -3.0
    engine.METHODS = tuple(
        engine.MethodSpec(
            name=cfg.name, backbone=cfg.backbone, classifier=cfg.classifier,
            domain_mode=cfg.domain_mode, use_center=cfg.use_center,
            use_entropy=cfg.use_entropy, use_se=cfg.use_se, wavelet=cfg.wavelet,
            use_wavelet_translation=cfg.use_wavelet_translation,
            normalize_wavelet=cfg.normalize_wavelet,
        )
        for cfg in engine.base.FACTORIAL_ABLATIONS
    )
    engine.METHOD_BY_NAME = {method.name: method for method in engine.METHODS}
    engine.NOISE_POLICY = {
        **engine.NOISE_POLICY, "snr_db": -3.0,
        "target_train": "first 80% of target record; unlabeled loader for all four models",
        "post_noise_normalization": "independent per-window z-score after AWGN; SNR audited at injection before this normalization",
    }

    def definition(method):
        return {
            "noise_condition": "-3 dB exact-power AWGN followed by per-window z-score",
            "feature_extractor": method.backbone,
            "classifier": method.classifier,
            "wavelet": method.wavelet,
            "SE": method.use_se,
            "objective": "source CE + CDAN + 0.05 center + 0.05*alpha target entropy",
            "target_training_loader": "signals only",
            "target_test_policy": "one evaluation after fixed final epoch",
        }

    engine.method_definition = definition
    return engine


engine = load_engine()
TASKS = tuple(engine.DEFAULT_TASKS)
SEEDS = tuple(engine.DEFAULT_SEEDS)


def build_partition(config, metadata, rpm, role, seed):
    partition = engine.build_noisy_partition(config, metadata, rpm, role, seed)
    noisy = partition.x
    normalized = ((noisy - noisy.mean(axis=1, keepdims=True)) /
                  (noisy.std(axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    return replace(partition, x=normalized,
                   noisy_fingerprint=engine.base.fingerprint_arrays(normalized, partition.y))


def make_manifest(config, tasks, seeds, inventory):
    sources = (Path(__file__), ENGINE_PATH, Path(engine.base.__file__),
               Path(engine.pu_core.__file__), Path(engine.pu_core.STRICT_DATA_SCRIPT))
    return {
        "frozen_signature": {
            "protocol_version": PROTOCOL_VERSION,
            "sources": {str(p.resolve()): engine.sha256_file(p) for p in sources},
            "metadata_sha256": engine.sha256_file(Path(config.metadata_path)),
            "config": asdict(config),
            "method_specs": [asdict(method) for method in engine.METHODS],
            "method_definitions": {m.name: engine.method_definition(m) for m in engine.METHODS},
            "noise_policy": engine.NOISE_POLICY,
            "tasks": [list(t) for t in tasks], "seeds": list(seeds),
            "expected_runs": len(tasks) * len(seeds) * len(engine.METHODS),
            "dataset_inventory": inventory,
            "parameter_inventory": engine.parameter_inventory(),
            "split_policy": "chronological 80/20 split before windowing; 1024 samples, 50% overlap",
            "selection_rule": "fixed_final_epoch; no early stopping or target-test selection",
            "target_test_policy": "partition constructed after training and evaluated once",
            "statistics_policy": "mean of all 12 tasks within each seed, then mean and sample SD over five seeds",
            "reportable": config.reportable,
        },
        "environment": engine.base.runtime_environment(),
        "command": [sys.executable, *sys.argv],
    }


def write_progress(output_dir, manifest, rows, status, active=None):
    value = {**manifest, "status": status,
             "completed_runs": len(rows),
             "completed_keys": [list(engine.row_key(row)) for row in rows],
             "active_run": active, "updated_unix": time.time()}
    engine.write_manifest(output_dir / "manifest.json", value)


def write_factorial_summary(output_dir, rows, tasks, seeds, config):
    _, seed_rows, method_rows = engine.write_outputs(rows, output_dir, tasks)
    indexed = {(r["method"], int(r["seed"])): float(r["cross_accuracy_mean"])
               for r in seed_rows if r["complete_seed_block"]}
    names = [m.name for m in engine.METHODS]
    paired_seeds = [s for s in seeds if all((name, s) in indexed for name in names)]
    contrasts = {
        "WKAN minus Conv with MLP (C-A)": (-1, 0, 1, 0),
        "WKAN minus Conv with Wavelet KAN (D-B)": (0, -1, 0, 1),
        "Wavelet KAN minus MLP with Conv (B-A)": (-1, 1, 0, 0),
        "Wavelet KAN minus MLP with WKAN (D-C)": (0, 0, -1, 1),
        "Interaction (D-C-B+A)": (1, -1, -1, 1),
    }
    effects = []
    for label, coefficients in contrasts.items():
        deltas = [sum(c * indexed[(name, s)] for c, name in zip(coefficients, names))
                  for s in paired_seeds]
        effects.append({"contrast": label, "paired_seeds": len(paired_seeds),
                        "mean_delta_pp": mean(deltas) if deltas else "",
                        "sample_sd_delta_pp": stdev(deltas) if len(deltas) > 1 else ""})
    engine._write_csv(output_dir / "factorial_effects.csv", effects, tuple(effects[0]))
    engine._write_csv(output_dir / "factorial_summary.csv", method_rows, tuple(method_rows[0]))
    lines = ["# CWRU -3 dB structural ablation", "", "## Material Passport", "",
             "- Origin Skill: academic-research-suite / experiment-agent", "- Origin Mode: run",
             "- Origin Date: 2026-10-10 (Asia/Hong_Kong)",
             "- Verification Status: SMOKE ONLY; not reportable" if not config.reportable
             else "- Verification Status: RUNNING" if len(rows) < len(tasks)*len(seeds)*4
             else "- Verification Status: COMPLETED; artifact integrity checked",
             f"- Version Label: {PROTOCOL_VERSION}",
             "- Repro Lock: manifest.json contains source and dataset SHA-256 hashes.", "",
             f"Completed fits: {len(rows)}/{len(tasks)*len(seeds)*4}.",
             "All four models share CDAN, SE, center and entropy losses.",
             f"{config.epochs} fixed epochs, batch {config.batch_size}, Adam lr {config.lr}, weight decay {config.weight_decay}.",
             "Only complete seed blocks enter the main summary; SD is across seed means.", "",
             "| Configuration | Mean accuracy (%) | Sample SD | Complete seeds |", "|---|---:|---:|---:|"]
    for r in method_rows:
        acc, sd = r["cross_accuracy_mean_across_seeds"], r["cross_accuracy_std_across_seeds"]
        lines.append(f"| {r['method']} | {acc if acc == '' else f'{acc:.4f}'} | {sd if sd == '' else f'{sd:.4f}'} | {r['complete_seed_blocks']} |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.log_every <= 0:
        raise ValueError("log-every must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    output_dir = args.output_dir.resolve()
    if args.smoke and output_dir == DEFAULT_OUTPUT_DIR.resolve():
        output_dir = output_dir.with_name(output_dir.name + "_smoke")
    engine.protect_output_directory(output_dir, args.resume)
    config = engine.StressConfig(output_dir=str(output_dir), snr_db=-3.0,
                                 device=args.device, log_every=args.log_every)
    tasks, seeds = TASKS, SEEDS
    if args.smoke:
        tasks, seeds = TASKS[:1], SEEDS[:1]
        config = replace(config, epochs=1, batch_size=10, max_windows_per_class=2,
                         max_steps_per_epoch=1, reportable=False, log_every=1)
    metadata = engine.base.load_metadata_map(config.metadata_path)
    inventory = engine.build_dataset_inventory(config, metadata)
    manifest = make_manifest(config, tasks, seeds, inventory)
    if args.resume:
        engine.validate_resume_manifest(output_dir / "manifest.json", manifest)
    rows = engine.read_rows(output_dir / "raw_results.csv") if args.resume else []
    completed = engine.validate_existing_rows(rows, config, output_dir, tasks, seeds) if rows else set()
    pending = [(m, s, src, tgt) for s in seeds for src, tgt in tasks for m in engine.METHODS
               if (m.name, s, src, tgt) not in completed]
    write_progress(output_dir, manifest, rows, "running")
    print(f"Protocol: {PROTOCOL_VERSION}\nDevice: {args.device}\nSNR: -3 dB\nPending fits: {len(pending)}", flush=True)
    try:
        for index, (method, seed, source, target) in enumerate(pending, 1):
            active = {"method": method.name, "seed": seed, "source_rpm": source, "target_rpm": target}
            write_progress(output_dir, manifest, rows, "running", active)
            print(f"[{index}/{len(pending)}] {method.name} | seed={seed} | {engine.base.LOAD_MAP[source]}->{engine.base.LOAD_MAP[target]}", flush=True)
            source_train = build_partition(config, metadata, source, "source_train", seed)
            target_train = build_partition(config, metadata, target, "target_train", seed)
            result = engine.train_one(config, method, source, target, seed, source_train,
                        lambda: build_partition(config, metadata, target, "target_test", seed),
                        target_train=target_train)
            rows.append(result)
            engine.validate_existing_rows([result], config, output_dir, tasks, seeds)
            write_factorial_summary(output_dir, rows, tasks, seeds, config)
            write_progress(output_dir, manifest, rows, "running")
            print(f"  accuracy={result['accuracy']:.2f}% macro-F1={result['macro_f1']:.2f}%", flush=True)
        engine.validate_existing_rows(rows, config, output_dir, tasks, seeds)
        write_progress(output_dir, manifest, rows, "complete")
    except BaseException:
        write_progress(output_dir, manifest, rows, "interrupted" if isinstance(sys.exc_info()[1], KeyboardInterrupt) else "failed", active)
        raise
    print(f"Outputs: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
