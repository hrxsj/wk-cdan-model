"""Dataset preparation, WK-CDAN training, baseline studies, and evaluation."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def data_root(args):
    env = "CWRU_DATA_ROOT" if args.dataset == "cwru" else "PU_DATA_ROOT"
    return Path(args.data_root or os.environ.get(env, ROOT / "Data" / args.dataset.upper())).resolve()


def code_fingerprints():
    files = [*ROOT.glob("*.py"), *(ROOT / "experiments").rglob("*.py")]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def configure_session(args, destination):
    from runtime import DEFAULT_RUN_SEEDS, parse_seeds
    session_path = destination / "session.json"
    options = {key: value for key, value in vars(args).items() if key != "resume"}
    signature = {"options": options, "code_sha256": code_fingerprints()}
    supplied = args.seeds or os.environ.get("WKCDAN_RUN_SEEDS", "")
    if args.resume:
        if not session_path.is_file():
            raise FileNotFoundError(f"Resume requires {session_path}")
        previous = json.loads(session_path.read_text(encoding="utf-8"))
        # The saved local values are reused automatically on resume.
        old_options = dict(previous["signature"]["options"])
        old_options["seeds"] = options["seeds"]
        if old_options != options or previous["signature"]["code_sha256"] != signature["code_sha256"]:
            raise ValueError("Resume requires the same experiment options and source files")
        values = tuple(previous["runtime_initializations"])
        if supplied and parse_seeds(supplied) != values:
            raise ValueError("Runtime initialization settings differ from the saved session")
    else:
        if session_path.exists() or any(destination.rglob("raw_results.csv")):
            raise FileExistsError("Use --resume or choose a new output directory")
        values = parse_seeds(supplied) if supplied else DEFAULT_RUN_SEEDS
        write_json(session_path, {"signature": signature, "runtime_initializations": values})
    os.environ["WKCDAN_RUN_SEEDS"] = ",".join(map(str, values))
    import runtime
    runtime.DEFAULT_RUN_SEEDS = tuple(values)
    return values


def prepare(args):
    destination = Path(args.output_dir or ROOT / "runs" / f"{args.dataset}_data").resolve()
    folder = data_root(args)
    inventory = []
    if args.dataset == "cwru":
        from experiments import cwru_ieee_revision as cwru
        metadata = cwru.load_metadata_map(str(ROOT / "cwru/metadata.txt"))
        for rpm in cwru.ALL_RPMS:
            for label, (group, filename) in cwru.CLASS_FILES.items():
                path = Path(cwru.ensure_mat_file(str(folder), metadata, group, rpm, label, filename))
                signal = cwru.load_1d_signal(str(path))
                if signal.size < 1024:
                    raise ValueError(f"CWRU record is too short: {path}")
                inventory.append({"rpm": rpm, "class": label, "path": str(path),
                                  "samples": int(signal.size), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    else:
        from experiments.pu import pu_frontend_ablation as pu
        for condition in pu.CONDITIONS:
            split = pu.build_domain_split(str(folder), condition, 2048, 0.5, 0.8)
            inventory.append({"condition": condition, "records": split.record_count,
                              "train_windows": len(split.train_y), "test_windows": len(split.test_y),
                              "fingerprint": split.fingerprint})
    write_json(destination / "dataset_inventory.json", inventory)
    print(f"Dataset prepared: {folder}; inventory: {destination / 'dataset_inventory.json'}")


def run_module(module, arguments):
    subprocess.run([sys.executable, "-m", module, *map(str, arguments)], cwd=ROOT, check=True)


def train_cwru(args, destination, values):
    from dataclasses import asdict
    from experiments import cwru_ieee_revision as cwru
    suites = ("wkcdan", "baselines") if args.suite == "all" else (args.suite,)
    tasks = cwru.parse_task_pairs(args.tasks) if args.tasks else [
        (source, target) for source in cwru.ALL_RPMS for target in cwru.ALL_RPMS if source != target]
    if not tasks or any(source == target for source, target in tasks):
        raise ValueError("Choose at least one cross-load task")
    if args.smoke:
        tasks, values = tasks[:1], values[:1]
    for suite in suites:
        if suite == "baselines":
            if args.protocol != "strict_holdout":
                raise ValueError("The core baseline benchmark uses strict_holdout")
            for name in ("source_only", "uda"):
                if args.batch_size is not None or args.lr is not None:
                    raise ValueError("Core baselines use their fixed batch size and learning rate")
                arguments = ["--data-root", data_root(args), "--metadata-path", ROOT / "cwru/metadata.txt",
                             "--output-dir", destination / name, "--device", args.device,
                             "--tasks", ",".join(f"{s}:{t}" for s, t in tasks)]
                if args.epochs is not None and not args.smoke:
                    arguments += ["--epochs", args.epochs]
                if args.resume:
                    arguments += ["--resume"]
                if args.smoke:
                    arguments += ["--smoke"]
                run_module(f"experiments.cwru.cwru_{name}_core_strict", arguments)
            continue
        if suite == "noise-baselines":
            arguments = ["--data-root", data_root(args), "--metadata-path", ROOT / "cwru/metadata.txt",
                         "--output-dir", destination / suite, "--device", args.device]
            if args.smoke:
                arguments += ["--smoke"]
            if args.resume:
                arguments += ["--resume"]
            run_module("experiments.cwru.cwru_noise_baselines_strict", arguments)
            continue
        experiments = {
            "wkcdan": [item for item in cwru.EXPERIMENTS if item.name == "WK-CDAN Full"],
            "frontend": [item for item in cwru.BASIS_ABLATIONS
                         if item.name in {"Conv-ReLU Front-end", "Mexican Hat KANConv"}],
            "basis": cwru.BASIS_ABLATIONS,
            "structure": cwru.STRUCTURE_ABLATIONS,
            "factorial": cwru.FACTORIAL_ABLATIONS,
            "noise": cwru.SNR_COMPARISON,
            "wavelet": cwru.WAVELET_ABLATIONS,
            "loss": cwru.NOISE_ABLATIONS,
        }.get(suite)
        if not experiments:
            raise ValueError(f"Unsupported CWRU suite: {suite}")
        policy = args.checkpoint_policy or ("final_epoch" if args.protocol == "strict_holdout" else "source_val_best")
        if args.protocol == "strict_holdout" and policy != "final_epoch":
            raise ValueError("A source validation split is required for early stopping")
        config = cwru.Config(data_root=str(data_root(args)), metadata_path=str(ROOT / "cwru/metadata.txt"),
                             output_dir=str(destination / suite), data_protocol=args.protocol,
                             checkpoint_policy=policy, device=args.device,
                             epochs=1 if args.smoke else (args.epochs or 100),
                             batch_size=8 if args.smoke else (args.batch_size or 32),
                             lr=args.lr or 5e-4, max_windows_per_class=2 if args.smoke else None)
        signature = {"config": asdict(config), "experiments": [asdict(item) for item in experiments],
                     "tasks": tasks, "initializations": values, "code_sha256": code_fingerprints()}
        manifest_path = Path(config.output_dir) / "manifest.json"
        if args.resume and manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous["signature"] != json.loads(json.dumps(signature)):
                raise ValueError("Saved CWRU configuration differs from this run")
        write_json(manifest_path, {"signature": signature, "status": "running"})
        snr = args.snr_db if suite in {"basis", "frontend", "structure", "factorial", "loss"} else None
        levels = cwru.parse_snr_levels(args.snr_levels)
        if args.smoke:
            levels = levels[:1]
        rows = cwru.run_experiments(config, experiments, cwru.ALL_RPMS, cwru.ALL_RPMS, values, snr,
                                    existing_rows=cwru.load_existing_rows(config.output_dir) if args.resume else [],
                                    snr_levels=levels, wavelet_snr_levels=levels, task_pairs=tasks)
        cwru.write_outputs(rows, config.output_dir,
                           expected_factorial_methods=[item.name for item in experiments if item.suite == "factorial_ablation"])
        write_json(manifest_path, {"signature": signature, "status": "complete", "completed_runs": len(rows)})


def train_pu(args, destination):
    suites = ("frontend", "baselines") if args.suite == "all" else (args.suite,)
    for suite in suites:
        module = {"wkcdan": "pu_frontend_ablation", "frontend": "pu_frontend_ablation",
                  "baselines": "pu_core_baselines_strict", "basis": "pu_basis_ablation"}.get(suite)
        if module is None:
            raise ValueError(f"Unsupported PU suite: {suite}")
        arguments = ["--data-root", data_root(args), "--output-dir", destination / suite, "--device", args.device]
        if suite == "wkcdan":
            arguments += ["--methods", "Mexican Hat WKAN-Conv1D"]
        if args.tasks:
            if suite == "basis":
                parts = args.tasks.replace("->", ":").split(":")
                if len(parts) != 2 or "," in args.tasks:
                    raise ValueError("PU basis study accepts one task")
                arguments += ["--source", parts[0], "--target", parts[1]]
            else:
                arguments += ["--tasks", args.tasks]
        for flag, value in (("--epochs", args.epochs), ("--batch-size", args.batch_size), ("--lr", args.lr)):
            if value is not None:
                arguments += [flag, value]
        if args.smoke:
            arguments += ["--smoke"]
        if args.resume:
            arguments += ["--resume"]
        run_module("experiments.pu." + module, arguments)


def train(args):
    destination = Path(args.output_dir or ROOT / "runs" / f"{args.dataset}_{args.suite}").resolve()
    values = configure_session(args, destination)
    if args.dataset == "cwru":
        train_cwru(args, destination, values)
    else:
        train_pu(args, destination)
    from reporting import summarize
    summarize(destination, destination / "report")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Download/check dataset records and write an inventory")
    fitting = commands.add_parser("train", help="Train and evaluate experiments, then export summaries")
    for item in (prep, fitting):
        item.add_argument("--dataset", required=True, choices=("cwru", "pu"))
        item.add_argument("--data-root")
        item.add_argument("--output-dir")
    fitting.add_argument("--suite", default="wkcdan", choices=("wkcdan", "baselines", "all", "frontend", "basis", "structure", "factorial", "noise", "noise-baselines", "wavelet", "loss"))
    fitting.add_argument("--device", default="cuda" if _cuda_available() else "cpu")
    fitting.add_argument("--epochs", type=int)
    fitting.add_argument("--batch-size", type=int)
    fitting.add_argument("--lr", type=float)
    fitting.add_argument("--tasks", default="")
    fitting.add_argument("--protocol", choices=("strict_holdout", "strict_721"), default="strict_holdout")
    fitting.add_argument("--checkpoint-policy", choices=("final_epoch", "source_val_best", "early_stop_last"))
    fitting.add_argument("--snr-db", type=float, default=None, help="AWGN SNR for ablations; default is clean")
    fitting.add_argument("--snr-levels", default="clean,-1,-3,-5")
    fitting.add_argument("--seeds", default="", help="Optional user-selected initialization values")
    fitting.add_argument("--smoke", action="store_true", help="Run one task and initialization for a short pipeline check")
    fitting.add_argument("--resume", action="store_true")
    evaluation = commands.add_parser("evaluate", help="Evaluate a saved checkpoint on held-out target records")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data-root", required=True)
    evaluation.add_argument("--output-dir", required=True)
    evaluation.add_argument("--device", default="cpu")
    summary = commands.add_parser("summarize", help="Rebuild tables and figures from completed result CSVs")
    summary.add_argument("--input-dir", required=True)
    summary.add_argument("--output-dir", required=True)
    return parser


def _cuda_available():
    import torch
    return torch.cuda.is_available()


def main():
    args = build_parser().parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "train":
        train(args)
    elif args.command == "summarize":
        from reporting import summarize
        summarize(Path(args.input_dir).resolve(), Path(args.output_dir).resolve())
    else:
        from evaluation import evaluate_checkpoint
        evaluate_checkpoint(args)


if __name__ == "__main__":
    main()
