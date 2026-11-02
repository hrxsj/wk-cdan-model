"""Profile CWRU model parameters, computation, and inference latency."""
import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from runtime import DEFAULT_RUN_SEEDS, resolve_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.cwru_ieee_revision import (  # noqa: E402
    CDANModel,
    KANConv1d,
    KANConv1dBSpline,
    KANLinearBSpline,
    KANLinearWavelet,
    NUM_CLASSES,
    count_parameters_m,
)


PAPER_PARAMS_M = {
    "1D-CNN": 0.04,
    "WDCNN": 0.06,
    "ResNet-18": 1.00,
    "Transformer": 0.27,
    "Vanilla KAN": 1.02,
    "CNN-1D-KAN": 0.043,
    "KANConv-based": 0.056,
    "DANN + ResNet-18": 1.031,
    "CDAN + ResNet-18": 1.330,
    "CDANN-JMMD": 1.326,
    "GCDAN-fair": 1.326,
    "WK-CDAN": 0.37,
}


DEPLOYMENT_GROUPS = {
    "ResNet-18": "resnet18-classification-path",
    "DANN + ResNet-18": "resnet18-classification-path",
    "CDAN + ResNet-18": "resnet18-classification-path",
    "CDANN-JMMD": "resnet18-classification-path",
    "GCDAN-fair": "resnet18-classification-path",
}


class PureCNNModel(nn.Module):
    """Exact deployment graph from baseline-other/1D-CNN.py."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=31, stride=2, padding=15),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(torch.tanh(self.features(x)))


class WDCNNModel(nn.Module):
    """Exact deployment graph from baseline-other/WDCNN_Baseline.py."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 64, 16, 24),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, 3, 1, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(32, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, 1, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(64, 64, 3, 1, 0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 100), nn.ReLU(), nn.Linear(100, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(1), :])


class TransformerModel(nn.Module):
    """Exact deployment graph from baseline-other/transformer.py."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=1024, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        return self.classifier(x)


class VanillaKANModel(nn.Module):
    """Exact deployment graph from baseline-other/kan.py."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
            nn.Flatten(),
        )
        self.proj = nn.Linear(1024, 256)
        self.kan1 = KANLinearBSpline(256, 128, grid_size=5, spline_order=3)
        self.drop = nn.Dropout(0.5)
        self.classifier = KANLinearBSpline(128, num_classes, grid_size=5, spline_order=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.proj(self.stem(x)))
        return self.classifier(self.drop(self.kan1(x)))


def make_model_specs() -> List[Tuple[str, Callable[[], nn.Module], Callable[[], nn.Module]]]:
    """Return (name, params_model, inference_model) specs.

    Parameter counts follow the full trained model. For WK-CDAN this includes the
    training-time domain discriminator, while latency/GFLOPs use the deployment
    classification path only.
    """

    return [
        (
            "1D-CNN",
            PureCNNModel,
            PureCNNModel,
        ),
        (
            "WDCNN",
            WDCNNModel,
            WDCNNModel,
        ),
        (
            "ResNet-18",
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
        ),
        ("Transformer", TransformerModel, TransformerModel),
        (
            "Vanilla KAN",
            VanillaKANModel,
            VanillaKANModel,
        ),
        (
            "CNN-1D-KAN",
            lambda: CDANModel("cnn", use_cdan=False, domain_mode="none", classifier="b_spline_kan", use_se=False),
            lambda: CDANModel("cnn", use_cdan=False, domain_mode="none", classifier="b_spline_kan", use_se=False),
        ),
        (
            "KANConv-based",
            lambda: CDANModel("bkan", use_cdan=False, domain_mode="none", classifier="b_spline_kan", use_se=True),
            lambda: CDANModel("bkan", use_cdan=False, domain_mode="none", classifier="b_spline_kan", use_se=True),
        ),
        (
            "DANN + ResNet-18",
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="dann", classifier="linear"),
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
        ),
        (
            "CDAN + ResNet-18",
            lambda: CDANModel("resnet", use_cdan=True, domain_mode="cdan", classifier="linear"),
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
        ),
        (
            "CDANN-JMMD",
            lambda: CDANModel("resnet", use_cdan=True, domain_mode="cdan", classifier="linear"),
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
        ),
        (
            "GCDAN-fair",
            lambda: CDANModel("resnet", use_cdan=True, domain_mode="cdan", classifier="linear"),
            lambda: CDANModel("resnet", use_cdan=False, domain_mode="none", classifier="linear"),
        ),
        (
            "WK-CDAN",
            lambda: CDANModel("wkan", use_cdan=True, domain_mode="cdan", classifier="wavelet_kan", use_se=True),
            lambda: CDANModel("wkan", use_cdan=False, domain_mode="none", classifier="wavelet_kan", use_se=True),
        ),
    ]


def _tensor_size(tensor_or_tuple) -> int:
    if isinstance(tensor_or_tuple, torch.Tensor):
        return int(tensor_or_tuple.numel())
    if isinstance(tensor_or_tuple, (tuple, list)):
        return sum(_tensor_size(item) for item in tensor_or_tuple if isinstance(item, torch.Tensor))
    return 0


def estimate_flops(model: nn.Module, example: torch.Tensor) -> int:
    flops: Dict[str, int] = {"total": 0}
    handles = []

    def add(value: int) -> None:
        flops["total"] += int(value)

    def hook_conv1d(module: nn.Conv1d, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        out_len = output.shape[-1]
        kernel_ops = module.kernel_size[0] * module.in_channels // module.groups
        add(2 * batch * module.out_channels * out_len * kernel_ops)

    def hook_linear(module: nn.Linear, inputs, output):
        out_elems = _tensor_size(output)
        add(2 * out_elems * module.in_features)

    def hook_bn(module: nn.BatchNorm1d, inputs, output):
        add(2 * _tensor_size(output))

    def hook_mha(module: nn.MultiheadAttention, inputs, output):
        query = inputs[0]
        if not module.batch_first:
            seq_len, batch, embed_dim = query.shape
        else:
            batch, seq_len, embed_dim = query.shape
        heads = module.num_heads
        head_dim = embed_dim // heads
        qkv = 3 * 2 * batch * seq_len * embed_dim * embed_dim
        attn_scores = 2 * batch * heads * seq_len * seq_len * head_dim
        attn_values = 2 * batch * heads * seq_len * seq_len * head_dim
        out_proj = 2 * batch * seq_len * embed_dim * embed_dim
        add(qkv + attn_scores + attn_values + out_proj)

    def hook_kan_conv(module: KANConv1d, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        out_len = output.shape[-1]
        base = 2 * batch * out_len * module.out_channels * module.in_channels * module.kernel_size
        wavelet = base * module.grid_size
        basis = 6 * batch * out_len * module.in_channels * module.kernel_size * module.grid_size
        add(base + wavelet + basis)

    def hook_kan_linear(module: KANLinearWavelet, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        out_features, in_features, grid = module.wavelet_weight.shape
        base = 2 * batch * out_features * in_features
        wavelet = base * grid
        basis = 6 * batch * in_features * grid
        add(base + wavelet + basis)

    def hook_bkan_conv(module: KANConv1dBSpline, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        out_len = output.shape[-1]
        basis_count = module.grid_size + module.spline_order
        base = 2 * batch * out_len * module.out_channels * module.in_channels * module.kernel_size
        spline = base * basis_count
        basis = 8 * module.spline_order * batch * out_len * module.in_channels * module.kernel_size * basis_count
        add(base + spline + basis)

    def hook_bkan_linear(module: KANLinearBSpline, inputs, output):
        x = inputs[0]
        batch = x.shape[0]
        basis_count = module.grid_size + module.spline_order
        base = 2 * batch * module.out_features * module.in_features
        spline = base * basis_count
        basis = 8 * module.spline_order * batch * module.in_features * basis_count
        add(base + spline + basis)

    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            handles.append(module.register_forward_hook(hook_mha))
        elif isinstance(module, nn.Linear):
            if name.endswith("self_attn.out_proj"):
                continue
            handles.append(module.register_forward_hook(hook_linear))
        elif isinstance(module, nn.Conv1d):
            handles.append(module.register_forward_hook(hook_conv1d))
        elif isinstance(module, nn.BatchNorm1d):
            handles.append(module.register_forward_hook(hook_bn))
        elif isinstance(module, KANConv1d):
            handles.append(module.register_forward_hook(hook_kan_conv))
        elif isinstance(module, KANLinearWavelet):
            handles.append(module.register_forward_hook(hook_kan_linear))
        elif isinstance(module, KANConv1dBSpline):
            handles.append(module.register_forward_hook(hook_bkan_conv))
        elif isinstance(module, KANLinearBSpline):
            handles.append(module.register_forward_hook(hook_bkan_linear))

    model.eval()
    with torch.no_grad():
        _ = model(example)

    for handle in handles:
        handle.remove()
    return flops["total"]


def run_forward(model: nn.Module, x: torch.Tensor):
    output = model(x)
    if isinstance(output, tuple):
        return output[0]
    return output


def deployment_signature(model: nn.Module) -> Tuple:
    """Describe the executable module graph used to validate shared timings."""

    return tuple(
        (
            name,
            module.__class__.__qualname__,
            tuple(tuple(parameter.shape) for parameter in module.parameters(recurse=False)),
        )
        for name, module in model.named_modules()
    )


def measure_latency_ms(
    model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    repeat: int,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = run_forward(model, x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        timings = []
        with torch.no_grad():
            for _ in range(repeat):
                starter.record()
                _ = run_forward(model, x)
                ender.record()
                torch.cuda.synchronize(device)
                timings.append(starter.elapsed_time(ender))
        return float(statistics.fmean(timings)), float(statistics.pstdev(timings))

    timings = []
    with torch.no_grad():
        for _ in range(repeat):
            start = time.perf_counter()
            _ = run_forward(model, x)
            timings.append((time.perf_counter() - start) * 1000.0)
    return float(statistics.fmean(timings)), float(statistics.pstdev(timings))


def format_latex_rows(rows: Iterable[Dict[str, float]]) -> List[str]:
    output = []
    for row in rows:
        output.append(
            f"{row['model']} & {row['params_m']:.3f} & {row['gflops']:.3f} & {row['inference_ms']:.3f} \\\\"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile CWRU model complexity for the paper.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "complexity_profile"))
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    example = torch.randn(1, 1, 1024, device=device)
    rows: List[Dict[str, float]] = []
    deployment_cache = {}

    for name, make_params_model, make_infer_model in make_model_specs():
        params_model = make_params_model().to(device)
        infer_model = make_infer_model().to(device)
        counted_params_m = count_parameters_m(params_model)
        deployment_params_m = count_parameters_m(infer_model)
        params_m = PAPER_PARAMS_M.get(name, counted_params_m)
        deployment_group = DEPLOYMENT_GROUPS.get(name, name)
        signature = deployment_signature(infer_model)
        if deployment_group in deployment_cache:
            (
                cached_signature,
                gflops,
                latency_ms,
                latency_std_ms,
                latency_source_model,
            ) = deployment_cache[deployment_group]
            if signature != cached_signature:
                raise RuntimeError(f"Deployment group {deployment_group!r} contains different inference graphs.")
        else:
            gflops = estimate_flops(infer_model, example) / 1e9
            latency_ms, latency_std_ms = measure_latency_ms(
                infer_model,
                example,
                args.warmup,
                args.repeat,
                device,
            )
            latency_source_model = name
            deployment_cache[deployment_group] = (
                signature,
                gflops,
                latency_ms,
                latency_std_ms,
                latency_source_model,
            )
        gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        rows.append(
            {
                "model": name,
                "params_m": round(params_m, 6),
                "counted_params_m": round(counted_params_m, 6),
                "deployment_params_m": round(deployment_params_m, 6),
                "gflops": round(gflops, 6),
                "inference_ms": round(latency_ms, 6),
                "inference_std_ms": round(latency_std_ms, 6),
                "device": str(device),
                "gpu_name": gpu_name,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "",
                "input_shape": "1x1x1024",
                "warmup": args.warmup,
                "repeat": args.repeat,
                "deployment_group": deployment_group,
                "latency_source_model": latency_source_model,
                "notes": "params_m is the paper-reported full training model value; counted_params_m and deployment_params_m are measured from the instantiated training and deployment models. GFLOPs and latency count the deployment classification path. Identical deployment graphs share one benchmark result and latency_source_model identifies the measured representative.",
            }
        )
        print(
            f"{name}: params={params_m:.3f}M, counted={counted_params_m:.3f}M, "
            f"deployment={deployment_params_m:.3f}M, gflops={gflops:.3f}, "
            f"inference={latency_ms:.3f}+/-{latency_std_ms:.3f} ms "
            f"(source={latency_source_model})"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cwru_complexity.csv"
    json_path = output_dir / "cwru_complexity.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {json_path}")
    print("\nLaTeX rows:")
    for latex_row in format_latex_rows(rows):
        print(latex_row)


if __name__ == "__main__":
    main()
