"""WK-CDAN for cross-domain bearing fault diagnosis.

Input: per-window normalized vibration signals of shape (batch, 1, length).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

NUM_CLASSES = 10


def wavelet_basis(x, wavelet="mexhat"):
    if wavelet.lower() not in {"mexhat", "mexican", "mexican_hat", "mexican-hat"}:
        raise ValueError(f"Unknown Mexican Hat wavelet alias: {wavelet}")
    return (1 - x**2) * torch.exp(-0.5 * x**2)



class GradientReversalLayer(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class CenterLoss(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, feat_dim=128, use_gpu=True):
        super().__init__()
        centers = torch.randn(num_classes, feat_dim)
        if use_gpu:
            centers = centers.cuda()
        self.centers = nn.Parameter(centers)
        self.num_classes = num_classes

    def forward(self, x, labels):
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        )
        distmat.addmm_(x, self.centers.t(), beta=1, alpha=-2)
        classes = torch.arange(self.num_classes, device=x.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        return (distmat * mask.float()).clamp(min=1e-12, max=1e12).sum() / batch_size


def normalize_wavelet_response(basis, eps=1e-6):
    rms = torch.sqrt(torch.mean(basis.pow(2), dim=-1, keepdim=True) + eps)
    return basis / rms


class KANLinearWavelet(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        scale_base=1.0,
        wavelet="mexhat",
        use_translation=False,
        normalize_response=False,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.use_translation = use_translation
        self.normalize_response = normalize_response
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.wavelet_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size))
        if use_translation:
            scale_init = torch.arange(1, grid_size + 1, dtype=torch.float32).view(1, grid_size).repeat(in_features, 1)
            translation_init = torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32).view(1, grid_size).repeat(
                in_features, 1
            )
            self.scales = nn.Parameter(scale_init)
            self.translations = nn.Parameter(translation_init)
        else:
            self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))
        self.silu = nn.SiLU()
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * scale_base)
        nn.init.normal_(self.wavelet_weight, mean=0.0, std=0.1)

    def forward(self, x):
        base_output = F.linear(self.silu(x), self.base_weight)
        if self.use_translation:
            x_scaled = (x.unsqueeze(-1) - self.translations.unsqueeze(0)) * self.scales.unsqueeze(0)
        else:
            x_scaled = x.unsqueeze(-1) * self.scales
        basis = wavelet_basis(x_scaled, self.wavelet)
        if self.normalize_response:
            basis = normalize_wavelet_response(basis)
        wavelet_output = torch.einsum("big,oig->bo", basis, self.wavelet_weight)
        return base_output + wavelet_output


class KANConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        grid_size=3,
        wavelet="mexhat",
        use_translation=False,
        normalize_response=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = grid_size
        self.wavelet = wavelet
        self.use_translation = use_translation
        self.normalize_response = normalize_response
        self.wavelet_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, grid_size) * 0.1)
        self.base_weights = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size) * 0.1)
        self.base_activation = nn.SiLU()
        if use_translation:
            scale_init = (
                torch.arange(1, grid_size + 1, dtype=torch.float32)
                .view(1, 1, grid_size)
                .repeat(in_channels, kernel_size, 1)
            )
            translation_init = (
                torch.linspace(-1.0, 1.0, grid_size, dtype=torch.float32)
                .view(1, 1, grid_size)
                .repeat(in_channels, kernel_size, 1)
            )
            self.scales = nn.Parameter(scale_init)
            self.translations = nn.Parameter(translation_init)
        else:
            self.scales = nn.Parameter(torch.arange(1, grid_size + 1, dtype=torch.float32))

    def forward(self, x):
        batch, _, _ = x.shape
        unfolded = F.unfold(
            x.unsqueeze(-1),
            kernel_size=(self.kernel_size, 1),
            padding=(self.padding, 0),
            stride=(self.stride, 1),
        )
        unfolded = unfolded.view(batch, self.in_channels, self.kernel_size, -1).permute(0, 3, 1, 2)
        base_output = torch.einsum("blic,oic->blo", self.base_activation(unfolded), self.base_weights)
        if self.use_translation:
            x_scaled = (
                unfolded.unsqueeze(-1)
                - self.translations.view(1, 1, self.in_channels, self.kernel_size, self.grid_size)
            ) * self.scales.view(1, 1, self.in_channels, self.kernel_size, self.grid_size)
        else:
            x_scaled = unfolded.unsqueeze(-1) * self.scales.view(1, 1, 1, 1, self.grid_size)
        basis = wavelet_basis(x_scaled, self.wavelet)
        if self.normalize_response:
            basis = normalize_wavelet_response(basis)
        wavelet_output = torch.einsum("blicg,oicg->blo", basis, self.wavelet_weights)
        return (base_output + wavelet_output).permute(0, 2, 1)


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excitation(self.squeeze(x)).expand_as(x)


class Conv1DBackbone(nn.Module):
    def __init__(self, use_se=False):
        super().__init__()
        self.first = nn.Conv1d(1, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.mid_cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock1D(128, reduction=8) if use_se else nn.Identity()
        self.tail = nn.Sequential(nn.MaxPool1d(2, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5))

    def forward(self, x):
        x = self.pool1(self.bn1(self.first(x)))
        x = self.mid_cnn(x)
        x = self.se(x)
        return torch.tanh(self.tail(x))


class WKANBackbone(Conv1DBackbone):
    def __init__(self, wavelet="mexhat", use_se=True, use_translation=False, normalize_response=False):
        super().__init__(use_se=use_se)
        self.first = KANConv1d(
            1,
            32,
            kernel_size=31,
            stride=2,
            padding=15,
            wavelet=wavelet,
            use_translation=use_translation,
            normalize_response=normalize_response,
        )


class CDANModel(nn.Module):
    """Wavelet-KAN backbone/classifier with conditional domain alignment.

    Returns class logits, domain logits, and 128-dimensional features.
    The domain discriminator receives the outer product of features and
    class probabilities, flattened to 128 * num_classes dimensions.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.domain_mode = "cdan"
        self.use_cdan = True
        self.num_classes = num_classes
        self.backbone = WKANBackbone(
            wavelet="mexhat", use_se=True,
            use_translation=False, normalize_response=False,
        )
        self.classifier = KANLinearWavelet(
            128, num_classes, wavelet="mexhat",
            use_translation=False, normalize_response=False,
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(128 * num_classes, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2),
        )

    def forward(self, x, alpha=0.0):
        features = self.backbone(x)
        class_logits = self.classifier(features)
        softmax_prob = F.softmax(class_logits, dim=1)
        cond_features = torch.bmm(
            features.unsqueeze(2), softmax_prob.unsqueeze(1)
        ).view(-1, 128 * self.num_classes)
        reverse_features = GradientReversalLayer.apply(cond_features, alpha)
        domain_logits = self.domain_classifier(reverse_features)
        return class_logits, domain_logits, features


def target_entropy_loss(class_logits):
    """Unweighted target entropy term from the main training objective."""
    probabilities = F.softmax(class_logits, dim=1)
    return -(probabilities * torch.log(probabilities + 1e-5)).sum(dim=1).mean()

