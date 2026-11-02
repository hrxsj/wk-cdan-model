"""Run a WK-CDAN forward/backward example with synthetic inputs."""

import torch
import torch.nn.functional as F

from model import CDANModel, CenterLoss, target_entropy_loss
from runtime import resolve_seed


def main():
    torch.manual_seed(resolve_seed())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CDANModel(num_classes=10).to(device)
    center_loss = CenterLoss(num_classes=10, use_gpu=False).to(device)
    # Create synthetic source and target batches.
    source = torch.randn(4, 1, 1024, device=device)
    target = torch.randn(4, 1, 1024, device=device)
    source_labels = torch.tensor([0, 1, 2, 3], device=device)
    alpha = 0.5
    source_logits, source_domain, source_features = model(source, alpha)
    target_logits, target_domain, _ = model(target, alpha)
    classification = F.cross_entropy(source_logits, source_labels)
    domain = F.cross_entropy(source_domain, torch.zeros(4, dtype=torch.long, device=device))
    domain = domain + F.cross_entropy(target_domain, torch.ones(4, dtype=torch.long, device=device))
    center = center_loss(source_features, source_labels)
    entropy = target_entropy_loss(target_logits)
    total = classification + domain + 0.05 * center + 0.05 * alpha * entropy
    total.backward()
    assert source_logits.shape == (4, 10)
    assert source_domain.shape == (4, 2)
    assert source_features.shape == (4, 128)
    assert torch.isfinite(total)
    assert model.backbone.first.wavelet_weights.grad is not None
    assert center_loss.centers.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"Device: {device}")
    print("Class logits: (4, 10); domain logits: (4, 2); features: (4, 128)")
    print("Synthetic forward/backward example completed successfully.")


if __name__ == "__main__":
    main()
