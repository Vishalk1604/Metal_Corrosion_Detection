"""
losses/focal_loss.py

Paper Eq.9: L_focal(pt) = (1 − pt)^γ · log(pt),  γ = 2

WHY WE NOW ADD MILD ALPHA
──────────────────────────
Pure focal loss (no alpha) failed because:
  - Background = 64% of pixels  → huge gradient
  - Severe     =  2.4% of pixels → tiny gradient  → model ignores it

Even with γ=2, the raw class imbalance is too extreme (27:1 ratio).
The paper reports Severe IoU ~68% on val but never mentions alpha explicitly.
Given that result is impossible without addressing the 2.4% class, they almost
certainly used some form of class balancing.

FIX: Use SQRT inverse-frequency weights.
  - Inverse-frequency gives huge weights (e.g. Severe=3.0x, BG=0.13x)
    → over-penalises BG, model obsesses over Severe → bad mIoU
  - SQRT inverse-frequency is a compromise:
    → Severe gets moderate boost (~1.7x) instead of 3x
    → Background still counted (~0.36x) instead of 0.13x
    → All classes within a 5x range instead of 23x range
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Focal Loss with optional mild sqrt-inverse-frequency class weights.

    Args:
        gamma        : focusing exponent. Paper: γ=2.
        alpha        : per-class weight tensor (num_classes,) or None.
        ignore_index : pixel label to exclude.
        reduction    : "mean" | "sum" | "none"
    """

    def __init__(self,
                 gamma:        float             = 2.0,
                 alpha:        torch.Tensor       = None,
                 ignore_index: int               = 255,
                 reduction:    str               = "mean"):
        super().__init__()
        self.gamma        = gamma
        self.alpha        = alpha
        self.ignore_index = ignore_index
        self.reduction    = reduction

    def forward(self,
                logits:  torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W)
        targets : (B, H, W)   integer class labels
        """
        B, C, H, W = logits.shape

        probs        = F.softmax(logits, dim=1)
        probs_flat   = probs.permute(0, 2, 3, 1).reshape(-1, C)
        targets_flat = targets.reshape(-1)

        valid        = targets_flat != self.ignore_index
        probs_flat   = probs_flat[valid]
        targets_flat = targets_flat[valid]

        if targets_flat.numel() == 0:
            return logits.sum() * 0.0

        # pt = probability of the ground-truth class
        pt     = probs_flat.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
        pt     = pt.clamp(min=1e-8)

        # Focal weight
        focal_w = (1.0 - pt) ** self.gamma

        # Optional mild class weight
        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets_flat]
            focal_w = focal_w * alpha_t

        loss = -(focal_w * torch.log(pt))

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def compute_class_weights(dataset,
                          num_classes: int   = 4,
                          mode:        str   = "sqrt",
                          smoothing:   float = 1e-6) -> torch.Tensor:
    """
    Compute class weights from pixel frequency.

    Args:
        dataset     : CorrosionDataset with .class_pixel_counts()
        num_classes : number of classes
        mode        : "sqrt"   — sqrt(total / (C * count))  [recommended]
                      "linear" — total / (C * count)        [too aggressive]
                      "none"   — uniform weights (all 1.0)
        smoothing   : avoids div/0

    Returns:
        weights : (num_classes,) float32 tensor, mean-normalised to 1
    """
    print("  Computing pixel counts (one-time) ...")
    counts = dataset.class_pixel_counts(num_classes).astype(np.float64)
    total  = counts.sum()

    names = ["Background", "Fair", "Poor", "Severe"]
    print(f"  Pixel %: " + " | ".join(
        f"{n}={100*c/total:.1f}%" for n, c in zip(names, counts)))

    if mode == "none":
        weights = np.ones(num_classes)
    elif mode == "sqrt":
        # Milder than pure inverse-frequency
        inv_freq = total / (num_classes * counts + smoothing)
        weights  = np.sqrt(inv_freq)
    elif mode == "linear":
        weights = total / (num_classes * counts + smoothing)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Normalise: mean = 1
    weights = weights / weights.mean()

    print(f"  Weights ({mode}): " + " | ".join(
        f"{n}={w:.3f}" for n, w in zip(names, weights)))

    return torch.tensor(weights, dtype=torch.float32)