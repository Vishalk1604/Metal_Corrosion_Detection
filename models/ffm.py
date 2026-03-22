"""
models/ffm.py
Feature Fusion Module (FFM) — with dropout for regularisation.

Paper Eq.5–8:
  F'  = F_edge ⊕ F_C
  F_A = σ( Conv1( DSConv3x3([F_edge, F']) ) )
  F_B = σ( Conv1( DSConv3x3([F_C,    F']) ) )
  Y   = Conv1([F_A ⊙ F_edge, F_B ⊙ F_C])

Dropout added to final fusion to fight overfitting on small dataset.
Dropout is OFF during eval (model.eval() handles this automatically).
"""

import torch
import torch.nn as nn


class FFM(nn.Module):
    """
    Feature Fusion Module.

    Args:
        low_channels  : channels of the low-level (edge/shallow) feature
        high_channels : channels of the high-level (semantic) feature
        out_channels  : output channels
        dropout       : dropout probability on final fusion (0 = disabled)
    """

    def __init__(self,
                 low_channels:  int,
                 high_channels: int,
                 out_channels:  int,
                 dropout:       float = 0.1):
        super().__init__()

        # ── Align both inputs to out_channels ────────────────────────────────
        self.low_align = nn.Sequential(
            nn.Conv2d(low_channels,  out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )
        self.high_align = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        )

        # ── Attention branch A (for edge/low-level stream) ───────────────────
        self.attn_A = nn.Sequential(
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1,
                      groups=2*out_channels, bias=False),
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # ── Attention branch B (for semantic/high-level stream) ──────────────
        self.attn_B = nn.Sequential(
            nn.Conv2d(2*out_channels, 2*out_channels, 3, padding=1,
                      groups=2*out_channels, bias=False),
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

        # ── Final fusion with optional dropout ───────────────────────────────
        layers = [
            nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(p=dropout))
        self.final = nn.Sequential(*layers)

    def forward(self,
                f_low:  torch.Tensor,
                f_high: torch.Tensor) -> torch.Tensor:
        """
        f_low  : (B, low_ch,  H, W)
        f_high : (B, high_ch, H, W)   already upsampled to match f_low
        out    : (B, out_ch,  H, W)
        """
        f_edge  = self.low_align(f_low)
        f_c     = self.high_align(f_high)

        f_prime = f_edge + f_c                                        # Eq.5

        f_a = self.attn_A(torch.cat([f_edge, f_prime], dim=1))       # Eq.6
        f_b = self.attn_B(torch.cat([f_c,    f_prime], dim=1))       # Eq.7

        y = self.final(torch.cat([f_a * f_edge, f_b * f_c], dim=1))  # Eq.8
        return y