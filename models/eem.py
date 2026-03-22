"""
models/eem.py
Edge-Feature Extraction Module (EEM)

Parallel branches:
  Branch 1 — Sobel (fixed kernel) + multi-scale DSConv (3x3, 5x5, 7x7)
  Branch 2 — Laplacian (fixed kernel)

Fused via 1x1 conv → edge feature map Y_E

Paper equations:
  F_sobel = Conv1x1(FX) ⊗ (G_sobel * S_sobel) + B_sobel        [Eq.1]
  F1      = Conv1x1([DSConv3(F_sobel), DSConv5(F_sobel),
                     DSConv7(F_sobel)])                          [Eq.2]
  F_lp    = Conv1x1(FX) ⊗ (G_lp * S_lp) + B_lp                 [Eq.3]
  Y_E     = Conv1x1([F1, F_lp])                                  [Eq.4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fixed kernel builders ──────────────────────────────────────────────────────

def _sobel_kernel(in_channels: int) -> torch.Tensor:
    """
    Returns a depth-wise Sobel kernel of shape (2*C, 1, 3, 3).
    Two filters per channel: horizontal Gx and vertical Gy.
    Used with groups=in_channels.
    """
    kx = torch.tensor([[-1.,  0.,  1.],
                        [-2.,  0.,  2.],
                        [-1.,  0.,  1.]])
    ky = torch.tensor([[-1., -2., -1.],
                        [ 0.,  0.,  0.],
                        [ 1.,  2.,  1.]])
    # (2, 1, 3, 3)  then repeat C times → (2C, 1, 3, 3)
    kernel = torch.stack([kx, ky], dim=0).unsqueeze(1)
    return kernel.repeat(in_channels, 1, 1, 1)


def _laplacian_kernel(in_channels: int) -> torch.Tensor:
    """
    Returns a depth-wise Laplacian kernel of shape (C, 1, 3, 3).
    Used with groups=in_channels.
    """
    lap = torch.tensor([[0.,  1., 0.],
                         [1., -4., 1.],
                         [0.,  1., 0.]])
    kernel = lap.unsqueeze(0).unsqueeze(0)        # (1, 1, 3, 3)
    return kernel.repeat(in_channels, 1, 1, 1)   # (C, 1, 3, 3)


# ── Depthwise Separable Convolution ───────────────────────────────────────────

class DSConv(nn.Module):
    """Depthwise-separable conv: DW(k×k) → PW(1×1) → BN → ReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pad: int):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, kernel, padding=pad,
                             groups=in_ch, bias=False)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn  = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


# ── EEM ────────────────────────────────────────────────────────────────────────

class EdgeFeatureExtractionModule(nn.Module):
    """
    Args:
        in_channels  : channels of Transformer Block-1 output  (C1 = 32 for MiT-B0)
        out_channels : channels of the output edge feature map
    """

    def __init__(self, in_channels: int, out_channels: int = 64):
        super().__init__()
        C = in_channels
        self.in_channels  = C
        self.out_channels = out_channels

        # ── Channel adjustment before edge filters ───────────────────────────
        self.adjust = nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # ── Branch 1: Sobel ──────────────────────────────────────────────────
        # Fixed Sobel kernel (not trained)
        sobel_k = _sobel_kernel(C)                     # (2C, 1, 3, 3)
        self.register_buffer("sobel_kernel", sobel_k)
        # Learnable scale & bias applied to the Sobel response
        self.sobel_scale = nn.Parameter(torch.ones(1))
        self.sobel_bias  = nn.Parameter(torch.zeros(1))
        # Reduce 2C → C after Sobel
        self.sobel_reduce = nn.Conv2d(2 * C, C, 1, bias=False)

        # Multi-scale DSConv on Sobel features
        self.ds3 = DSConv(C, C, 3, 1)
        self.ds5 = DSConv(C, C, 5, 2)
        self.ds7 = DSConv(C, C, 7, 3)
        # Fuse 3 scales → C
        self.sobel_fuse = nn.Sequential(
            nn.Conv2d(3 * C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        # ── Branch 2: Laplacian ──────────────────────────────────────────────
        lap_k = _laplacian_kernel(C)                   # (C, 1, 3, 3)
        self.register_buffer("laplacian_kernel", lap_k)
        self.lp_scale = nn.Parameter(torch.ones(1))
        self.lp_bias  = nn.Parameter(torch.zeros(1))

        # ── Final fusion: [F1, F_lp] → out_channels ─────────────────────────
        self.final = nn.Sequential(
            nn.Conv2d(2 * C, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _sobel(self, x: torch.Tensor) -> torch.Tensor:
        """Depth-wise Sobel (groups=C, two kernels per channel) then scale."""
        # F.conv2d with groups=C, kernel (2C,1,3,3) → out (B, 2C, H, W)
        out = F.conv2d(x, self.sobel_kernel, padding=1, groups=self.in_channels)
        out = out * self.sobel_scale + self.sobel_bias
        return self.sobel_reduce(out)    # (B, C, H, W)

    def _laplacian(self, x: torch.Tensor) -> torch.Tensor:
        out = F.conv2d(x, self.laplacian_kernel, padding=1,
                       groups=self.in_channels)
        return out * self.lp_scale + self.lp_bias    # (B, C, H, W)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x   : (B, C, H/4, W/4)  — output of Transformer Block 1
        out : (B, out_channels, H/4, W/4)
        """
        fx = self.adjust(x)

        # Branch 1
        f_sobel = self._sobel(fx)
        f1 = self.sobel_fuse(
            torch.cat([self.ds3(f_sobel),
                       self.ds5(f_sobel),
                       self.ds7(f_sobel)], dim=1)
        )

        # Branch 2
        f_lp = self._laplacian(fx)

        # Fuse
        y_e = self.final(torch.cat([f1, f_lp], dim=1))
        return y_e    # (B, out_channels, H/4, W/4)