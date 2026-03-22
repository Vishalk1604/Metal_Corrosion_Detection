"""
models/segformer_eem_ffm.py
SegFormer + EEM + FFM — paper's full architecture (Fig 1)

Decoder flow verified against Fig 1 (right column, bottom-up):
  Block 4 (H/32×W/32×C4)  → MLP proj → p4
  Block 3 (H/16×W/16×C3)  → MLP proj → p3  + upsample(p4)  → FFM3 → d3
  Block 2 (H/8 ×W/8 ×C2)  → MLP proj → p2  + upsample(d3)  → FFM2 → d2
  Block 1 (H/4 ×W/4 ×C1)  → EEM → edge     + upsample(d2)  → FFM1 → d1
  d1  → seg_head (1×1)  → bilinear upsample → (B, num_cls, H, W)

Key paper facts (Table 5/6):
  Baseline SegFormer  : 3.72 MB params, 6.78 GFLOPs, mIoU 66.20%
  + EEM only          : 3.87 MB params, 9.13 GFLOPs, mIoU 67.45%
  + FFM only          : 3.58 MB params, 2.79 GFLOPs, mIoU 66.25%
  + EEM + FFM (ours)  : 3.60 MB params, 3.06 GFLOPs, mIoU 67.69%

MiT-B0 hidden sizes: [32, 64, 160, 256]  (C1, C2, C3, C4)
decoder_channels D = 64  (paper default)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel, SegformerConfig

from models.eem import EdgeFeatureExtractionModule
from models.ffm import FFM


# ── Backbone registry ─────────────────────────────────────────────────────────

BACKBONES = {
    "b0": {"hf_name": "nvidia/mit-b0",
            "hidden_sizes": [32, 64, 160, 256]},
    "b1": {"hf_name": "nvidia/mit-b1",
            "hidden_sizes": [64, 128, 320, 512]},
    "b2": {"hf_name": "nvidia/mit-b2",
            "hidden_sizes": [64, 128, 320, 512]},
}


# ── Hidden-state reshape ─────────────────────────────────────────────────────

def _to_spatial(hidden: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Converts encoder hidden states to (B, C, H, W).
    Different transformers versions return either:
      - 3D (B, H*W, C)  → needs reshape
      - 4D (B, C, H, W) → already spatial
    """
    if hidden.dim() == 4:
        return hidden
    B, N, C = hidden.shape
    return hidden.permute(0, 2, 1).reshape(B, C, h, w)


# ── Full model ────────────────────────────────────────────────────────────────

class SegFormerEEMFFM(nn.Module):
    """
    Args:
        num_classes      : 4  (Background / Fair / Poor / Severe)
        decoder_channels : 64  (paper default, D in equations)
        pretrained       : load ImageNet pre-trained MiT weights
        backbone         : "b0" (paper) | "b1" | "b2"
    """

    def __init__(self,
                 num_classes:      int   = 4,
                 decoder_channels: int   = 64,
                 pretrained:       bool  = True,
                 backbone:         str   = "b0",
                 ffm_dropout:      float = 0.1):
        super().__init__()

        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone '{backbone}'. "
                             f"Choose from {list(BACKBONES.keys())}")

        cfg = BACKBONES[backbone]
        self.backbone_name    = backbone
        self.num_classes      = num_classes
        self.decoder_channels = decoder_channels
        C1, C2, C3, C4        = cfg["hidden_sizes"]
        D                     = decoder_channels
        self._hidden_sizes    = cfg["hidden_sizes"]

        # ── Encoder  (pre-trained MiT backbone) ──────────────────────────────
        if pretrained:
            self.encoder = SegformerModel.from_pretrained(cfg["hf_name"])
        else:
            seg_cfg = SegformerConfig(hidden_sizes=cfg["hidden_sizes"])
            self.encoder = SegformerModel(seg_cfg)

        # ── EEM: auxiliary spatial branch at Block-1 output ──────────────────
        # Placed between Transformer Block 1 and the topmost FFM (FFM1).
        # Block-1 features are richest in local edge detail (paper Sec 3.1).
        self.eem = EdgeFeatureExtractionModule(in_channels=C1,
                                               out_channels=D)

        # ── MLP projections: unify all encoder stages to D channels ──────────
        # These are the "MLP" blocks shown in the right decoder column of Fig 1.
        def _mlp(in_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, D, 1, bias=False),
                nn.BatchNorm2d(D),
                nn.ReLU(inplace=True),
            )

        self.proj4 = _mlp(C4)   # Block 4 → D
        self.proj3 = _mlp(C3)   # Block 3 → D
        self.proj2 = _mlp(C2)   # Block 2 → D
        # Block 1 goes through EEM (not a plain MLP projection)

        self.ffm_dropout = ffm_dropout

        # ── Hierarchical FFM decoder (bottom-up) ─────────────────────────────
        # FFM3: fuses Block-3 features + upsampled Block-4 features
        self.ffm3 = FFM(low_channels=D, high_channels=D, out_channels=D,
                        dropout=ffm_dropout)
        # FFM2: fuses Block-2 features + upsampled FFM3 output
        self.ffm2 = FFM(low_channels=D, high_channels=D, out_channels=D,
                        dropout=ffm_dropout)
        # FFM1: fuses EEM edge features + upsampled FFM2 output  ← topmost
        self.ffm1 = FFM(low_channels=D, high_channels=D, out_channels=D,
                        dropout=ffm_dropout)

        # ── Segmentation head ─────────────────────────────────────────────────
        self.seg_head = nn.Conv2d(D, num_classes, 1)

        # Initialise new (non-pretrained) modules
        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        new_mods = [self.eem,
                    self.proj4, self.proj3, self.proj2,
                    self.ffm3,  self.ffm2,  self.ffm1,
                    self.seg_head]
        for mod in new_mods:
            for layer in mod.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out",
                                            nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x      : (B, 3, H, W)  — H = W = 512
        returns: (B, num_classes, H, W)  raw logits
        """
        B, _, H, W = x.shape
        # Spatial resolutions for 512×512 input:
        #   stride-4  → 128×128  (Block 1)
        #   stride-8  →  64×64   (Block 2)
        #   stride-16 →  32×32   (Block 3)
        #   stride-32 →  16×16   (Block 4)
        s = H // 4

        # ── Encoder ───────────────────────────────────────────────────────────
        enc = self.encoder(x, output_hidden_states=True)
        h1, h2, h3, h4 = enc.hidden_states   # tuple of 4

        # Reshape to spatial tensors (handles 3D and 4D output formats)
        f1 = _to_spatial(h1, s,    s   )   # (B, C1, 128, 128)
        f2 = _to_spatial(h2, s//2, s//2)   # (B, C2,  64,  64)
        f3 = _to_spatial(h3, s//4, s//4)   # (B, C3,  32,  32)
        f4 = _to_spatial(h4, s//8, s//8)   # (B, C4,  16,  16)

        # ── EEM: edge feature from shallowest encoder block ───────────────────
        edge = self.eem(f1)                 # (B, D, 128, 128)

        # ── MLP projections → D channels ──────────────────────────────────────
        p4 = self.proj4(f4)                 # (B, D, 16, 16)
        p3 = self.proj3(f3)                 # (B, D, 32, 32)
        p2 = self.proj2(f2)                 # (B, D, 64, 64)

        # ── Decoder: progressive FFM upsampling (bottom-up) ──────────────────
        # Stage 3: upsample Block-4 proj → Block-3 spatial size, then FFM
        up4 = F.interpolate(p4, size=p3.shape[2:],
                             mode="bilinear", align_corners=False)
        d3  = self.ffm3(f_low=p3, f_high=up4)       # (B, D, 32, 32)

        # Stage 2: upsample FFM3 output → Block-2 spatial size, then FFM
        up3 = F.interpolate(d3, size=p2.shape[2:],
                             mode="bilinear", align_corners=False)
        d2  = self.ffm2(f_low=p2, f_high=up3)       # (B, D, 64, 64)

        # Stage 1 (top FFM): upsample FFM2 output → Block-1 spatial size,
        # fuse with EEM edge features (the skip connection in the paper)
        up2 = F.interpolate(d2, size=edge.shape[2:],
                             mode="bilinear", align_corners=False)
        d1  = self.ffm1(f_low=edge, f_high=up2)     # (B, D, 128, 128)

        # ── Segmentation head + final upsample to input resolution ───────────
        logits = self.seg_head(d1)                   # (B, num_cls, 128, 128)
        logits = F.interpolate(logits, size=(H, W),
                                mode="bilinear", align_corners=False)
        return logits                                # (B, num_cls, 512, 512)

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        return {"total_M": total / 1e6,
                "trainable_M": trainable / 1e6}

    def __repr__(self):
        p = self.count_parameters()
        return (f"SegFormerEEMFFM("
                f"backbone=MiT-{self.backbone_name.upper()}, "
                f"num_classes={self.num_classes}, "
                f"decoder_ch={self.decoder_channels}, "
                f"params={p['total_M']:.2f}M)")