"""
datasets/corrosion_dataset.py

Stronger augmentation to combat overfitting on only ~356 source training images.

Key additions vs before:
  - ElasticTransform: deforms corrosion boundaries → helps edge generalisation
  - RandomGridDistortion: texture-level geometric noise
  - CoarseDropout (cutout): forces model to infer from partial views
  - Sharpen / Emboss: simulates different camera/lighting conditions
  - More aggressive colour jitter ranges

Class labels: 0=Background, 1=Fair, 2=Poor, 3=Severe
"""

import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _train_transforms() -> A.Compose:
    return A.Compose([
        # ── Heavy geometric ────────────────────────────────────────────────────
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.2, rotate_limit=45,
            border_mode=cv2.BORDER_REFLECT, p=0.6),

        # ── Elastic / grid distortion (helps edge generalisation) ─────────────
        A.OneOf([
            A.ElasticTransform(
                alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,
                border_mode=cv2.BORDER_REFLECT, p=1.0),
            A.GridDistortion(
                num_steps=5, distort_limit=0.3,
                border_mode=cv2.BORDER_REFLECT, p=1.0),
        ], p=0.3),

        # ── Colour jitter (more aggressive) ────────────────────────────────────
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.4, contrast_limit=0.4, p=1.0),
            A.HueSaturationValue(
                hue_shift_limit=25, sat_shift_limit=40,
                val_shift_limit=25, p=1.0),
            A.CLAHE(clip_limit=6.0, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.7),

        # ── Texture augmentation ───────────────────────────────────────────────
        A.OneOf([
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7), p=1.0),
        ], p=0.3),

        # ── Noise / blur ───────────────────────────────────────────────────────
        A.OneOf([
            A.GaussianBlur(blur_limit=5, p=1.0),
            A.GaussNoise(var_limit=(10.0, 60.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
        ], p=0.4),

        # ── Cutout / dropout (forces partial-view learning) ────────────────────
        A.CoarseDropout(
            max_holes=8, max_height=64, max_width=64,
            min_holes=1, min_height=16, min_width=16,
            fill_value=0, p=0.3),

        # ── Normalise + tensor ─────────────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def _val_transforms() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


class CorrosionDataset(Dataset):
    """
    Args:
        img_dir  : path to images folder (512×512 PNG patches)
        mask_dir : path to masks folder  (grayscale 0–3 PNG)
        split    : "train" | "val" | "test"
    """

    def __init__(self, img_dir: str, mask_dir: str, split: str = "train"):
        self.img_dir  = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.split    = split

        self.img_paths  = sorted(self.img_dir.glob("*.png"))
        self.mask_paths = sorted(self.mask_dir.glob("*.png"))

        if len(self.img_paths) == 0:
            raise RuntimeError(f"No PNG images found in {img_dir}")
        if len(self.img_paths) != len(self.mask_paths):
            raise RuntimeError(
                f"Image/mask count mismatch: "
                f"{len(self.img_paths)} images vs "
                f"{len(self.mask_paths)} masks")

        for ip, mp in zip(self.img_paths, self.mask_paths):
            if ip.name != mp.name:
                raise RuntimeError(
                    f"Filename mismatch: {ip.name} vs {mp.name}")

        self.transform = (_train_transforms()
                          if split == "train" else _val_transforms())

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        img = cv2.imread(str(self.img_paths[idx]))
        if img is None:
            raise IOError(f"Cannot read {self.img_paths[idx]}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Cannot read {self.mask_paths[idx]}")
        mask = np.clip(mask, 0, 3).astype(np.uint8)

        out   = self.transform(image=img, mask=mask)
        image = out["image"]
        mask  = out["mask"].long()
        return image, mask

    def class_pixel_counts(self, num_classes: int = 4) -> np.ndarray:
        counts = np.zeros(num_classes, dtype=np.int64)
        for p in self.mask_paths:
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if m is not None:
                for c in range(num_classes):
                    counts[c] += int(np.sum(m == c))
        return counts