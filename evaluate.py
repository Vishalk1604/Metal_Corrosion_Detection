"""
evaluate.py — Evaluate with optional Test Time Augmentation (TTA)

TTA runs each image through 8 augmentations (original + 3 rotations + their flips)
and averages the softmax predictions before argmax. This typically adds 5-8% mIoU
at zero training cost.

Usage:
    python evaluate.py                         # standard eval
    python evaluate.py --tta                   # with TTA (~8x slower but higher mIoU)
    python evaluate.py --save_visuals --tta    # TTA + save comparison images
    python evaluate.py --split val             # evaluate val set
"""

import argparse
import os
import logging
import warnings
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*pin_memory.*")

from models.segformer_eem_ffm import SegFormerEEMFFM
from datasets.corrosion_dataset import CorrosionDataset
from utils import (compute_mIoU, compute_per_class_IoU, compute_accuracy,
                   batch_to_numpy, load_checkpoint, print_metrics, CLASS_NAMES)


# ── Color map ─────────────────────────────────────────────────────────────────
CLASS_BGR = {0: (0,0,0), 1: (0,200,0), 2: (0,200,200), 3: (0,0,200)}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def mask_to_color(mask):
    c = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, bgr in CLASS_BGR.items():
        c[mask == cls] = bgr
    return c


def unnormalize(t):
    img = t.cpu().numpy().transpose(1, 2, 0)
    img = (img * IMAGENET_STD + IMAGENET_MEAN) * 255
    return cv2.cvtColor(img.clip(0,255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def make_comparison(img_t, gt_mask, pred_mask):
    H = 400
    def rh(im):
        s = H / im.shape[0]
        return cv2.resize(im, (int(im.shape[1]*s), H),
                          interpolation=cv2.INTER_NEAREST)
    orig  = rh(unnormalize(img_t))
    gt_c  = rh(mask_to_color(gt_mask))
    pr_c  = rh(mask_to_color(pred_mask))
    ov    = cv2.addWeighted(orig, 0.5, pr_c, 0.5, 0)
    div   = np.full((H, 3, 3), 200, dtype=np.uint8)
    panel = np.hstack([orig, div, gt_c, div, pr_c, div, ov])
    for i, lbl in enumerate(["Original","Ground Truth","Prediction","Overlay"]):
        cv2.putText(panel, lbl, (i*(orig.shape[1]+3)+4, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    return panel


# ── TTA helpers ───────────────────────────────────────────────────────────────

def _tta_augments():
    """
    8 deterministic augmentations:
    original, hflip, rot90, rot90+hflip, rot180, rot180+hflip, rot270, rot270+hflip
    Returns list of (forward_fn, inverse_fn) pairs operating on (B,C,H,W) tensors.
    """
    def hflip(x):  return torch.flip(x, [3])
    def rot90(x):  return torch.rot90(x, 1, [2, 3])
    def rot180(x): return torch.rot90(x, 2, [2, 3])
    def rot270(x): return torch.rot90(x, 3, [2, 3])

    def ihflip(x):  return torch.flip(x, [3])
    def irot90(x):  return torch.rot90(x, -1, [2, 3])
    def irot180(x): return torch.rot90(x, -2, [2, 3])
    def irot270(x): return torch.rot90(x, -3, [2, 3])

    identity = lambda x: x

    return [
        (identity, identity),
        (hflip,    ihflip),
        (rot90,    irot90),
        (lambda x: hflip(rot90(x)),   lambda x: irot90(ihflip(x))),
        (rot180,   irot180),
        (lambda x: hflip(rot180(x)),  lambda x: irot180(ihflip(x))),
        (rot270,   irot270),
        (lambda x: hflip(rot270(x)),  lambda x: irot270(ihflip(x))),
    ]


def predict_with_tta(model, images, autocast_ctx, device, num_classes):
    """
    Run inference with 8 TTA augmentations and average softmax probabilities.
    Returns logits-like tensor (B, C, H, W) — averaged probs, not raw logits.
    """
    augments = _tta_augments()
    acc_probs = None

    for fwd, inv in augments:
        aug_imgs = fwd(images)
        with autocast_ctx:
            logits = model(aug_imgs)                 # (B, C, H, W)
        # Invert spatial transform on the output
        logits_inv = inv(logits)
        probs      = F.softmax(logits_inv, dim=1)    # (B, C, H, W)
        if acc_probs is None:
            acc_probs = probs
        else:
            acc_probs = acc_probs + probs

    # Return average (pseudo-logits for argmax)
    return acc_probs / len(augments)


def predict_standard(model, images, autocast_ctx, device):
    with autocast_ctx:
        return model(images)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp    = (device_str == "cuda")

    # Load backbone name from checkpoint
    raw_ckpt = torch.load(args.checkpoint, map_location="cpu")
    backbone = raw_ckpt.get("backbone", "b0")

    model = SegFormerEEMFFM(
        num_classes      = args.num_classes,
        decoder_channels = args.decoder_channels,
        pretrained       = False,
        backbone         = backbone,
    ).to(device)
    model.load_state_dict(raw_ckpt["model_state"])
    model.eval()

    trained_epoch = raw_ckpt.get("epoch", "?")
    best_val_mIoU = raw_ckpt.get("best_mIoU", 0.0)

    print("=" * 60)
    print("  Evaluation — SegFormer + EEM + FFM")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Backbone     : MiT-{backbone.upper()}")
    print(f"  Trained epoch: {trained_epoch}")
    print(f"  Best val mIoU: {best_val_mIoU:.2f}%")
    print(f"  Eval split   : {args.split}")
    print(f"  TTA          : {'YES (8 augmentations)' if args.tta else 'NO'}")
    print(f"  Device       : {device}")

    img_dir  = getattr(args, f"{args.split}_img")
    mask_dir = getattr(args, f"{args.split}_mask")
    ds       = CorrosionDataset(img_dir, mask_dir, split="test")
    loader   = DataLoader(ds, batch_size=args.batch_size,
                          shuffle=False, num_workers=args.workers,
                          pin_memory=use_amp)
    print(f"  Patches      : {len(ds)}")
    print("=" * 60)

    pred_dir = Path(args.out_dir) / args.split / "predictions"
    comp_dir = Path(args.out_dir) / args.split / "comparisons"
    if args.save_visuals:
        pred_dir.mkdir(parents=True, exist_ok=True)
        comp_dir.mkdir(parents=True, exist_ok=True)

    all_preds, all_labels = [], []
    img_idx = 0
    autocast_ctx = torch.amp.autocast(device_type=device_str, enabled=use_amp)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loader):
            images_gpu = images.to(device, non_blocking=True)

            if args.tta:
                probs  = predict_with_tta(model, images_gpu, autocast_ctx,
                                          device, args.num_classes)
                preds_b = probs.argmax(dim=1).cpu().numpy()
            else:
                logits  = predict_standard(model, images_gpu, autocast_ctx, device)
                preds_b = logits.argmax(dim=1).cpu().numpy()

            labels_b = masks.numpy()
            all_preds.append(preds_b.reshape(-1))
            all_labels.append(labels_b.reshape(-1))

            if args.save_visuals:
                for b in range(images.size(0)):
                    fname = f"{img_idx:05d}.png"
                    cv2.imwrite(str(pred_dir / fname), mask_to_color(preds_b[b]))
                    comp = make_comparison(images[b], labels_b[b], preds_b[b])
                    cv2.imwrite(str(comp_dir / fname), comp,
                                [cv2.IMWRITE_PNG_COMPRESSION, 6])
                    img_idx += 1

            if (batch_idx + 1) % 5 == 0:
                print(f"  Processed {min((batch_idx+1)*args.batch_size, len(ds))}"
                      f"/{len(ds)}", end="\r")

    print()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    mIoU    = compute_mIoU(all_preds, all_labels, args.num_classes)
    acc     = compute_accuracy(all_preds, all_labels)
    per_cls = compute_per_class_IoU(all_preds, all_labels, args.num_classes)

    tta_label = " (with TTA)" if args.tta else ""
    print(f"\n{args.split.upper()} SET RESULTS{tta_label}:")
    print_metrics(mIoU, acc, per_cls)

    print("\n  Pixel distribution:")
    total_px = len(all_labels)
    for c, name in enumerate(CLASS_NAMES):
        gt_n  = int(np.sum(all_labels == c))
        pr_n  = int(np.sum(all_preds  == c))
        print(f"    {name:10s}  GT={100*gt_n/total_px:.1f}%  "
              f"Pred={100*pr_n/total_px:.1f}%")

    if args.save_visuals:
        print(f"\n  Visuals → {args.out_dir}/{args.split}/")

    return mIoU, acc


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",       default="checkpoints/best.pth")
    p.add_argument("--split",            default="test",
                   choices=["test", "val"])
    p.add_argument("--test_img",         default="data/final/test/images")
    p.add_argument("--test_mask",        default="data/final/test/masks")
    p.add_argument("--val_img",          default="data/final/val/images")
    p.add_argument("--val_mask",         default="data/final/val/masks")
    p.add_argument("--num_classes",      type=int, default=4)
    p.add_argument("--decoder_channels", type=int, default=64)
    p.add_argument("--batch_size",       type=int, default=4,
                   help="Use 4 for TTA (8x memory usage), 8 for standard")
    p.add_argument("--workers",          type=int, default=4)
    p.add_argument("--tta",              action="store_true", default=False,
                   help="Test Time Augmentation: 8 flips/rotations averaged. "
                        "Adds ~5-8 percent mIoU, 8x slower.")
    p.add_argument("--save_visuals",     action="store_true", default=False)
    p.add_argument("--out_dir",          default="results")
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    evaluate(args)