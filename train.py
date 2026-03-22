"""
train.py — Two-Stage Training for SegFormer + EEM + FFM

WHY TWO-STAGE TRAINING
───────────────────────
Problem: With only 396 source training images, the pretrained MiT encoder
gets fine-tuned too aggressively and memorises training image patterns.
Result: val mIoU=63% but test mIoU=40% (23% gap = overfitting).

Fix: Two-stage training:
  Stage 1 (epochs 1–freeze_epochs): Encoder FROZEN
    - Only EEM + FFM + decoder train (randomly initialised weights)
    - Higher LR (1e-3) since starting from scratch for new layers
    - Model learns what corrosion LOOKS LIKE using pretrained features
    - Cannot overfit encoder to training images yet

  Stage 2 (freeze_epochs+1 – end): Entire model trains (encoder unfrozen)
    - Lower LR (1e-4) for fine-tuning
    - Encoder adapts gently to corrosion domain
    - Decoder is already well-trained → stable gradient signal

This is standard practice for small datasets with pretrained backbones.

Usage:
  python train.py                    # two-stage, B0, sqrt alpha
  python train.py --freeze_epochs 0  # disable two-stage (single stage)
  python train.py --resume checkpoints/last.pth
"""

import argparse
import os
import sys
import logging
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from pathlib import Path
from tqdm import tqdm

logging.getLogger("albumentations.check_version").setLevel(logging.ERROR)
logging.getLogger("albumentations").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*blur_limit.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*pin_memory.*")

from models.segformer_eem_ffm import SegFormerEEMFFM
from datasets.corrosion_dataset import CorrosionDataset
from losses.focal_loss import FocalLoss, compute_class_weights
from utils import (AverageMeter, Timer,
                   compute_mIoU, compute_per_class_IoU, compute_accuracy,
                   batch_to_numpy, save_checkpoint, load_checkpoint,
                   print_metrics, vram_usage, CLASS_NAMES)

BACKBONE_CFG = {
    "b0": {"hf_name": "nvidia/mit-b0", "approx": "~3.5M"},
    "b1": {"hf_name": "nvidia/mit-b1", "approx": "~13.7M"},
    "b2": {"hf_name": "nvidia/mit-b2", "approx": "~24.7M"},
}


def get_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train_img",  default="data/final/train/images")
    p.add_argument("--train_mask", default="data/final/train/masks")
    p.add_argument("--val_img",    default="data/final/val/images")
    p.add_argument("--val_mask",   default="data/final/val/masks")
    # Model
    p.add_argument("--backbone",         default="b0",
                   choices=list(BACKBONE_CFG.keys()))
    p.add_argument("--num_classes",      type=int,  default=4)
    p.add_argument("--decoder_channels", type=int,  default=64)
    p.add_argument("--ffm_dropout",      type=float, default=0.15)
    p.add_argument("--pretrained",       action="store_true", default=True)
    # Training
    p.add_argument("--epochs",        type=int,   default=300)
    p.add_argument("--batch_size",    type=int,   default=16)
    p.add_argument("--weight_decay",  type=float, default=5e-4,
                   help="Higher weight decay (5e-4) helps generalisation")
    p.add_argument("--focal_gamma",   type=float, default=2.0)
    p.add_argument("--alpha_mode",    default="sqrt",
                   choices=["sqrt", "linear", "none"])
    p.add_argument("--workers",       type=int,   default=4)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--accum_steps",   type=int,   default=1)
    # Two-stage training
    p.add_argument("--freeze_epochs", type=int,   default=30,
                   help="Epochs to train with encoder FROZEN. "
                        "0 = disable two-stage. "
                        "Recommended: 20-40 for small datasets.")
    p.add_argument("--stage1_lr",     type=float, default=1e-3,
                   help="LR for Stage 1 (decoder only, encoder frozen)")
    p.add_argument("--stage2_lr",     type=float, default=1e-4,
                   help="LR for Stage 2 (full model fine-tuning)")
    # Early stopping
    p.add_argument("--patience",      type=int,   default=0,
                   help="0 = disabled (recommended)")
    # Checkpoints
    p.add_argument("--ckpt_dir", default="checkpoints")
    p.add_argument("--resume",   default=None)
    return p.parse_args()


def get_decoder_params(model):
    """Parameters NOT in the encoder — EEM, FFM, projections, head."""
    encoder_ids = set(id(p) for p in model.encoder.parameters())
    return [p for p in model.parameters() if id(p) not in encoder_ids]


def freeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False


def unfreeze_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = True


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def _train_one_epoch(model, loader, criterion, optimizer, scaler,
                     autocast_ctx, device, accum_steps, epoch, total_epochs):
    model.train()
    loss_meter = AverageMeter()
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(loader,
                desc=f"  Train [{epoch:03d}/{total_epochs}]",
                ncols=90, bar_format="{l_bar}{bar:28}{r_bar}",
                leave=False, file=sys.stdout)

    for step, (images, masks) in enumerate(pbar, 1):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        with autocast_ctx:
            logits = model(images)
            loss   = criterion(logits, masks) / accum_steps

        scaler.scale(loss).backward()

        if step % accum_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        loss_meter.update(loss.item() * accum_steps, images.size(0))
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", refresh=False)

    if len(loader) % accum_steps != 0:
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    pbar.close()
    return loss_meter.avg


def _val_one_epoch(model, loader, criterion, autocast_ctx,
                   device, num_classes, epoch, total_epochs):
    model.eval()
    loss_meter            = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(loader,
                desc=f"  Val   [{epoch:03d}/{total_epochs}]",
                ncols=90, bar_format="{l_bar}{bar:28}{r_bar}",
                leave=False, file=sys.stdout)

    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast_ctx:
                logits = model(images)
                loss   = criterion(logits, masks)

            loss_meter.update(loss.item(), images.size(0))
            p, l = batch_to_numpy(logits, masks)
            all_preds.append(p)
            all_labels.append(l)
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", refresh=False)

    pbar.close()

    all_preds  = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return (loss_meter.avg,
            compute_mIoU(all_preds, all_labels, num_classes),
            compute_accuracy(all_preds, all_labels),
            compute_per_class_IoU(all_preds, all_labels, num_classes))


def main():
    args = get_args()
    cfg  = BACKBONE_CFG[args.backbone]

    if torch.cuda.is_available():
        device, device_str = torch.device("cuda"), "cuda"
    else:
        device, device_str = torch.device("cpu"), "cpu"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device_str == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.ckpt_dir, exist_ok=True)

    use_amp      = (device_str == "cuda")
    scaler       = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = torch.amp.autocast(device_type=device_str, enabled=use_amp)

    W = 64
    print("═" * W)
    print("  SegFormer + EEM + FFM  —  Two-Stage Training")
    print("═" * W)
    print(f"  Backbone    : MiT-{args.backbone.upper()}  ({cfg['hf_name']})  {cfg['approx']}")
    print(f"  Device      : {device}")
    if device_str == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM        : {vram_gb:.1f} GB")
    print(f"  AMP (fp16)  : {use_amp}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Weight decay: {args.weight_decay}  (higher = less overfitting)")
    print(f"  Alpha mode  : {args.alpha_mode}")
    print(f"  FFM dropout : {args.ffm_dropout}")
    if args.freeze_epochs > 0:
        print(f"  ── Two-Stage Training ──")
        print(f"  Stage 1     : epochs 1–{args.freeze_epochs}  "
              f"(encoder FROZEN, decoder LR={args.stage1_lr})")
        print(f"  Stage 2     : epochs {args.freeze_epochs+1}–{args.epochs}  "
              f"(full model, LR={args.stage2_lr})")
    else:
        print(f"  Two-stage   : DISABLED  (single-stage training)")
    print("═" * W)

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading datasets ...")
    train_ds = CorrosionDataset(args.train_img, args.train_mask, split="train")
    val_ds   = CorrosionDataset(args.val_img,   args.val_mask,   split="val")
    print(f"  Train : {len(train_ds):>5} patches")
    print(f"  Val   : {len(val_ds):>5} patches")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=use_amp,
        drop_last=True, persistent_workers=(args.workers>0))
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=use_amp,
        persistent_workers=(args.workers>0))

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\n[2/4] Building model ...")
    model = SegFormerEEMFFM(
        num_classes      = args.num_classes,
        decoder_channels = args.decoder_channels,
        pretrained       = args.pretrained,
        backbone         = args.backbone,
        ffm_dropout      = args.ffm_dropout,
    ).to(device)

    # ── Loss ──────────────────────────────────────────────────────────────────
    print(f"\n[3/4] Setting up loss ...")
    if args.alpha_mode == "none":
        criterion = FocalLoss(gamma=args.focal_gamma)
        print(f"  FocalLoss(γ={args.focal_gamma}, alpha=None)")
    else:
        alpha     = compute_class_weights(train_ds, args.num_classes,
                                          mode=args.alpha_mode).to(device)
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=alpha)
        print(f"  FocalLoss(γ={args.focal_gamma}, alpha={args.alpha_mode})")

    # ── Stage 1 setup (encoder frozen) ────────────────────────────────────────
    start_epoch  = 1
    best_mIoU    = 0.0
    no_improve   = 0
    current_stage = 1

    if args.freeze_epochs > 0:
        freeze_encoder(model)
        print(f"\n  Stage 1: encoder FROZEN  "
              f"(trainable: {count_trainable(model):.2f} M params)")
        optimizer = torch.optim.AdamW(
            get_decoder_params(model),
            lr=args.stage1_lr,
            weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.freeze_epochs, eta_min=args.stage1_lr * 0.1)
    else:
        print(f"\n  Single-stage (trainable: {count_trainable(model):.2f} M)")
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.stage2_lr,
            weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ────────────────────────────────────────────────────────────────
    if args.resume and Path(args.resume).exists():
        print(f"\nResuming from {args.resume} ...")
        ckpt         = load_checkpoint(args.resume, model, optimizer,
                                       device=str(device))
        start_epoch  = ckpt.get("epoch", 0) + 1
        best_mIoU    = ckpt.get("best_mIoU", 0.0)
        no_improve   = ckpt.get("no_improve", 0)
        current_stage = ckpt.get("stage", 1)

        # Restore correct stage state
        if start_epoch > args.freeze_epochs and args.freeze_epochs > 0:
            unfreeze_encoder(model)
            current_stage = 2
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.stage2_lr,
                weight_decay=args.weight_decay)
            s2_epochs  = args.epochs - args.freeze_epochs
            done_in_s2 = start_epoch - args.freeze_epochs - 1
            scheduler  = CosineAnnealingLR(
                optimizer, T_max=s2_epochs, eta_min=1e-6)
            for _ in range(done_in_s2):
                scheduler.step()

        print(f"  Resumed epoch {start_epoch-1}  stage={current_stage}  "
              f"best={best_mIoU:.2f}%")

    print(f"\n[4/4] Training ...")
    print("─" * W)

    timer     = Timer()
    epoch_bar = tqdm(
        range(start_epoch, args.epochs + 1),
        desc="  Overall", ncols=90,
        bar_format="{l_bar}{bar:28}{r_bar}", unit="ep",
        file=sys.stdout, initial=start_epoch-1, total=args.epochs)

    for epoch in epoch_bar:

        # ── Stage transition ──────────────────────────────────────────────────
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1 \
                and current_stage == 1:
            current_stage = 2
            unfreeze_encoder(model)
            s2_epochs = args.epochs - args.freeze_epochs
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=args.stage2_lr,
                weight_decay=args.weight_decay)
            scheduler = CosineAnnealingLR(
                optimizer, T_max=s2_epochs, eta_min=1e-6)
            tqdm.write(f"\n  ── Stage 2: encoder UNFROZEN ──  "
                       f"(trainable: {count_trainable(model):.2f} M  "
                       f"LR={args.stage2_lr})")

        # ── Train / Val ───────────────────────────────────────────────────────
        train_loss = _train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            autocast_ctx, device, args.accum_steps, epoch, args.epochs)

        val_loss, mIoU, acc, per_cls = _val_one_epoch(
            model, val_loader, criterion, autocast_ctx,
            device, args.num_classes, epoch, args.epochs)

        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        done   = epoch - start_epoch + 1
        total  = args.epochs - start_epoch + 1

        stage_tag = f"[S{current_stage}]"
        epoch_bar.set_postfix(
            stage=current_stage,
            tr=f"{train_loss:.4f}", val=f"{val_loss:.4f}",
            mIoU=f"{mIoU:.2f}%", lr=f"{lr_now:.2e}",
            VRAM=vram_usage())

        tqdm.write(
            f"  {stage_tag} [{epoch:03d}/{args.epochs}] "
            f"loss {train_loss:.4f} | val {val_loss:.4f} | "
            f"mIoU {mIoU:.2f}% | acc {acc:.2f}% | "
            f"lr {lr_now:.2e} | {vram_usage()} | "
            f"ETA {timer.eta(done, total)}")

        if epoch % 10 == 0 or epoch == 1:
            row = "  " + " | ".join(
                f"{c[:3]}({n[:3]})={v:.1f}%"
                for c, n, v in zip(
                    ["Blk","Grn","Yel","Red"], CLASS_NAMES, per_cls))
            tqdm.write(row)

        # ── Checkpoints ────────────────────────────────────────────────────────
        state = {
            "epoch": epoch, "stage": current_stage,
            "backbone": args.backbone,
            "model_state": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_mIoU": best_mIoU, "mIoU": mIoU, "acc": acc,
            "per_cls": per_cls, "no_improve": no_improve,
        }
        save_checkpoint(state, os.path.join(args.ckpt_dir, "last.pth"))

        if mIoU > best_mIoU:
            best_mIoU          = mIoU
            no_improve         = 0
            state["best_mIoU"] = best_mIoU
            save_checkpoint(state, os.path.join(args.ckpt_dir, "best.pth"))
            tqdm.write(f"  ✔ Best mIoU: {best_mIoU:.2f}% → saved best.pth")
        else:
            no_improve += 1

        if args.patience > 0 and no_improve >= args.patience:
            tqdm.write(f"\n  ⏹ Early stop: no improvement for {no_improve} epochs")
            break

        if epoch % 50 == 0:
            save_checkpoint(state,
                os.path.join(args.ckpt_dir, f"epoch_{epoch:03d}.pth"))

    epoch_bar.close()

    print("\n" + "═" * W)
    print(f"  Training complete!")
    print(f"  Backbone   : SegFormer MiT-{args.backbone.upper()}")
    print(f"  Best mIoU  : {best_mIoU:.2f}%  (val, target ~67.69%)")
    print(f"  Duration   : {timer.elapsed()}")
    print(f"  Checkpoint : {args.ckpt_dir}/best.pth")
    print("═" * W)
    print("\n  Evaluate without TTA:  python evaluate.py")
    print("  Evaluate with TTA:     python evaluate.py --tta --batch_size 4")


if __name__ == "__main__":
    main()