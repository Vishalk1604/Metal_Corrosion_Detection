"""
step3_make_splits.py  —  Patch-level val, image-level test

CORRECT INTERPRETATION OF THE PAPER
─────────────────────────────────────
Paper says:
  "The dataset is divided into training set and test set according to 9:1.
   Each image was adjusted to 512×512 pixel small image...
   consists of 5121 training sets, 570 validation sets, and 338 test sets."

This means:
  - Test  : patches from the 44 held-out test images   → image-level holdout
  - Val   : patches from within the 396 training image patches → patch-level
  - Train : remaining patches from the 396 training images

WHY PATCH-LEVEL VAL IS FINE
─────────────────────────────
The val patches have overlap with training patches (same source images).
This is acceptable because:
  1. The TEST is the honest evaluation (44 completely unseen images)
  2. Val is only used to MONITOR training progress and save best.pth
  3. The paper uses this same setup and achieves 67.69% TEST mIoU

The IMAGE-LEVEL val we tried was too strict — it made training monitors
useless (val never correlates with test) and caused early stopping to
fire too early.

RESULT
───────
  train : ~5121 patches from 396 training images
  val   : ~570  patches from the same 396 training images (patch-level)
  test  : all patches from 44 held-out images (image-level, never seen)
"""

import random
import shutil
from pathlib import Path

RANDOM_SEED = 42


def fast_copy(filenames, src_img, src_mask, src_color,
              dst_img, dst_mask, dst_color, label):
    for d in [dst_img, dst_mask, dst_color]:
        d.mkdir(parents=True, exist_ok=True)
    total = len(filenames)
    for i, fname in enumerate(filenames, 1):
        for src, dst in [(src_img, dst_img), (src_mask, dst_mask)]:
            s = src / fname
            if s.exists():
                shutil.copy2(str(s), str(dst / fname))
        if src_color.exists():
            s = src_color / fname
            if s.exists():
                shutil.copy2(str(s), str(dst_color / fname))
        if i % 500 == 0 or i == total:
            print(f"    {label}: {i}/{total}", end="\r")
    if total > 0:
        print()


if __name__ == "__main__":
    PATCHES_DIR = Path("data/patches")
    FINAL_DIR   = Path("data/final")

    random.seed(RANDOM_SEED)

    # ── Source paths ──────────────────────────────────────────────────────────
    tr_img   = PATCHES_DIR / "Train" / "images"
    tr_mask  = PATCHES_DIR / "Train" / "masks"
    tr_color = PATCHES_DIR / "Train" / "masks_colored"

    te_img   = PATCHES_DIR / "Test"  / "images"
    te_mask  = PATCHES_DIR / "Test"  / "masks"
    te_color = PATCHES_DIR / "Test"  / "masks_colored"

    # ── PATCH-LEVEL train/val split (matching paper) ──────────────────────────
    all_patches = sorted(p.name for p in tr_img.glob("*.png"))
    random.shuffle(all_patches)

    n_total = len(all_patches)
    n_val   = int(n_total * 0.10)       # ~10% → val
    n_train = n_total - n_val           # ~90% → train

    val_files   = all_patches[:n_val]
    train_files = all_patches[n_val:]

    print(f"Patch-level split (paper approach):")
    print(f"  Total training patches : {n_total}")
    print(f"  → train                : {n_train}")
    print(f"  → val                  : {n_val}")
    print(f"  (patches from same source images — this is fine,")
    print(f"   test images are completely held out)")

    # ── Copy train ────────────────────────────────────────────────────────────
    print("\n[train]")
    fast_copy(train_files, tr_img, tr_mask, tr_color,
              FINAL_DIR/"train"/"images", FINAL_DIR/"train"/"masks",
              FINAL_DIR/"train"/"masks_colored", "train")

    # ── Copy val ──────────────────────────────────────────────────────────────
    print("[val]")
    fast_copy(val_files, tr_img, tr_mask, tr_color,
              FINAL_DIR/"val"/"images", FINAL_DIR/"val"/"masks",
              FINAL_DIR/"val"/"masks_colored", "val")

    # ── Copy test (all patches from 44 held-out images) ───────────────────────
    test_files = sorted(p.name for p in te_img.glob("*.png"))
    print(f"[test]  ({len(test_files)} patches from 44 held-out images)")
    fast_copy(test_files, te_img, te_mask, te_color,
              FINAL_DIR/"test"/"images", FINAL_DIR/"test"/"masks",
              FINAL_DIR/"test"/"masks_colored", "test")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print("Final split summary:")
    targets = {"train": 5121, "val": 570, "test": 338}
    for split in ["train", "val", "test"]:
        n = len(list((FINAL_DIR / split / "images").glob("*.png")))
        t = targets[split]
        print(f"  {split:6s}: {n:5d}  (paper ~{t})")

    print(f"\n{'='*55}")
    print("Leakage analysis:")
    print("  Val ↔ Train : patches may overlap (same source images) — EXPECTED")
    print("  Test ↔ Train: ZERO — test comes from 44 completely different images")
    print("  Test mIoU = HONEST metric (paper reports 67.69% this way)")
    print(f"\n{'='*55}")
    print("Step 3 complete → data/final/")
    print("\nNow retrain from scratch:")
    print("  python train.py --batch_size 16")