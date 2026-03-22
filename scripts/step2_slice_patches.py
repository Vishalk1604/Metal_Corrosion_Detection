"""
step2_slice_patches.py — Sliding-window 512×512 patch extraction.

IMPORTANT FIX vs previous version:
────────────────────────────────────
The previous version calibrated SEPARATE thresholds for Train and Test,
forcing the test set to hit exactly 338 patches. This caused the test patches
to be heavily biased toward corrosion-dense regions:
    Our test: BG=36.6%  (should be ~67% like the paper)

The paper uses the SAME processing for all images. The 338/5121/570 counts
come from applying one consistent quality filter across all images.

CORRECT APPROACH:
  1. Calibrate BG_THRESHOLD on the Train split only (to hit ~5691 patches)
  2. Apply the SAME threshold to Test (we get however many we get)
  3. This gives representative test patches matching the paper's distribution

Expected results with this fix:
  Train + val : ~5691 patches (BG ≈ 64%)
  Test        : varies (paper got 338, we may get more — that is fine)
  Test BG %   : ~60-70%  (matches paper's 67.2%)
"""

import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
PATCH_SIZE = 512
STRIDE     = 256          # 0.5 overlap

# Calibration target for the TRAIN split only
TARGET_TRAIN_TOTAL = 5691   # 5121 train + 570 val  (paper target)
TOLERANCE          = 30

# ── Color map (BGR) for visualization ─────────────────────────────────────────
CLASS_COLORS_BGR = {
    0: (0,   0,   0),     # Background → Black
    1: (0,   200, 0),     # Fair       → Green
    2: (0,   200, 200),   # Poor       → Yellow
    3: (0,   0,   200),   # Severe     → Red
}


def integer_mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, bgr in CLASS_COLORS_BGR.items():
        color[mask == cls_idx] = bgr
    return color


def pad_to_minimum(img: np.ndarray, is_mask: bool) -> np.ndarray:
    H, W = img.shape[:2]
    ph = max(0, PATCH_SIZE - H)
    pw = max(0, PATCH_SIZE - W)
    if ph > 0 or pw > 0:
        border = cv2.BORDER_CONSTANT if is_mask else cv2.BORDER_REFLECT
        img = cv2.copyMakeBorder(img, 0, ph, 0, pw, border, value=0)
    return img


def collect_patch_stats(conv_split_dir: Path):
    """Collect bg_ratio for every candidate patch without writing anything."""
    img_dir  = conv_split_dir / "images"
    mask_dir = conv_split_dir / "masks"
    stats    = []

    for img_path in sorted(img_dir.glob("*.png")):
        mask_path = mask_dir / img_path.name
        if not mask_path.exists():
            continue

        img  = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue

        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        img  = pad_to_minimum(img,  is_mask=False)
        mask = pad_to_minimum(mask, is_mask=True)
        H, W = img.shape[:2]

        y = 0
        while y + PATCH_SIZE <= H:
            x = 0
            while x + PATCH_SIZE <= W:
                patch_mask = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                bg_ratio   = float(np.sum(patch_mask == 0)) / patch_mask.size
                stats.append((bg_ratio, img_path.stem, y, x))
                x += STRIDE
            y += STRIDE

    return stats


def calibrate_threshold(stats: list, target: int) -> float:
    """Binary-search for BG_THRESHOLD that yields ~target patches."""
    bg_ratios = sorted(s[0] for s in stats)
    total     = len(bg_ratios)

    print(f"  Total candidate patches (no filter): {total}")
    print(f"  Target patches                      : {target}")

    print("  Distribution:")
    for t_pct in [50, 70, 80, 90, 95, 99, 100]:
        n = sum(1 for r in bg_ratios if r <= t_pct/100)
        print(f"    bg <= {t_pct:3d}%  →  {n:5d} patches")

    if target >= total:
        print("  All patches kept (threshold = 1.0)")
        return 1.0

    lo, hi      = 0.0, 1.0
    best_thresh = hi
    for _ in range(60):
        mid   = (lo + hi) / 2.0
        count = sum(1 for r in bg_ratios if r <= mid)
        if count >= target:
            best_thresh = mid
            hi = mid
        else:
            lo = mid

    final = sum(1 for r in bg_ratios if r <= best_thresh)
    print(f"\n  Calibrated threshold = {best_thresh:.6f}  →  {final} patches")
    return best_thresh


def write_patches(conv_split_dir: Path, out_split_dir: Path,
                  stats: list, threshold: float) -> int:
    """Write image patches + integer masks + color masks."""
    img_dir  = conv_split_dir / "images"
    mask_dir = conv_split_dir / "masks"

    out_img   = out_split_dir / "images";        out_img.mkdir(parents=True, exist_ok=True)
    out_mask  = out_split_dir / "masks";         out_mask.mkdir(parents=True, exist_ok=True)
    out_color = out_split_dir / "masks_colored"; out_color.mkdir(parents=True, exist_ok=True)

    image_cache = {}
    kept = 0

    for (bg_ratio, stem, y, x) in stats:
        if bg_ratio > threshold:
            continue

        if stem not in image_cache:
            img  = cv2.imread(str(img_dir  / (stem + ".png")))
            mask = cv2.imread(str(mask_dir / (stem + ".png")),
                              cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                continue
            if img.shape[:2] != mask.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            img  = pad_to_minimum(img,  is_mask=False)
            mask = pad_to_minimum(mask, is_mask=True)
            image_cache[stem] = (img, mask)

        img, mask   = image_cache[stem]
        img_patch   = img [y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        mask_patch  = mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
        color_patch = integer_mask_to_color(mask_patch)

        fname = f"{stem}_{y:04d}_{x:04d}.png"
        cv2.imwrite(str(out_img   / fname), img_patch)
        cv2.imwrite(str(out_mask  / fname), mask_patch)
        cv2.imwrite(str(out_color / fname), color_patch)
        kept += 1

    return kept


def print_class_distribution(out_split_dir: Path, label: str):
    counts = np.zeros(4, dtype=np.int64)
    mask_dir = out_split_dir / "masks"
    for p in sorted(mask_dir.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            for c in range(4):
                counts[c] += int(np.sum(m == c))
    total  = counts.sum()
    names  = ["Background", "Fair",  "Poor",   "Severe"]
    colors = ["Black",      "Green", "Yellow", "Red"]
    print(f"\n  {label} class distribution (pixel %):")
    for c, (name, col) in enumerate(zip(names, colors)):
        pct = 100.0 * counts[c] / total if total > 0 else 0.0
        bar = "#" * int(pct / 2)
        print(f"    {col:7s} ({name:10s}): {pct:5.2f}%  {bar}")
    print(f"  Paper reference — Train: BG=64.5% Fair=21% Poor=10.9% Severe=2.4%")
    print(f"  Paper reference — Test:  BG=67.2% Fair=19.9% Poor=12.8% Severe=1.7%")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    CONV_DIR    = Path("data/converted")
    PATCHES_DIR = Path("data/patches")

    print("Color legend:  Black=BG | Green=Fair | Yellow=Poor | Red=Severe\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP A: Calibrate threshold on TRAIN only
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("[Train]  — calibrating threshold ...")
    print("=" * 60)

    train_stats = collect_patch_stats(CONV_DIR / "Train")
    threshold   = calibrate_threshold(train_stats, TARGET_TRAIN_TOTAL)

    print("\nWriting Train patches ...")
    n_train = write_patches(
        CONV_DIR / "Train", PATCHES_DIR / "Train", train_stats, threshold)
    print(f"  Written: {n_train} patches")
    print_class_distribution(PATCHES_DIR / "Train", "Train")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP B: Apply the SAME threshold to Test
    # This gives a representative test set (not forced to exactly 338)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[Test]  — applying SAME threshold as Train ...")
    print("=" * 60)

    test_stats = collect_patch_stats(CONV_DIR / "Test")

    # Show how many test patches we'd get at the train threshold
    n_at_train_thresh = sum(1 for (r, *_) in test_stats if r <= threshold)
    print(f"  Test patches at train threshold ({threshold:.4f}): {n_at_train_thresh}")
    print(f"  Paper reports 338 test patches.")
    print(f"  Using SAME threshold — keeping {n_at_train_thresh} patches")
    print(f"  (Different count from paper is OK — what matters is representative distribution)")

    print("\nWriting Test patches ...")
    n_test = write_patches(
        CONV_DIR / "Test", PATCHES_DIR / "Test", test_stats, threshold)
    print(f"  Written: {n_test} patches")
    print_class_distribution(PATCHES_DIR / "Test", "Test")

    print("\n" + "=" * 60)
    print("Step 2 complete → data/patches/")
    print(f"  Train patches: {n_train}")
    print(f"  Test patches : {n_test}")
    print("\nCritical check: Test BG% should now be ~60-70% (not 36%)")
    print("If Test BG is still low, increase TARGET_TRAIN_TOTAL to get")
    print("a more permissive threshold.")
    print("\nNext: run scripts/step3_make_splits.py")