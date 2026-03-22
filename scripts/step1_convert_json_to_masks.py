"""
Step 1 (Revised): Convert LabelMe JSON → integer masks + color-coded visualization masks.

Class mapping:
  0 = Background  →  Black        (0,   0,   0)
  1 = Fair        →  Green        (0,   200, 0)
  2 = Poor        →  Yellow       (0,   200, 200)  [BGR = yellow]
  3 = Severe      →  Red          (0,   0,   200)  [BGR = red]

Outputs per split:
  data/converted/{Train,Test}/images/          ← full-res images (PNG)
  data/converted/{Train,Test}/masks/           ← grayscale integer masks (0/1/2/3) — used for training
  data/converted/{Train,Test}/masks_colored/   ← color-coded PNG for visual inspection
"""

import os
import json
import shutil
import numpy as np
import cv2
from pathlib import Path


# ── Class → BGR color mapping ──────────────────────────────────────────────────
CLASS_COLORS_BGR = {
    0: (0,   0,   0),     # Background  → Black
    1: (0,   200, 0),     # Fair        → Green
    2: (0,   200, 200),   # Poor        → Yellow  (green+red in BGR = green+green = yellow)
    3: (0,   0,   200),   # Severe      → Red
}
# Note: OpenCV uses BGR not RGB
# BGR (0,200,200) = RGB (200,200,0) = Yellow ✓
# BGR (0,0,200)   = RGB (200,0,0)   = Red    ✓
# BGR (0,200,0)   = RGB (0,200,0)   = Green  ✓


def label_to_class(label: str) -> int:
    """Extract class index from label string."""
    label_lower = label.lower()
    if "fair"   in label_lower: return 1
    if "poor"   in label_lower: return 2
    if "severe" in label_lower: return 3
    return 0


def integer_mask_to_color(mask: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) uint8 integer mask (values 0-3)
    to a (H, W, 3) BGR color image for visualization.
    """
    color = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_idx, bgr in CLASS_COLORS_BGR.items():
        color[mask == cls_idx] = bgr
    return color


def make_side_by_side(img: np.ndarray, color_mask: np.ndarray,
                      alpha: float = 0.5) -> np.ndarray:
    """
    Create a 3-panel visualization:
      [Original image] | [Color mask] | [Overlay (image + mask blended)]
    All panels are resized to the same height for display.
    """
    TARGET_H = 400   # display height per panel

    def resize_to_h(im, h):
        scale = h / im.shape[0]
        w = int(im.shape[1] * scale)
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_NEAREST)

    img_r    = resize_to_h(img,        TARGET_H)
    mask_r   = resize_to_h(color_mask, TARGET_H)

    # Overlay: blend image with color mask (only on non-background pixels)
    img_float  = img_r.astype(np.float32)
    mask_float = mask_r.astype(np.float32)
    overlay    = np.where(mask_r.any(axis=2, keepdims=True),
                          img_float * (1 - alpha) + mask_float * alpha,
                          img_float).astype(np.uint8)

    # Add white divider lines
    divider = np.full((TARGET_H, 3, 3), 255, dtype=np.uint8)
    panel   = np.hstack([img_r, divider, mask_r, divider, overlay])
    return panel


def json_to_mask(json_path: Path, height: int, width: int) -> np.ndarray:
    """Parse LabelMe JSON → (H, W) integer class mask."""
    with open(json_path, "r") as f:
        data = json.load(f)

    mask = np.zeros((height, width), dtype=np.uint8)

    for shape in data["shapes"]:
        label      = shape["label"]
        shape_type = shape.get("shape_type", "polygon")
        points     = shape["points"]
        cls_idx    = label_to_class(label)

        if shape_type == "polygon" and len(points) >= 3:
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], color=int(cls_idx))

    return mask


def convert_split(orig_split_dir: Path, out_split_dir: Path,
                  save_previews: bool = True):
    """
    Convert one split (Train or Test).
    """
    src_img_dir  = orig_split_dir / "images"
    src_json_dir = orig_split_dir / "json"

    out_img_dir     = out_split_dir / "images"
    out_mask_dir    = out_split_dir / "masks"
    out_color_dir   = out_split_dir / "masks_colored"
    out_preview_dir = out_split_dir / "previews"   # side-by-side panels

    for d in [out_img_dir, out_mask_dir, out_color_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if save_previews:
        out_preview_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(src_json_dir.glob("*.json"))
    print(f"  Found {len(json_files)} JSON files in {orig_split_dir.name}")

    ok = skipped = 0
    for json_path in json_files:
        stem = json_path.stem

        # Find matching image
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
            candidate = src_img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"    [SKIP] No image for {stem}")
            skipped += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    [SKIP] Cannot read {img_path.name}")
            skipped += 1
            continue

        H, W = img.shape[:2]

        # Read JSON dimensions
        with open(json_path) as f:
            jdata = json.load(f)
        json_h = jdata.get("imageHeight", H)
        json_w = jdata.get("imageWidth",  W)

        if (json_h != H) or (json_w != W):
            print(f"    [WARN] {stem}: image {H}x{W} vs JSON {json_h}x{json_w} — using JSON dims")
            H, W = json_h, json_w
            img  = cv2.resize(img, (W, H))

        # Generate integer mask
        int_mask   = json_to_mask(json_path, H, W)
        color_mask = integer_mask_to_color(int_mask)

        # Save outputs
        cv2.imwrite(str(out_img_dir  / (stem + ".png")), img)
        cv2.imwrite(str(out_mask_dir / (stem + ".png")), int_mask)
        cv2.imwrite(str(out_color_dir / (stem + ".png")), color_mask)

        if save_previews:
            preview = make_side_by_side(img, color_mask)
            cv2.imwrite(str(out_preview_dir / (stem + "_preview.jpg")), preview,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

        ok += 1

    print(f"  → Converted: {ok} | Skipped: {skipped}")


def print_legend():
    print("\n  Color legend:")
    print("    ■ Black  → Background  (class 0)")
    print("    ■ Green  → Fair        (class 1)")
    print("    ■ Yellow → Poor        (class 2)")
    print("    ■ Red    → Severe      (class 3)")


def verify_class_distribution(mask_dir: Path):
    """Print pixel-level class distribution."""
    counts = np.zeros(4, dtype=np.int64)
    for p in sorted(mask_dir.glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            for c in range(4):
                counts[c] += int(np.sum(m == c))
    total = counts.sum()
    names = ["Background", "Fair", "Poor", "Severe"]
    print("\n  Class distribution (pixel %):")
    for c, name in enumerate(names):
        pct = 100.0 * counts[c] / total if total > 0 else 0.0
        bar = "█" * int(pct / 2)
        print(f"    Class {c} ({name:10s}): {pct:5.2f}%  {bar}")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ORIG_DIR = Path("original")
    OUT_DIR  = Path("data/converted")

    print_legend()

    for split in ["Train", "Test"]:
        print(f"\n{'='*55}")
        print(f"[{split}]")
        print(f"{'='*55}")
        convert_split(ORIG_DIR / split, OUT_DIR / split, save_previews=True)
        verify_class_distribution(OUT_DIR / split / "masks")

    print(f"\n{'='*55}")
    print("Step 1 complete.")
    print("Outputs:")
    print("  data/converted/{Train,Test}/images/         ← images")
    print("  data/converted/{Train,Test}/masks/          ← integer masks (for training)")
    print("  data/converted/{Train,Test}/masks_colored/  ← color-coded (for inspection)")
    print("  data/converted/{Train,Test}/previews/       ← side-by-side panels")