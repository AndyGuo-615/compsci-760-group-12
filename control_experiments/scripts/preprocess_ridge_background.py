"""
Generate ridge-only and background-only versions of every fingerprint image.

Improved version:
    1. Convert to grayscale.
    2. Gaussian blur.
    3. Otsu threshold -> binary mask.
    4. Morphological closing + opening.
    5. Keep all large connected components (not just the largest).
    6. DILATE the mask outward to absorb boundary ridges.
    7. Ridge-only: keep pixels inside the (dilated) mask.
    8. Background-only: keep pixels outside the mask, then apply
       a strong Gaussian blur to remove any residual ridge texture.

Input:
    /Users/a75027/Desktop/compsci 760/datasets/

Output:
    data/datasets_processed/ridge_only/
    data/datasets_processed/background_only/

Run:
    python control_experiments/scripts/preprocess_ridge_background.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.filters import threshold_otsu, gaussian
from skimage.morphology import (
    binary_closing,
    binary_opening,
    binary_dilation,
    disk,
)
from skimage.measure import label as sk_label


# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CONTROL_ROOT.parent

DATA_ROOT = Path("/Users/a75027/Desktop/compsci 760/datasets")
OUTPUT_ROOT = PROJECT_ROOT / "data" / "datasets_processed"

RIDGE_OUTPUT = OUTPUT_ROOT / "ridge_only"
BG_OUTPUT = OUTPUT_ROOT / "background_only"

SPLIT_DIR = CONTROL_ROOT / "splits"


# ============================================================
# Segmentation parameters
# ============================================================

GAUSSIAN_SIGMA = 1.0        # for pre-threshold smoothing
MORPH_KERNEL = 2            # disk radius for closing/opening
DILATE_KERNEL = 8           # disk radius for expanding the ridge mask
MIN_COMPONENT_SIZE = 100    # ignore connected components smaller than this
BG_BLUR_SIGMA = 4.0         # extra blur on background image


# ============================================================
# Collect images from split files
# ============================================================

def collect_image_paths_from_splits():
    paths = set()
    for seed in [111, 222, 333, 444, 555]:
        csv = SPLIT_DIR / f"protocol_b_seed{seed}.csv"
        if not csv.exists():
            print(f"[WARN] Missing split file: {csv}")
            continue
        df = pd.read_csv(csv)
        paths.update(df["filepath"].tolist())
    return sorted(paths)


# ============================================================
# Segmentation
# ============================================================

def segment_ridge_mask(gray: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask where True = ridge (foreground) region.

    Improvements:
        - Keep ALL large connected components (not just the largest).
        - Dilate the mask outward to absorb boundary ridges.
    """
    # 1. Smooth
    blurred = gaussian(gray, sigma=GAUSSIAN_SIGMA, preserve_range=True)

    # 2. Otsu threshold
    try:
        thresh = threshold_otsu(blurred)
    except Exception:
        # Degenerate image: return all-True mask
        return np.ones_like(gray, dtype=bool)

    # Ridges are darker than the background
    binary = blurred < thresh

    # 3. Morphological cleanup
    selem = disk(MORPH_KERNEL)
    binary = binary_closing(binary, selem)
    binary = binary_opening(binary, selem)

    # 4. Keep ALL large connected components
    labeled = sk_label(binary, connectivity=2)
    if labeled.max() == 0:
        return np.ones_like(gray, dtype=bool)

    # Compute component sizes
    counts = np.bincount(labeled.ravel())
    counts[0] = 0  # ignore background label

    # Keep components larger than MIN_COMPONENT_SIZE
    keep_labels = np.where(counts >= MIN_COMPONENT_SIZE)[0]
    if len(keep_labels) == 0:
        # Fallback: keep the largest one
        keep_labels = [counts.argmax()]

    mask = np.isin(labeled, keep_labels)

    # 5. Dilate the mask outward to absorb boundary ridges
    mask = binary_dilation(mask, disk(DILATE_KERNEL))

    # Sanity check: if the mask covers >90% of the image, treat as failed
    if mask.mean() > 0.90:
        return np.ones_like(gray, dtype=bool)

    return mask


# ============================================================
# Process one image
# ============================================================

def process_one_image(src_path: Path, label: str, filename: str) -> bool:
    """
    Read one image, produce cleaner ridge-only and background-only images.
    """
    try:
        img = Image.open(src_path).convert("L")
    except Exception as e:
        print(f"[SKIP] Cannot read {src_path}: {e}")
        return False

    arr = np.asarray(img).astype(np.float32) / 255.0

    mask = segment_ridge_mask(arr)

    # Ridge-only: keep ridge pixels, black out background
    ridge_arr = np.where(mask, arr, 0.0)

    # Background-only: keep background pixels, black out ridge
    bg_arr = np.where(mask, 0.0, arr)

    # Extra: blur the background image to remove residual ridge texture
    # (Only blur where the background is non-zero, to avoid blurring
    #  the blacked-out ridge region into the background.)
    bg_nonzero = bg_arr > 0.01
    if bg_nonzero.any():
        blurred_bg = gaussian(bg_arr, sigma=BG_BLUR_SIGMA, preserve_range=True)
        # Keep original where background is zero (ridge region stays black)
        bg_arr = np.where(bg_nonzero, blurred_bg, 0.0)

    # Convert to uint8
    ridge_uint8 = np.clip(ridge_arr * 255, 0, 255).astype(np.uint8)
    bg_uint8 = np.clip(bg_arr * 255, 0, 255).astype(np.uint8)

    # Output dirs
    ridge_dir = RIDGE_OUTPUT / label
    bg_dir = BG_OUTPUT / label
    ridge_dir.mkdir(parents=True, exist_ok=True)
    bg_dir.mkdir(parents=True, exist_ok=True)

    # Save
    Image.fromarray(ridge_uint8).save(ridge_dir / filename, format="BMP")
    Image.fromarray(bg_uint8).save(bg_dir / filename, format="BMP")

    return True


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Ridge vs Background preprocessing (improved)")
    print("=" * 60)

    if not DATA_ROOT.exists():
        print(f"[ERROR] Data root not found: {DATA_ROOT}")
        sys.exit(1)

    print(f"Input:  {DATA_ROOT}")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"Dilate kernel: disk({DILATE_KERNEL})")
    print(f"Min component size: {MIN_COMPONENT_SIZE}")
    print(f"Background blur sigma: {BG_BLUR_SIGMA}")
    print()

    paths = collect_image_paths_from_splits()
    print(f"Found {len(paths)} unique image paths across all splits")
    print()

    success = 0
    failed = 0

    for i, rel_path in enumerate(paths, 1):
        prefix = "data/datasets/"
        if rel_path.startswith(prefix):
            rel_path = rel_path[len(prefix):]

        src_path = DATA_ROOT / rel_path

        if not src_path.exists():
            print(f"[MISS] {src_path}")
            failed += 1
            continue

        label = src_path.parent.name
        filename = src_path.name

        ok = process_one_image(src_path, label, filename)
        if ok:
            success += 1
        else:
            failed += 1

        if i % 200 == 0 or i == len(paths):
            print(f"  Processed {i}/{len(paths)} "
                  f"(ok={success}, failed={failed})")

    print()
    print("=" * 60)
    print(f"Done. Success: {success}, Failed: {failed}")
    print(f"Ridge images:      {RIDGE_OUTPUT}")
    print(f"Background images: {BG_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()