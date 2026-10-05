"""
Analyze mask coverage for the ridge/background segmentation.

This script uses the same segmentation procedure as
preprocess_ridge_background.py and checks how much of each
image is covered by the segmentation mask.

Important:
    coverage = mask.mean()

    coverage == 1.0
    usually indicates the fallback:
        np.ones_like(gray, dtype=bool)

Output:
    control_experiments/results/mask_coverage.csv
    control_experiments/results/mask_coverage_histogram.png
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

DATA_ROOT = Path(
    "/Users/a75027/Desktop/compsci 760/datasets"
)

SPLIT_DIR = CONTROL_ROOT / "splits"

RESULT_DIR = CONTROL_ROOT / "results"
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Parameters
# ============================================================

GAUSSIAN_SIGMA = 1.0
MORPH_KERNEL = 2
DILATE_KERNEL = 8
MIN_COMPONENT_SIZE = 100

# Same threshold used in your segmentation code
FAILURE_COVERAGE = 0.90


# ============================================================
# Collect image paths
# ============================================================

def collect_image_paths_from_splits():

    paths = set()

    for seed in [111, 222, 333, 444, 555]:

        csv = (
            SPLIT_DIR
            / f"protocol_b_seed{seed}.csv"
        )

        if not csv.exists():

            print(
                f"[WARN] Missing split file: {csv}"
            )

            continue

        df = pd.read_csv(csv)

        paths.update(
            df["filepath"].tolist()
        )

    return sorted(paths)


# ============================================================
# Segmentation
# ============================================================

def segment_ridge_mask(gray):

    """
    Same segmentation procedure as the original
    ridge/background preprocessing code.

    Returns:
        mask
        fallback
        reason
    """

    # --------------------------------------------------------
    # 1. Gaussian smoothing
    # --------------------------------------------------------

    blurred = gaussian(
        gray,
        sigma=GAUSSIAN_SIGMA,
        preserve_range=True
    )

    # --------------------------------------------------------
    # 2. Otsu threshold
    # --------------------------------------------------------

    try:

        thresh = threshold_otsu(
            blurred
        )

    except Exception:

        # Otsu failed
        return (
            np.ones_like(
                gray,
                dtype=bool
            ),
            True,
            "otsu_failed"
        )

    # Ridges are darker
    binary = blurred < thresh

    # --------------------------------------------------------
    # 3. Morphological cleanup
    # --------------------------------------------------------

    selem = disk(MORPH_KERNEL)

    binary = binary_closing(
        binary,
        selem
    )

    binary = binary_opening(
        binary,
        selem
    )

    # --------------------------------------------------------
    # 4. Connected components
    # --------------------------------------------------------

    labeled = sk_label(
        binary,
        connectivity=2
    )

    if labeled.max() == 0:

        return (
            np.ones_like(
                gray,
                dtype=bool
            ),
            True,
            "no_components"
        )

    # Component sizes
    counts = np.bincount(
        labeled.ravel()
    )

    counts[0] = 0

    # Keep large components
    keep_labels = np.where(
        counts >= MIN_COMPONENT_SIZE
    )[0]

    if len(keep_labels) == 0:

        # Same fallback as original code:
        # keep largest component
        keep_labels = [
            counts.argmax()
        ]

        mask = np.isin(
            labeled,
            keep_labels
        )

        mask = binary_dilation(
            mask,
            disk(DILATE_KERNEL)
        )

        coverage = mask.mean()

        if coverage > FAILURE_COVERAGE:

            return (
                np.ones_like(
                    gray,
                    dtype=bool
                ),
                True,
                "coverage_too_high_after_largest"
            )

        return (
            mask,
            False,
            "largest_component_fallback"
        )

    # --------------------------------------------------------
    # 5. Keep all large components
    # --------------------------------------------------------

    mask = np.isin(
        labeled,
        keep_labels
    )

    # --------------------------------------------------------
    # 6. Dilate
    # --------------------------------------------------------

    mask = binary_dilation(
        mask,
        disk(DILATE_KERNEL)
    )

    # --------------------------------------------------------
    # 7. Check coverage
    # --------------------------------------------------------

    coverage = mask.mean()

    if coverage > FAILURE_COVERAGE:

        return (
            np.ones_like(
                gray,
                dtype=bool
            ),
            True,
            "coverage_too_high"
        )

    return (
        mask,
        False,
        "normal"
    )


# ============================================================
# Analyze one image
# ============================================================

def analyze_image(src_path):

    try:

        img = Image.open(
            src_path
        ).convert("L")

    except Exception as e:

        return {
            "filepath": str(src_path),
            "status": "read_failed",
            "coverage": np.nan,
            "fallback": False,
            "reason": "read_failed",
            "width": np.nan,
            "height": np.nan,
        }

    arr = (
        np.asarray(img)
        .astype(np.float32)
        / 255.0
    )

    mask, fallback, reason = (
        segment_ridge_mask(arr)
    )

    coverage = mask.mean()

    height, width = mask.shape

    return {
        "filepath": str(src_path),
        "status": "ok",
        "coverage": coverage,
        "coverage_percent": coverage * 100,
        "fallback": fallback,
        "reason": reason,
        "width": width,
        "height": height,
        "mask_pixels": int(mask.sum()),
        "total_pixels": int(mask.size),
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("MASK COVERAGE ANALYSIS")
    print("=" * 70)

    paths = (
        collect_image_paths_from_splits()
    )

    print(
        f"Unique images: {len(paths)}"
    )

    print()

    results = []

    for i, rel_path in enumerate(
        paths,
        1
    ):

        prefix = "data/datasets/"

        if rel_path.startswith(prefix):

            rel_path = (
                rel_path[len(prefix):]
            )

        src_path = (
            DATA_ROOT
            / rel_path
        )

        if not src_path.exists():

            print(
                f"[MISS] {src_path}"
            )

            continue

        result = analyze_image(
            src_path
        )

        # Keep relative filepath
        result["relative_filepath"] = (
            rel_path
        )

        # Extract label
        result["label"] = (
            src_path.parent.name
        )

        results.append(result)

        if (
            i % 500 == 0
            or i == len(paths)
        ):

            print(
                f"Processed "
                f"{i}/{len(paths)}"
            )

    df = pd.DataFrame(results)

    # ========================================================
    # Save individual results
    # ========================================================

    output_csv = (
        RESULT_DIR
        / "mask_coverage.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    valid = df[
        df["status"] == "ok"
    ]

    coverage = valid[
        "coverage"
    ]

    print()
    print("=" * 70)
    print("OVERALL MASK COVERAGE")
    print("=" * 70)

    print(
        f"Images analyzed: {len(valid)}"
    )

    print(
        f"Mean coverage:   "
        f"{coverage.mean():.4f} "
        f"({coverage.mean()*100:.2f}%)"
    )

    print(
        f"Median coverage: "
        f"{coverage.median():.4f} "
        f"({coverage.median()*100:.2f}%)"
    )

    print(
        f"Minimum:         "
        f"{coverage.min():.4f} "
        f"({coverage.min()*100:.2f}%)"
    )

    print(
        f"Maximum:         "
        f"{coverage.max():.4f} "
        f"({coverage.max()*100:.2f}%)"
    )

    print(
        f"Std:             "
        f"{coverage.std():.4f}"
    )

    # ========================================================
    # Fallback statistics
    # ========================================================

    fallback_count = (
        valid["fallback"]
        .sum()
    )

    fallback_rate = (
        fallback_count
        / len(valid)
    )

    print()
    print("=" * 70)
    print("FALLBACK STATISTICS")
    print("=" * 70)

    print(
        f"Fallback images: "
        f"{fallback_count}"
    )

    print(
        f"Fallback rate:   "
        f"{fallback_rate:.4f} "
        f"({fallback_rate*100:.2f}%)"
    )

    # ========================================================
    # Coverage > 90%
    # ========================================================

    high_coverage = valid[
        valid["coverage"] > 0.90
    ]

    print()
    print(
        f"Coverage > 90%: "
        f"{len(high_coverage)} "
        f"({len(high_coverage)/len(valid)*100:.2f}%)"
    )

    # ========================================================
    # Coverage == 100%
    # ========================================================

    full_coverage = valid[
        valid["coverage"] >= 0.999999
    ]

    print(
        f"Coverage = 100%: "
        f"{len(full_coverage)} "
        f"({len(full_coverage)/len(valid)*100:.2f}%)"
    )

    # ========================================================
    # Coverage < 10%
    # ========================================================

    low_coverage = valid[
        valid["coverage"] < 0.10
    ]

    print(
        f"Coverage < 10%: "
        f"{len(low_coverage)} "
        f"({len(low_coverage)/len(valid)*100:.2f}%)"
    )

    # ========================================================
    # Reasons
    # ========================================================

    print()
    print("=" * 70)
    print("SEGMENTATION REASONS")
    print("=" * 70)

    print(
        df["reason"]
        .value_counts()
        .to_string()
    )

    # ========================================================
    # Coverage by blood group
    # ========================================================

    print()
    print("=" * 70)
    print("COVERAGE BY CLASS")
    print("=" * 70)

    class_summary = (
        valid
        .groupby("label")["coverage"]
        .agg([
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ])
        .sort_index()
    )

    print(
        class_summary.to_string()
    )

    # ========================================================
    # Histogram
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        coverage,
        bins=50
    )

    plt.axvline(
        0.90,
        linestyle="--",
        linewidth=2,
        label="90% threshold"
    )

    plt.xlabel(
        "Mask coverage"
    )

    plt.ylabel(
        "Number of images"
    )

    plt.title(
        "Distribution of Fingerprint Mask Coverage"
    )

    plt.legend()

    plt.tight_layout()

    histogram_path = (
        RESULT_DIR
        / "mask_coverage_histogram.png"
    )

    plt.savefig(
        histogram_path,
        dpi=300
    )

    plt.close()

    print()
    print("=" * 70)
    print("FILES")
    print("=" * 70)

    print(
        f"CSV:       {output_csv}"
    )

    print(
        f"Histogram: {histogram_path}"
    )


if __name__ == "__main__":
    main()