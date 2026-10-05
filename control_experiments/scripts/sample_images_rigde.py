from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    disk,
    binary_closing,
    binary_opening,
    binary_dilation
)
from skimage.measure import label, regionprops


# ============================================================
# 1. Settings
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "control_experiments"
    / "results"
    / "random_segmentation_check"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# Random sample size
N_SAMPLES = 50

# Reproducible random sampling
RANDOM_SEED = 111

# Segmentation parameters
DILATION_RADIUS = 8
MIN_COMPONENT_SIZE = 100

# Background blur
BACKGROUND_BLUR_SIGMA = 4

# QC thresholds
HIGH_COVERAGE = 0.90
LOW_COVERAGE = 0.10


# ============================================================
# 2. Find all images
# ============================================================

print("=" * 75)
print("RANDOM SEGMENTATION QUALITY CHECK")
print("=" * 75)

print(f"\nDataset directory:")
print(DATASET_DIR)

print(
    f"Directory exists: "
    f"{DATASET_DIR.exists()}"
)

if not DATASET_DIR.exists():
    raise FileNotFoundError(
        f"\nDataset directory does not exist:\n"
        f"{DATASET_DIR}"
    )


image_paths = [
    p
    for p in DATASET_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() == ".bmp"
]

print(
    f"\nTotal BMP images found: "
    f"{len(image_paths)}"
)

if len(image_paths) == 0:
    raise RuntimeError(
        "No BMP images were found."
    )


# ============================================================
# 3. Random sampling
# ============================================================

random.seed(RANDOM_SEED)

sample_size = min(
    N_SAMPLES,
    len(image_paths)
)

sample_paths = random.sample(
    image_paths,
    sample_size
)

print(
    f"Randomly sampled: "
    f"{sample_size} images"
)


# ============================================================
# 4. Segmentation function
# ============================================================

def segment_image(image_path):

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("L")

    gray = (
        np.asarray(image)
        .astype(np.float32)
        / 255.0
    )

    # --------------------------------------------------------
    # Gaussian smoothing
    # --------------------------------------------------------

    blurred = gaussian(
        gray,
        sigma=1
    )

    # --------------------------------------------------------
    # Otsu threshold
    # --------------------------------------------------------

    threshold = threshold_otsu(
        blurred
    )

    # Fingerprint assumed darker
    binary = (
        blurred < threshold
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    binary = binary_closing(
        binary,
        footprint=disk(2)
    )

    binary = binary_opening(
        binary,
        footprint=disk(2)
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    labeled = label(
        binary,
        connectivity=2
    )

    regions = regionprops(
        labeled
    )

    # Keep sufficiently large components
    keep_labels = [
        region.label
        for region in regions
        if region.area >= MIN_COMPONENT_SIZE
    ]

    # --------------------------------------------------------
    # Handle segmentation failure
    # --------------------------------------------------------

    if len(keep_labels) == 0:

        if len(regions) == 0:

            raise ValueError(
                "No connected components detected."
            )

        # Keep largest component as fallback
        largest = max(
            regions,
            key=lambda x: x.area
        )

        binary = (
            labeled == largest.label
        )

        fallback_used = True

    else:

        binary = np.isin(
            labeled,
            keep_labels
        )

        fallback_used = False

    # --------------------------------------------------------
    # Raw coverage
    # --------------------------------------------------------

    raw_coverage = binary.mean()

    # --------------------------------------------------------
    # Dilation
    # --------------------------------------------------------

    mask = binary_dilation(
        binary,
        footprint=disk(DILATION_RADIUS)
    )

    # --------------------------------------------------------
    # Final coverage
    # --------------------------------------------------------

    final_coverage = mask.mean()

    # --------------------------------------------------------
    # Ridge / foreground only
    # --------------------------------------------------------

    ridge_only = np.where(
        mask,
        gray,
        0
    )

    # --------------------------------------------------------
    # Background only
    # --------------------------------------------------------

    background_only = np.where(
        mask,
        0,
        gray
    )

    # Blur background to remove residual ridge texture
    background_only = gaussian(
        background_only,
        sigma=BACKGROUND_BLUR_SIGMA
    )

    return {
        "gray": gray,
        "mask": mask,
        "ridge_only": ridge_only,
        "background_only": background_only,
        "raw_coverage": raw_coverage,
        "final_coverage": final_coverage,
        "n_components": len(keep_labels),
        "fallback_used": fallback_used
    }


# ============================================================
# 5. Process sampled images
# ============================================================

results = []

successful = 0
failed = 0

print("\nProcessing images...\n")

for i, image_path in enumerate(
    sample_paths,
    start=1
):

    try:

        output = segment_image(
            image_path
        )

        successful += 1

        results.append({
            "filepath": str(image_path),
            "filename": image_path.name,
            "label": image_path.parent.name,

            "status": "SUCCESS",

            "raw_coverage":
                output["raw_coverage"],

            "final_coverage":
                output["final_coverage"],

            "n_components":
                output["n_components"],

            "fallback_used":
                output["fallback_used"],

            "high_coverage":
                output["final_coverage"]
                >= HIGH_COVERAGE,

            "low_coverage":
                output["final_coverage"]
                <= LOW_COVERAGE,

            "error": ""
        })

    except Exception as e:

        failed += 1

        results.append({
            "filepath": str(image_path),
            "filename": image_path.name,
            "label": image_path.parent.name,

            "status": "FAILED",

            "raw_coverage":
                np.nan,

            "final_coverage":
                np.nan,

            "n_components":
                np.nan,

            "fallback_used":
                False,

            "high_coverage":
                False,

            "low_coverage":
                False,

            "error": str(e)
        })

    if i % 10 == 0:
        print(
            f"Processed "
            f"{i}/{sample_size}"
        )


# ============================================================
# 6. Save results
# ============================================================

df = pd.DataFrame(
    results
)

csv_path = (
    OUTPUT_DIR
    / "random_segmentation_results.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# 7. Print summary
# ============================================================

print("\n")
print("=" * 75)
print("SEGMENTATION SUMMARY")
print("=" * 75)

print(
    f"Total dataset images : "
    f"{len(image_paths)}"
)

print(
    f"Random sample size   : "
    f"{sample_size}"
)

print(
    f"Successful           : "
    f"{successful}"
)

print(
    f"Failed               : "
    f"{failed}"
)

print(
    f"Success rate         : "
    f"{successful / sample_size:.2%}"
)

print(
    f"\nResults saved to:\n"
    f"{csv_path}"
)


# ============================================================
# 8. Coverage statistics
# ============================================================

valid_df = df[
    df["status"] == "SUCCESS"
]

if len(valid_df) > 0:

    high_count = (
        valid_df["high_coverage"]
        .sum()
    )

    low_count = (
        valid_df["low_coverage"]
        .sum()
    )

    fallback_count = (
        valid_df["fallback_used"]
        .sum()
    )

    print("\n")
    print("=" * 75)
    print("COVERAGE QUALITY")
    print("=" * 75)

    print(
        f"High coverage (>=90%) : "
        f"{high_count}"
    )

    print(
        f"Low coverage (<=10%)  : "
        f"{low_count}"
    )

    print(
        f"Fallback used         : "
        f"{fallback_count}"
    )

    print("\nRaw coverage:")
    print(
        f"  Mean   : "
        f"{valid_df['raw_coverage'].mean():.4f}"
    )

    print(
        f"  Median : "
        f"{valid_df['raw_coverage'].median():.4f}"
    )

    print(
        f"  Min    : "
        f"{valid_df['raw_coverage'].min():.4f}"
    )

    print(
        f"  Max    : "
        f"{valid_df['raw_coverage'].max():.4f}"
    )

    print("\nFinal coverage:")
    print(
        f"  Mean   : "
        f"{valid_df['final_coverage'].mean():.4f}"
    )

    print(
        f"  Median : "
        f"{valid_df['final_coverage'].median():.4f}"
    )

    print(
        f"  Min    : "
        f"{valid_df['final_coverage'].min():.4f}"
    )

    print(
        f"  Max    : "
        f"{valid_df['final_coverage'].max():.4f}"
    )


# ============================================================
# 9. Display random samples
# ============================================================

successful_paths = []

for image_path in sample_paths:

    row = df[
        df["filepath"] == str(image_path)
    ]

    if len(row) == 0:
        continue

    if row.iloc[0]["status"] == "SUCCESS":
        successful_paths.append(
            image_path
        )


# Limit visualization if necessary
DISPLAY_N = min(
    len(successful_paths),
    N_SAMPLES
)


# ------------------------------------------------------------
# Use the SAME random order as sampling
# ------------------------------------------------------------

display_paths = successful_paths[
    :DISPLAY_N
]


# ============================================================
# 10. Create visualization
# ============================================================

if DISPLAY_N > 0:

    print("\n")
    print("=" * 75)
    print("CREATING VISUALIZATION")
    print("=" * 75)

    print(
        f"Displaying "
        f"{DISPLAY_N} images"
    )

    fig, axes = plt.subplots(
        DISPLAY_N,
        4,
        figsize=(14, 3.2 * DISPLAY_N)
    )

    if DISPLAY_N == 1:
        axes = np.expand_dims(
            axes,
            axis=0
        )

    for row_idx, image_path in enumerate(
        display_paths
    ):

        output = segment_image(
            image_path
        )

        gray = output["gray"]
        mask = output["mask"]
        ridge_only = output["ridge_only"]
        background_only = output[
            "background_only"
        ]

        final_coverage = output[
            "final_coverage"
        ]

        raw_coverage = output[
            "raw_coverage"
        ]

        filename = image_path.name

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        axes[row_idx, 0].imshow(
            gray,
            cmap="gray"
        )

        axes[row_idx, 0].set_title(
            f"Original\n{filename}"
        )

        # ----------------------------------------------------
        # Mask
        # ----------------------------------------------------

        axes[row_idx, 1].imshow(
            mask,
            cmap="gray"
        )

        axes[row_idx, 1].set_title(
            f"Mask\n"
            f"raw={raw_coverage:.1%}, "
            f"final={final_coverage:.1%}"
        )

        # ----------------------------------------------------
        # Ridge only
        # ----------------------------------------------------

        axes[row_idx, 2].imshow(
            ridge_only,
            cmap="gray"
        )

        axes[row_idx, 2].set_title(
            "Ridge / Foreground Only"
        )

        # ----------------------------------------------------
        # Background only
        # ----------------------------------------------------

        axes[row_idx, 3].imshow(
            background_only,
            cmap="gray"
        )

        axes[row_idx, 3].set_title(
            "Background Only"
        )

        # ----------------------------------------------------
        # Remove axes
        # ----------------------------------------------------

        for col in range(4):

            axes[
                row_idx,
                col
            ].axis("off")

    plt.tight_layout()

    figure_path = (
        OUTPUT_DIR
        / "random_segmentation_examples.png"
    )

    plt.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(
        f"\nVisualization saved to:\n"
        f"{figure_path}"
    )


# ============================================================
# 11. Failed images
# ============================================================

failed_df = df[
    df["status"] == "FAILED"
]

if len(failed_df) > 0:

    failed_csv = (
        OUTPUT_DIR
        / "failed_images.csv"
    )

    failed_df.to_csv(
        failed_csv,
        index=False
    )

    print("\n")
    print("=" * 75)
    print("FAILED IMAGES")
    print("=" * 75)

    print(
        failed_df[
            [
                "filename",
                "label",
                "error"
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nFailed image list saved to:\n"
        f"{failed_csv}"
    )

else:

    print("\n")
    print("=" * 75)
    print("FAILED IMAGES")
    print("=" * 75)

    print(
        "No segmentation failures "
        "in the random sample."
    )