"""
Automatic segmentation quality-control (QC) for fingerprint masks.

Purpose:
    Automatically detect suspicious segmentation results before
    running the ridge-vs-background control experiment.

Checks:
    1. Mask coverage
    2. Boundary contact ratio
    3. Number of connected components
    4. Very small / very large masks
    5. Segmentation fallback conditions

Outputs:
    control_experiments/results/mask_qc.csv
    control_experiments/results/mask_qc_summary.txt
    control_experiments/results/qc_examples/

No original images are modified.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

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

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "datasets"
)

SPLIT_DIR = CONTROL_ROOT / "splits"

RESULT_DIR = (
    CONTROL_ROOT / "results"
)

QC_EXAMPLE_DIR = (
    RESULT_DIR / "qc_examples"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

QC_EXAMPLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# Segmentation parameters
# ============================================================

GAUSSIAN_SIGMA = 1.0
MORPH_KERNEL = 2
DILATE_KERNEL = 6
MIN_COMPONENT_SIZE = 100


# ============================================================
# QC thresholds
# ============================================================

# Mask coverage
HIGH_COVERAGE = 0.90
LOW_COVERAGE = 0.10

# Boundary contact
MAX_BOUNDARY_CONTACT = 0.80

# Number of components
MAX_COMPONENTS_WARNING = 100

# Save examples
MAX_EXAMPLES_PER_CATEGORY = 20


# ============================================================
# Collect images
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
                f"[WARN] Missing split: {csv}"
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
    Same basic segmentation method as the original code.

    IMPORTANT:
        This version does NOT replace suspicious masks
        with an all-True mask.

    Instead it returns the raw mask and diagnostic
    information.
    """

    # --------------------------------------------------------
    # Gaussian smoothing
    # --------------------------------------------------------

    blurred = gaussian(
        gray,
        sigma=GAUSSIAN_SIGMA,
        preserve_range=True
    )

    # --------------------------------------------------------
    # Otsu
    # --------------------------------------------------------

    try:

        threshold = threshold_otsu(
            blurred
        )

    except Exception:

        return (
            None,
            {
                "segmentation_failed": True,
                "failure_reason": "otsu_failed",
            }
        )

    binary = (
        blurred < threshold
    )

    # --------------------------------------------------------
    # Morphological cleanup
    # --------------------------------------------------------

    selem = disk(
        MORPH_KERNEL
    )

    binary = binary_closing(
        binary,
        selem
    )

    binary = binary_opening(
        binary,
        selem
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    labeled = sk_label(
        binary,
        connectivity=2
    )

    n_components = labeled.max()

    if n_components == 0:

        return (
            None,
            {
                "segmentation_failed": True,
                "failure_reason": "no_components",
            }
        )

    counts = np.bincount(
        labeled.ravel()
    )

    counts[0] = 0

    keep_labels = np.where(
        counts >= MIN_COMPONENT_SIZE
    )[0]

    if len(keep_labels) == 0:

        # Keep largest component as diagnostic
        largest = counts.argmax()

        mask = (
            labeled == largest
        )

        component_fallback = True

    else:

        mask = np.isin(
            labeled,
            keep_labels
        )

        component_fallback = False

    # --------------------------------------------------------
    # Dilation
    # --------------------------------------------------------

    mask = binary_dilation(
        mask,
        disk(DILATE_KERNEL)
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    coverage = mask.mean()

    # Boundary pixels
    boundary = np.zeros_like(
        mask,
        dtype=bool
    )

    boundary[0, :] = True
    boundary[-1, :] = True
    boundary[:, 0] = True
    boundary[:, -1] = True

    boundary_pixels = (
        mask & boundary
    ).sum()

    total_boundary = (
        boundary.sum()
    )

    boundary_contact = (
        boundary_pixels
        / total_boundary
    )

    return (
        mask,
        {
            "segmentation_failed": False,
            "failure_reason": "",
            "coverage": coverage,
            "n_components": len(
                keep_labels
            ),
            "boundary_contact": boundary_contact,
            "component_fallback":
                component_fallback,
        }
    )


# ============================================================
# Create QC visualization
# ============================================================

def save_qc_example(
    image,
    mask,
    filepath,
    category,
):

    filename = Path(
        filepath
    ).stem

    output_dir = (
        QC_EXAMPLE_DIR
        / category
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original = (
        image * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Mask
    # --------------------------------------------------------

    mask_img = (
        mask.astype(np.uint8)
        * 255
    )

    # --------------------------------------------------------
    # Ridge-only
    # --------------------------------------------------------

    ridge = np.where(
        mask,
        image,
        0
    )

    ridge_img = (
        ridge * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Background-only
    # --------------------------------------------------------

    background = np.where(
        mask,
        0,
        image
    )

    background_img = (
        background * 255
    ).astype(np.uint8)

    # --------------------------------------------------------
    # Combine horizontally
    # --------------------------------------------------------

    h, w = image.shape

    combined = np.zeros(
        (h, w * 4),
        dtype=np.uint8
    )

    combined[:, 0:w] = original
    combined[:, w:2*w] = mask_img
    combined[:, 2*w:3*w] = ridge_img
    combined[:, 3*w:4*w] = background_img

    output_path = (
        output_dir
        / f"{filename}_qc.BMP"
    )

    Image.fromarray(
        combined
    ).save(
        output_path,
        format="BMP"
    )


# ============================================================
# Determine QC status
# ============================================================

def determine_status(info):

    if info["segmentation_failed"]:

        return "FAIL"

    coverage = (
        info["coverage"]
    )

    boundary_contact = (
        info["boundary_contact"]
    )

    n_components = (
        info["n_components"]
    )

    # --------------------------------------------------------
    # Strong failure
    # --------------------------------------------------------

    if coverage >= HIGH_COVERAGE:

        return "FAIL"

    if coverage <= LOW_COVERAGE:

        return "FAIL"

    # --------------------------------------------------------
    # Boundary suspicious
    # --------------------------------------------------------

    if boundary_contact >= MAX_BOUNDARY_CONTACT:

        return "WARNING"

    # --------------------------------------------------------
    # Too many components
    # --------------------------------------------------------

    if n_components >= MAX_COMPONENTS_WARNING:

        return "WARNING"

    # --------------------------------------------------------
    # Component fallback
    # --------------------------------------------------------

    if info["component_fallback"]:

        return "WARNING"

    return "PASS"


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print(
        "AUTOMATIC FINGERPRINT SEGMENTATION QC"
    )
    print("=" * 70)

    paths = (
        collect_image_paths_from_splits()
    )

    print(
        f"Images found: {len(paths)}"
    )

    print()

    results = []

    category_counter = {
        "high_coverage": 0,
        "low_coverage": 0,
        "boundary": 0,
        "components": 0,
        "failure": 0,
    }

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
            DATA_ROOT / rel_path
        )

        if not src_path.exists():

            continue

        try:

            img = Image.open(
                src_path
            ).convert("L")

        except Exception:

            continue

        arr = (
            np.asarray(img)
            .astype(np.float32)
            / 255.0
        )

        mask, info = (
            segment_ridge_mask(arr)
        )

        # ----------------------------------------------------
        # Segmentation failure
        # ----------------------------------------------------

        if mask is None:

            row = {
                "filepath": rel_path,
                "label": src_path.parent.name,
                "status": "FAIL",
                "coverage": np.nan,
                "coverage_percent": np.nan,
                "n_components": np.nan,
                "boundary_contact": np.nan,
                "component_fallback": False,
                "failure_reason":
                    info["failure_reason"],
            }

            category_counter[
                "failure"
            ] += 1

            results.append(row)

            continue

        # ----------------------------------------------------
        # QC status
        # ----------------------------------------------------

        status = determine_status(
            info
        )

        coverage = (
            info["coverage"]
        )

        boundary_contact = (
            info["boundary_contact"]
        )

        n_components = (
            info["n_components"]
        )

        # ----------------------------------------------------
        # Flags
        # ----------------------------------------------------

        high_coverage = (
            coverage >= HIGH_COVERAGE
        )

        low_coverage = (
            coverage <= LOW_COVERAGE
        )

        boundary_flag = (
            boundary_contact
            >= MAX_BOUNDARY_CONTACT
        )

        component_flag = (
            n_components
            >= MAX_COMPONENTS_WARNING
        )

        if high_coverage:

            category_counter[
                "high_coverage"
            ] += 1

        if low_coverage:

            category_counter[
                "low_coverage"
            ] += 1

        if boundary_flag:

            category_counter[
                "boundary"
            ] += 1

        if component_flag:

            category_counter[
                "components"
            ] += 1

        row = {
            "filepath": rel_path,
            "label": src_path.parent.name,
            "status": status,
            "coverage": coverage,
            "coverage_percent":
                coverage * 100,
            "n_components":
                n_components,
            "boundary_contact":
                boundary_contact,
            "boundary_contact_percent":
                boundary_contact * 100,
            "component_fallback":
                info["component_fallback"],
            "high_coverage":
                high_coverage,
            "low_coverage":
                low_coverage,
            "boundary_flag":
                boundary_flag,
            "component_flag":
                component_flag,
            "failure_reason":
                info["failure_reason"],
        }

        results.append(row)

        # ----------------------------------------------------
        # Save suspicious examples
        # ----------------------------------------------------

        if status != "PASS":

            if high_coverage:

                category = (
                    "high_coverage"
                )

            elif low_coverage:

                category = (
                    "low_coverage"
                )

            elif boundary_flag:

                category = (
                    "boundary"
                )

            elif component_flag:

                category = (
                    "many_components"
                )

            else:

                category = (
                    "other"
                )

            category_dir = (
                QC_EXAMPLE_DIR
                / category
            )

            existing = (
                list(
                    category_dir.glob(
                        "*.BMP"
                    )
                )
                if category_dir.exists()
                else []
            )

            if (
                len(existing)
                < MAX_EXAMPLES_PER_CATEGORY
            ):

                save_qc_example(
                    arr,
                    mask,
                    rel_path,
                    category
                )

        if (
            i % 500 == 0
            or i == len(paths)
        ):

            print(
                f"Processed "
                f"{i}/{len(paths)}"
            )

    # ========================================================
    # DataFrame
    # ========================================================

    df = pd.DataFrame(
        results
    )

    output_csv = (
        RESULT_DIR
        / "mask_qc.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("QC SUMMARY")
    print("=" * 70)

    print(
        f"Total analyzed: {len(df)}"
    )

    print()

    print(
        df["status"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"High coverage >= 90%: "
        f"{category_counter['high_coverage']}"
    )

    print(
        f"Low coverage <= 10%: "
        f"{category_counter['low_coverage']}"
    )

    print(
        f"High boundary contact: "
        f"{category_counter['boundary']}"
    )

    print(
        f"Many components: "
        f"{category_counter['components']}"
    )

    print(
        f"Segmentation failures: "
        f"{category_counter['failure']}"
    )

    # ========================================================
    # Coverage statistics
    # ========================================================

    valid = df[
        df["coverage"].notna()
    ]

    if len(valid) > 0:

        print()

        print(
            "Coverage statistics:"
        )

        print(
            f"Mean:   "
            f"{valid['coverage'].mean()*100:.2f}%"
        )

        print(
            f"Median: "
            f"{valid['coverage'].median()*100:.2f}%"
        )

        print(
            f"Min:    "
            f"{valid['coverage'].min()*100:.2f}%"
        )

        print(
            f"Max:    "
            f"{valid['coverage'].max()*100:.2f}%"
        )

    # ========================================================
    # Problem images
    # ========================================================

    problem = df[
        df["status"] != "PASS"
    ]

    problem_csv = (
        RESULT_DIR
        / "mask_qc_problem_images.csv"
    )

    problem.to_csv(
        problem_csv,
        index=False
    )

    # ========================================================
    # Final
    # ========================================================

    print()
    print("=" * 70)

    print(
        f"Full QC report:"
        f"\n{output_csv}"
    )

    print(
        f"\nProblem images:"
        f"\n{problem_csv}"
    )

    print(
        f"\nQC examples:"
        f"\n{QC_EXAMPLE_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()