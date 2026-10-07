"""
2. Image/file metadata control
------------------------------
This experiment uses only seven low-level image/file properties:
    - width
    - height
    - file_size
    - brightness
    - contrast
    - background_intensity
    - image_quality

The purpose is to test whether blood-group labels are associated with
low-level image characteristics or acquisition/processing artefacts rather
than fingerprint ridge information.

This experiment therefore asks:

    "Can the blood-group label be predicted from low-level image/file
     properties alone, without using the fingerprint ridge pattern?"

The image_quality feature is used as a simple sharpness proxy based on the
variance of the Laplacian (or a gradient-based fallback when OpenCV is not
available). background_intensity is a simple border-pixel estimate and
should be interpreted as a background proxy rather than a segmentation
measure.
"""


from pathlib import Path

import pandas as pd
import numpy as np

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)

# ============================================================
# Project paths
# ============================================================

# Directory containing this script:
# .../compsci-760-group-12/control_experiments/scripts/

# control_experiments/
CONTROL_ROOT = Path(__file__).resolve().parents[1]

# Project root:
# .../compsci-760-group-12/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Existing Protocol B split files
SPLIT_DIR = CONTROL_ROOT / "splits"

# Original image dataset
DATA_DIR = PROJECT_ROOT / "data" / "datasets"

# Results
RESULT_DIR = CONTROL_ROOT / "metadata_image"

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# Settings
# ============================================================

SEEDS = [
    111,
    222,
    333,
    444,
    555
]

PROTOCOL = "B"

CLASS_NAMES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

FEATURE_COLUMNS = [
    "width",
    "height",
    "file_size",
    "brightness",
    "contrast",
    "background_intensity",
    "image_quality"
]


# ============================================================
# Helper: Resolve image filepath
# ============================================================

def resolve_image_path(filepath):
    filepath = Path(str(filepath))

    if filepath.is_absolute():
        return filepath

    return PROJECT_ROOT / filepath

# ============================================================
# Helper: Extract image/file metadata
# ============================================================

def extract_image_metadata(filepath):
    """
    Extract image/file-level metadata.

    Features:
        width
        height
        file_size
        brightness
        contrast
        background_intensity
        image_quality

    IMPORTANT:
    The complete fingerprint image is NOT supplied to the
    classifier. Only these summary-level features are used.
    """

    image_path = resolve_image_path(filepath)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image file not found:\n{image_path}"
        )

    # --------------------------------------------------------
    # File size
    # --------------------------------------------------------

    file_size = image_path.stat().st_size

    # --------------------------------------------------------
    # Open image
    # --------------------------------------------------------

    with Image.open(image_path) as image:

        width, height = image.size

        # Convert to grayscale
        gray = image.convert("L")

        pixels = np.asarray(
            gray,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness = float(
        pixels.mean()
    )

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = float(
        pixels.std()
    )

    # --------------------------------------------------------
    # Background intensity
    # --------------------------------------------------------
    #
    # Estimate background intensity using pixels near the
    # image borders.
    #
    # This is a simple background proxy, not a segmentation mask.
    # --------------------------------------------------------

    min_dimension = min(
        pixels.shape[0],
        pixels.shape[1]
    )

    border_width = max(
        1,
        min_dimension // 20
    )

    top = pixels[
        :border_width,
        :
    ]

    bottom = pixels[
        -border_width:,
        :
    ]

    left = pixels[
        :,
        :border_width
    ]

    right = pixels[
        :,
        -border_width:
    ]

    border_pixels = np.concatenate(
        [
            top.ravel(),
            bottom.ravel(),
            left.ravel(),
            right.ravel()
        ]
    )

    background_intensity = float(
        border_pixels.mean()
    )

    # --------------------------------------------------------
    # Image quality
    # --------------------------------------------------------
    #
    # Use variance of Laplacian as a simple sharpness proxy.
    #
    # OpenCV is preferred if available.
    # --------------------------------------------------------

    try:

        import cv2

        laplacian = cv2.Laplacian(
            pixels,
            cv2.CV_64F
        )

        image_quality = float(
            laplacian.var()
        )

    except ImportError:

        # Fallback gradient-based sharpness measure

        vertical_difference = np.diff(
            pixels,
            axis=0
        )

        horizontal_difference = np.diff(
            pixels,
            axis=1
        )

        image_quality = float(
            np.var(vertical_difference)
            +
            np.var(horizontal_difference)
        )

    return {
        "width": width,
        "height": height,
        "file_size": file_size,
        "brightness": brightness,
        "contrast": contrast,
        "background_intensity": background_intensity,
        "image_quality": image_quality
    }


# ============================================================
# Build image metadata table
# ============================================================

def build_metadata_table(split_files):
    """
    Extract image/file metadata from all unique filepaths
    appearing in the Protocol B split files.
    """

    # --------------------------------------------------------
    # Collect unique filepaths
    # --------------------------------------------------------

    all_filepaths = set()

    for split_file in split_files:

        df = pd.read_csv(
            split_file
        )

        if "filepath" not in df.columns:
            raise ValueError(
                f"'filepath' column missing in {split_file}"
            )

        all_filepaths.update(
            df["filepath"]
            .astype(str)
            .tolist()
        )

    all_filepaths = sorted(
        all_filepaths
    )

    print()
    print("=" * 70)
    print("EXTRACTING IMAGE/FILE METADATA")
    print("=" * 70)

    print(
        f"Unique images: {len(all_filepaths)}"
    )

    records = []

    total = len(all_filepaths)

    for index, filepath in enumerate(
        all_filepaths,
        start=1
    ):

        features = extract_image_metadata(
            filepath
        )

        record = {
            "filepath": filepath,
            **features
        }

        records.append(
            record
        )

        if index % 500 == 0 or index == total:

            print(
                f"Processed "
                f"{index}/{total}"
            )

    metadata_df = pd.DataFrame(
        records
    )

    return metadata_df


# ============================================================
# Validate metadata table
# ============================================================

def validate_metadata_table(metadata_df):

    required_columns = {
        "filepath",
        *FEATURE_COLUMNS
    }

    missing_columns = (
        required_columns
        - set(metadata_df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Missing metadata columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Check duplicate filepath
    # --------------------------------------------------------

    if metadata_df[
        "filepath"
    ].duplicated().any():

        raise ValueError(
            "Duplicate filepaths found "
            "in metadata table."
        )

    # --------------------------------------------------------
    # Check missing values
    # --------------------------------------------------------

    if metadata_df[
        FEATURE_COLUMNS
    ].isna().any().any():

        raise ValueError(
            "Missing metadata feature values found."
        )

    # --------------------------------------------------------
    # Check finite values
    # --------------------------------------------------------

    values = metadata_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():

        raise ValueError(
            "Non-finite metadata values found."
        )


# ============================================================
# Load one Protocol B split
# ============================================================

def load_protocol_b_split(seed):

    split_file = (
        SPLIT_DIR
        / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
    )

    if not split_file.is_file():

        raise FileNotFoundError(
            f"Protocol B split not found:\n"
            f"{split_file}"
        )

    split_df = pd.read_csv(
        split_file
    )

    required_columns = {
        "filepath",
        "label",
        "group_id",
        "split"
    }

    missing_columns = (
        required_columns
        - set(split_df.columns)
    )

    if missing_columns:

        raise ValueError(
            f"Missing columns in "
            f"{split_file}: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Check valid split names
    # --------------------------------------------------------

    valid_splits = {
        "train",
        "validation",
        "test"
    }

    observed_splits = set(
        split_df["split"].dropna().unique()
    )

    invalid_splits = (
        observed_splits
        - valid_splits
    )

    if invalid_splits:

        raise ValueError(
            f"Invalid split labels: "
            f"{invalid_splits}"
        )

    return split_df


# ============================================================
# Merge metadata with Protocol B split
# ============================================================

def prepare_dataset(
    split_df,
    metadata_df
):
    """
    Merge metadata features onto the existing Protocol B split.

    No new train/validation/test split is created.
    """

    data = split_df.merge(
        metadata_df,
        on="filepath",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Check missing metadata after merge
    # --------------------------------------------------------

    if data[
        FEATURE_COLUMNS
    ].isna().any().any():

        missing_rows = data[
            data[
                FEATURE_COLUMNS
            ].isna().any(axis=1)
        ]

        raise ValueError(
            "Some images have missing metadata.\n"
            f"Number of affected rows: "
            f"{len(missing_rows)}"
        )

    return data


# ============================================================
# Evaluate metadata-only classifier
# ============================================================

def evaluate_metadata_classifier(
    data,
    seed
):

    train_df = data[
        data["split"] == "train"
    ].copy()

    validation_df = data[
        data["split"] == "validation"
    ].copy()

    test_df = data[
        data["split"] == "test"
    ].copy()

    print()
    print(
        f"Train size:      {len(train_df)}"
    )

    print(
        f"Validation size: {len(validation_df)}"
    )

    print(
        f"Test size:       {len(test_df)}"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    X_test = test_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    y_train = train_df[
        "label"
    ].astype(str).to_numpy()

    y_test = test_df[
        "label"
    ].astype(str).to_numpy()

    # --------------------------------------------------------
    # Label encoding
    # --------------------------------------------------------
    #
    # Use the training labels to define the class encoding.
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_train_encoded = (
        label_encoder.fit_transform(
            y_train
        )
    )

    y_test_encoded = (
        label_encoder.transform(
            y_test
        )
    )

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------
    #
    # This is deliberately a simple classifier.
    #
    # The purpose is to test whether the metadata itself
    # contains class-related information, not to optimise
    # a high-capacity image classifier.
    # --------------------------------------------------------

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=seed,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train_encoded
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred_encoded = model.predict(
        X_test
    )

    # Convert back to original labels

    y_pred = (
        label_encoder.inverse_transform(
            y_pred_encoded
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    balanced_accuracy = (
        balanced_accuracy_score(
            y_test,
            y_pred
        )
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    feature_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_
    })

    feature_importance = (
        feature_importance
        .sort_values(
            "importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "train_size": len(train_df),
        "validation_size": len(validation_df),
        "test_size": len(test_df),
        "feature_importance": feature_importance
    }


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("IMAGE/FILE METADATA-ONLY CONTROL EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Find Protocol B split files
    # --------------------------------------------------------

    split_files = []

    for seed in SEEDS:

        split_file = (
            SPLIT_DIR
            / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
        )

        if not split_file.is_file():

            raise FileNotFoundError(
                f"Missing split file:\n"
                f"{split_file}"
            )

        split_files.append(
            split_file
        )

    # --------------------------------------------------------
    # Metadata feature cache
    # --------------------------------------------------------

    metadata_file = (
        RESULT_DIR
        / "image_metadata_features.csv"
    )

    if metadata_file.is_file():

        print()
        print(
            "Existing metadata feature file found."
        )

        print(
            f"Loading:\n{metadata_file}"
        )

        metadata_df = pd.read_csv(
            metadata_file
        )

    else:

        metadata_df = (
            build_metadata_table(
                split_files
            )
        )

        validate_metadata_table(
            metadata_df
        )

        metadata_df.to_csv(
            metadata_file,
            index=False
        )

        print()
        print(
            "Metadata features saved to:"
        )

        print(
            metadata_file
        )

    # --------------------------------------------------------
    # Validate metadata
    # --------------------------------------------------------

    validate_metadata_table(
        metadata_df
    )

    # --------------------------------------------------------
    # Run all seeds
    # --------------------------------------------------------

    results = []

    all_feature_importance = []

    for seed in SEEDS:

        print()
        print("=" * 70)
        print(
            f"Protocol {PROTOCOL} | Seed {seed}"
        )
        print("=" * 70)

        # Load existing Protocol B split

        split_df = (
            load_protocol_b_split(
                seed
            )
        )

        # Merge metadata with split

        data = prepare_dataset(
            split_df,
            metadata_df
        )

        # Evaluate

        metrics = (
            evaluate_metadata_classifier(
                data,
                seed
            )
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        result = {
            "experiment": "metadata_image_control",
            "protocol": PROTOCOL,
            "seed": seed,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": (
                metrics["balanced_accuracy"]
            ),
            "macro_f1": metrics["macro_f1"],
            "train_size": metrics["train_size"],
            "validation_size": (
                metrics["validation_size"]
            ),
            "test_size": metrics["test_size"],
            "num_features": len(
                FEATURE_COLUMNS
            )
        }

        results.append(
            result
        )

        # ----------------------------------------------------
        # Store feature importance
        # ----------------------------------------------------

        importance_df = (
            metrics["feature_importance"]
            .copy()
        )

        importance_df.insert(
            0,
            "seed",
            seed
        )

        all_feature_importance.append(
            importance_df
        )

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print()
        print(
            f"Accuracy:           "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"Balanced Accuracy:  "
            f"{metrics['balanced_accuracy']:.4f}"
        )

        print(
            f"Macro F1:            "
            f"{metrics['macro_f1']:.4f}"
        )

        print()
        print("Feature importance:")

        print(
            metrics[
                "feature_importance"
            ].to_string(
                index=False
            )
        )

    # ========================================================
    # Save results
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    result_file = (
        RESULT_DIR
        / "metadata_image_control.csv"
    )

    results_df.to_csv(
        result_file,
        index=False
    )

    # ========================================================
    # Save feature importance
    # ========================================================

    importance_all_df = pd.concat(
        all_feature_importance,
        ignore_index=True
    )

    importance_file = (
        RESULT_DIR
        / "metadata_feature_importance.csv"
    )

    importance_all_df.to_csv(
        importance_file,
        index=False
    )

    # ========================================================
    # Summary
    # ========================================================

    print()
    print("=" * 70)
    print("METADATA-ONLY SUMMARY")
    print("=" * 70)

    summary_metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1"
    ]

    for metric in summary_metrics:

        mean_value = (
            results_df[metric].mean()
        )

        std_value = (
            results_df[metric].std(
                ddof=1
            )
        )

        print(
            f"{metric:20s}"
            f"mean = {mean_value:.4f}, "
            f"SD = {std_value:.4f}"
        )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    summary_rows = []

    for metric in summary_metrics:

        summary_rows.append({
            "metric": metric,
            "mean": results_df[
                metric
            ].mean(),
            "sd": results_df[
                metric
            ].std(ddof=1)
        })

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_file = (
        RESULT_DIR
        / "metadata_image_control_summary.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False
    )

    # ========================================================
    # Final output
    # ========================================================

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(
        f"Metadata features:\n"
        f"{metadata_file}"
    )

    print(
        f"\nResults:\n"
        f"{result_file}"
    )

    print(
        f"\nFeature importance:\n"
        f"{importance_file}"
    )

    print(
        f"\nSummary:\n"
        f"{summary_file}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()

