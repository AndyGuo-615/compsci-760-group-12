from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
SPLIT_DIR = PROJECT_ROOT / "splits"

# Five fixed random seeds
SEEDS = [111, 222, 333, 444, 555]

# Expected blood-group classes
EXPECTED_CLASSES = [
    "A+",
    "A-",
    "AB+",
    "AB-",
    "B+",
    "B-",
    "O+",
    "O-"
]

# Create output directory if it does not already exist
SPLIT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Build a manifest of all images
# --------------------------------------------------

records = []

for class_name in EXPECTED_CLASSES:
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Class folder not found: {class_dir}"
        )

    image_files = sorted(
    file
    for file in class_dir.iterdir()
    if file.is_file() and file.suffix.lower() == ".bmp"
    )

    for image_path in image_files:
        records.append(
            {
                "filepath": image_path.relative_to(PROJECT_ROOT).as_posix(),
                "label": class_name
            }
        )

df = pd.DataFrame(records)

print(f"Total images found: {len(df)}")
print()
print("Class counts:")
print(df["label"].value_counts().sort_index())
print()


# --------------------------------------------------
# 2. Generate five stratified 70/15/15 random splits
# --------------------------------------------------

for seed in SEEDS:

    # First split:
    # 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=seed,
        stratify=df["label"]
    )

    # Second split:
    # Divide the remaining 30% equally
    # -> 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=seed,
        stratify=temp_df["label"]
    )

    # Add split labels
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"

    # Combine all three sets
    split_df = pd.concat(
        [train_df, val_df, test_df],
        ignore_index=True
    )

    # Safety check:
    # every image must appear exactly once
    assert len(split_df) == len(df)
    assert split_df["filepath"].nunique() == len(df)

    # Save split file
    output_file = SPLIT_DIR / f"protocol_a_seed{seed}.csv"
    split_df.to_csv(output_file, index=False)

    print(f"Seed {seed}")
    print(f"  Train:      {len(train_df)}")
    print(f"  Validation: {len(val_df)}")
    print(f"  Test:       {len(test_df)}")
    print(f"  Saved to:   {output_file.name}")
    print()