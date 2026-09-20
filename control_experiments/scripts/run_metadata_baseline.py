from pathlib import Path
import re
import pandas as pd
import numpy as np
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = PROJECT_ROOT / "splits"
RESULT_DIR = PROJECT_ROOT / "results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

SEEDS = [111, 222, 333, 444, 555]

PROTOCOL = "B"

CLASS_NAMES = [
    "A+", "A-", "AB+", "AB-",
    "B+", "B-", "O+", "O-"
]


# ============================================================
# Helper: Extract metadata features
# ============================================================

def extract_metadata_features(df):
    """
    Extract metadata features from the dataset.

    Features:
    - subject_id_num: numerical subject ID
    - gender: M / F parsed from socofing_files
    - finger_position: hand + finger parsed from socofing_files

    Expected format in socofing_files:
        "382__M_Left_index_finger.BMP"
    """
    df = df.copy()

    # 1. Subject ID (as numeric)
    df["subject_id_num"] = pd.to_numeric(
        df["subject_ids"], errors="coerce"
    )

    # 2. Parse Gender and Finger Position
    def parse_socofing(filename):
        if pd.isna(filename):
            return pd.Series({"gender": "Unknown", "finger": "Unknown"})

        match = re.search(
            r"__([MF])_(Left|Right)_(\w+)_finger",
            str(filename)
        )
        if match:
            gender = match.group(1)
            hand = match.group(2)
            finger = match.group(3)
            return pd.Series({
                "gender": gender,
                "finger": f"{hand}_{finger}"
            })
        return pd.Series({"gender": "Unknown", "finger": "Unknown"})

    parsed = df["socofing_files"].apply(parse_socofing)
    df["gender"] = parsed["gender"]
    df["finger_position"] = parsed["finger"]

    return df


# ============================================================
# Evaluate Metadata-only Baseline for one split
# ============================================================

def evaluate_metadata_baseline(split_df):
    """
    Train a Random Forest using ONLY metadata features.
    Metadata: subject_id_num, gender, finger_position
    """
    # Extract metadata features
    split_df = extract_metadata_features(split_df)

    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    # === Feature encoding ===
    feature_cols = ["subject_id_num", "gender", "finger_position"]

    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        # Handle unseen categories in test set
        test_df[col] = test_df[col].astype(str).map(
            lambda x: le.transform([x])[0] if x in le.classes_ else -1
        )
        encoders[col] = le

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values

    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    # === Train Random Forest ===
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # === Predict ===
    y_pred = model.predict(X_test)

    # === Metrics ===
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(
        y_test, y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "train_size": len(train_df),
        "test_size": len(test_df)
    }


# ============================================================
# Main
# ============================================================

def main():

    results = []

    for seed in SEEDS:

        split_file = (
            SPLIT_DIR
            / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
        )

        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                "Please run merge_metadata.py first."
            )

        split_df = pd.read_csv(split_file)

        required_columns = {
            "filepath", "label", "split",
            "subject_ids", "socofing_files"
        }

        missing_columns = required_columns - set(split_df.columns)

        if missing_columns:
            raise ValueError(
                f"Missing columns in {split_file}: {missing_columns}\n"
                "Please run merge_metadata.py first."
            )

        metrics = evaluate_metadata_baseline(split_df)

        result = {
            "experiment": "metadata_baseline",
            "protocol": PROTOCOL,
            "seed": seed,
            **metrics
        }

        results.append(result)

        print("=" * 60)
        print(f"Protocol {PROTOCOL} | Seed {seed}")
        print(f"Accuracy:           {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1:            {metrics['macro_f1']:.4f}")

    results_df = pd.DataFrame(results)

    output_file = RESULT_DIR / "metadata_baseline.csv"
    results_df.to_csv(output_file, index=False)

    print()
    print("=" * 60)
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()