from pathlib import Path

import pandas as pd
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

PROTOCOLS = ["A", "B"]

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


# ============================================================
# Evaluate one split
# ============================================================

def evaluate_majority_baseline(split_df):

    train_df = split_df[split_df["split"] == "train"].copy()
    test_df = split_df[split_df["split"] == "test"].copy()

    # Find majority class using training data only
    majority_class = (
        train_df["label"]
        .value_counts()
        .reindex(CLASS_NAMES, fill_value=0)
        .idxmax()
    )

    y_true = test_df["label"].to_numpy()
    y_pred = [majority_class] * len(test_df)

    accuracy = accuracy_score(y_true, y_pred)

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0
    )

    return {
        "majority_class": majority_class,
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

    for protocol in PROTOCOLS:

        for seed in SEEDS:

            split_file = (
                SPLIT_DIR
                / f"protocol_{protocol.lower()}_seed{seed}.csv"
            )

            if not split_file.exists():
                raise FileNotFoundError(
                    f"Split file not found: {split_file}"
                )

            split_df = pd.read_csv(split_file)

            required_columns = {
                "filepath",
                "label",
                "split"
            }

            missing_columns = (
                required_columns - set(split_df.columns)
            )

            if missing_columns:
                raise ValueError(
                    f"Missing columns in {split_file}: "
                    f"{missing_columns}"
                )

            metrics = evaluate_majority_baseline(split_df)

            result = {
                "experiment": "majority_baseline",
                "protocol": protocol,
                "seed": seed,
                **metrics
            }

            results.append(result)

            print("=" * 60)
            print(
                f"Protocol {protocol} | Seed {seed}"
            )
            print(
                f"Majority class: {metrics['majority_class']}"
            )
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

    results_df = pd.DataFrame(results)

    output_file = (
        RESULT_DIR / "majority_baseline.csv"
    )

    results_df.to_csv(
        output_file,
        index=False
    )

    print()
    print("=" * 60)
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()