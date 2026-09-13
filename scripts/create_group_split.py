from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "main_subject_groups.csv"
)

SPLIT_DIR = PROJECT_ROOT / "splits"

SEEDS = [111, 222, 333, 444, 555]

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

SPLIT_NAMES = [
    "train",
    "validation",
    "test"
]

TARGET_RATIOS = np.array([
    0.70,
    0.15,
    0.15
])

NUMBER_OF_ATTEMPTS = 500


def objective(current_counts, target_counts):
    """
    Measure how far the current class counts are
    from the target class counts.
    """

    scale = np.maximum(
        target_counts,
        1.0
    )

    difference = (
        current_counts - target_counts
    ) / scale

    return float(
        np.square(difference).sum()
    )


def create_group_split(df, seed):
    """
    Assign complete subject groups to train,
    validation or test.

    A group is never divided between splits.
    Groups may contain multiple class labels.
    """

    # One row per group and one column per class
    group_class_counts = pd.crosstab(
        df["group_id"],
        df["label"]
    )

    group_class_counts = (
        group_class_counts
        .reindex(
            columns=CLASS_NAMES,
            fill_value=0
        )
    )

    group_sizes = group_class_counts.sum(
        axis=1
    )

    total_class_counts = (
        group_class_counts
        .to_numpy()
        .sum(axis=0)
    )

    target_counts = (
        TARGET_RATIOS[:, None]
        * total_class_counts[None, :]
    )

    best_assignment = None
    best_score = float("inf")

    # Try multiple group orders and keep
    # the most balanced valid result
    for attempt in range(
        NUMBER_OF_ATTEMPTS
    ):
        rng = np.random.default_rng(
            np.random.SeedSequence(
        	[seed, attempt]
    		)
        )

        random_values = {
            group_id: rng.random()
            for group_id
            in group_class_counts.index
        }

        # Larger groups are assigned first.
        # Equal-size groups receive random order.
        ordered_groups = sorted(
            group_class_counts.index,
            key=lambda group_id: (
                -group_sizes[group_id],
                random_values[group_id]
            )
        )

        current_counts = np.zeros_like(
            target_counts,
            dtype=float
        )

        assignment = {}

        for group_id in ordered_groups:
            group_vector = (
                group_class_counts
                .loc[group_id]
                .to_numpy(dtype=float)
            )

            candidate_scores = []

            for split_index in range(
                len(SPLIT_NAMES)
            ):
                candidate_counts = (
                    current_counts.copy()
                )

                candidate_counts[
                    split_index
                ] += group_vector

                score = objective(
                    candidate_counts,
                    target_counts
                )

                candidate_scores.append(
                    score
                )

            minimum_score = min(
                candidate_scores
            )

            tied_choices = [
                index
                for index, score
                in enumerate(candidate_scores)
                if np.isclose(
                    score,
                    minimum_score
                )
            ]

            selected_index = int(
                rng.choice(tied_choices)
            )

            selected_split = (
                SPLIT_NAMES[selected_index]
            )

            assignment[group_id] = (
                selected_split
            )

            current_counts[
                selected_index
            ] += group_vector

        final_score = objective(
            current_counts,
            target_counts
        )

        if final_score < best_score:
            best_score = final_score
            best_assignment = assignment

    result = df.copy()

    result["split"] = (
        result["group_id"]
        .map(best_assignment)
    )

    return result


def validate_split(split_df):
    """
    Mandatory Protocol B leakage checks.
    """

    if split_df["filepath"].duplicated().any():
        raise ValueError(
            "A filepath appears more than once."
        )

    if split_df["split"].isna().any():
        raise ValueError(
            "Some images were not assigned."
        )

    # Every group must occur in exactly one split
    group_split_counts = (
        split_df
        .groupby("group_id")["split"]
        .nunique()
    )

    crossing_groups = (
        group_split_counts[
            group_split_counts > 1
        ]
    )

    if len(crossing_groups) > 0:
        raise ValueError(
            "LEAKAGE DETECTED: "
            "some subject groups cross splits."
        )

    expected_splits = {
        "train",
        "validation",
        "test"
    }

    if set(split_df["split"]) != expected_splits:
        raise ValueError(
            "One or more splits are empty."
        )


def print_summary(split_df):
    print("Images by split:")

    print(
        split_df["split"]
        .value_counts()
        .reindex(SPLIT_NAMES)
    )

    print()
    print("Groups by split:")

    print(
        split_df
        .groupby("split")["group_id"]
        .nunique()
        .reindex(SPLIT_NAMES)
    )

    print()
    print("Images by class and split:")

    class_table = pd.crosstab(
        split_df["label"],
        split_df["split"]
    )

    class_table = class_table.reindex(
        index=CLASS_NAMES,
        columns=SPLIT_NAMES,
        fill_value=0
    )

    print(class_table)


def main():
    df = pd.read_csv(
        MANIFEST_FILE,
        dtype={
            "filepath": str,
            "label": str,
            "group_id": str
        }
    )

    required_columns = {
        "filepath",
        "label",
        "group_id"
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    if len(df) != 5837:
        raise ValueError(
            f"Expected 5837 images, found {len(df)}."
        )

    unknown_labels = (
        set(df["label"]) - set(CLASS_NAMES)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown labels: {unknown_labels}"
        )

    # Check that every referenced image exists
    missing_files = [
        filepath
        for filepath in df["filepath"]
        if not (
            PROJECT_ROOT / filepath
        ).is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Manifest contains missing images. "
            f"Examples: {missing_files[:10]}"
        )

    SPLIT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(
        "Images:",
        len(df)
    )

    print(
        "Leakage-control groups:",
        df["group_id"].nunique()
    )

    print()

    for seed in SEEDS:
        split_df = create_group_split(
            df,
            seed
        )

        validate_split(split_df)

        output_file = (
            SPLIT_DIR
            / f"protocol_b_seed{seed}.csv"
        )

        split_df.to_csv(
            output_file,
            index=False
        )

        print("=" * 50)
        print(f"Seed: {seed}")
        print("Leakage check: PASS")
        print(f"Saved: {output_file.name}")
        print()

        print_summary(split_df)

        print()


if __name__ == "__main__":
    main()