from pathlib import Path
import re
import pandas as pd
import numpy as np


# ============================================================
# Project paths
# ============================================================

CONTROL_ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = CONTROL_ROOT / "splits"

RESULT_DIR = CONTROL_ROOT / "metadata_audit"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

SEEDS = [111, 222, 333, 444, 555]

CLASS_NAMES = [
    "A+", "A-", "AB+", "AB-",
    "B+", "B-", "O+", "O-"
]


# ============================================================
# Parse SOCofing filename
# ============================================================

def parse_socofing(filename):
    """
    Extract gender, hand and finger type from a SOCofing filename.

    Example:
        382__M_Left_index_finger.BMP

    Returns:
        gender = M
        hand = Left
        finger_type = index
    """

    if pd.isna(filename):
        return pd.Series({
            "gender": "Unknown",
            "hand": "Unknown",
            "finger_type": "Unknown"
        })

    filename = str(filename)

    match = re.search(
        r"__([MF])_(Left|Right)_(\w+)_finger",
        filename,
        flags=re.IGNORECASE
    )

    if match:

        gender = match.group(1).upper()
        hand = match.group(2).capitalize()
        finger_type = match.group(3).lower()

        return pd.Series({
            "gender": gender,
            "hand": hand,
            "finger_type": finger_type
        })

    return pd.Series({
        "gender": "Unknown",
        "hand": "Unknown",
        "finger_type": "Unknown"
    })


# ============================================================
# Load one Protocol B split
# ============================================================

def load_split(seed):

    split_file = (
        SPLIT_DIR /
        f"protocol_b_seed{seed}.csv"
    )

    if not split_file.is_file():
        raise FileNotFoundError(
            f"Split file not found:\n{split_file}"
        )

    df = pd.read_csv(split_file)

    required_columns = {
        "filepath",
        "label",
        "subject_ids",
        "socofing_files",
        "split"
    }

    missing = (
        required_columns -
        set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing columns in {split_file}: {missing}"
        )

    return df


# ============================================================
# Add parsed metadata
# ============================================================

def add_metadata(df):

    df = df.copy()

    # --------------------------------------------------------
    # Subject ID
    # --------------------------------------------------------

    df["subject_id_num"] = pd.to_numeric(
        df["subject_ids"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Parse SOCofing filename
    # --------------------------------------------------------

    parsed = df["socofing_files"].apply(
        parse_socofing
    )

    df[
        ["gender", "hand", "finger_type"]
    ] = parsed[
        ["gender", "hand", "finger_type"]
    ]

    return df


# ============================================================
# Check basic metadata
# ============================================================

def check_basic_metadata(df, seed):

    print()
    print("=" * 70)
    print(f"METADATA AUDIT | SEED {seed}")
    print("=" * 70)

    print()
    print("Number of images:")
    print(len(df))

    print()
    print("Subject ID range:")
    print(
        f"Minimum: {df['subject_id_num'].min():.0f}"
    )
    print(
        f"Maximum: {df['subject_id_num'].max():.0f}"
    )

    print()
    print("Number of unique subjects:")
    print(
        df["subject_id_num"].nunique()
    )

    print()
    print("Hand distribution:")
    print(
        df["hand"].value_counts(
            dropna=False
        )
    )

    print()
    print("Finger type distribution:")
    print(
        df["finger_type"].value_counts(
            dropna=False
        )
    )

    print()
    print("Gender distribution:")
    print(
        df["gender"].value_counts(
            dropna=False
        )
    )


# ============================================================
# Hand by blood group
# ============================================================

def check_hand_by_label(df):

    print()
    print("=" * 70)
    print("HAND × BLOOD GROUP")
    print("=" * 70)

    counts = pd.crosstab(
        df["label"],
        df["hand"]
    ).reindex(
        CLASS_NAMES,
        fill_value=0
    )

    print()
    print("Counts:")
    print(counts)

    proportions = counts.div(
        counts.sum(axis=1),
        axis=0
    )

    print()
    print("Row proportions:")
    print(
        proportions.round(3)
    )

    return counts, proportions


# ============================================================
# Finger type by blood group
# ============================================================

def check_finger_by_label(df):

    print()
    print("=" * 70)
    print("FINGER TYPE × BLOOD GROUP")
    print("=" * 70)

    counts = pd.crosstab(
        df["label"],
        df["finger_type"]
    ).reindex(
        CLASS_NAMES,
        fill_value=0
    )

    print()
    print("Counts:")
    print(counts)

    proportions = counts.div(
        counts.sum(axis=1),
        axis=0
    )

    print()
    print("Row proportions:")
    print(
        proportions.round(3)
    )

    return counts, proportions


# ============================================================
# Subject ID range by blood group
# ============================================================

def check_subject_id_by_label(df):

    print()
    print("=" * 70)
    print("SUBJECT ID RANGE × BLOOD GROUP")
    print("=" * 70)

    summary = (
        df.groupby("label")[
            "subject_id_num"
        ]
        .agg(
            min_subject="min",
            max_subject="max",
            n_unique_subjects="nunique",
            n_images="count"
        )
        .reindex(CLASS_NAMES)
    )

    print()
    print(summary)

    return summary


# ============================================================
# Subject ID distribution by blood group
# ============================================================

def check_subject_id_overlap(df):

    print()
    print("=" * 70)
    print("SUBJECT ID OVERLAP")
    print("=" * 70)

    subject_label = (
        df[
            [
                "subject_id_num",
                "label"
            ]
        ]
        .drop_duplicates()
    )

    subject_counts = pd.crosstab(
        subject_label["subject_id_num"],
        subject_label["label"]
    )

    # Number of labels represented by each subject
    labels_per_subject = (
        (subject_counts > 0)
        .sum(axis=1)
    )

    print()
    print(
        "Subjects appearing in multiple blood groups:"
    )

    multi_label_subjects = (
        labels_per_subject[
            labels_per_subject > 1
        ]
    )

    print(
        f"{len(multi_label_subjects)} "
        f"out of "
        f"{len(labels_per_subject)} "
        f"unique subjects"
    )

    if len(multi_label_subjects) > 0:

        print()
        print("Examples:")

        print(
            subject_counts.loc[
                multi_label_subjects.index
            ].head(20)
        )

    return subject_counts


# ============================================================
# Subject ID range by split
# ============================================================

def check_subject_id_by_split(df):

    print()
    print("=" * 70)
    print("SUBJECT ID RANGE BY SPLIT")
    print("=" * 70)

    summary = (
        df.groupby("split")[
            "subject_id_num"
        ]
        .agg(
            min_subject="min",
            max_subject="max",
            n_unique_subjects="nunique",
            n_images="count"
        )
    )

    print(summary)

    return summary


# ============================================================
# Save audit tables
# ============================================================

def save_tables(
    seed,
    df,
    hand_counts,
    hand_props,
    finger_counts,
    finger_props,
    subject_summary,
    subject_split_summary
):

    df.to_csv(
        RESULT_DIR /
        f"metadata_audit_seed{seed}.csv",
        index=False
    )

    hand_counts.to_csv(
        RESULT_DIR /
        f"hand_by_label_counts_seed{seed}.csv"
    )

    hand_props.to_csv(
        RESULT_DIR /
        f"hand_by_label_proportions_seed{seed}.csv"
    )

    finger_counts.to_csv(
        RESULT_DIR /
        f"finger_by_label_counts_seed{seed}.csv"
    )

    finger_props.to_csv(
        RESULT_DIR /
        f"finger_by_label_proportions_seed{seed}.csv"
    )

    subject_summary.to_csv(
        RESULT_DIR /
        f"subject_id_by_label_seed{seed}.csv"
    )

    subject_split_summary.to_csv(
        RESULT_DIR /
        f"subject_id_by_split_seed{seed}.csv"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("HAND / FINGER / SUBJECT ID METADATA AUDIT")
    print("=" * 70)

    for seed in SEEDS:

        df = load_split(seed)

        df = add_metadata(df)

        # ----------------------------------------------------
        # Basic checks
        # ----------------------------------------------------

        check_basic_metadata(
            df,
            seed
        )

        # ----------------------------------------------------
        # Hand
        # ----------------------------------------------------

        (
            hand_counts,
            hand_props
        ) = check_hand_by_label(df)

        # ----------------------------------------------------
        # Finger
        # ----------------------------------------------------

        (
            finger_counts,
            finger_props
        ) = check_finger_by_label(df)

        # ----------------------------------------------------
        # Subject ID range
        # ----------------------------------------------------

        subject_summary = (
            check_subject_id_by_label(
                df
            )
        )

        # ----------------------------------------------------
        # Subject ID overlap
        # ----------------------------------------------------

        check_subject_id_overlap(
            df
        )

        # ----------------------------------------------------
        # Subject ID by split
        # ----------------------------------------------------

        subject_split_summary = (
            check_subject_id_by_split(
                df
            )
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        save_tables(
            seed,
            df,
            hand_counts,
            hand_props,
            finger_counts,
            finger_props,
            subject_summary,
            subject_split_summary
        )

    print()
    print("=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"Results saved to:\n{RESULT_DIR}"
    )


if __name__ == "__main__":
    main()