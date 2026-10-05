"""
Within-split image-level Label Permutation Control Experiment for Protocol B.

- Reuses src/ modules for full consistency with the main experiment.
- Original Protocol B train/validation/test split is kept unchanged.
- Only the label assignment is changed.
- Labels are permuted at the GROUP level:
    all images belonging to the same group_id receive the same
    permuted label.

Expected result:
- Performance should be close to the no-information / chance-level
  performance of the 8-class classification problem.

Run directly:
    python control_experiments/scripts/run_group_label_permutation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ============================================================
# Paths
# ============================================================

# Script location:
# compsci-760-group-12/control_experiments/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent

CONTROL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CONTROL_ROOT.parent

# Make src/ importable
sys.path.insert(0, str(PROJECT_ROOT))

# Import main experiment modules
from src.dataset import FingerprintDataset
from src.model import build_resnet18
from src.transforms import get_train_transform, get_eval_transform
from src.train import (
    train_one_epoch,
    validate_one_epoch,
    EarlyStopping,
)
from src.evaluate import evaluate_model


# ============================================================
# Directories
# ============================================================

SPLIT_DIR = CONTROL_ROOT / "splits"

PERMUTED_SPLIT_DIR = (
    CONTROL_ROOT / "splits_permuted"
)

CHECKPOINT_DIR = (
    CONTROL_ROOT / "checkpoints_label_permutation"
)

RESULT_DIR = CONTROL_ROOT / "results"

for d in (
    PERMUTED_SPLIT_DIR,
    CHECKPOINT_DIR,
    RESULT_DIR,
):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

SEEDS = [111, 222, 333, 444, 555]

PROTOCOL = "B"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
OPTIMIZER_NAME = "adam"      # "adam" or "sgd"
PRETRAINED = True
PATIENCE = 3
NUM_WORKERS = 0


# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")

elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")

else:
    DEVICE = torch.device("cpu")


# ============================================================
# Group-Level Label Permutation
# ============================================================

def make_group_permuted_split(src_csv, dst_csv, seed):
    """
    Permute labels within each original split.

    The following remain unchanged:
        filepath
        group_id
        subject_ids
        socofing_files
        sha256
        split

    Labels are randomly permuted within:
        train
        validation
        test

    This preserves the label distribution within each split
    while breaking the original image-label association.
    """

    df = pd.read_csv(src_csv)

    required_columns = [
        "filepath",
        "label",
        "group_id",
        "split",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep a copy for validation
    original_df = df.copy()

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Permute labels separately within each split
    # --------------------------------------------------------

    for split_name in ["train", "validation", "test"]:

        mask = df["split"] == split_name

        labels = df.loc[mask, "label"].to_numpy()

        # Randomly permute labels within this split
        permuted_labels = rng.permutation(labels)

        df.loc[mask, "label"] = permuted_labels

    # --------------------------------------------------------
    # Validation 1:
    # Number of rows unchanged
    # --------------------------------------------------------

    if len(df) != len(original_df):
        raise RuntimeError(
            "Number of rows changed during permutation."
        )

    # --------------------------------------------------------
    # Validation 2:
    # filepath unchanged
    # --------------------------------------------------------

    if not df["filepath"].equals(
        original_df["filepath"]
    ):
        raise RuntimeError(
            "filepath was changed."
        )

    # --------------------------------------------------------
    # Validation 3:
    # group_id unchanged
    # --------------------------------------------------------

    if not df["group_id"].equals(
        original_df["group_id"]
    ):
        raise RuntimeError(
            "group_id was changed."
        )

    # --------------------------------------------------------
    # Validation 4:
    # split unchanged
    # --------------------------------------------------------

    if not df["split"].equals(
        original_df["split"]
    ):
        raise RuntimeError(
            "split was changed."
        )

    # --------------------------------------------------------
    # Validation 5:
    # Label counts within each split unchanged
    # --------------------------------------------------------

    original_counts = (
        original_df
        .groupby(["split", "label"])
        .size()
        .sort_index()
    )

    permuted_counts = (
        df
        .groupby(["split", "label"])
        .size()
        .sort_index()
    )

    if not original_counts.equals(
        permuted_counts
    ):
        raise RuntimeError(
            "Label counts within splits changed."
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_csv(
        dst_csv,
        index=False
    )

    print(
        f"Label-permuted split saved to:\n"
        f"{dst_csv}"
    )

    print()
    print("Original label distribution:")
    print(
        original_df
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )

    print()
    print("Permuted label distribution:")
    print(
        df
        .groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
    )


# ============================================================
# Run One Seed
# ============================================================

def run_one_seed(seed):

    # --------------------------------------------------------
    # Set random seeds
    # --------------------------------------------------------

    torch.manual_seed(seed)
    np.random.seed(seed)

    print()
    print("=" * 60)
    print(
        f"Protocol {PROTOCOL} | "
        f"GROUP-Level Label Permutation | "
        f"Seed {seed}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Original split
    # --------------------------------------------------------

    original_csv = (
        SPLIT_DIR
        / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
    )

    if not original_csv.exists():
        print(
            f"[SKIP] Split file not found:\n"
            f"{original_csv}"
        )
        return None

    # --------------------------------------------------------
    # Permuted split
    # --------------------------------------------------------

    permuted_csv = (
        PERMUTED_SPLIT_DIR
        / (
            f"protocol_{PROTOCOL.lower()}"
            f"_seed{seed}_group_permuted.csv"
        )
    )

    make_group_permuted_split(
        original_csv,
        permuted_csv,
        seed=seed,
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = FingerprintDataset(
        permuted_csv,
        split="train",
        transform=get_train_transform(
            PRETRAINED
        ),
        project_root=PROJECT_ROOT,
    )

    val_dataset = FingerprintDataset(
        permuted_csv,
        split="validation",
        transform=get_eval_transform(
            PRETRAINED
        ),
        project_root=PROJECT_ROOT,
    )

    test_dataset = FingerprintDataset(
        permuted_csv,
        split="test",
        transform=get_eval_transform(
            PRETRAINED
        ),
        project_root=PROJECT_ROOT,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Print settings
    # --------------------------------------------------------

    print(f"Device:     {DEVICE}")

    print(
        f"Train: {len(train_dataset)} | "
        f"Val: {len(val_dataset)} | "
        f"Test: {len(test_dataset)}"
    )

    print(
        f"Batch size: {BATCH_SIZE} | "
        f"LR: {LEARNING_RATE} | "
        f"Optimizer: {OPTIMIZER_NAME} | "
        f"Pretrained: {PRETRAINED} | "
        f"Patience: {PATIENCE}"
    )

    print("-" * 60)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_resnet18(
        num_classes=8,
        pretrained=PRETRAINED,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    if OPTIMIZER_NAME == "sgd":

        optimizer = optim.SGD(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    else:

        optimizer = optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            f"protocol_{PROTOCOL.lower()}"
            f"_seed{seed}"
            f"_group_permuted_best.pt"
        )
    )

    early_stopping = EarlyStopping(
        patience=PATIENCE,
        save_path=checkpoint_path,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(NUM_EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            DEVICE,
        )

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        early_stopping.step(
            val_loss,
            model,
        )

        if early_stopping.should_stop:

            print(
                f"Early stopping at epoch "
                f"{epoch + 1}"
            )

            break

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    results = evaluate_model(
        model,
        test_loader,
        DEVICE,
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "Test Results "
        "(Group-Level Label Permutation)"
    )

    print("-" * 60)

    print(
        f"Accuracy:          "
        f"{results['accuracy']:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{results['balanced_accuracy']:.4f}"
    )

    print(
        f"Macro F1:          "
        f"{results['macro_f1']:.4f}"
    )

    print()

    print("Per-Class Recall:")

    for class_name, recall in (
        results["per_class_recall"].items()
    ):

        print(
            f"  {class_name}: "
            f"{recall:.4f}"
        )

    print()

    print("Confusion Matrix:")

    print(
        results["confusion_matrix"]
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "experiment": "group_label_permutation",
        "protocol": PROTOCOL,
        "seed": seed,
        "accuracy": results["accuracy"],
        "balanced_accuracy": (
            results["balanced_accuracy"]
        ),
        "macro_f1": results["macro_f1"],
        "train_size": len(train_dataset),
        "test_size": len(test_dataset),
    }


# ============================================================
# Main
# ============================================================

def main():

    all_results = []

    for seed in SEEDS:

        result = run_one_seed(seed)

        if result is not None:
            all_results.append(result)

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if not all_results:

        print(
            "No seeds ran successfully."
        )

        return

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results_file = (
        RESULT_DIR
        / "label_permutation.csv"
    )

    df_new = pd.DataFrame(
        all_results
    )

    if results_file.exists():

        df_old = pd.read_csv(
            results_file
        )

        df_combined = pd.concat(
            [
                df_old,
                df_new,
            ],
            ignore_index=True,
        )

        df_combined = (
            df_combined
            .drop_duplicates(
                subset=[
                    "experiment",
                    "protocol",
                    "seed",
                ],
                keep="last",
            )
        )

        df_combined.to_csv(
            results_file,
            index=False,
        )

    else:

        df_new.to_csv(
            results_file,
            index=False,
        )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print("=" * 60)

    print(
        "All permutation results "
        f"saved to:\n{results_file}"
    )

    print("=" * 60)

    print(
        df_new.to_string(
            index=False
        )
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()