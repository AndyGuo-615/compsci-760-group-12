"""
Label Permutation Control Experiment for Protocol B.

- Reuses src/ modules for full consistency with the main experiment.
- Only change from the main experiment: the 'label' column is randomly permuted.
- Expected test accuracy: ~12.5% (8-class random chance).

Run directly:
    python control_experiments/scripts/run_label_permutation.py
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

# Script location: compsci-760-group-12/control_experiments/scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = SCRIPT_DIR.parent                    # control_experiments/
PROJECT_ROOT = CONTROL_ROOT.parent                  # compsci-760-group-12/

# Make src/ importable
sys.path.insert(0, str(PROJECT_ROOT))

# Import main experiment modules
from src.dataset import FingerprintDataset
from src.model import build_resnet18
from src.transforms import get_train_transform, get_eval_transform
from src.train import train_one_epoch, validate_one_epoch, EarlyStopping
from src.evaluate import evaluate_model


# ============================================================
# Directories
# ============================================================

SPLIT_DIR = CONTROL_ROOT / "splits"
PERMUTED_SPLIT_DIR = CONTROL_ROOT / "splits_permuted"
CHECKPOINT_DIR = CONTROL_ROOT / "checkpoints_label_permutation"
RESULT_DIR = CONTROL_ROOT / "results"

for d in (PERMUTED_SPLIT_DIR, CHECKPOINT_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings (align with the main experiment)
# ============================================================

SEEDS = [111, 222, 333, 444, 555]
PROTOCOL = "B"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
OPTIMIZER_NAME = "adam"      # "adam" or "sgd"
PRETRAINED = True
PATIENCE = 5
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
# Label permutation
# ============================================================

def make_permuted_split(src_csv, dst_csv, seed):
    """Permute the 'label' column; keep everything else unchanged."""
    df = pd.read_csv(src_csv)
    rng = np.random.default_rng(seed)
    df["label"] = rng.permutation(df["label"].values)
    df.to_csv(dst_csv, index=False)
    print(f"Permuted split saved to: {dst_csv}")


# ============================================================
# Run one seed
# ============================================================

def run_one_seed(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)

    print()
    print("=" * 60)
    print(f"Protocol {PROTOCOL} | Label Permutation | Seed {seed}")
    print("=" * 60)

    # Split files
    original_csv = SPLIT_DIR / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
    if not original_csv.exists():
        print(f"[SKIP] Split file not found: {original_csv}")
        return None

    permuted_csv = (
        PERMUTED_SPLIT_DIR
        / f"protocol_{PROTOCOL.lower()}_seed{seed}_permuted.csv"
    )
    make_permuted_split(original_csv, permuted_csv, seed=seed)

    # Datasets (using src/ modules)
    train_dataset = FingerprintDataset(
        permuted_csv,
        split="train",
        transform=get_train_transform(PRETRAINED),
        project_root=PROJECT_ROOT,
    )
    val_dataset = FingerprintDataset(
        permuted_csv,
        split="validation",
        transform=get_eval_transform(PRETRAINED),
        project_root=PROJECT_ROOT,
    )
    test_dataset = FingerprintDataset(
        permuted_csv,
        split="test",
        transform=get_eval_transform(PRETRAINED),
        project_root=PROJECT_ROOT,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    print(f"Device:     {DEVICE}")
    print(f"Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | "
          f"Test: {len(test_dataset)}")
    print(f"Batch size: {BATCH_SIZE} | "
          f"LR: {LEARNING_RATE} | "
          f"Optimizer: {OPTIMIZER_NAME} | "
          f"Pretrained: {PRETRAINED} | "
          f"Patience: {PATIENCE}")
    print("-" * 60)

    # Model
    model = build_resnet18(
        num_classes=8,
        pretrained=PRETRAINED,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_NAME == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"protocol_{PROTOCOL.lower()}_seed{seed}_permuted_best.pt"
    )
    early_stopping = EarlyStopping(
        patience=PATIENCE,
        save_path=checkpoint_path,
    )

    # Training
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch {epoch+1:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Test with best checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    results = evaluate_model(model, test_loader, DEVICE)

    print()
    print("Test Results (Label Permutation)")
    print("-" * 60)
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")
    print()
    print("Per-Class Recall:")
    for class_name, recall in results["per_class_recall"].items():
        print(f"  {class_name}: {recall:.4f}")
    print()
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    return {
        "experiment": "label_permutation",
        "protocol": PROTOCOL,
        "seed": seed,
        "accuracy": results["accuracy"],
        "balanced_accuracy": results["balanced_accuracy"],
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

    if not all_results:
        print("No seeds ran successfully.")
        return

    results_file = RESULT_DIR / "label_permutation.csv"
    df_new = pd.DataFrame(all_results)

    if results_file.exists():
        df_old = pd.read_csv(results_file)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["experiment", "protocol", "seed"],
            keep="last",
        )
        df_combined.to_csv(results_file, index=False)
    else:
        df_new.to_csv(results_file, index=False)

    print()
    print("=" * 60)
    print(f"All results saved to: {results_file}")
    print("=" * 60)
    print(df_new.to_string(index=False))


if __name__ == "__main__":
    main()