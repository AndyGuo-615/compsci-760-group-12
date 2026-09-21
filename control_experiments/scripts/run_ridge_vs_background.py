"""
Ridge vs Background control experiment for Protocol B.

Trains two sets of models:
    - ridge_only       (only fingerprint ridge region, background black)
    - background_only  (only background, ridge black)

Compares results to check whether the main model relies on ridge features
or on background artifacts.

Run:
    python control_experiments/scripts/run_ridge_vs_background.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ============================================================
# Paths
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CONTROL_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = CONTROL_ROOT.parent

sys.path.insert(0, str(PROJECT_ROOT))

from src.model import build_resnet18
from src.train import train_one_epoch, validate_one_epoch, EarlyStopping
from src.evaluate import evaluate_model


# ============================================================
# Directories
# ============================================================

SPLIT_DIR = CONTROL_ROOT / "splits"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "datasets_processed"
CHECKPOINT_DIR = CONTROL_ROOT / "checkpoints_ridge_vs_background"
RESULT_DIR = CONTROL_ROOT / "results"

for d in (CHECKPOINT_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

SEEDS = [111, 222, 333, 444, 555]
PROTOCOL = "B"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
NUM_EPOCHS = 30
OPTIMIZER_NAME = "adam"
PRETRAINED = True
PATIENCE = 5
NUM_WORKERS = 0

# ImageNet normalization for pretrained ResNet-18
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)

CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# Transforms (same as main experiment: no augmentation)
# ============================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ============================================================
# Dataset (reads from processed image folders)
# ============================================================

class ProcessedFingerprintDataset(Dataset):
    """
    Dataset that reads images from a processed root directory
    (ridge_only or background_only), using the split CSV to know
    which files belong to train/val/test.
    """

    def __init__(self, dataframe, image_root, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # row["filepath"] = "data/datasets/A+/cluster_0_2609.BMP"
        # We need to strip "data/datasets/" and re-root at image_root.
        rel = row["filepath"]
        prefix = "data/datasets/"
        if rel.startswith(prefix):
            rel = rel[len(prefix):]

        img_path = self.image_root / rel

        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]
        return image, label


# ============================================================
# Run one seed for one condition
# ============================================================

def run_one_condition(seed, condition):
    """
    condition: "ridge_only" or "background_only"
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    print()
    print("=" * 60)
    print(f"Protocol {PROTOCOL} | Ridge vs Background | "
          f"Seed {seed} | {condition}")
    print("=" * 60)

    split_csv = SPLIT_DIR / f"protocol_{PROTOCOL.lower()}_seed{seed}.csv"
    if not split_csv.exists():
        print(f"[SKIP] Split file not found: {split_csv}")
        return None

    df = pd.read_csv(split_csv)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    image_root = PROCESSED_ROOT / condition

    train_ds = ProcessedFingerprintDataset(
        train_df, image_root, get_train_transform()
    )
    val_ds = ProcessedFingerprintDataset(
        val_df, image_root, get_eval_transform()
    )
    test_ds = ProcessedFingerprintDataset(
        test_df, image_root, get_eval_transform()
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
    )

    print(f"Device:     {DEVICE}")
    print(f"Train: {len(train_ds)} | "
          f"Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Image root: {image_root}")
    print("-" * 60)

    model = build_resnet18(
        num_classes=len(CLASS_NAMES),
        pretrained=PRETRAINED,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    if OPTIMIZER_NAME == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    else:
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"protocol_{PROTOCOL.lower()}_seed{seed}_{condition}_best.pt"
    )
    early_stopping = EarlyStopping(
        patience=PATIENCE, save_path=checkpoint_path,
    )

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

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    results = evaluate_model(model, test_loader, DEVICE)

    print()
    print(f"Test Results ({condition})")
    print("-" * 60)
    print(f"Accuracy:          {results['accuracy']:.4f}")
    print(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}")
    print(f"Macro F1:          {results['macro_f1']:.4f}")

    return {
        "experiment": "ridge_vs_background",
        "protocol": PROTOCOL,
        "condition": condition,
        "seed": seed,
        "accuracy": results["accuracy"],
        "balanced_accuracy": results["balanced_accuracy"],
        "macro_f1": results["macro_f1"],
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }


# ============================================================
# Main
# ============================================================

def main():
    all_results = []

    for seed in SEEDS:
        for condition in ["ridge_only", "background_only"]:
            result = run_one_condition(seed, condition)
            if result is not None:
                all_results.append(result)

    if not all_results:
        print("No runs completed.")
        return

    results_file = RESULT_DIR / "ridge_vs_background.csv"
    df_new = pd.DataFrame(all_results)

    if results_file.exists() and results_file.stat().st_size > 0:
        try:
            df_old = pd.read_csv(results_file)
        except pd.errors.EmptyDataError:
            df_old = None
    else:
        df_old = None

    if df_old is not None and len(df_old) > 0:
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(
            subset=["experiment", "protocol", "condition", "seed"],
            keep="last",
        )
        df_combined.to_csv(results_file, index=False)
    else:
        df_new.to_csv(results_file, index=False)

    print()
    print("=" * 60)
    print(f"All results saved to: {results_file}")
    print("=" * 60)

    # Print summary
    summary = df_new.groupby("condition").agg({
        "accuracy": ["mean", "std"],
        "balanced_accuracy": ["mean", "std"],
        "macro_f1": ["mean", "std"],
    }).round(4)
    print()
    print("Summary (mean ± std over seeds):")
    print(summary.to_string())


if __name__ == "__main__":
    main()