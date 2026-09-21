from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path("/Users/a75027/Documents/GitHub/compsci-760-group-12")

SPLIT_DIR = PROJECT_ROOT / "control_experiments" / "splits"
PERMUTED_SPLIT_DIR = PROJECT_ROOT / "control_experiments" / "splits_permuted"
CHECKPOINT_DIR = PROJECT_ROOT / "control_experiments" / "checkpoints_label_permutation"
RESULT_DIR = PROJECT_ROOT / "control_experiments" / "results"

for d in (PERMUTED_SPLIT_DIR, CHECKPOINT_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Local dataset root
DATA_ROOT = Path("/Users/a75027/Desktop/compsci 760/datasets")

# Split file stores paths as "data/datasets/A+/..."; strip this prefix.
SPLIT_PATH_PREFIX = "data/datasets/"


# ============================================================
# Experiment configuration (matches main experiment)
# ============================================================

SEEDS = [111, 222, 333, 444, 555]

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
OPTIMIZER_NAME = "adam"      # "adam" or "sgd"
PRETRAINED = True
PATIENCE = 5
NUM_WORKERS = 0              # macOS 建议 0

CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


# ============================================================
# Dataset
# ============================================================

class FingerprintDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rel_path = row["filepath"]

        if rel_path.startswith(SPLIT_PATH_PREFIX):
            rel_path = rel_path[len(SPLIT_PATH_PREFIX):]
        img_path = DATA_ROOT / rel_path

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label = CLASS_TO_IDX[row["label"]]
        return image, label


# ============================================================
# Transforms
# ============================================================

def get_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def get_eval_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ============================================================
# Model
# ============================================================

def build_resnet18(num_classes=8, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ============================================================
# Train / Eval
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def evaluate_model(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        ),
        "per_class_recall": {
            name: r for name, r in zip(
                CLASS_NAMES,
                recall_score(
                    y_true, y_pred,
                    labels=list(range(len(CLASS_NAMES))),
                    average=None,
                    zero_division=0,
                ),
            )
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred,
            labels=list(range(len(CLASS_NAMES))),
        ),
    }


# ============================================================
# Early stopping
# ============================================================

class EarlyStopping:
    def __init__(self, patience=5, save_path=None):
        self.patience = patience
        self.save_path = save_path
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.save_path is not None:
                torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True


# ============================================================
# Permute labels
# ============================================================

def make_permuted_split(src_csv, dst_csv, seed):
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
    print(f"Protocol B | Label Permutation | Seed {seed}")
    print("=" * 60)

    original_csv = SPLIT_DIR / f"protocol_b_seed{seed}.csv"
    if not original_csv.exists():
        print(f"[SKIP] Split file not found: {original_csv}")
        return None

    permuted_csv = PERMUTED_SPLIT_DIR / f"protocol_b_seed{seed}_permuted.csv"
    make_permuted_split(original_csv, permuted_csv, seed=seed)

    split_df = pd.read_csv(permuted_csv)
    train_df = split_df[split_df["split"] == "train"]
    val_df = split_df[split_df["split"] == "validation"]
    test_df = split_df[split_df["split"] == "test"]

    train_ds = FingerprintDataset(train_df, get_train_transform())
    val_ds = FingerprintDataset(val_df, get_eval_transform())
    test_ds = FingerprintDataset(test_df, get_eval_transform())

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

    print(f"Device:  {DEVICE}")
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | "
          f"Optimizer: {OPTIMIZER_NAME} | Pretrained: {PRETRAINED} | "
          f"Patience: {PATIENCE}")
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

    checkpoint_path = CHECKPOINT_DIR / f"protocol_b_seed{seed}_permuted_best.pt"
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
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

        early_stopping.step(val_loss, model)
        if early_stopping.should_stop:
            print(f"Early stopping at epoch {epoch+1}")
            break

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
    for name, r in results["per_class_recall"].items():
        print(f"  {name}: {r:.4f}")
    print()
    print("Confusion Matrix:")
    print(results["confusion_matrix"])

    return {
        "experiment": "label_permutation",
        "protocol": "B",
        "seed": seed,
        "accuracy": results["accuracy"],
        "balanced_accuracy": results["balanced_accuracy"],
        "macro_f1": results["macro_f1"],
        "train_size": len(train_ds),
        "test_size": len(test_ds),
    }


# ============================================================
# Main: run all seeds
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
        # Avoid duplicate (experiment, protocol, seed) rows
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