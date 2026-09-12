import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Allow this script to import modules from src/
sys.path.insert(0, str(PROJECT_ROOT))


from src.dataset import FingerprintDataset
from src.model import build_resnet18
from src.transforms import get_train_transform, get_eval_transform
from src.train import train_one_epoch, validate_one_epoch, EarlyStopping
from src.evaluate import evaluate_model


def main():

    # --------------------------------------------------
    # Command-line arguments
    # --------------------------------------------------

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        required=True
    )

    parser.add_argument(
        "--epochs",
        type=int,
        required=True
    )

    parser.add_argument(
        "--lr",
        type=float,
        required=True
    )

    parser.add_argument(
        "--pretrained",
        action="store_true"
    )

    parser.add_argument(
       "--patience",
       type=int,
       required=True
    )

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)

    # --------------------------------------------------
    # Split file
    # --------------------------------------------------

    split_csv = (
        PROJECT_ROOT
        / "splits"
        / f"protocol_a_seed{args.seed}.csv"
    )

    if not split_csv.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_csv}"
        )

    # --------------------------------------------------
    # Experiment information
    # --------------------------------------------------

    print("Protocol A")
    print("-------------------------")
    print("Seed:", args.seed)
    print("Batch size:", args.batch_size)
    print("Epochs:", args.epochs)
    print("Learning rate:", args.lr)
    print("Pretrained:", args.pretrained)
    print("Patience:", args.patience)
    print("-------------------------")
    print()

    # --------------------------------------------------
    # Datasets
    # --------------------------------------------------

    train_dataset = FingerprintDataset(
        split_csv,
        split="train",
        transform=get_train_transform()
    )

    val_dataset = FingerprintDataset(
        split_csv,
        split="validation",
        transform=get_eval_transform()
    )

    test_dataset = FingerprintDataset(
        split_csv,
        split="test",
        transform=get_eval_transform()
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print("Train images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print()

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print()

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = build_resnet18(
        num_classes=8,
        pretrained=args.pretrained
    )

    model = model.to(device)

    # --------------------------------------------------
    # Loss and optimizer
    #
    # TEMPORARY:
    # Final settings still need group confirmation.
    # --------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr
    )

    # --------------------------------------------------
    # Early stopping and model saving
    # --------------------------------------------------

    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = (
        checkpoint_dir
        / f"protocol_a_seed{args.seed}_best.pt"
    )

    early_stopping = EarlyStopping(
        patience=args.patience,
        save_path=checkpoint_path
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    for epoch in range(args.epochs):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss, val_acc = validate_one_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        early_stopping.step(
            val_loss,
            model
        )

        if early_stopping.should_stop:
            print(
                f"Early stopping triggered at epoch {epoch + 1}."
            )
            break


    # --------------------------------------------------
    # Test evaluation using the best saved model
    # --------------------------------------------------

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location=device
        )
    )

    test_results = evaluate_model(
        model,
        test_loader,
        device
    )

    print()
    print("Test Results")
    print("-------------------------")
    print(f"Accuracy: {test_results['accuracy']:.4f}")
    print(
        f"Balanced Accuracy: "
        f"{test_results['balanced_accuracy']:.4f}"
    )
    print(f"Macro F1: {test_results['macro_f1']:.4f}")
    print("Confusion Matrix:")
    print(test_results["confusion_matrix"])

    
if __name__ == "__main__":
    main()