import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CLASS_NAMES = ["A+", "A-", "AB+", "AB-", "B+", "B-", "O+", "O-"]


def evaluate_model(model, data_loader, device):
    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    confusion = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(8))
    )

    recalls = recall_score(
        true_labels,
        predicted_labels,
        labels=list(range(8)),
        average=None,
        zero_division=0
    )

    per_class_recall = {
        class_name: recall
        for class_name, recall in zip(CLASS_NAMES, recalls)
    }

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_recall": per_class_recall,
        "confusion_matrix": confusion,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels
    }