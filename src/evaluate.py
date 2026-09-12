import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(
    model,
    data_loader,
    device
):
    """
    Evaluate a trained model.

    Returns:
        Dictionary containing:
        - accuracy
        - balanced_accuracy
        - macro_f1
        - confusion_matrix
        - true_labels
        - predicted_labels
    """

    model.eval()

    true_labels = []
    predicted_labels = []

    with torch.no_grad():

        for images, labels in data_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            true_labels.extend(
                labels.cpu().numpy()
            )

            predicted_labels.extend(
                predictions.cpu().numpy()
            )

    accuracy = accuracy_score(
        true_labels,
        predicted_labels
    )

    balanced_accuracy = balanced_accuracy_score(
        true_labels,
        predicted_labels
    )

    macro_f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0
    )

    cm = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(8))
    )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "true_labels": true_labels,
        "predicted_labels": predicted_labels
    }