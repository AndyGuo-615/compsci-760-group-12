import torch


def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
    device
):
    """
    Train the model for one epoch.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def validate_one_epoch(
    model,
    data_loader,
    criterion,
    device
):
    """
    Evaluate the model on a validation set.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


class EarlyStopping:
    """
    Stop training when validation loss stops improving.

    The best model is saved automatically.
    """

    def __init__(
        self,
        patience,
        save_path,
        min_delta=0.0
    ):
        self.patience = patience
        self.save_path = save_path
        self.min_delta = min_delta

        self.best_loss = None
        self.counter = 0
        self.should_stop = False

    def step(self, validation_loss, model):

        # First validation result
        if self.best_loss is None:
            self.best_loss = validation_loss
            torch.save(model.state_dict(), self.save_path)
            return

        # Validation loss improved
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0

            torch.save(
                model.state_dict(),
                self.save_path
            )

        # No improvement
        else:
            self.counter += 1

            if self.counter >= self.patience:
                self.should_stop = True