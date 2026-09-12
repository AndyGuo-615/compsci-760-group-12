import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18(num_classes=8, pretrained=True):
    """
    Build a ResNet-18 model for blood-group classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet pretrained weights.

    Returns:
        ResNet-18 model.
    """

    if pretrained:
        weights = ResNet18_Weights.DEFAULT
    else:
        weights = None

    model = resnet18(weights=weights)

    # Replace the original ImageNet classifier
    # with an 8-class classifier
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model