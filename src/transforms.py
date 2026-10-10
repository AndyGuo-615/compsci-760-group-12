from torchvision import transforms

# ResNet-18 input resolution used in the main experiments.
# 224 x 224 is consistent with the standard input crop size used by
# torchvision's ImageNet-pretrained ResNet-18 weights.
IMAGE_SIZE = (224, 224)

# ImageNet normalization statistics used with pretrained ResNet-18 weights.
# Order: RGB channels.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform(pretrained=False):
    transform_list = [
        # Resize the full fingerprint image without cropping or padding.
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ]

    if pretrained:
        transform_list.append(
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        )

    return transforms.Compose(transform_list)


def get_eval_transform(pretrained=False):
    transform_list = [
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ]

    if pretrained:
        transform_list.append(
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD
            )
        )

    return transforms.Compose(transform_list)