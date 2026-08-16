"""
preprocessing.py

Defines the image transforms used before images enter the model:
resizing, tensor conversion, and normalization.

Uses ImageNet mean/std since the model is pretrained on ImageNet
(see model.py) — this keeps the input distribution consistent with
what the pretrained backbone expects.
"""

from torchvision import transforms

IMG_SIZE = 224

# ImageNet normalization stats (standard for torchvision pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Transform pipeline used during training.
    Includes light data augmentation to help generalization.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """
    Transform pipeline used during evaluation/prediction.
    No augmentation — deterministic resize + normalize only.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
