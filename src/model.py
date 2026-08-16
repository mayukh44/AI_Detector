"""
model.py

Defines the image classification model.

Uses transfer learning: a ResNet18 backbone pretrained on ImageNet, with
its final fully-connected layer replaced to output 2 classes
(0 = REAL, 1 = AI-generated).

Transfer learning is used instead of training a CNN from scratch because
it needs far less data and far less training time to reach good accuracy.
"""

import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 2, freeze_backbone: bool = True) -> nn.Module:
    """
    Builds a ResNet18-based classifier.

    Args:
        num_classes: number of output classes (2: real vs ai).
        freeze_backbone: if True, freezes all pretrained layers and only
            trains the new final classification layer. This trains faster
            and works well with smaller datasets. Set to False to fine-tune
            the whole network once you have more data.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final fully-connected layer.
    # This new layer is always trainable, even when the backbone is frozen.
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )

    return model
