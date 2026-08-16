"""
predict.py

Takes a new image and returns a prediction (AI-GENERATED or REAL) along
with a confidence score, using the trained model in ../models/ai_detector.pth.

Usage:
    python predict.py --image ../test_images/test1.jpg

Can also be imported and used directly:
    from predict import predict_image
    label, confidence = predict_image("path/to/image.jpg")
"""

import argparse
import os

import torch
import torch.nn.functional as F
from PIL import Image

from model import build_model
from preprocessing import get_eval_transforms

MODEL_PATH = os.path.join("..", "models", "ai_detector.pth")
IMG_SIZE = 224
CLASS_NAMES = {0: "REAL", 1: "AI-GENERATED"}

_model = None  # lazily loaded and cached so repeated calls are fast


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


_device = _get_device()


def _load_model() -> torch.nn.Module:
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"No trained model found at {MODEL_PATH}. Run train.py first."
            )
        model = build_model(num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        model.to(_device)
        model.eval()
        _model = model
    return _model


def predict_image(image_path: str):
    """
    Returns a tuple: (label_str, confidence_float)
    e.g. ("AI-GENERATED", 0.924)
    """
    model = _load_model()
    transform = get_eval_transforms(IMG_SIZE)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(_device)  # add batch dimension

    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, dim=1)

    label = CLASS_NAMES[predicted_idx.item()]
    return label, confidence.item()


def main():
    parser = argparse.ArgumentParser(description="Predict whether an image is AI-generated or REAL.")
    parser.add_argument("--image", required=True, help="Path to the image file to classify.")
    args = parser.parse_args()

    label, confidence = predict_image(args.image)
    short_label = "AI" if label == "AI-GENERATED" else "REAL"
    print(f"{short_label} {confidence * 100:.1f}%")


if __name__ == "__main__":
    main()
    