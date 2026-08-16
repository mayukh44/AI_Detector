"""
evaluate.py

Loads the trained model from ../models/ai_detector.pth and measures its
performance (accuracy, precision, recall, F1, confusion matrix) on a
validation split of ../dataset/.

Uses the same random split/seed logic as train.py so this reports on
data the model was validated against during training.
"""

import os

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from dataset import AIImageDataset
from model import build_model
from preprocessing import get_eval_transforms

CONFIG = {
    "dataset_dir": os.path.join("..", "dataset"),
    "model_path": os.path.join("..", "models", "ai_detector.pth"),
    "img_size": 224,
    "batch_size": 16,
    "val_split": 0.2,
    "seed": 42,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def evaluate():
    device = get_device()

    if not os.path.exists(CONFIG["model_path"]):
        raise FileNotFoundError(
            f"No trained model found at {CONFIG['model_path']}. Run train.py first."
        )

    full_ds = AIImageDataset(CONFIG["dataset_dir"], transform=get_eval_transforms(CONFIG["img_size"]))

    total = len(full_ds)
    val_size = int(total * CONFIG["val_split"])
    val_size = max(1, min(val_size, total - 1)) if total > 1 else 0
    train_size = total - val_size
    generator = torch.Generator().manual_seed(CONFIG["seed"])
    _, val_indices = random_split(range(len(full_ds)), [train_size, val_size], generator=generator)

    val_subset = torch.utils.data.Subset(full_ds, val_indices.indices)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = build_model(num_classes=2)
    model.load_state_dict(torch.load(CONFIG["model_path"], map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    print(f"Validation samples: {len(all_labels)}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision (AI class): {precision:.4f}")
    print(f"Recall (AI class):    {recall:.4f}")
    print(f"F1-score (AI class):  {f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred, order=[REAL, AI]):")
    print(cm)


if __name__ == "__main__":
    evaluate()