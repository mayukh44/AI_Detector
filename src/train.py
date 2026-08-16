# """
# train.py

# Trains the AI-vs-REAL image classifier using the labeled images in
# ../dataset/ai/ and ../dataset/real/, then saves the trained model to
# ../models/ai_detector.pth.
# """

# import os

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# from dataset import AIImageDataset
# from model import build_model
# from preprocessing import get_train_transforms, get_eval_transforms

# CONFIG = {
#     "dataset_dir": os.path.join("..", "dataset"),
#     "model_out_path": os.path.join("..", "models", "ai_detector.pth"),
#     "img_size": 224,
#     "batch_size": 16,
#     "epochs": 10,
#     "learning_rate": 1e-3,
#     "val_split": 0.2,
#     "freeze_backbone": True,
#     "seed": 42,
# }


# def get_device() -> torch.device:
#     return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def train():
#     torch.manual_seed(CONFIG["seed"])
#     device = get_device()
#     print(f"Using device: {device}")

#     # NOTE: we build the dataset twice on purpose — once with train transforms
#     # (augmentation) and once with eval transforms (no augmentation) — then
#     # split indices so the validation subset never sees augmented images.
#     full_train_ds = AIImageDataset(CONFIG["dataset_dir"], transform=get_train_transforms(CONFIG["img_size"]))
#     full_eval_ds = AIImageDataset(CONFIG["dataset_dir"], transform=get_eval_transforms(CONFIG["img_size"]))

#     print(f"Class counts: {full_train_ds.class_counts()}")

#     val_size = int(len(full_train_ds) * CONFIG["val_split"])
#     train_size = len(full_train_ds) - val_size

#     generator = torch.Generator().manual_seed(CONFIG["seed"])
#     train_indices, val_indices = random_split(
#         range(len(full_train_ds)), [train_size, val_size], generator=generator
#     )

#     train_subset = torch.utils.data.Subset(full_train_ds, train_indices.indices)
#     val_subset = torch.utils.data.Subset(full_eval_ds, val_indices.indices)

#     train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
#     val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)

#     model = build_model(num_classes=2, freeze_backbone=CONFIG["freeze_backbone"]).to(device)

#     criterion = nn.CrossEntropyLoss()
#     # Only the trainable params (the new head, if backbone is frozen) get optimized.
#     trainable_params = [p for p in model.parameters() if p.requires_grad]
#     optimizer = torch.optim.Adam(trainable_params, lr=CONFIG["learning_rate"])

#     best_val_acc = 0.0
#     os.makedirs(os.path.dirname(CONFIG["model_out_path"]), exist_ok=True)

#     for epoch in range(1, CONFIG["epochs"] + 1):
#         model.train()
#         running_loss, correct, total = 0.0, 0, 0

#         loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
#         for images, labels in loop:
#             images, labels = images.to(device), labels.to(device)

#             optimizer.zero_grad()
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item() * images.size(0)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)

#             loop.set_postfix(loss=running_loss / total, acc=correct / total)

#         val_acc = _evaluate(model, val_loader, device)
#         print(f"Epoch {epoch}: train_acc={correct / total:.4f}  val_acc={val_acc:.4f}")

#         if val_acc >= best_val_acc:
#             best_val_acc = val_acc
#             torch.save(model.state_dict(), CONFIG["model_out_path"])
#             print(f"  -> Saved new best model (val_acc={val_acc:.4f}) to {CONFIG['model_out_path']}")

#     print(f"Training complete. Best val_acc={best_val_acc:.4f}")


# def _evaluate(model, loader, device) -> float:
#     model.eval()
#     correct, total = 0, 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             preds = outputs.argmax(dim=1)
#             correct += (preds == labels).sum().item()
#             total += labels.size(0)
#     return correct / total if total > 0 else 0.0


# if __name__ == "__main__":
#     train()


"""
train.py

Trains the AI-vs-REAL image classifier using the labeled images in
../dataset/ai/ and ../dataset/real/, then saves the trained model to
../models/ai_detector.pth.
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import AIImageDataset
from model import build_model
from preprocessing import get_train_transforms, get_eval_transforms

CONFIG = {
    "dataset_dir": os.path.join("..", "dataset"),
    "model_out_path": os.path.join("..", "models", "ai_detector.pth"),
    "img_size": 224,
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 1e-3,
    "val_split": 0.2,
    "freeze_backbone": True,
    "seed": 42,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    return torch.device("cpu")


def train():
    torch.manual_seed(CONFIG["seed"])
    device = get_device()
    print(f"Using device: {device}")

    # NOTE: we build the dataset twice on purpose — once with train transforms
    # (augmentation) and once with eval transforms (no augmentation) — then
    # split indices so the validation subset never sees augmented images.
    full_train_ds = AIImageDataset(CONFIG["dataset_dir"], transform=get_train_transforms(CONFIG["img_size"]))
    full_eval_ds = AIImageDataset(CONFIG["dataset_dir"], transform=get_eval_transforms(CONFIG["img_size"]))

    print(f"Class counts: {full_train_ds.class_counts()}")

    total = len(full_train_ds)
    val_size = int(total * CONFIG["val_split"])
    # Guarantee at least 1 sample in each split so tiny/smoke-test datasets
    # don't crash with an empty validation set. NOTE: with datasets this
    # small the resulting model is not meaningful — this is purely so the
    # pipeline can be verified end-to-end.
    val_size = max(1, min(val_size, total - 1)) if total > 1 else 0
    train_size = total - val_size

    generator = torch.Generator().manual_seed(CONFIG["seed"])
    train_indices, val_indices = random_split(
        range(len(full_train_ds)), [train_size, val_size], generator=generator
    )

    train_subset = torch.utils.data.Subset(full_train_ds, train_indices.indices)
    val_subset = torch.utils.data.Subset(full_eval_ds, val_indices.indices)

    train_loader = DataLoader(train_subset, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=CONFIG["batch_size"], shuffle=False)

    model = build_model(num_classes=2, freeze_backbone=CONFIG["freeze_backbone"]).to(device)

    criterion = nn.CrossEntropyLoss()
    # Only the trainable params (the new head, if backbone is frozen) get optimized.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=CONFIG["learning_rate"])

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(CONFIG["model_out_path"]), exist_ok=True)

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{CONFIG['epochs']}")
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=running_loss / total, acc=correct / total)

        val_acc = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_acc={correct / total:.4f}  val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CONFIG["model_out_path"])
            print(f"  -> Saved new best model (val_acc={val_acc:.4f}) to {CONFIG['model_out_path']}")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")


def _evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


if __name__ == "__main__":
    train()