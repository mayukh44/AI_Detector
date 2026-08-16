"""
dataset.py

PyTorch Dataset that loads images from:
    dataset/ai/    -> label 1 (AI-generated)
    dataset/real/  -> label 0 (REAL)

and applies the given transform pipeline (see preprocessing.py).
"""

import os
from typing import Callable, List, Tuple

from PIL import Image
from torch.utils.data import Dataset

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

LABEL_MAP = {
    "ai": 1,
    "real": 0,
}


class AIImageDataset(Dataset):
    """
    Expects a directory structure like:

        root_dir/
            ai/
                image1.jpg
                image2.jpg
            real/
                image1.jpg
                image2.jpg
    """

    def __init__(self, root_dir: str, transform: Callable = None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples: List[Tuple[str, int]] = self._load_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found in '{root_dir}'. Make sure images are placed "
                f"inside '{root_dir}/ai/' and '{root_dir}/real/'."
            )

    def _load_samples(self) -> List[Tuple[str, int]]:
        samples = []
        for folder_name, label in LABEL_MAP.items():
            folder_path = os.path.join(self.root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith(VALID_EXTENSIONS):
                    samples.append((os.path.join(folder_path, fname), label))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def class_counts(self) -> dict:
        """Returns {'ai': n, 'real': n} — useful for spotting class imbalance."""
        counts = {"ai": 0, "real": 0}
        for _, label in self.samples:
            key = "ai" if label == 1 else "real"
            counts[key] += 1
        return counts
