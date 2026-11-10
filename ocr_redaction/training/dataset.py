"""IIIT-5K dataset loader with aspect-ratio preserving preprocessing.

The CSV files live under DATASETS/IIIT-5K/train_test while images live under
DATASETS/IIIT-5K/IIIT5K-Word_V3.0/IIIT5K/{train,test}.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def resolve_dataset_layout(data_root):
    p = Path(data_root).expanduser().resolve()

    csv_candidates = [
        p,
        p / "train_test",
        p / "IIIT5K-Word_V3.0" / "IIIT5K",
        p / "IIIT5K-Word_V3.0" / "IIIT5K" / "train_test",
    ]
    csv_dir = next(
        (c for c in csv_candidates
         if (c / "traindata.csv").exists() and (c / "testdata.csv").exists()),
        None,
    )

    image_candidates = [
        p,
        p / "IIIT5K-Word_V3.0" / "IIIT5K",
        p / "IIIT5K",
    ]
    image_root = next(
        (c for c in image_candidates if (c / "train").is_dir() and (c / "test").is_dir()),
        None,
    )

    if csv_dir is None or image_root is None:
        raise FileNotFoundError(
            "Could not resolve IIIT-5K layout. Pass the DATASETS\\IIIT-5K directory."
        )
    return csv_dir, image_root


class IIIT5KDataset(Dataset):
    def __init__(self, data_root, split="train", height=32, width=192, augment=False):
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        self.csv_dir, self.image_root = resolve_dataset_layout(data_root)
        self.split = split
        self.height = int(height)
        self.width = int(width)
        self.augment = bool(augment)
        csv_name = "traindata.csv" if split == "train" else "testdata.csv"
        self.df = pd.read_csv(self.csv_dir / csv_name)

    def __len__(self):
        return len(self.df)

    def _image_path(self, rel):
        rel = str(rel).replace("\\", "/")
        for candidate in (self.image_root / rel, self.image_root / rel.lstrip("./")):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Image referenced by CSV was not found: {rel}")

    def _augment(self, img):
        if np.random.rand() < 0.35:
            alpha = np.random.uniform(0.90, 1.10)
            beta = np.random.uniform(-10, 10)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        if np.random.rand() < 0.15:
            img = cv2.GaussianBlur(img, (3, 3), 0)
        return img

    def _load(self, path):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(path)
        if self.augment:
            img = self._augment(img)

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"Invalid image shape for {path}")

        scale = self.height / float(h)
        new_w = max(1, int(round(w * scale)))
        new_h = self.height

        if new_w > self.width:
            scale = self.width / float(w)
            new_w = self.width
            new_h = max(1, int(round(h * scale)))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((self.height, self.width), 255, dtype=np.uint8)
        y0 = max(0, (self.height - new_h) // 2)
        canvas[y0:y0 + new_h, :new_w] = resized

        # valid_width is the number of image columns containing the word.
        # The CRNN reduces width by exactly 4x, so CTC can use only the
        # corresponding number of sequence steps and ignore right-side padding.
        valid_width = max(4, new_w)

        x = canvas.astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).unsqueeze(0), valid_width

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image, valid_width = self._load(self._image_path(row["ImgName"]))
        text = str(row["GroundTruth"]).strip().upper()
        return image, text, valid_width
