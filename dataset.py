import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch.utils.data import Dataset

CLASS_NAMES = ["Unlabeled", "Car", "Person", "Bicycle", "Curve", "Car stop",
               "Guardrail", "Color cone", "Bump"]


def original_ids(root, split):
    ids = [s.strip().removesuffix("_flip") for s in (Path(root) / f"{split}.txt").read_text().splitlines() if s.strip()]
    return list(dict.fromkeys(ids))


class MFNetDataset(Dataset):
    def __init__(self, root, split, size=(384, 512), augment=False):
        self.root, self.split = Path(root), split
        self.ids, self.size, self.augment = original_ids(root, split), size, augment
        for sid in self.ids:
            for sub in ("images", "labels"):
                if not (self.root / sub / f"{sid}.png").is_file():
                    raise FileNotFoundError(f"Missing {sub}/{sid}.png")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sid = self.ids[index]
        raw = np.array(Image.open(self.root / "images" / f"{sid}.png"))
        if raw.shape != (480, 640, 4):
            raise ValueError(f"Unexpected raw RGB-T shape for {sid}: {raw.shape}")
        rgb, thermal = Image.fromarray(raw[:, :, :3]), Image.fromarray(raw[:, :, 3])
        gt = Image.open(self.root / "labels" / f"{sid}.png")
        if gt.mode not in ("L", "P"):
            raise ValueError(f"Expected indexed single-channel labels, got {gt.mode}")
        width, height = self.size[1], self.size[0]
        rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
        thermal = thermal.resize((width, height), Image.Resampling.BILINEAR)
        gt = gt.resize((width, height), Image.Resampling.NEAREST)
        if self.augment:
            if random.random() < 0.5:
                rgb, thermal, gt = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in (rgb, thermal, gt)]
            rgb = ImageEnhance.Brightness(rgb).enhance(random.uniform(0.85, 1.15))
        r = np.array(rgb, dtype=np.float32) / 255
        r = (r - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        t = np.array(thermal, dtype=np.float32) / 255
        # Per-image contrast normalization is also used at validation/inference.
        lo, hi = np.percentile(t, [1, 99])
        t = np.clip((t - lo) / max(float(hi - lo), 1 / 255), 0, 1)
        y = np.array(gt, dtype=np.int64)
        if y.min() < 0 or y.max() > 8:
            raise ValueError(f"Invalid label IDs for {sid}")
        return (torch.from_numpy(r.transpose(2, 0, 1).copy()),
                torch.from_numpy(t[None].copy()), torch.from_numpy(y.copy()), sid)


def verify_dataset(root):
    sets = {s: set(original_ids(root, s)) for s in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        if sets[a] & sets[b]:
            raise ValueError(f"Data split overlap: {a}/{b}")
    counts = np.zeros(9, dtype=np.int64)
    for sid in sorted(sets["train"]):
        label = np.array(Image.open(Path(root) / "labels" / f"{sid}.png"))
        if label.ndim != 2 or label.max() > 8:
            raise ValueError(f"Invalid training label {sid}")
        counts += np.bincount(label.ravel(), minlength=9)
    return {"counts": counts.tolist(), "split_counts": {s: len(v) for s, v in sets.items()}}
