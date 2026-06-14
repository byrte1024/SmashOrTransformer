from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def canonical_render(sprite_rgba: np.ndarray, resolution: int) -> np.ndarray:
    """Scale longest side to fit, center on a white res x res canvas -> RGB uint8."""
    im = Image.fromarray(sprite_rgba, "RGBA")
    w, h = im.size
    scale = resolution / max(w, h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
    canvas.paste(im, ((resolution - nw) // 2, (resolution - nh) // 2), im)
    bg = Image.new("RGB", (resolution, resolution), (255, 255, 255))
    bg.paste(canvas, (0, 0), canvas)
    return np.asarray(bg, dtype=np.uint8)


def to_tensor(img_rgb_uint8: np.ndarray, mean, std) -> torch.Tensor:
    arr = img_rgb_uint8.astype(np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (t - m) / s


class TrainDataset(Dataset):
    """Augmented training samples from a DataSampler -> (tensor, label)."""

    def __init__(self, sampler, mean, std):
        self.sampler = sampler
        self.mean = mean
        self.std = std

    def set_epoch(self, epoch: int) -> None:
        self.sampler.set_epoch(epoch)

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, i: int):
        img, label = self.sampler[i]
        return to_tensor(img, self.mean, self.std), torch.tensor(label, dtype=torch.float32)


class EvalDataset(Dataset):
    """Canonical (non-augmented) val/eval renders -> (tensor, label, pokemon_id)."""

    def __init__(self, dataset_dir, split, mean, std, resolution):
        d = Path(dataset_dir)
        with np.load(d / "data.npz", allow_pickle=True) as data:
            self._images = data["images"]
            self._pid = data["pokemon_id"]
            self._smash = data["smash_pct"]
        self._rows = list(json.loads((d / "split.json").read_text())[split])
        self.mean, self.std, self.resolution = mean, std, resolution

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, i: int):
        r = self._rows[i]
        img = canonical_render(self._images[r], self.resolution)
        t = to_tensor(img, self.mean, self.std)
        return t, torch.tensor(float(self._smash[r]), dtype=torch.float32), int(self._pid[r])
