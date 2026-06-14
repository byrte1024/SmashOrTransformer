from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from .config import DataConfig

_RESAMPLE = {
    "nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
}


class Augmentation(ABC):
    @abstractmethod
    def apply(self, img: Image.Image, rng: np.random.Generator,
              resolution: int) -> Image.Image:
        ...


class Scale(Augmentation):
    def __init__(self, w_range, h_range, method):
        self.w_range, self.h_range, self.method = w_range, h_range, method

    def apply(self, img, rng, resolution):
        w = max(1, round(rng.uniform(*self.w_range) * resolution))
        h = max(1, round(rng.uniform(*self.h_range) * resolution))
        return img.resize((w, h), _RESAMPLE[self.method])


class Rotate(Augmentation):
    def __init__(self, deg_range):
        self.deg_range = deg_range

    def apply(self, img, rng, resolution):
        deg = rng.uniform(*self.deg_range)
        return img.rotate(deg, resample=Image.BILINEAR, expand=True)


class Position(Augmentation):
    def __init__(self, x_range, y_range):
        self.x_range, self.y_range = x_range, y_range

    def apply(self, img, rng, resolution):
        canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        cx = rng.uniform(*self.x_range) * resolution
        cy = rng.uniform(*self.y_range) * resolution
        x = round(cx - img.width / 2)
        y = round(cy - img.height / 2)
        canvas.paste(img, (x, y), img)
        return canvas


class CompositeBackground(Augmentation):
    def __init__(self, mode="white"):
        self.mode = mode

    def apply(self, img, rng, resolution):
        if self.mode == "white":
            bg = Image.new("RGB", img.size, (255, 255, 255))
        else:
            bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, (0, 0), img)
        return bg


class Compose(Augmentation):
    def __init__(self, steps: list[Augmentation]):
        self.steps = steps

    def apply(self, img, rng, resolution):
        for step in self.steps:
            img = step.apply(img, rng, resolution)
        return img


def build_augmentations(cfg: DataConfig) -> Compose:
    a = cfg.augmentations
    return Compose([
        Scale(a.scale.w, a.scale.h, a.scale_method),
        Rotate(a.rotation),
        Position(a.position.x, a.position.y),
        CompositeBackground(a.background.mode),
    ])
