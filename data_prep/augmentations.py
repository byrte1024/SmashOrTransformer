from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance
from .config import DataConfig

_RESAMPLE = {
    "nearest": Image.NEAREST, "bilinear": Image.BILINEAR,
    "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS,
}
_BG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
        return img.rotate(rng.uniform(*self.deg_range), resample=Image.BILINEAR, expand=True)


class Position(Augmentation):
    def __init__(self, x_range, y_range):
        self.x_range, self.y_range = x_range, y_range

    def apply(self, img, rng, resolution):
        canvas = Image.new("RGBA", (resolution, resolution), (0, 0, 0, 0))
        cx = rng.uniform(*self.x_range) * resolution
        cy = rng.uniform(*self.y_range) * resolution
        canvas.paste(img, (round(cx - img.width / 2), round(cy - img.height / 2)), img)
        return canvas


class HorizontalFlip(Augmentation):
    def __init__(self, prob):
        self.prob = prob

    def apply(self, img, rng, resolution):
        if self.prob and rng.random() < self.prob:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img


class RandomResizedCrop(Augmentation):
    def __init__(self, scale_range, ratio=(0.8, 1.25)):
        self.scale_range, self.ratio = scale_range, ratio

    def apply(self, img, rng, resolution):
        W, H = img.size
        area = W * H
        crop = img
        for _ in range(10):
            a = rng.uniform(*self.scale_range) * area
            ar = rng.uniform(*self.ratio)
            w, h = int(round((a * ar) ** 0.5)), int(round((a / ar) ** 0.5))
            if 0 < w <= W and 0 < h <= H:
                x = int(rng.integers(0, W - w + 1))
                y = int(rng.integers(0, H - h + 1))
                crop = img.crop((x, y, x + w, y + h))
                break
        return crop.resize((resolution, resolution), Image.BILINEAR)


class ColorJitter(Augmentation):
    def __init__(self, brightness, contrast, saturation, hue):
        self.b, self.c, self.s, self.h = brightness, contrast, saturation, hue

    def apply(self, img, rng, resolution):
        im = img.convert("RGB")
        if self.b:
            im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-self.b, self.b))
        if self.c:
            im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-self.c, self.c))
        if self.s:
            im = ImageEnhance.Color(im).enhance(1 + rng.uniform(-self.s, self.s))
        if self.h:
            hsv = np.asarray(im.convert("HSV")).astype(np.int16)
            hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(-self.h, self.h) * 255)) % 256
            im = Image.fromarray(hsv.astype(np.uint8), "HSV").convert("RGB")
        return im


class ToRGB(Augmentation):
    def apply(self, img, rng, resolution):
        return img.convert("RGB")


class BackgroundPool:
    """Lazily lists background images across dirs; caches each chosen image
    resized + center-cropped to the canvas. Missing dirs are skipped."""
    def __init__(self, dirs, weights=None):
        self.groups, gw = [], []
        for i, d in enumerate(dirs or []):
            p = Path(d)
            files = sorted(f for f in p.glob("*") if f.suffix.lower() in _BG_EXTS) \
                if p.is_dir() else []
            if files:
                self.groups.append(files)
                gw.append(float(weights[i]) if weights and i < len(weights) else 1.0)
        self._gw = np.array(gw, dtype=float) if self.groups else None
        self._cache = {}

    def __len__(self):
        return sum(len(g) for g in self.groups)

    def empty(self):
        return not self.groups

    def sample(self, rng, resolution):
        if not self.groups:
            return None
        gi = int(rng.choice(len(self.groups), p=self._gw / self._gw.sum()))
        group = self.groups[gi]
        path = group[int(rng.integers(len(group)))]
        key = (str(path), resolution)
        if key not in self._cache:
            self._cache[key] = self._load(path, resolution)
        return self._cache[key]

    @staticmethod
    def _load(path, res):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        s = res / min(w, h)
        nw, nh = max(res, round(w * s)), max(res, round(h * s))
        im = im.resize((nw, nh), Image.BILINEAR)
        l, t = (nw - res) // 2, (nh - res) // 2
        return im.crop((l, t, l + res, t + res))


class CompositeBackground(Augmentation):
    """Flatten an RGBA res x res canvas onto a background (real with prob, else
    white) -> RGB."""
    def __init__(self, pool=None, prob=0.0):
        self.pool, self.prob = pool, prob

    def apply(self, img, rng, resolution):
        bg = None
        if self.pool is not None and self.prob > 0 and not self.pool.empty() \
                and rng.random() < self.prob:
            sampled = self.pool.sample(rng, resolution)
            bg = sampled.copy() if sampled is not None else None
        if bg is None:
            bg = Image.new("RGB", (resolution, resolution), (255, 255, 255))
        bg.paste(img, (0, 0), img)
        return bg


class Compose(Augmentation):
    def __init__(self, steps: list[Augmentation]):
        self.steps = steps

    def apply(self, img, rng, resolution):
        for step in self.steps:
            img = step.apply(img, rng, resolution)
        return img


def build_sprite_aug(cfg: DataConfig) -> Compose:
    s = cfg.augmentations.sprite
    pool = BackgroundPool(s.background.dirs, s.background.weights)
    return Compose([
        Scale(s.scale.w, s.scale.h, s.scale_method),
        Rotate(s.rotation),
        Position(s.position.x, s.position.y),
        HorizontalFlip(s.flip),
        CompositeBackground(pool=pool, prob=s.background.prob),
    ])


def build_photo_aug(cfg: DataConfig) -> Compose:
    p = cfg.augmentations.photo
    return Compose([
        RandomResizedCrop(p.crop_scale),
        ColorJitter(p.color.brightness, p.color.contrast, p.color.saturation, p.color.hue),
        HorizontalFlip(p.flip),
        ToRGB(),
    ])


def build_augmentations(cfg: DataConfig) -> Compose:
    """Back-compat alias: the sprite pipeline."""
    return build_sprite_aug(cfg)
