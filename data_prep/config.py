from __future__ import annotations
from dataclasses import dataclass, field, asdict

CATEGORIES = ("portrait", "in-game", "animated", "booru")
SCALE_METHODS = ("nearest", "bilinear", "bicubic", "lanczos")
SPLIT_STRATEGIES = ("pokemon", "image")

DEFAULT_SCALE_W = (0.65, 1.10)
DEFAULT_SCALE_H = (0.65, 1.10)
DEFAULT_POS_X = (0.10, 0.90)
DEFAULT_POS_Y = (0.10, 0.90)
DEFAULT_ROTATION = (-10.0, 10.0)
DEFAULT_BG_DIRS = ["backgrounds/real", "backgrounds/pokemon_battle"]
DEFAULT_BG_PROB = 0.8
DEFAULT_FLIP = 0.5
DEFAULT_CROP_SCALE = (0.6, 1.0)
DEFAULT_PHOTO_SCALE = (0.9, 1.2)
DEFAULT_PHOTO_POS_X = (0.4, 0.6)
DEFAULT_PHOTO_POS_Y = (0.4, 0.6)
_FLAT_AUG_KEYS = {"scale", "scale_method", "position", "rotation", "background", "flip"}


def _pair(v, default):
    if v is None:
        return tuple(float(x) for x in default)
    return (float(v[0]), float(v[1]))


@dataclass
class NamesCfg:
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class GensCfg:
    include: list[int] = field(default_factory=list)
    exclude: list[int] = field(default_factory=list)


@dataclass
class SelectionCfg:
    categories: list[str] = field(default_factory=list)
    names: NamesCfg = field(default_factory=NamesCfg)
    gens: GensCfg = field(default_factory=GensCfg)


@dataclass
class SplitCfg:
    strategy: str = "pokemon"
    val_frac: float = 0.1


@dataclass
class VariationsCfg:
    mode: str = "flat"
    n: int | None = 1
    target: int | None = None


@dataclass
class ScaleAugCfg:
    w: tuple[float, float] = DEFAULT_SCALE_W
    h: tuple[float, float] = DEFAULT_SCALE_H


@dataclass
class PositionAugCfg:
    x: tuple[float, float] = DEFAULT_POS_X
    y: tuple[float, float] = DEFAULT_POS_Y


@dataclass
class BackgroundCfg:
    prob: float = DEFAULT_BG_PROB
    dirs: list[str] = field(default_factory=lambda: list(DEFAULT_BG_DIRS))
    weights: list[float] | None = None


@dataclass
class SpriteAugCfg:
    scale: ScaleAugCfg = field(default_factory=ScaleAugCfg)
    scale_method: str = "bilinear"
    position: PositionAugCfg = field(default_factory=PositionAugCfg)
    rotation: tuple[float, float] = DEFAULT_ROTATION
    flip: float = DEFAULT_FLIP
    background: BackgroundCfg = field(default_factory=BackgroundCfg)


@dataclass
class ColorCfg:
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.05


@dataclass
class PhotoBgCfg:
    white: bool = True
    black: bool = True
    dirs: list[str] = field(default_factory=lambda: list(DEFAULT_BG_DIRS))
    weights: list[float] | None = None


@dataclass
class PhotoAugCfg:
    scale: tuple[float, float] = DEFAULT_PHOTO_SCALE
    position: PositionAugCfg = field(
        default_factory=lambda: PositionAugCfg(x=DEFAULT_PHOTO_POS_X, y=DEFAULT_PHOTO_POS_Y))
    color: ColorCfg = field(default_factory=lambda: ColorCfg(
        brightness=0.2, contrast=0.3, saturation=0.2, hue=0.1))
    flip: float = DEFAULT_FLIP
    background: PhotoBgCfg = field(default_factory=PhotoBgCfg)


@dataclass
class AugmentationsCfg:
    sprite: SpriteAugCfg = field(default_factory=SpriteAugCfg)
    photo: PhotoAugCfg = field(default_factory=PhotoAugCfg)


def _parse_sprite(sp: dict) -> SpriteAugCfg:
    sc = sp.get("scale", {}) or {}
    po = sp.get("position", {}) or {}
    bg = sp.get("background", {}) or {}
    if "prob" in bg:
        prob = float(bg["prob"])
    elif bg.get("mode") == "white":           # back-compat: old white stub
        prob = 0.0
    else:
        prob = DEFAULT_BG_PROB
    return SpriteAugCfg(
        scale=ScaleAugCfg(w=_pair(sc.get("w"), DEFAULT_SCALE_W),
                          h=_pair(sc.get("h"), DEFAULT_SCALE_H)),
        scale_method=sp.get("scale_method", "bilinear"),
        position=PositionAugCfg(x=_pair(po.get("x"), DEFAULT_POS_X),
                                y=_pair(po.get("y"), DEFAULT_POS_Y)),
        rotation=_pair(sp.get("rotation"), DEFAULT_ROTATION),
        flip=float(sp.get("flip", DEFAULT_FLIP)),
        background=BackgroundCfg(prob=prob,
                                dirs=list(bg.get("dirs", DEFAULT_BG_DIRS)),
                                weights=bg.get("weights")),
    )


def _parse_photo(ph: dict) -> PhotoAugCfg:
    col = ph.get("color", {}) or {}
    po = ph.get("position", {}) or {}
    bg = ph.get("background", {}) or {}
    return PhotoAugCfg(
        scale=_pair(ph.get("scale"), DEFAULT_PHOTO_SCALE),
        position=PositionAugCfg(x=_pair(po.get("x"), DEFAULT_PHOTO_POS_X),
                                y=_pair(po.get("y"), DEFAULT_PHOTO_POS_Y)),
        color=ColorCfg(brightness=float(col.get("brightness", 0.2)),
                       contrast=float(col.get("contrast", 0.3)),
                       saturation=float(col.get("saturation", 0.2)),
                       hue=float(col.get("hue", 0.1))),
        flip=float(ph.get("flip", DEFAULT_FLIP)),
        background=PhotoBgCfg(white=bool(bg.get("white", True)),
                              black=bool(bg.get("black", True)),
                              dirs=list(bg.get("dirs", DEFAULT_BG_DIRS)),
                              weights=bg.get("weights")),
    )


@dataclass
class DataConfig:
    name: str
    resolution: int
    seed: int = 0
    minimages: int = 1
    selection: SelectionCfg = field(default_factory=SelectionCfg)
    variations: VariationsCfg = field(default_factory=VariationsCfg)
    split: SplitCfg = field(default_factory=SplitCfg)
    augmentations: AugmentationsCfg = field(default_factory=AugmentationsCfg)

    @staticmethod
    def from_dict(d: dict) -> "DataConfig":
        sel = d.get("selection", {}) or {}
        names = sel.get("names", {}) or {}
        gens = sel.get("gens", {}) or {}
        selection = SelectionCfg(
            categories=list(sel.get("categories", []) or []),
            names=NamesCfg(include=list(names.get("include", []) or []),
                           exclude=list(names.get("exclude", []) or [])),
            gens=GensCfg(include=list(gens.get("include", []) or []),
                         exclude=list(gens.get("exclude", []) or [])),
        )

        var_raw = d.get("variations", 1)
        if isinstance(var_raw, dict):
            if "mode" in var_raw:
                variations = VariationsCfg(mode=var_raw.get("mode", "flat"),
                                           n=var_raw.get("n"), target=var_raw.get("target"))
            else:
                variations = VariationsCfg(mode="fill_so", n=None,
                                           target=var_raw.get("fill_so", None))
        else:
            variations = VariationsCfg(mode="flat", n=int(var_raw), target=None)

        sp = d.get("split", {}) or {}
        split = SplitCfg(strategy=sp.get("strategy", "pokemon"),
                         val_frac=float(sp.get("val_frac", 0.1)))

        au = d.get("augmentations", {}) or {}
        sprite_raw = au.get("sprite")
        if sprite_raw is None:                    # back-compat: old flat aug == sprite
            sprite_raw = au if (set(au) & _FLAT_AUG_KEYS) else {}
        aug = AugmentationsCfg(sprite=_parse_sprite(sprite_raw),
                               photo=_parse_photo(au.get("photo", {}) or {}))

        cfg = DataConfig(
            name=d["name"], resolution=int(d["resolution"]),
            seed=int(d.get("seed", 0)), minimages=int(d.get("minimages", 1)),
            selection=selection, variations=variations, split=split, augmentations=aug,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.minimages < 0:
            raise ValueError("minimages must be >= 0")
        if self.split.strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split.strategy must be one of {SPLIT_STRATEGIES}")
        if self.augmentations.sprite.scale_method not in SCALE_METHODS:
            raise ValueError(f"scale_method must be one of {SCALE_METHODS}")
        if not (0.0 <= self.augmentations.sprite.background.prob <= 1.0):
            raise ValueError("background.prob must be in [0,1]")
        for c in self.selection.categories:
            if c not in CATEGORIES:
                raise ValueError(f"unknown category {c!r}")
        lo, hi = self.augmentations.photo.scale
        if not (0.0 < lo <= hi):
            raise ValueError("photo.scale must satisfy 0 < lo <= hi")
        for fl in (self.augmentations.sprite.flip, self.augmentations.photo.flip):
            if not (0.0 <= fl <= 1.0):
                raise ValueError("flip probability must be in [0,1]")
        if self.variations.mode == "flat" and (self.variations.n or 0) < 1:
            raise ValueError("flat variations must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
