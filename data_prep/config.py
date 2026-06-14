from __future__ import annotations
from dataclasses import dataclass, field, asdict

CATEGORIES = ("portrait", "in-game", "animated")
SCALE_METHODS = ("nearest", "bilinear", "bicubic", "lanczos")
SPLIT_STRATEGIES = ("pokemon", "image")


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
    w: tuple[float, float] = (0.65, 1.10)
    h: tuple[float, float] = (0.65, 1.10)


@dataclass
class PositionAugCfg:
    x: tuple[float, float] = (0.10, 0.90)
    y: tuple[float, float] = (0.10, 0.90)


@dataclass
class BackgroundCfg:
    mode: str = "white"


@dataclass
class AugmentationsCfg:
    scale: ScaleAugCfg = field(default_factory=ScaleAugCfg)
    scale_method: str = "bilinear"
    position: PositionAugCfg = field(default_factory=PositionAugCfg)
    rotation: tuple[float, float] = (-10.0, 10.0)
    background: BackgroundCfg = field(default_factory=BackgroundCfg)


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
            # Handle both user input format {"fill_so": value} and roundtrip format {"mode": "fill_so", ...}
            if "mode" in var_raw:
                # Roundtrip format from to_dict()
                variations = VariationsCfg(mode=var_raw.get("mode", "flat"),
                                           n=var_raw.get("n"),
                                           target=var_raw.get("target"))
            else:
                # User input format {"fill_so": value}
                variations = VariationsCfg(mode="fill_so", n=None,
                                           target=var_raw.get("fill_so", None))
        else:
            variations = VariationsCfg(mode="flat", n=int(var_raw), target=None)

        sp = d.get("split", {}) or {}
        split = SplitCfg(strategy=sp.get("strategy", "pokemon"),
                         val_frac=float(sp.get("val_frac", 0.1)))

        au = d.get("augmentations", {}) or {}
        sc = au.get("scale", {}) or {}
        po = au.get("position", {}) or {}
        bg = au.get("background", {}) or {}
        aug = AugmentationsCfg(
            scale=ScaleAugCfg(w=_pair(sc.get("w"), (0.65, 1.10)),
                              h=_pair(sc.get("h"), (0.65, 1.10))),
            scale_method=au.get("scale_method", "bilinear"),
            position=PositionAugCfg(x=_pair(po.get("x"), (0.10, 0.90)),
                                    y=_pair(po.get("y"), (0.10, 0.90))),
            rotation=_pair(au.get("rotation"), (-10.0, 10.0)),
            background=BackgroundCfg(mode=bg.get("mode", "white")),
        )

        cfg = DataConfig(
            name=d["name"], resolution=int(d["resolution"]),
            seed=int(d.get("seed", 0)), minimages=int(d.get("minimages", 1)),
            selection=selection, variations=variations, split=split,
            augmentations=aug,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.resolution <= 0:
            raise ValueError("resolution must be positive")
        if self.split.strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"split.strategy must be one of {SPLIT_STRATEGIES}")
        if self.augmentations.scale_method not in SCALE_METHODS:
            raise ValueError(f"scale_method must be one of {SCALE_METHODS}")
        for c in self.selection.categories:
            if c not in CATEGORIES:
                raise ValueError(f"unknown category {c!r}")
        if self.variations.mode == "flat" and (self.variations.n or 0) < 1:
            raise ValueError("flat variations must be >= 1")

    def to_dict(self) -> dict:
        return asdict(self)
