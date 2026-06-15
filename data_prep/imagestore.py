"""Memory-bounded image storage for prepared datasets.

`prepare` streams each selected image's original (compressed) bytes into a
packed `images.bin` blob and forgets them ("append-and-forget"), recording only
per-image (offset, length). This keeps prep memory O(1) instead of holding every
decoded RGBA array in one giant list (which OOMs on large/booru datasets).

`DatasetImages` reads a row's image on demand: it mmaps `images.bin` and decodes
the byte slice to RGBA. It also transparently supports the legacy format where
decoded arrays were stored directly in `data.npz` under the `images` key.

For training the originals are often multi-thousand-pixel images that get shrunk
to a ~224 canvas, so decoding them in full is wasteful (it dominates data-loading
time). `build_decode_cache` writes a one-time `cache_s{N}.bin` of every image
re-encoded WEBP at <= N px (lossless where alpha matters, lossy otherwise); the
result is tiny (fits the OS page cache) and decodes ~16x faster. `DatasetImages`
uses it automatically when `cache_side` is requested and the cache is current.
"""
from __future__ import annotations
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm


def decode_rgba(source, max_side: int | None = None) -> np.ndarray:
    """Decode a path or raw bytes to a uint8 RGBA array; first frame for
    animated images (gif/webp).

    When `max_side` is given the image is decoded no larger than necessary: for
    JPEGs `draft()` performs a near-free DCT-scaled decode (1/2, 1/4, 1/8) and
    any residual oversize is bilinear-shrunk preserving aspect. Training never
    needs more than ~1.2x the canvas, so capping here makes every downstream
    augmentation (color, resize, paste) run on a small image instead of the
    full multi-thousand-pixel original -- the dominant cost.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        im = Image.open(io.BytesIO(source))
    else:
        im = Image.open(source)
    if max_side and not getattr(im, "is_animated", False):
        # draft only affects JPEG; harmless no-op otherwise. Decode to >= cap
        # so the follow-up resize has clean source pixels.
        im.draft("RGB", (max_side, max_side))
    if getattr(im, "is_animated", False):
        im = next(ImageSequence.Iterator(im)).copy()
    if max_side:
        w, h = im.size
        if max(w, h) > max_side:
            f = max_side / max(w, h)
            im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
    return np.asarray(im.convert("RGBA"), dtype=np.uint8)


class ImageStoreWriter:
    """Append raw image bytes to a packed blob, freeing each after writing.
    Returns (offsets, lengths) int64 arrays on close()."""
    def __init__(self, bin_path):
        self._f = open(bin_path, "wb")
        self._offsets: list[int] = []
        self._lengths: list[int] = []

    def add_bytes(self, data: bytes) -> None:
        self._offsets.append(self._f.tell())
        self._lengths.append(len(data))
        self._f.write(data)

    def close(self):
        if not self._f.closed:
            self._f.close()
        return (np.array(self._offsets, dtype=np.int64),
                np.array(self._lengths, dtype=np.int64))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._f.closed:
            self._f.close()


def _encode_capped(arr: np.ndarray, max_side: int, quality: int) -> bytes:
    """RGBA uint8 -> WEBP bytes, capped to max_side. Lossless when the image has
    real transparency (so composite edges stay crisp), lossy otherwise."""
    im = Image.fromarray(arr, "RGBA")
    w, h = im.size
    if max(w, h) > max_side:
        f = max_side / max(w, h)
        im = im.resize((max(1, round(w * f)), max(1, round(h * f))), Image.BILINEAR)
    has_alpha = bool(arr.shape[2] == 4 and arr[..., 3].min() < 255)
    buf = io.BytesIO()
    if has_alpha:
        im.save(buf, "WEBP", lossless=True, method=4)
    else:
        im.save(buf, "WEBP", quality=quality, method=4)
    return buf.getvalue()


def cache_paths(dataset_dir, max_side: int):
    d = Path(dataset_dir)
    return d / f"cache_s{max_side}.bin", d / f"cache_s{max_side}.npz"


def build_decode_cache(dataset_dir, max_side: int, quality: int = 92,
                       workers: int = 6) -> None:
    """Build cache_s{max_side}.{bin,npz}: every image re-encoded WEBP <= max_side.
    Idempotent and skipped if already current (keyed to images.bin size)."""
    d = Path(dataset_dir)
    bin_path, meta_path = cache_paths(d, max_side)
    src_size = (d / "images.bin").stat().st_size
    if bin_path.exists() and meta_path.exists():
        meta = np.load(meta_path)
        if int(meta["src_bytes"]) == src_size and int(meta["max_side"]) == max_side:
            return                                     # already current
    data = np.load(d / "data.npz", allow_pickle=True)
    src = DatasetImages(d, data)                       # read originals (no cache)
    n = len(src)
    # decode+encode in threads (PIL releases the GIL in its C codecs), then write
    # the blob sequentially to keep offsets ordered.
    def work(i):
        return _encode_capped(src.get(i, max_side), max_side, quality)
    with ThreadPoolExecutor(max_workers=workers) as ex, \
            ImageStoreWriter(bin_path) as w:
        for b in tqdm(ex.map(work, range(n)), total=n, desc=f"caching @{max_side}px",
                      unit="img"):
            w.add_bytes(b)
        offs, lens = w._offsets, w._lengths
    np.savez(meta_path, offsets=np.array(offs, dtype=np.int64),
             lengths=np.array(lens, dtype=np.int64),
             max_side=np.int64(max_side), src_bytes=np.int64(src_size))


class DatasetImages:
    """Row-indexed RGBA image access for a prepared dataset. Supports the packed
    blob format (images.bin + offsets/lengths in the npz) and the legacy in-npz
    object array (npz['images']).

    When `cache_side` is given and a current cache_s{cache_side} exists, reads go
    through the small capped WEBP cache instead of the full-resolution blob."""
    def __init__(self, dataset_dir, npz, cache_side: int | None = None):
        self.dir = Path(dataset_dir)
        if "images" in npz.files:                      # legacy: decoded arrays in npz
            self._legacy = npz["images"]
            self._mm = None
        else:                                          # packed blob
            self._legacy = None
            self._off = np.asarray(npz["offsets"])
            self._len = np.asarray(npz["lengths"])
            self._mm = np.memmap(self.dir / "images.bin", dtype=np.uint8, mode="r")
        self._cache_mm = self._cache_off = self._cache_len = None
        self._cache_side = None
        if cache_side is not None and self._legacy is None:
            self._attach_cache(cache_side)

    def _attach_cache(self, cache_side: int) -> None:
        bin_path, meta_path = cache_paths(self.dir, cache_side)
        if not (bin_path.exists() and meta_path.exists()):
            return
        meta = np.load(meta_path)
        if int(meta["src_bytes"]) != (self.dir / "images.bin").stat().st_size:
            return                                     # stale cache, fall back
        self._cache_off = np.asarray(meta["offsets"])
        self._cache_len = np.asarray(meta["lengths"])
        self._cache_mm = np.memmap(bin_path, dtype=np.uint8, mode="r")
        self._cache_side = cache_side

    def __len__(self) -> int:
        return len(self._legacy) if self._legacy is not None else len(self._off)

    def __getitem__(self, i: int) -> np.ndarray:
        return self.get(i)

    def get(self, i: int, max_side: int | None = None) -> np.ndarray:
        """Decode row i to RGBA, optionally capping the longest side (see
        decode_rgba). Uses the capped cache when attached and sufficient."""
        if self._legacy is not None:
            return self._legacy[i]
        if self._cache_mm is not None and (max_side is None or max_side >= self._cache_side):
            o, n = int(self._cache_off[i]), int(self._cache_len[i])
            return decode_rgba(memoryview(self._cache_mm)[o:o + n])
        if self._cache_mm is not None:                 # cache covers a larger cap
            o, n = int(self._cache_off[i]), int(self._cache_len[i])
            return decode_rgba(memoryview(self._cache_mm)[o:o + n], max_side=max_side)
        o, n = int(self._off[i]), int(self._len[i])
        return decode_rgba(memoryview(self._mm)[o:o + n], max_side=max_side)
