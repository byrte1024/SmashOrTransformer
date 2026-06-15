"""Memory-bounded image storage for prepared datasets.

`prepare` streams each selected image's original (compressed) bytes into a
packed `images.bin` blob and forgets them ("append-and-forget"), recording only
per-image (offset, length). This keeps prep memory O(1) instead of holding every
decoded RGBA array in one giant list (which OOMs on large/booru datasets).

`DatasetImages` reads a row's image on demand: it mmaps `images.bin` and decodes
the byte slice to RGBA. It also transparently supports the legacy format where
decoded arrays were stored directly in `data.npz` under the `images` key.
"""
from __future__ import annotations
import io
from pathlib import Path
import numpy as np
from PIL import Image, ImageSequence


def decode_rgba(source) -> np.ndarray:
    """Decode a path or raw bytes to a uint8 RGBA array; first frame for
    animated images (gif/webp)."""
    if isinstance(source, (bytes, bytearray)):
        im = Image.open(io.BytesIO(source))
    else:
        im = Image.open(source)
    if getattr(im, "is_animated", False):
        im = next(ImageSequence.Iterator(im)).copy()
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


class DatasetImages:
    """Row-indexed RGBA image access for a prepared dataset. Supports the packed
    blob format (images.bin + offsets/lengths in the npz) and the legacy in-npz
    object array (npz['images'])."""
    def __init__(self, dataset_dir, npz):
        if "images" in npz.files:                      # legacy: decoded arrays in npz
            self._legacy = npz["images"]
            self._mm = None
        else:                                          # packed blob
            self._legacy = None
            self._off = np.asarray(npz["offsets"])
            self._len = np.asarray(npz["lengths"])
            self._mm = np.memmap(Path(dataset_dir) / "images.bin", dtype=np.uint8, mode="r")

    def __len__(self) -> int:
        return len(self._legacy) if self._legacy is not None else len(self._off)

    def __getitem__(self, i: int) -> np.ndarray:
        if self._legacy is not None:
            return self._legacy[i]
        o, n = int(self._off[i]), int(self._len[i])
        return decode_rgba(bytes(self._mm[o:o + n]))
