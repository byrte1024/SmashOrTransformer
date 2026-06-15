import io
import numpy as np
from PIL import Image
from data_prep.imagestore import (decode_rgba, ImageStoreWriter, DatasetImages,
                                  build_decode_cache, cache_paths)


def _png_bytes(color, size=(8, 8)):
    buf = io.BytesIO()
    Image.new("RGBA", size, (*color, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_rgba_from_bytes_and_gif_first_frame():
    arr = decode_rgba(_png_bytes((10, 20, 30)))
    assert arr.dtype == np.uint8 and arr.shape == (8, 8, 4)
    assert tuple(arr[0, 0]) == (10, 20, 30, 255)
    # 2-frame gif: first frame red, second blue -> decode takes the first
    buf = io.BytesIO()
    f0 = Image.new("RGB", (6, 6), (255, 0, 0)).convert("P")
    f1 = Image.new("RGB", (6, 6), (0, 0, 255)).convert("P")
    f0.save(buf, format="GIF", save_all=True, append_images=[f1], loop=0)
    g = decode_rgba(buf.getvalue())
    assert g.shape == (6, 6, 4) and g[0, 0, 0] > 200 and g[0, 0, 2] < 50  # red, not blue


def test_writer_then_packed_reader_roundtrip(tmp_path):
    blobs = [_png_bytes((200, 0, 0)), _png_bytes((0, 200, 0)), _png_bytes((0, 0, 200))]
    w = ImageStoreWriter(tmp_path / "images.bin")
    for b in blobs:
        w.add_bytes(b)
    offsets, lengths = w.close()
    assert len(offsets) == 3 and offsets[0] == 0 and offsets[1] == lengths[0]

    npz_path = tmp_path / "data.npz"
    np.savez(npz_path, offsets=offsets, lengths=lengths)
    data = np.load(npz_path, allow_pickle=True)
    store = DatasetImages(tmp_path, data)
    assert len(store) == 3
    assert tuple(store[1][0, 0]) == (0, 200, 0, 255)   # decoded second image


def test_datasetimages_legacy_object_array(tmp_path):
    imgs = np.empty(2, dtype=object)
    imgs[0] = np.full((4, 4, 4), 7, dtype=np.uint8)
    imgs[1] = np.full((4, 4, 4), 9, dtype=np.uint8)
    np.savez(tmp_path / "data.npz", images=imgs)
    data = np.load(tmp_path / "data.npz", allow_pickle=True)
    store = DatasetImages(tmp_path, data)        # no images.bin needed for legacy
    assert len(store) == 2
    assert store[1][0, 0, 0] == 9


def _big_png(color, size=(800, 400)):
    buf = io.BytesIO()
    Image.new("RGBA", size, (*color, 255)).save(buf, format="PNG")
    return buf.getvalue()


def test_decode_rgba_max_side_caps_longest_side():
    arr = decode_rgba(_big_png((1, 2, 3), (800, 400)), max_side=100)
    assert max(arr.shape[0], arr.shape[1]) == 100      # longest side capped
    assert arr.shape[1] == 100 and arr.shape[0] == 50  # aspect preserved (W,H)->(H,W)
    # without a cap the full size comes back
    full = decode_rgba(_big_png((1, 2, 3), (800, 400)))
    assert full.shape[:2] == (400, 800)


def _make_dataset(tmp_path, blobs):
    w = ImageStoreWriter(tmp_path / "images.bin")
    for b in blobs:
        w.add_bytes(b)
    offsets, lengths = w.close()
    np.savez(tmp_path / "data.npz", offsets=offsets, lengths=lengths)
    return np.load(tmp_path / "data.npz", allow_pickle=True)


def test_build_decode_cache_roundtrip_and_caps(tmp_path):
    data = _make_dataset(tmp_path, [_big_png((200, 50, 10), (800, 400)),
                                    _big_png((10, 180, 60), (400, 800))])
    build_decode_cache(tmp_path, max_side=64, workers=2)
    binp, metap = cache_paths(tmp_path, 64)
    assert binp.exists() and metap.exists()
    store = DatasetImages(tmp_path, data, cache_side=64)
    a = store.get(0)                                   # served from cache
    assert max(a.shape[0], a.shape[1]) == 64
    assert np.allclose(a[0, 0, :3], (200, 50, 10), atol=4)  # color ~preserved (lossy webp)
    # a tighter cap than the cache downsizes further
    b = store.get(1, max_side=32)
    assert max(b.shape[0], b.shape[1]) == 32


def test_decode_cache_preserves_alpha_losslessly(tmp_path):
    # half-transparent sprite: alpha must survive exactly (lossless webp path)
    im = Image.new("RGBA", (200, 200), (120, 200, 40, 255))
    im.paste((0, 0, 0, 0), (0, 0, 100, 200))           # left half fully transparent
    buf = io.BytesIO(); im.save(buf, "PNG")
    data = _make_dataset(tmp_path, [buf.getvalue()])
    build_decode_cache(tmp_path, max_side=64, workers=2)
    a = DatasetImages(tmp_path, data, cache_side=64).get(0)
    assert a[0, 0, 3] == 0 and a[0, -1, 3] == 255       # hard alpha edge intact


def test_decode_cache_is_idempotent_and_detects_staleness(tmp_path):
    data = _make_dataset(tmp_path, [_big_png((9, 9, 9), (300, 300))])
    build_decode_cache(tmp_path, max_side=48, workers=2)
    binp, _ = cache_paths(tmp_path, 48)
    mtime = binp.stat().st_mtime_ns
    build_decode_cache(tmp_path, max_side=48, workers=2)   # current -> skipped
    assert binp.stat().st_mtime_ns == mtime
    # changing images.bin size invalidates the cache -> rebuild attaches nothing stale
    _make_dataset(tmp_path, [_big_png((9, 9, 9), (300, 300)),
                             _big_png((1, 1, 1), (300, 300))])
    data2 = np.load(tmp_path / "data.npz", allow_pickle=True)
    store = DatasetImages(tmp_path, data2, cache_side=48)  # stale cache ignored
    assert store._cache_mm is None                         # fell back to originals
    assert len(store) == 2 and store.get(1, 48).shape[2] == 4
