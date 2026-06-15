import io
import numpy as np
from PIL import Image
from data_prep.imagestore import decode_rgba, ImageStoreWriter, DatasetImages


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
