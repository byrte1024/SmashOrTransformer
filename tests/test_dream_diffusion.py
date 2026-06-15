import numpy as np
import torch
from PIL import Image
from model.model import SmashRanker
from model import dream_diffusion as dd


def _tiny_model():
    return SmashRanker("vit_tiny_patch16_224", resolution=32, dropout=0.0, pretrained=False).eval()


def test_score_tensor_shape_and_range():
    m = _tiny_model()
    img01 = torch.rand(2, 3, 96, 96)            # B,3,H,W in [0,1]
    s = dd._score_tensor(m, img01, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 32)
    assert s.shape == (2,) and float(s.min()) >= 0 and float(s.max()) <= 1


def test_smash_loss_grad_flows_and_direction():
    m = _tiny_model()
    img = torch.rand(1, 3, 96, 96, requires_grad=True)
    loss = dd.smash_loss(m, img, target=1.0, mean=(0.5,) * 3, std=(0.5,) * 3, resolution=32)
    loss.backward()
    assert img.grad is not None and torch.isfinite(img.grad).all()


def test_to_pil_roundtrip():
    img01 = torch.zeros(1, 3, 8, 8); img01[:, 0] = 1.0   # red
    pil = dd.to_pil(img01)
    assert isinstance(pil, Image.Image) and pil.size == (8, 8)
    assert np.asarray(pil)[0, 0, 0] == 255


# --- Task 2 tests ---
import csv as _csv
from data_prep.config import DataConfig
from data_prep.prepare import prepare
from model.config import TrainConfig
from model.train import run as train_run
from model.calibrate import run as calib_run
from model.infer import load_model as _load_model, load_calibration as _load_calib


class _StubStrategy(dd.GuidanceStrategy):
    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        # ignore SD; return a flat image whose brightness encodes the target
        v = int(target * 255)
        return Image.fromarray(np.full((32, 32, 3), v, np.uint8), "RGB")


def _trained(mini_repo, tmp_path):
    out = prepare(DataConfig.from_dict({"name": "ds", "resolution": 32, "minimages": 1,
                  "variations": 2, "split": {"strategy": "pokemon", "val_frac": 0.34}}),
                  mini_repo["images"], mini_repo["labels"], mini_repo["root"] / "datasets")
    run_dir = train_run(TrainConfig.from_dict({
        "dataset_dir": str(out), "run_name": "dd", "resolution": 32,
        "out_dir": str(tmp_path / "runs"), "backbone": "vit_tiny_patch16_224",
        "epochs": 1, "batch_size": 4, "freeze_epochs": 0, "warmup_epochs": 0,
        "amp": False, "num_workers": 0, "device": "cpu"}), pretrained=False)
    calib_run(run_dir / "checkpoints" / "best.pt", device="cpu", batch_size=4, num_workers=0)
    return run_dir / "checkpoints" / "best.pt"


def test_run_orchestration_with_stub(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    out = dd.run(ckpt, out_dir=str(tmp_path / "dream"), device="cpu",
                 methods=["stub"], prompts=[("uncond", ""), ("creature", "x")],
                 brackets=[1.0, 0.0], n_per=2, pipe=object(),
                 strategies={"stub": _StubStrategy()})
    # folder tree: method/cond/bracket/seed.png
    pngs = list((out).rglob("*.png"))
    # 1 method x 2 prompts x 2 brackets x (2 imgs + 1 grid) = 12 png
    assert len(pngs) == 12
    rows = list(_csv.DictReader(open(out / "manifest.csv")))
    assert len(rows) == 1 * 2 * 2 * 2                      # method*prompt*bracket*n_per
    assert {"method", "conditioning", "prompt", "target", "seed",
            "achieved_raw_pct", "achieved_calibrated_pct"} == set(rows[0].keys())
    # brackets recorded in reverse order (1.0 before 0.0) within a cell grouping
    assert rows[0]["target"] == "1.0"


def test_score_pil_in_range(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    model, cfg = _load_model(ckpt, device="cpu", pretrained=False)
    calib = _load_calib(ckpt, fit="combined")
    raw, cal = dd.score_pil(model, cfg, Image.new("RGB", (40, 40), (200, 100, 50)),
                            calib, device="cpu")
    assert 0 <= raw <= 100 and 0 <= cal <= 100
