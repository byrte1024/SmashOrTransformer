from pathlib import Path
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


class _CountingStub(dd.GuidanceStrategy):
    def __init__(self): self.n = 0
    def generate(self, pipe, smash_model, prompt, target, seed, mean, std):
        self.n += 1
        return Image.fromarray(np.full((32, 32, 3), int(target * 255), np.uint8), "RGB")


def test_run_skips_existing_images_on_resume(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    out_dir = str(tmp_path / "dream")
    kwargs = dict(out_dir=out_dir, device="cpu", methods=["stub"],
                  prompts=[("uncond", "")], brackets=[1.0, 0.0], n_per=2)

    first = _CountingStub()
    dd.run(ckpt, strategies={"stub": first}, pipe=object(), **kwargs)
    assert first.n == 4                                   # 1*1*2*2 generated fresh
    manifest1 = list(_csv.DictReader(open(Path(out_dir) / "manifest.csv")))
    assert len(manifest1) == 4

    # second run over the same out_dir: every image already exists -> 0 regenerated,
    # manifest still complete (rows carried forward from the prior manifest)
    second = _CountingStub()
    dd.run(ckpt, strategies={"stub": second}, pipe=object(), **kwargs)
    assert second.n == 0
    manifest2 = list(_csv.DictReader(open(Path(out_dir) / "manifest.csv")))
    assert len(manifest2) == 4
    assert {r["seed"] for r in manifest2} == {r["seed"] for r in manifest1}


def test_score_pil_in_range(mini_repo, tmp_path):
    ckpt = _trained(mini_repo, tmp_path)
    model, cfg = _load_model(ckpt, device="cpu", pretrained=False)
    calib = _load_calib(ckpt, fit="combined")
    raw, cal = dd.score_pil(model, cfg, Image.new("RGB", (40, 40), (200, 100, 50)),
                            calib, device="cpu")
    assert 0 <= raw <= 100 and 0 <= cal <= 100


# --- Task 3 tests ---
import torch.nn as _nn
from types import SimpleNamespace


class _FakeUNet(_nn.Module):
    def __init__(self): super().__init__(); self.c = _nn.Conv2d(4, 4, 3, padding=1)
    def forward(self, x, t, encoder_hidden_states=None): return SimpleNamespace(sample=self.c(x))
    __call__ = forward


class _FakeVAE(_nn.Module):
    def __init__(self):
        super().__init__(); self.w = _nn.Conv2d(4, 3, 1)
        self.config = SimpleNamespace(scaling_factor=0.18215)
    def decode(self, z):
        img = _nn.functional.interpolate(self.w(z), size=512, mode="nearest")
        return SimpleNamespace(sample=torch.tanh(img))


class _FakeScheduler:
    def __init__(self, n=1000):
        self.alphas_cumprod = torch.linspace(0.9999, 0.02, n)
        self.init_noise_sigma = 1.0
        self.config = SimpleNamespace(num_train_timesteps=n)
    def set_timesteps(self, steps, device=None):
        self.timesteps = torch.linspace(self.config.num_train_timesteps - 1, 0, steps).long()
    def scale_model_input(self, x, t): return x
    def step(self, noise, t, latents): return SimpleNamespace(prev_sample=latents - 0.01 * noise)


class _FakePipe:
    def __init__(self):
        self.unet = _FakeUNet(); self.vae = _FakeVAE(); self.scheduler = _FakeScheduler()
        self.device = torch.device("cpu")
    def encode_prompt(self, prompt, device, n, cfg, negative_prompt=""):
        e = torch.zeros(1, 4, 8)
        return e, e   # (prompt_embeds, negative_prompt_embeds)


def test_xhat_generate_returns_image_and_uses_model():
    pipe = _FakePipe()
    m = _tiny_model()
    calls = {"n": 0}
    orig = dd._score_tensor
    def counting(*a, **k):
        calls["n"] += 1; return orig(*a, **k)
    dd._score_tensor = counting
    try:
        img = dd.XHatGuidance(steps=3, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
            pipe, m, "a creature", target=1.0, seed=0, mean=(0.5,) * 3, std=(0.5,) * 3)
    finally:
        dd._score_tensor = orig
    assert isinstance(img, Image.Image) and img.size == (512, 512)
    assert calls["n"] >= 3            # guidance evaluated the smash model each step


# --- Task 4 tests ---
def test_doodl_generate_returns_image():
    pipe = _FakePipe()
    m = _tiny_model()
    img = dd.DoodlGuidance(steps=3, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
        pipe, m, "a creature", target=1.0, seed=1, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert isinstance(img, Image.Image) and img.size == (512, 512)


# --- Task 5 tests ---
def test_sds_generate_returns_image():
    pipe = _FakePipe()
    m = _tiny_model()
    img = dd.SdsGuidance(steps=4, guidance_scale=10.0, sd_guidance_scale=7.5).generate(
        pipe, m, "a creature", target=1.0, seed=2, mean=(0.5,) * 3, std=(0.5,) * 3)
    assert isinstance(img, Image.Image) and img.size == (512, 512)
