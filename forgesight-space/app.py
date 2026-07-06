"""
ForgeSight — AI Image Forensics
FastAPI backend serving a 3-model deep-learning ensemble (fine-tuned DINOv2-Base,
SigLIP-So400m, CLIP ViT-L/14 — fold-0 weights) plus live ELA/FFT/Wavelet forensic
heuristics, for a single-image AI-vs-real verdict with rich visual explanations.

Architecture and preprocessing mirror the original training notebook
(version_11_final.ipynb) exactly for the DL branch. The forensic branch's XGBoost +
isotonic calibrator were never serialized during training, so — same as the reference
demo this project is patterned after — ELA/FFT/Wavelet are computed live as heuristic
proxies rather than through the trained classifier.
"""
import os
import io
import time
import base64
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ExifTags
from scipy.fft import fft2, fftshift
from scipy.stats import kurtosis
import pywt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
WEIGHTS_REPO = "rahulraj1406/forgesight-weights"
HF_TOKEN = os.environ.get("HF_TOKEN")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD = [0.229, 0.224, 0.225]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]

# Hill-climbed ensemble weights from training (v11), simplified to fold-0-only demo form
DL_WEIGHTS = {"dino": 0.45, "siglip": 0.30, "clip": 0.25}
FINAL_WEIGHTS = {"dl": 0.80, "forensic": 0.20}
THRESHOLD = 0.47

DEVICE = "cpu"

# ----------------------------------------------------------------------------
# Model architectures (must match training-time classes exactly for state_dict load)
# ----------------------------------------------------------------------------
class SigLIPFT(nn.Module):
    def __init__(self, vm, d=1152):
        super().__init__()
        self.visual = vm
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.visual(pixel_values=x).pooler_output.float()).squeeze(-1)


class DINOv2FT(nn.Module):
    def __init__(self, bb, d=768):
        super().__init__()
        self.backbone = bb
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class CLIPFT(nn.Module):
    def __init__(self, vis, d=768):
        super().__init__()
        self.visual = vis
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Dropout(0.3), nn.Linear(d, 256), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.head(self.visual(x).float()).squeeze(-1)


# ----------------------------------------------------------------------------
# Model loading (once, at startup)
# ----------------------------------------------------------------------------
_MODELS = {}


def _download(filename):
    return hf_hub_download(repo_id=WEIGHTS_REPO, filename=filename, token=HF_TOKEN)


def load_models():
    print("Loading DINOv2-Base...", flush=True)
    import timm
    dino_bb = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
    dino = DINOv2FT(dino_bb)
    dino.load_state_dict(torch.load(_download("dino_fold_0.pt"), map_location="cpu"))
    dino.eval()
    _MODELS["dino"] = dino
    print("DINOv2-Base ready.", flush=True)

    print("Loading SigLIP-So400m...", flush=True)
    from transformers import SiglipModel
    sig_base = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    sig = SigLIPFT(sig_base.vision_model)
    del sig_base
    sig.load_state_dict(torch.load(_download("sig_fold_0.pt"), map_location="cpu"))
    sig.eval()
    _MODELS["siglip"] = sig
    print("SigLIP-So400m ready.", flush=True)

    print("Loading CLIP ViT-L/14...", flush=True)
    import clip as openai_clip
    clip_model, _ = openai_clip.load("ViT-L/14", device="cpu")
    clip_ft = CLIPFT(clip_model.float().visual)
    clip_ft.load_state_dict(torch.load(_download("clip_fold_0.pt"), map_location="cpu"))
    clip_ft.eval()
    _MODELS["clip"] = clip_ft
    print("CLIP ViT-L/14 ready.", flush=True)


# ----------------------------------------------------------------------------
# Preprocessing (mirrors make_val_tfm: square resize + normalize, no TTA/crop)
# ----------------------------------------------------------------------------
def to_tensor(pil_img, size, mean, std):
    img = pil_img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    t = torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)
    return t


def sigmoid(x):
    return float(1.0 / (1.0 + np.exp(-x)))


# ----------------------------------------------------------------------------
# Forensic heuristics (ported from notebook CELL 8 — _ela / _fft / _noise)
# ----------------------------------------------------------------------------
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def ela_branch(pil256):
    arr = np.array(pil256, dtype=np.float64)
    buf = io.BytesIO()
    pil256.save(buf, "JPEG", quality=90)
    buf.seek(0)
    recompressed = np.array(Image.open(buf), dtype=np.float64)
    err = np.abs(arr - recompressed).mean(axis=2)
    spread = np.percentile(err, 95) - np.percentile(err, 5)
    score = sigmoid((spread - 12.0) / 4.0)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor("#0b0d14")
    ax.imshow(err, cmap="inferno")
    ax.axis("off")
    return score, fig_to_b64(fig)


def fft_branch(pil256):
    arr = np.array(pil256, dtype=np.float64)
    mag = np.log1p(np.abs(fftshift(fft2(arr.mean(2)))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    mr = min(cy, cx)
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    bins = np.array([
        mag[(r >= i * mr / 30) & (r < (i + 1) * mr / 30)].mean()
        if ((r >= i * mr / 30) & (r < (i + 1) * mr / 30)).any() else 0.0
        for i in range(30)
    ])
    kernel = np.ones(3) / 3
    smooth = np.convolve(bins, kernel, mode="same")
    residual = float(np.mean((bins - smooth) ** 2) / (np.mean(bins) ** 2 + 1e-6))
    score = sigmoid((residual - 0.013) / 0.004)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor("#0b0d14")
    ax.imshow(mag, cmap="magma")
    ax.axis("off")
    return score, fig_to_b64(fig)


def wavelet_branch(pil256):
    gray = np.array(pil256.convert("L"), dtype=np.float64)
    _, (cH, cV, cD) = pywt.dwt2(gray, "db2")
    k = float(kurtosis(cD.flatten(), fisher=True))
    # Theory (per training notebook): authentic sensor noise is heavy-tailed/sparse
    # (high kurtosis); diffusion-model outputs are more Gaussian-like (kurtosis -> 0).
    # Lower kurtosis => higher AI-likelihood.
    score = sigmoid((50.0 - k) / 15.0)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    fig.patch.set_facecolor("#0b0d14")
    ax.imshow(cD, cmap="twilight")
    ax.axis("off")
    return score, fig_to_b64(fig)


# ----------------------------------------------------------------------------
# DINOv2 saliency (vanilla input-gradient), used as an "attention" visualization
# ----------------------------------------------------------------------------
def dino_saliency(pil_img):
    try:
        t = to_tensor(pil_img, 518, DINO_MEAN, DINO_STD)
        t.requires_grad_(True)
        model = _MODELS["dino"]
        logit = model(t)
        model.zero_grad()
        logit.backward()
        grad = t.grad.detach().abs().amax(dim=1)[0].numpy()
        grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)

        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        fig.patch.set_facecolor("#0b0d14")
        ax.imshow(np.array(pil_img.convert("RGB").resize((518, 518))))
        ax.imshow(grad, cmap="jet", alpha=0.55)
        ax.axis("off")
        return fig_to_b64(fig)
    except Exception as e:
        print("saliency failed:", e, flush=True)
        return None


# ----------------------------------------------------------------------------
# EXIF
# ----------------------------------------------------------------------------
def read_exif(pil_img):
    out = {}
    try:
        exif = pil_img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                if isinstance(value, bytes):
                    continue
                sval = str(value)
                if len(sval) > 100:
                    continue
                out[str(tag)] = sval
    except Exception:
        pass
    out["dimensions"] = f"{pil_img.width} x {pil_img.height}"
    out["format"] = pil_img.format or "unknown"
    out["mode"] = pil_img.mode
    return out


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI(title="ForgeSight")


@app.on_event("startup")
def _startup():
    load_models()


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(_MODELS.keys())}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    t0 = time.time()
    raw = await file.read()
    pil_img = Image.open(io.BytesIO(raw))
    pil_img.load()
    exif = read_exif(pil_img)

    pil_rgb = pil_img.convert("RGB")
    pil256 = pil_rgb.resize((256, 256), Image.LANCZOS)

    with torch.no_grad():
        dino_logit = _MODELS["dino"](to_tensor(pil_rgb, 518, DINO_MEAN, DINO_STD))
        sig_logit = _MODELS["siglip"](to_tensor(pil_rgb, 224, SIGLIP_MEAN, SIGLIP_STD))
        clip_logit = _MODELS["clip"](to_tensor(pil_rgb, 224, CLIP_MEAN, CLIP_STD))

    dino_score = sigmoid(float(dino_logit.item()))
    sig_score = sigmoid(float(sig_logit.item()))
    clip_score = sigmoid(float(clip_logit.item()))

    ela_score, ela_img = ela_branch(pil256)
    fft_score, fft_img = fft_branch(pil256)
    wav_score, wav_img = wavelet_branch(pil256)
    saliency_img = dino_saliency(pil_rgb)

    dl_score = (DL_WEIGHTS["dino"] * dino_score + DL_WEIGHTS["siglip"] * sig_score
                + DL_WEIGHTS["clip"] * clip_score)
    forensic_score = (ela_score + fft_score + wav_score) / 3.0
    final_score = FINAL_WEIGHTS["dl"] * dl_score + FINAL_WEIGHTS["forensic"] * forensic_score

    is_ai = final_score >= THRESHOLD
    confidence = final_score * 100 if is_ai else (1 - final_score) * 100

    branches = {
        "dino": dino_score, "siglip": sig_score, "clip": clip_score,
        "ela": ela_score, "fft": fft_score, "wavelet": wav_score,
    }
    agree = sum(1 for v in branches.values() if (v >= 0.5) == is_ai)
    agreement_pct = agree / len(branches) * 100

    elapsed = time.time() - t0

    return JSONResponse({
        "verdict": "AI-GENERATED" if is_ai else "REAL",
        "confidence": round(confidence, 1),
        "final_score": round(final_score, 4),
        "threshold": THRESHOLD,
        "dl_score": round(dl_score, 4),
        "forensic_score": round(forensic_score, 4),
        "branches": {k: round(v * 100, 1) for k, v in branches.items()},
        "agreement_pct": round(agreement_pct, 1),
        "elapsed_sec": round(elapsed, 2),
        "exif": exif,
        "images": {
            "ela": ela_img,
            "fft": fft_img,
            "wavelet": wav_img,
            "saliency": saliency_img,
        },
    })


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
