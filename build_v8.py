#!/usr/bin/env python3
"""Build Version 8 notebook: Best-in-class AI image detection."""
import json

def cell(source, cell_type="code"):
    """Create a notebook cell."""
    lines = source.split("\n")
    # Each line needs \n except potentially the last
    src_list = [line + "\n" for line in lines[:-1]]
    if lines[-1]:  # non-empty last line
        src_list.append(lines[-1])
    elif src_list:  # empty last line, remove trailing \n from previous
        pass  # keep it, notebook format is fine

    c = {
        "cell_type": cell_type,
        "metadata": {},
        "source": src_list,
    }
    if cell_type == "code":
        c["execution_count"] = None
        c["outputs"] = []
    return c

cells = []

# ============================================================
# CELL 1: INSTALLS
# ============================================================
cells.append(cell(r"""# ============================================================
# VERSION 8: ULTIMATE AI IMAGE DETECTION
# ============================================================
# Architecture: SigLIP + DINOv2-B + CLIP ViT-L/14 + Forensic XGB
# Strategy: 3 fine-tuned vision models + 1 XGB → 4-model ensemble
# Target: Beat V5's Val F1 = 0.9248
# ============================================================

# Install dependencies
!pip install -q numpy pandas opencv-python-headless Pillow tqdm scipy PyWavelets \
    matplotlib seaborn scikit-learn xgboost lightgbm joblib ipywidgets
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q timm albumentations transformers
print("All packages installed.")"""))

# ============================================================
# CELL 2: IMPORTS, CONFIG, SEEDS, GPU
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 2: IMPORTS, CONFIG, SEEDS, GPU, MEMORY UTILITIES
# ============================================================
import os, sys, warnings, io, random, gc, copy, time
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from scipy.fft import fft2, fftshift
from scipy.ndimage import median_filter
import pywt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import clip
import timm
import albumentations as A

warnings.filterwarnings('ignore')

# ─── VRAM Management ────────────────────────────────────────
def gpu_cleanup():
    # Aggressive GPU memory cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_vram():
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated()) / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {used:.1f} GB used, {free:.1f} GB free")

def ensure_vram(min_gb=4.0):
    """Ensure minimum VRAM is available."""
    gpu_cleanup()
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated()) / 1e9
        if free < min_gb:
            print(f"WARNING: Only {free:.1f} GB free, need {min_gb} GB")
        print_vram()

# ─── Path Detection (Renku / Kaggle / Local) ────────────────
def detect_paths():
    """Auto-detect dataset paths for different environments."""
    candidates = [
        # Renku
        (Path("../DCU 2026 ML challenge - external 2"),
         "genai_image_challenge/images_final_sample",
         "train.csv", "test.csv"),
        # Kaggle dataset-easy
        (Path("/kaggle/input/datasets/rahulraj1406/dataset-easy"),
         "genai_image_challenge/images_final_sample",
         "train.csv", "test.csv"),
        # Kaggle ml-dataset-easy
        (Path("/kaggle/input/datasets/rahulraj1406/ml-dataset-easy/DCU 2026 ML challenge - external 2"),
         "genai_image_challenge/images_final_sample",
         "train.csv", "test.csv"),
    ]
    for base, img_sub, train_csv, test_csv in candidates:
        if (base / train_csv).exists():
            return (base / img_sub, base / train_csv, base / test_csv)
    raise FileNotFoundError("Cannot find dataset. Check paths.")

IMAGE_DIR, TRAIN_CSV, TEST_CSV = detect_paths()
print(f"IMAGE_DIR: {IMAGE_DIR}")
print(f"TRAIN_CSV: {TRAIN_CSV}")
print(f"TEST_CSV:  {TEST_CSV}")

# ─── Config ──────────────────────────────────────────────────
SEED         = 42
CACHE_DIR    = Path("./feature_cache_v8")
MODEL_DIR    = Path("./saved_models_v8")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

N_FOLDS      = 5
N_TTA        = 5
FORCE_FRESH  = False

SIGLIP_DIM   = 1152
CLIP_DIM     = 768
DINO_DIM     = 768
CNN_DIM      = 1280
FORENSIC_DIM = 114

# ─── Normalization constants ────────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# ─── Reproducibility ─────────────────────────────────────────
def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seeds()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    # Set memory allocation strategy
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"\nConfig: N_FOLDS={N_FOLDS}, N_TTA={N_TTA}")
print(f"Models: SigLIP({SIGLIP_DIM}d) + CLIP({CLIP_DIM}d) + DINOv2-B({DINO_DIM}d) + CNN({CNN_DIM}d) + Forensic({FORENSIC_DIM}d)")"""))

# ============================================================
# CELL 3: DATASET LOADING
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 3: DATASET LOADING
# ============================================================
set_seeds()

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

df_train['filepath'] = df_train['image_id'].apply(lambda x: str(IMAGE_DIR / x))
df_test['filepath']  = df_test['image_id'].apply(lambda x: str(IMAGE_DIR / x))

print(f"Train: {len(df_train)} images, Test: {len(df_test)} images")
print(f"Class balance: Real={sum(df_train['ground_truth']==0)} ({sum(df_train['ground_truth']==0)/len(df_train):.1%}), "
      f"AI={sum(df_train['ground_truth']==1)} ({sum(df_train['ground_truth']==1)/len(df_train):.1%})")

# Verify paths
found = sum(Path(p).exists() for p in df_train['filepath'][:50])
print(f"Path check: {found}/50 train images found")
assert found >= 45, "Too many missing images!"

train_paths = df_train['filepath'].tolist()
test_paths  = df_test['filepath'].tolist()
y_all       = df_train['ground_truth'].values

print(f"Ready: {len(train_paths)} train, {len(test_paths)} test")"""))

# ============================================================
# CELL 4: IMAGE LOADING & AUGMENTATION UTILITIES
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 4: IMAGE LOADING & AUGMENTATION UTILITIES
# ============================================================

def load_image_pil(path):
    """Load image as PIL RGB."""
    try:
        return Image.open(str(path)).convert('RGB')
    except Exception:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
    return None

def load_image_np(path, size=None):
    """Load image as numpy array (RGB, float64, 0-255)."""
    img = load_image_pil(path)
    if img is None:
        return None
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float64)

# ── Augmentation transforms (albumentations) ────────────────
def get_train_transform(mean, std, size=224):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Normalize(mean=mean, std=std),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
    ])

def get_val_transform(mean, std, size=224):
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(size, size),
        A.Normalize(mean=mean, std=std),
    ])

def get_tta_transform(mean, std, size=224):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
    ])

# ── Dataset class ────────────────────────────────────────────
class AlbuDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_pil(self.paths[idx])
        if img is None:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        img_tensor = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float()
        lbl = self.labels[idx] if self.labels is not None else -1
        return img_tensor, torch.tensor(lbl, dtype=torch.float32)

# ── Mixup ────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("Augmentation utilities ready.")"""))

# ============================================================
# CELL 5: FEATURE EXTRACTION — SigLIP (1152d)
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 5: FEATURE EXTRACTION — SigLIP So400m (1152-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(4.0)
set_seeds()

from transformers import SiglipModel, SiglipProcessor

def extract_siglip_features(image_paths, batch_size=32):
    print("Loading SigLIP So400m-patch14-224...")
    model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    model = model.to(DEVICE).eval()
    vision_model = model.vision_model

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='SigLIP'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is None:
                img = Image.new('RGB', (224, 224), (128, 128, 128))
            batch_imgs.append(img)
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(DEVICE)
        with torch.no_grad():
            outputs = vision_model(pixel_values=pixel_values)
            feats = outputs.pooler_output.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model, vision_model, processor
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

cache = CACHE_DIR
if FORCE_FRESH or not (cache/'siglip_train.npy').exists():
    print("Extracting SigLIP features...")
    siglip_train = extract_siglip_features(train_paths)
    siglip_test  = extract_siglip_features(test_paths)
    np.save(cache/'siglip_train.npy', siglip_train)
    np.save(cache/'siglip_test.npy', siglip_test)
else:
    siglip_train = np.load(cache/'siglip_train.npy')
    siglip_test  = np.load(cache/'siglip_test.npy')

print(f"SigLIP train: {siglip_train.shape}  test: {siglip_test.shape}")
print_vram()"""))

# ============================================================
# CELL 6: FEATURE EXTRACTION — CLIP (768d)
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 6: FEATURE EXTRACTION — CLIP ViT-L/14 (768-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(4.0)
set_seeds()

def extract_clip_features(image_paths, batch_size=64):
    model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    model = model.float().eval()

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CLIP'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is not None:
                batch_imgs.append(preprocess(img))
            else:
                batch_imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

if FORCE_FRESH or not (cache/'clip_train.npy').exists():
    print("Extracting CLIP features...")
    clip_train = extract_clip_features(train_paths)
    clip_test  = extract_clip_features(test_paths)
    np.save(cache/'clip_train.npy', clip_train)
    np.save(cache/'clip_test.npy', clip_test)
else:
    clip_train = np.load(cache/'clip_train.npy')
    clip_test  = np.load(cache/'clip_test.npy')

print(f"CLIP train: {clip_train.shape}  test: {clip_test.shape}")
print_vram()"""))

# ============================================================
# CELL 7: FEATURE EXTRACTION — DINOv2-B (768d)
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 7: FEATURE EXTRACTION — DINOv2 ViT-B/14 (768-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(2.0)
set_seeds()

def extract_dino_features(image_paths, batch_size=64):
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                               pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='DINOv2-B'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is not None:
                batch_imgs.append(tfm(img))
            else:
                batch_imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

if FORCE_FRESH or not (cache/'dino_train.npy').exists():
    print("Extracting DINOv2-B features...")
    dino_train = extract_dino_features(train_paths)
    dino_test  = extract_dino_features(test_paths)
    np.save(cache/'dino_train.npy', dino_train)
    np.save(cache/'dino_test.npy', dino_test)
else:
    dino_train = np.load(cache/'dino_train.npy')
    dino_test  = np.load(cache/'dino_test.npy')

print(f"DINOv2-B train: {dino_train.shape}  test: {dino_test.shape}")
print_vram()"""))

# ============================================================
# CELL 8: FEATURE EXTRACTION — EfficientNet-B0 (1280d)
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 8: FEATURE EXTRACTION — EfficientNet-B0 (1280-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(2.0)
set_seeds()

def extract_cnn_features(image_paths, batch_size=128):
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    model = model.to(DEVICE).eval()

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CNN'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is not None:
                batch_imgs.append(tfm(img))
            else:
                batch_imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

if FORCE_FRESH or not (cache/'cnn_train.npy').exists():
    print("Extracting CNN features...")
    cnn_train = extract_cnn_features(train_paths)
    cnn_test  = extract_cnn_features(test_paths)
    np.save(cache/'cnn_train.npy', cnn_train)
    np.save(cache/'cnn_test.npy', cnn_test)
else:
    cnn_train = np.load(cache/'cnn_train.npy')
    cnn_test  = np.load(cache/'cnn_test.npy')

print(f"CNN train: {cnn_train.shape}  test: {cnn_test.shape}")
print_vram()"""))

# ============================================================
# CELL 9: FORENSIC FEATURES
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 9: FORENSIC FEATURE EXTRACTION (~114 dims)
# ============================================================
set_seeds()

def extract_ela_features(img_np):
    features = []
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    for quality in [90, 75, 50]:
        buffer = io.BytesIO()
        img_pil.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer), dtype=np.float64)
        ela = np.abs(img_np - recompressed)
        for c in range(3):
            ch = ela[:, :, c]
            features.extend([np.mean(ch), np.std(ch)])
        ela_gray = np.mean(ela, axis=2)
        features.extend([np.percentile(ela_gray, 95), np.percentile(ela_gray, 5)])
    return np.array(features, dtype=np.float32)  # 24 dims

def extract_fft_features(img_np):
    gray = np.mean(img_np, axis=2)
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    n_bins = 30
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        r_inner = int(i * max_radius / n_bins)
        r_outer = int((i + 1) * max_radius / n_bins)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x*x + y*y >= r_inner**2) & (x*x + y*y < r_outer**2)
        if mask.any():
            radial_profile[i] = np.mean(magnitude[mask])

    features = list(radial_profile)
    features.extend([np.mean(magnitude), np.std(magnitude),
                     np.sum(magnitude[cy-10:cy+10, cx-10:cx+10]),
                     np.sum(magnitude) - np.sum(magnitude[cy-10:cy+10, cx-10:cx+10])])

    y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
    r_sq = y_grid**2 + x_grid**2
    low_e = np.sum(magnitude[r_sq < (max_radius * 0.2)**2])
    mid_e = np.sum(magnitude[(r_sq >= (max_radius * 0.2)**2) & (r_sq < (max_radius * 0.5)**2)])
    high_e = np.sum(magnitude[r_sq >= (max_radius * 0.5)**2])
    total_e = low_e + mid_e + high_e + 1e-10
    features.extend([low_e/total_e, mid_e/total_e, high_e/total_e, high_e/(low_e+1e-10)])

    valid = radial_profile > 0
    slope = np.polyfit(np.log(np.arange(1, n_bins+1)[valid]),
                       np.log(radial_profile[valid]), 1)[0] if valid.sum() > 5 else 0.0
    features.append(slope)

    phase = np.angle(f_shift)
    features.extend([np.mean(phase), np.std(phase),
                     np.mean(np.abs(np.diff(phase, axis=0))),
                     np.mean(np.abs(np.diff(phase, axis=1)))])
    features = features[:50]
    while len(features) < 50:
        features.append(0.0)
    return np.array(features, dtype=np.float32)

def extract_noise_features(img_np):
    gray = np.mean(img_np, axis=2)
    features = []
    for wavelet in ['db1', 'db2']:
        coeffs = pywt.dwt2(gray, wavelet)
        cA, (cH, cV, cD) = coeffs
        for detail in [cH, cV, cD]:
            features.extend([np.mean(np.abs(detail)), np.std(detail),
                             np.percentile(np.abs(detail), 99), np.mean(detail**2)])
    denoised = median_filter(gray, size=3)
    noise = gray - denoised
    features.extend([np.mean(noise), np.std(noise), np.mean(noise**2),
                     np.percentile(noise, 1), np.percentile(noise, 99)])
    block_size = 32
    h, w = gray.shape
    local_vars = []
    for yy in range(0, h - block_size, block_size):
        for xx in range(0, w - block_size, block_size):
            local_vars.append(np.var(noise[yy:yy+block_size, xx:xx+block_size]))
    if local_vars:
        features.extend([np.mean(local_vars), np.std(local_vars),
                         np.percentile(local_vars, 90) / (np.percentile(local_vars, 10) + 1e-10)])
    else:
        features.extend([0, 0, 0])
    features = features[:40]
    while len(features) < 40:
        features.append(0.0)
    return np.array(features, dtype=np.float32)

def extract_forensic_single(path, target_size=(256, 256)):
    img_np = load_image_np(path, size=target_size)
    if img_np is None:
        return np.zeros(FORENSIC_DIM, dtype=np.float32)
    ela = extract_ela_features(img_np)
    fft = extract_fft_features(img_np)
    noise = extract_noise_features(img_np)
    combined = np.concatenate([ela, fft, noise])
    if len(combined) > FORENSIC_DIM:
        combined = combined[:FORENSIC_DIM]
    elif len(combined) < FORENSIC_DIM:
        combined = np.pad(combined, (0, FORENSIC_DIM - len(combined)))
    return combined.astype(np.float32)

def extract_forensic_batch(paths):
    return np.vstack([extract_forensic_single(p) for p in tqdm(paths, desc='Forensic')])

if FORCE_FRESH or not (cache/'forensic_train.npy').exists():
    print("Extracting forensic features...")
    forensic_train = extract_forensic_batch(train_paths)
    forensic_test  = extract_forensic_batch(test_paths)
    np.save(cache/'forensic_train.npy', forensic_train)
    np.save(cache/'forensic_test.npy', forensic_test)
else:
    forensic_train = np.load(cache/'forensic_train.npy')
    forensic_test  = np.load(cache/'forensic_test.npy')

n_alive = (forensic_train.std(axis=0) > 1e-10).sum()
print(f"Forensic train: {forensic_train.shape}  Alive: {n_alive}/{forensic_train.shape[1]}")"""))

# ============================================================
# CELL 10: SHARED CV + FINE-TUNE INFRASTRUCTURE
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 10: SHARED CV + FINE-TUNE INFRASTRUCTURE
# ============================================================

results_tracker = {}
oof_store       = {}
test_pred_store = {}

def run_epoch(model, loader, optimizer, scaler, criterion, is_train, use_mixup=False):
    model.train() if is_train else model.eval()
    tot_loss = 0.0; preds = []; trues = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            if is_train and use_mixup and random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.3)
                with autocast():
                    logits = model(imgs)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    f1 = f1_score(trues, (np.array(preds) >= 0.5).astype(int), zero_division=0)
    return tot_loss / len(trues), f1, np.array(preds)

def save_state_cpu(model):
    """Save model state_dict to CPU to avoid VRAM leak."""
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def train_finetune_fold(model, tr_ds, val_ds, criterion,
                        freeze_fn, unfreeze_fn, get_opt_fn,
                        fold_num, batch_size=16):
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)

    # Stage 1: Head only
    freeze_fn(model)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-5)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_f1 = 0; best_state = None; best_vp = None
    patience = 5; no_improve = 0

    for ep in range(1, 11):
        tl, tf, _ = run_epoch(model, tr_ld, opt1, scaler, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler, criterion, False)
        sch1.step()
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = save_state_cpu(model)
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    s1_f1 = best_f1
    model.load_state_dict(best_state)

    # Stage 2: Unfreeze last blocks
    unfreeze_fn(model)
    opt2 = get_opt_fn(model)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-7)
    scaler2 = GradScaler(enabled=(DEVICE == "cuda"))
    no_improve = 0

    for ep in range(1, 16):
        tl, tf, _ = run_epoch(model, tr_ld, opt2, scaler2, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler2, criterion, False)
        sch2.step()
        gap = tf - vf
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = save_state_cpu(model)
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        if gap > 0.12:
            print(f"  Fold {fold_num}: gap {gap:.4f} > 0.12, stopping Stage 2")
            break

    model.load_state_dict(best_state)
    print(f"  Fold {fold_num}: Stage1 F1={s1_f1:.4f} -> Stage2 F1={best_f1:.4f}")
    return best_f1, best_vp, best_state

def run_test_tta(model_class, model_init_fn, fold_models, test_paths,
                 tta_tfm, batch_size=16, n_tta=N_TTA):
    """Run TTA inference across all fold models."""
    test_proba = np.zeros(len(test_paths))
    for fold, fold_state in enumerate(fold_models):
        model = model_init_fn()
        model.load_state_dict(fold_state)
        model = model.to(DEVICE).eval()

        test_ds = AlbuDataset(test_paths, [-1]*len(test_paths), tta_tfm)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        for t in range(n_tta):
            preds = []
            with torch.no_grad():
                for imgs, _ in test_ld:
                    imgs = imgs.to(DEVICE)
                    with autocast():
                        logits = model(imgs)
                    preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            test_proba += np.array(preds)

        del model
        gpu_cleanup()

    test_proba /= (len(fold_models) * n_tta)
    return test_proba

print("CV + Fine-tune infrastructure ready.")"""))

# ============================================================
# CELL 11: SigLIP FINE-TUNE — 5-Fold CV
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 11: SigLIP So400m FINE-TUNE — 5-Fold CV
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(6.0)
set_seeds()

from transformers import SiglipModel

class SigLIPFineTuner(nn.Module):
    def __init__(self, vision_model, embed_dim=1152):
        super().__init__()
        self.visual = vision_model
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        outputs = self.visual(pixel_values=x)
        f = outputs.pooler_output.float()
        return self.head(f).squeeze(-1)

def freeze_siglip(model):
    for p in model.visual.parameters():
        p.requires_grad = False

def unfreeze_siglip_blocks(model, n=2):
    layers = model.visual.encoder.layers
    for p in layers[-n:].parameters():
        p.requires_grad = True
    if hasattr(model.visual, 'post_layernorm'):
        for p in model.visual.post_layernorm.parameters():
            p.requires_grad = True

def get_siglip_optimizer(model, backbone_lr=5e-6, head_lr=1e-4):
    backbone_params = [p for n, p in model.visual.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 0.05},
        {'params': head_params,     'lr': head_lr,     'weight_decay': 0.01}
    ])

def make_siglip_model():
    """Create a fresh SigLIP model for inference."""
    base = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    model = SigLIPFineTuner(base.vision_model, embed_dim=1152)
    del base
    return model

# SigLIP uses its own processor normalization — approximate with ImageNet-like
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]
siglip_train_tfm = get_train_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)
siglip_val_tfm   = get_val_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)
siglip_tta_tfm   = get_tta_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
siglip_oof = np.zeros(len(y_all))
siglip_fold_f1s = []
siglip_fold_models = []

print("=" * 60)
print(f"SigLIP So400m FINE-TUNE — 5-Fold CV")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    print(f"\n-- Fold {fold+1}/{N_FOLDS} --")
    gpu_cleanup()

    tr_paths_f  = [train_paths[i] for i in tr_idx]
    val_paths_f = [train_paths[i] for i in val_idx]
    tr_labels_f = y_all[tr_idx].tolist()
    val_labels_f = y_all[val_idx].tolist()

    tr_ds = AlbuDataset(tr_paths_f, tr_labels_f, siglip_train_tfm)
    val_ds = AlbuDataset(val_paths_f, val_labels_f, siglip_val_tfm)

    # Fresh model each fold
    base = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    model = SigLIPFineTuner(base.vision_model, embed_dim=1152).to(DEVICE)
    del base; gpu_cleanup()

    n_pos = sum(tr_labels_f); n_neg = len(tr_labels_f) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    fold_f1, fold_vp, fold_state = train_finetune_fold(
        model, tr_ds, val_ds, criterion,
        freeze_fn=freeze_siglip,
        unfreeze_fn=lambda m: unfreeze_siglip_blocks(m, n=2),
        get_opt_fn=get_siglip_optimizer,
        fold_num=fold+1, batch_size=16
    )

    siglip_oof[val_idx] = fold_vp
    siglip_fold_f1s.append(fold_f1)
    siglip_fold_models.append(fold_state)
    del model; gpu_cleanup()

# Test inference
print(f"\n-- Test Inference (5 models x {N_TTA} TTA) --")
siglip_test_proba = run_test_tta(
    SigLIPFineTuner, make_siglip_model, siglip_fold_models,
    test_paths, siglip_tta_tfm, batch_size=16
)

siglip_val_f1 = np.mean(siglip_fold_f1s)
siglip_oof_f1 = f1_score(y_all, (siglip_oof >= 0.5).astype(int))
siglip_oof_auc = roc_auc_score(y_all, siglip_oof)

results_tracker['siglip_finetune'] = {
    'name': 'SigLIP-FT (5-fold)', 'val_f1_mean': siglip_val_f1,
    'val_f1_std': np.std(siglip_fold_f1s), 'val_auc_mean': siglip_oof_auc,
    'oof_proba': siglip_oof.copy()
}
oof_store['siglip_finetune'] = siglip_oof.copy()
test_pred_store['siglip_finetune'] = siglip_test_proba.copy()

print(f"\nSigLIP Fine-Tune: Val F1={siglip_val_f1:.4f}, OOF AUC={siglip_oof_auc:.4f}")
print(f"  Fold F1s: {[f'{f:.4f}' for f in siglip_fold_f1s]}")
gpu_cleanup()
print_vram()"""))

# ============================================================
# CELL 12: DINOv2-B FINE-TUNE — 5-Fold CV
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 12: DINOv2 ViT-B/14 FINE-TUNE — 5-Fold CV
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(4.0)
set_seeds()

class DINOv2FineTuner(nn.Module):
    def __init__(self, backbone, embed_dim=768):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        f = self.backbone(x)
        return self.head(f).squeeze(-1)

def freeze_dino(model):
    for p in model.backbone.parameters():
        p.requires_grad = False

def unfreeze_dino_blocks(model, n=3):
    blocks = model.backbone.blocks
    for p in blocks[-n:].parameters():
        p.requires_grad = True
    if hasattr(model.backbone, 'norm'):
        for p in model.backbone.norm.parameters():
            p.requires_grad = True

def get_dino_optimizer(model, backbone_lr=5e-6, head_lr=1e-4):
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 0.05},
        {'params': head_params,     'lr': head_lr,     'weight_decay': 0.01}
    ])

def make_dino_model():
    backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                  pretrained=True, num_classes=0)
    return DINOv2FineTuner(backbone)

dino_train_tfm = get_train_transform(DINO_MEAN, DINO_STD, size=224)
dino_val_tfm   = get_val_transform(DINO_MEAN, DINO_STD, size=224)
dino_tta_tfm   = get_tta_transform(DINO_MEAN, DINO_STD, size=224)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
dino_oof = np.zeros(len(y_all))
dino_fold_f1s = []
dino_fold_models = []

print("=" * 60)
print("DINOv2 ViT-B/14 FINE-TUNE — 5-Fold CV")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    print(f"\n-- Fold {fold+1}/{N_FOLDS} --")
    gpu_cleanup()

    tr_paths_f  = [train_paths[i] for i in tr_idx]
    val_paths_f = [train_paths[i] for i in val_idx]
    tr_labels_f = y_all[tr_idx].tolist()
    val_labels_f = y_all[val_idx].tolist()

    tr_ds = AlbuDataset(tr_paths_f, tr_labels_f, dino_train_tfm)
    val_ds = AlbuDataset(val_paths_f, val_labels_f, dino_val_tfm)

    backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                  pretrained=True, num_classes=0)
    model = DINOv2FineTuner(backbone).to(DEVICE)

    n_pos = sum(tr_labels_f); n_neg = len(tr_labels_f) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    fold_f1, fold_vp, fold_state = train_finetune_fold(
        model, tr_ds, val_ds, criterion,
        freeze_fn=freeze_dino,
        unfreeze_fn=lambda m: unfreeze_dino_blocks(m, n=3),
        get_opt_fn=get_dino_optimizer,
        fold_num=fold+1, batch_size=32
    )

    dino_oof[val_idx] = fold_vp
    dino_fold_f1s.append(fold_f1)
    dino_fold_models.append(fold_state)
    del model, backbone; gpu_cleanup()

print(f"\n-- Test Inference (5 models x {N_TTA} TTA) --")
dino_test_proba = run_test_tta(
    DINOv2FineTuner, make_dino_model, dino_fold_models,
    test_paths, dino_tta_tfm, batch_size=32
)

dino_val_f1 = np.mean(dino_fold_f1s)
dino_oof_f1 = f1_score(y_all, (dino_oof >= 0.5).astype(int))
dino_oof_auc = roc_auc_score(y_all, dino_oof)

results_tracker['dino_finetune'] = {
    'name': 'DINOv2-B-FT (5-fold)', 'val_f1_mean': dino_val_f1,
    'val_f1_std': np.std(dino_fold_f1s), 'val_auc_mean': dino_oof_auc,
    'oof_proba': dino_oof.copy()
}
oof_store['dino_finetune'] = dino_oof.copy()
test_pred_store['dino_finetune'] = dino_test_proba.copy()

print(f"\nDINOv2-B Fine-Tune: Val F1={dino_val_f1:.4f}, OOF AUC={dino_oof_auc:.4f}")
print(f"  Fold F1s: {[f'{f:.4f}' for f in dino_fold_f1s]}")
gpu_cleanup()
print_vram()"""))

# ============================================================
# CELL 13: CLIP FINE-TUNE — 5-Fold CV
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 13: CLIP ViT-L/14 FINE-TUNE — 5-Fold CV
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(6.0)
set_seeds()

class CLIPFineTuner(nn.Module):
    def __init__(self, clip_visual, embed_dim=768):
        super().__init__()
        self.visual = clip_visual
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.3),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        f = self.visual(x).float()
        return self.head(f).squeeze(-1)

def freeze_clip(model):
    for p in model.visual.parameters():
        p.requires_grad = False

def unfreeze_clip_blocks(model, n=2):
    blocks = model.visual.transformer.resblocks
    for p in blocks[-n:].parameters():
        p.requires_grad = True
    if hasattr(model.visual, 'ln_post'):
        for p in model.visual.ln_post.parameters():
            p.requires_grad = True

def get_clip_optimizer(model, backbone_lr=5e-6, head_lr=1e-4):
    backbone_params = [p for n, p in model.visual.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': 0.05},
        {'params': head_params,     'lr': head_lr,     'weight_decay': 0.01}
    ])

def make_clip_model():
    _cm, _ = clip.load('ViT-L/14', device='cpu')
    _cm = _cm.float()
    model = CLIPFineTuner(_cm.visual)
    del _cm
    return model

clip_train_tfm = get_train_transform(CLIP_MEAN, CLIP_STD, size=224)
clip_val_tfm   = get_val_transform(CLIP_MEAN, CLIP_STD, size=224)
clip_tta_tfm   = get_tta_transform(CLIP_MEAN, CLIP_STD, size=224)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
clip_oof = np.zeros(len(y_all))
clip_fold_f1s = []
clip_fold_models = []

print("=" * 60)
print("CLIP ViT-L/14 FINE-TUNE — 5-Fold CV")
print("=" * 60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    print(f"\n-- Fold {fold+1}/{N_FOLDS} --")
    gpu_cleanup()

    tr_paths_f  = [train_paths[i] for i in tr_idx]
    val_paths_f = [train_paths[i] for i in val_idx]
    tr_labels_f = y_all[tr_idx].tolist()
    val_labels_f = y_all[val_idx].tolist()

    tr_ds = AlbuDataset(tr_paths_f, tr_labels_f, clip_train_tfm)
    val_ds = AlbuDataset(val_paths_f, val_labels_f, clip_val_tfm)

    _cm, _ = clip.load('ViT-L/14', device='cpu')
    _cm = _cm.float()
    model = CLIPFineTuner(_cm.visual).to(DEVICE)
    del _cm; gpu_cleanup()

    n_pos = sum(tr_labels_f); n_neg = len(tr_labels_f) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    fold_f1, fold_vp, fold_state = train_finetune_fold(
        model, tr_ds, val_ds, criterion,
        freeze_fn=freeze_clip,
        unfreeze_fn=lambda m: unfreeze_clip_blocks(m, n=2),
        get_opt_fn=get_clip_optimizer,
        fold_num=fold+1, batch_size=16
    )

    clip_oof[val_idx] = fold_vp
    clip_fold_f1s.append(fold_f1)
    clip_fold_models.append(fold_state)
    del model; gpu_cleanup()

print(f"\n-- Test Inference (5 models x {N_TTA} TTA) --")
clip_test_proba = run_test_tta(
    CLIPFineTuner, make_clip_model, clip_fold_models,
    test_paths, clip_tta_tfm, batch_size=16
)

clip_val_f1 = np.mean(clip_fold_f1s)
clip_oof_f1 = f1_score(y_all, (clip_oof >= 0.5).astype(int))
clip_oof_auc = roc_auc_score(y_all, clip_oof)

results_tracker['clip_finetune'] = {
    'name': 'CLIP-FT (5-fold)', 'val_f1_mean': clip_val_f1,
    'val_f1_std': np.std(clip_fold_f1s), 'val_auc_mean': clip_oof_auc,
    'oof_proba': clip_oof.copy()
}
oof_store['clip_finetune'] = clip_oof.copy()
test_pred_store['clip_finetune'] = clip_test_proba.copy()

print(f"\nCLIP Fine-Tune: Val F1={clip_val_f1:.4f}, OOF AUC={clip_oof_auc:.4f}")
print(f"  Fold F1s: {[f'{f:.4f}' for f in clip_fold_f1s]}")
gpu_cleanup()
print_vram()"""))

# ============================================================
# CELL 14: FORENSIC + CNN → XGBoost
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 14: FORENSIC + CNN → XGBoost (5-Fold CV)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
set_seeds()

# Remove dead forensic features
vt = VarianceThreshold(threshold=1e-10)
forensic_train_clean = vt.fit_transform(forensic_train)
forensic_test_clean  = vt.transform(forensic_test)
print(f"Forensic: {forensic_train.shape[1]} -> {forensic_train_clean.shape[1]} features")

# Combine forensic + CNN
X_combo_train = np.hstack([forensic_train_clean, cnn_train])
X_combo_test  = np.hstack([forensic_test_clean, cnn_test])

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    use_label_encoder=False, eval_metric='logloss',
    random_state=SEED, tree_method='hist', device='cuda'
)

# 5-fold CV
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
xgb_oof = np.zeros(len(y_all))
xgb_fold_f1s = []

print("=" * 60)
print("XGB-Forensic+CNN — 5-Fold CV")
print("=" * 60)
print(f"{'Fold':<5} {'Tr-F1':<8} {'Va-F1':<8} {'Gap':<7} {'AUC':<8}")
print('-' * 40)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    Xtr = X_combo_train[tr_idx]; Xv = X_combo_train[val_idx]
    ytr = y_all[tr_idx]; yv = y_all[val_idx]

    pipe = Pipeline([
        ('sc', RobustScaler()),
        ('pca', PCA(n_components=128, random_state=SEED)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ])
    pipe.fit(Xtr, ytr)
    tp = pipe.predict_proba(Xtr)[:, 1]
    vp = pipe.predict_proba(Xv)[:, 1]
    tf = f1_score(ytr, (tp >= 0.5).astype(int))
    vf = f1_score(yv, (vp >= 0.5).astype(int))
    au = roc_auc_score(yv, vp)
    gap = tf - vf
    xgb_oof[val_idx] = vp
    xgb_fold_f1s.append(vf)
    print(f"{fold+1:<5} {tf:<8.4f} {vf:<8.4f} {gap:<7.4f} {au:<8.4f}")

xgb_val_f1 = np.mean(xgb_fold_f1s)
xgb_oof_auc = roc_auc_score(y_all, xgb_oof)
print('-' * 40)
print(f"MEAN  {'':8} {xgb_val_f1:<8.4f}")

# Train on all data for test predictions
pipe_final = Pipeline([
    ('sc', RobustScaler()),
    ('pca', PCA(n_components=128, random_state=SEED)),
    ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
])
pipe_final.fit(X_combo_train, y_all)
xgb_test_proba = pipe_final.predict_proba(X_combo_test)[:, 1]

results_tracker['xgb_combo'] = {
    'name': 'XGB-Forensic+CNN', 'val_f1_mean': xgb_val_f1,
    'val_f1_std': np.std(xgb_fold_f1s), 'val_auc_mean': xgb_oof_auc,
    'oof_proba': xgb_oof.copy()
}
oof_store['xgb_combo'] = xgb_oof.copy()
test_pred_store['xgb_combo'] = xgb_test_proba.copy()

print(f"\nXGB Forensic+CNN: Val F1={xgb_val_f1:.4f}, OOF AUC={xgb_oof_auc:.4f}")"""))

# ============================================================
# CELL 15: MULTI-MODEL ENSEMBLE + THRESHOLD TUNING
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 15: 4-MODEL ENSEMBLE + THRESHOLD TUNING
# ============================================================
set_seeds()

ensemble_keys = ['siglip_finetune', 'dino_finetune', 'clip_finetune']
# Add XGB if it has reasonable signal
if results_tracker['xgb_combo']['val_f1_mean'] > 0.55:
    ensemble_keys.append('xgb_combo')

print("=" * 60)
print("ENSEMBLE COMPOSITION")
print("=" * 60)
for k in ensemble_keys:
    r = results_tracker[k]
    print(f"  {r['name']:<30} | Val F1: {r['val_f1_mean']:.4f}")

# ── Strategy A: Weighted Average (F1-squared weights) ───────
print("\n-- Strategy A: Weighted Average --")
f1_sq = {k: results_tracker[k]['val_f1_mean']**2 for k in ensemble_keys}
total = sum(f1_sq.values())
weights = {k: v/total for k, v in f1_sq.items()}
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} w={w:.4f}")

oof_blend_A = np.zeros(len(y_all))
for k, w in weights.items():
    oof_blend_A += w * oof_store[k]

best_thr_A = 0.5; best_f1_A = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (oof_blend_A >= t).astype(int))
    if f > best_f1_A:
        best_f1_A = f; best_thr_A = round(t, 2)
print(f"Result A: OOF F1={best_f1_A:.4f} @ Thr={best_thr_A}")

# ── Strategy B: LogReg Meta-Learner ─────────────────────────
print("\n-- Strategy B: LogReg Meta-Learner --")
X_meta = np.column_stack([oof_store[k] for k in ensemble_keys])
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros(len(y_all))
for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(y_all, y_all)):
    meta_lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_lr.fit(X_meta[tr_idx], y_all[tr_idx])
    meta_oof[val_idx] = meta_lr.predict_proba(X_meta[val_idx])[:, 1]

best_thr_B = 0.5; best_f1_B = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (meta_oof >= t).astype(int))
    if f > best_f1_B:
        best_f1_B = f; best_thr_B = round(t, 2)
print(f"Result B: OOF F1={best_f1_B:.4f} @ Thr={best_thr_B}")

# ── Strategy C: Rank-based averaging ────────────────────────
print("\n-- Strategy C: Rank-Based Average --")
from scipy.stats import rankdata
rank_blend = np.zeros(len(y_all))
for k in ensemble_keys:
    ranks = rankdata(oof_store[k]) / len(y_all)
    rank_blend += weights[k] * ranks

best_thr_C = 0.5; best_f1_C = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (rank_blend >= t).astype(int))
    if f > best_f1_C:
        best_f1_C = f; best_thr_C = round(t, 2)
print(f"Result C: OOF F1={best_f1_C:.4f} @ Thr={best_thr_C}")

# ── Select best ──────────────────────────────────────────────
strategies = {'weighted_avg': (best_f1_A, best_thr_A),
              'meta': (best_f1_B, best_thr_B),
              'rank_avg': (best_f1_C, best_thr_C)}
best_strat = max(strategies, key=lambda k: strategies[k][0])
best_ensemble_f1, BEST_THRESHOLD = strategies[best_strat]

print(f"\n{'='*60}")
print(f"WINNER: {best_strat} (F1={best_ensemble_f1:.4f})")

# Generate test predictions with winning strategy
if best_strat == 'meta':
    meta_final = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_final.fit(X_meta, y_all)
    X_meta_test = np.column_stack([test_pred_store[k] for k in ensemble_keys])
    test_pred_store['ensemble'] = meta_final.predict_proba(X_meta_test)[:, 1]
    final_oof = meta_oof
elif best_strat == 'rank_avg':
    test_blend = np.zeros(len(test_paths))
    for k in ensemble_keys:
        # Use train ranks to calibrate test
        test_blend += weights[k] * (rankdata(test_pred_store[k]) / len(test_paths))
    test_pred_store['ensemble'] = test_blend
    final_oof = rank_blend
else:  # weighted_avg
    test_blend = np.zeros(len(test_paths))
    for k, w in weights.items():
        test_blend += w * test_pred_store[k]
    test_pred_store['ensemble'] = test_blend
    final_oof = oof_blend_A

results_tracker['ensemble'] = {
    'name': f'Ensemble ({best_strat})',
    'val_f1_mean': best_ensemble_f1,
    'val_f1_std': 0.0,
    'val_auc_mean': roc_auc_score(y_all, final_oof),
    'oof_proba': final_oof.copy()
}
oof_store['ensemble'] = final_oof.copy()

print(f"FINAL ENSEMBLE: {best_strat} | F1={best_ensemble_f1:.4f} | Thr={BEST_THRESHOLD}")
print(f"{'='*60}")"""))

# ============================================================
# CELL 16: ANALYSIS & VISUALIZATION
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 16: ANALYSIS & VISUALIZATION
# ============================================================

print("=" * 75)
print(f"{'Model':<32} {'Va-F1':<8} {'AUC':<8}")
print("=" * 75)
for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'xgb_combo', 'ensemble']:
    if k not in results_tracker:
        continue
    r = results_tracker[k]
    print(f"  {r['name']:<30} {r['val_f1_mean']:<8.4f} {r['val_auc_mean']:<8.4f}")

# Model Correlation
print("\nModel Correlation:")
corr_keys = [k for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'xgb_combo']
             if k in oof_store]
corr_data = np.column_stack([oof_store[k] for k in corr_keys])
corr_matrix = np.corrcoef(corr_data.T)
labels = [k[:15] for k in corr_keys]
print(f"{'':>18}", end='')
for l in labels:
    print(f" {l:>15}", end='')
print()
for i, l in enumerate(labels):
    print(f"{l:>18}", end='')
    for j in range(len(labels)):
        print(f" {corr_matrix[i,j]:>15.4f}", end='')
    print()
print("\nLower correlation = more diverse = better ensemble")

# Confusion matrix
best_oof = oof_store.get('ensemble', siglip_oof)
oof_preds = (best_oof >= BEST_THRESHOLD).astype(int)
cm = confusion_matrix(y_all, oof_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
axes[0].set_title(f'Confusion Matrix (OOF, thr={BEST_THRESHOLD})')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'ensemble']:
    if k not in oof_store:
        continue
    fpr, tpr, _ = roc_curve(y_all, oof_store[k])
    auc_val = roc_auc_score(y_all, oof_store[k])
    axes[1].plot(fpr, tpr, label=f"{k[:20]} (AUC={auc_val:.4f})")
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].set_title('ROC Curves'); axes[1].legend()
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')

plt.tight_layout()
plt.savefig('analysis_v8.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFINAL ENSEMBLE F1: {best_ensemble_f1:.4f}")"""))

# ============================================================
# CELL 17: FINAL SUBMISSION
# ============================================================
cells.append(cell(r"""# ============================================================
# CELL 17: FINAL SUBMISSION
# ============================================================
set_seeds()

# Use ensemble predictions
test_proba = test_pred_store['ensemble']
thr = BEST_THRESHOLD

# Also check if best single model beats ensemble
best_single_key = max(['siglip_finetune', 'dino_finetune', 'clip_finetune'],
                      key=lambda k: results_tracker[k]['val_f1_mean'])
best_single_f1 = results_tracker[best_single_key]['val_f1_mean']

if best_single_f1 > best_ensemble_f1:
    print(f"NOTE: {best_single_key} ({best_single_f1:.4f}) > ensemble ({best_ensemble_f1:.4f})")
    print(f"Using {best_single_key} for submission instead.")
    test_proba = test_pred_store[best_single_key]
    # Re-optimize threshold for single model
    single_oof = oof_store[best_single_key]
    thr = 0.5
    best_f1_single = 0
    for t in np.arange(0.30, 0.71, 0.01):
        f = f1_score(y_all, (single_oof >= t).astype(int))
        if f > best_f1_single:
            best_f1_single = f; thr = round(t, 2)
    final_model_name = best_single_key
    final_f1 = best_f1_single
else:
    final_model_name = 'ensemble'
    final_f1 = best_ensemble_f1

print(f"{'='*60}")
print(f"FINAL MODEL: {final_model_name}")
print(f"Val F1: {final_f1:.4f}")
print(f"Threshold: {thr}")
print(f"{'='*60}")

preds_binary = (test_proba >= thr).astype(int)
submission = pd.DataFrame({
    'image_id':     df_test['image_id'].values,
    'ground_truth': preds_binary
})

# Sanity checks
assert submission.shape == (len(df_test), 2), f"Wrong shape: {submission.shape}"
assert submission['ground_truth'].isin([0, 1]).all()
assert not submission.isnull().any().any()

n0 = (submission['ground_truth'] == 0).sum()
n1 = (submission['ground_truth'] == 1).sum()
print(f"\nSanity checks PASSED")
print(f"  Real (0): {n0} ({n0/len(submission):.1%})")
print(f"  AI   (1): {n1} ({n1/len(submission):.1%})")
print(f"\nFirst 5 rows:")
print(submission.head())

# Save — try multiple paths
for out_path in ['/kaggle/working/submission.csv',
                 '/home/jovyan/work/data/submission.csv',
                 './submission.csv']:
    try:
        os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else '.', exist_ok=True)
        submission.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        break
    except Exception:
        continue

print(f"\nFINAL: {final_model_name} | Val F1={final_f1:.4f} | Thr={thr}")"""))

# ============================================================
# BUILD NOTEBOOK
# ============================================================
notebook = {
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_minor": 5,
            "pygments_lexer": "ipython3",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5,
    "cells": cells
}

out_path = "Version 8 - Triple Vision Ensemble.ipynb"
with open(out_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Generated: {out_path}")
print(f"Total cells: {len(cells)}")
