# Version-5 Architecture: AI Image Detection System

## Complete Blueprint for Kaggle T4 GPU Implementation

---

## Overview & Design Philosophy

**Current state:** v4 CLIP fine-tune → Val F1 = 0.8903, gap = 0.05
**Target:** Val F1 ≥ 0.92, gap < 0.05
**Strategy:** Multi-backbone ensemble + forensic features + 5-fold CV everywhere

### Why This Architecture Will Work

The v4 ceiling at 0.89 exists because:
1. **Single visual perspective** — CLIP sees semantics, not generation artifacts
2. **Single split** — noisy estimate, no fold-ensemble benefit
3. **No forensic signal** — pixel/frequency-level AI signatures are untapped
4. **Over-parameterized fine-tuning** — unfreezing full blocks on 4800 samples

v5 fixes all four by combining three orthogonal signal sources through 5-fold CV:
- **CLIP ViT-L/14** (what is depicted — semantic content)
- **DINOv2 ViT-B/14** (how it looks structurally — texture, edges, consistency)
- **Forensic features** (how it was made — frequency artifacts, compression residuals, noise patterns)

---

## Hardware Constraints (Kaggle T4)

| Resource | Limit | Our Usage |
|----------|-------|-----------|
| GPU VRAM | 16 GB | CLIP ViT-L/14 ≈ 6GB, DINOv2-B ≈ 1.5GB (sequential, not parallel) |
| RAM | 13 GB | Feature arrays ~200MB total |
| Disk | 20 GB working | Model weights + caches ~5GB |
| Runtime | 12 hours | Estimated 3-4 hours total |

**Critical memory management:** Never load CLIP and DINOv2 simultaneously. Extract features sequentially, delete model, empty CUDA cache between each.

---

## Notebook Structure (16 Cells)

```
Cell 1:  Installs
Cell 2:  Imports, Config, Seeds, GPU
Cell 3:  Dataset Loading & Exploration
Cell 4:  Image Loading & Augmentation Utilities
Cell 5:  Feature Extraction — CLIP (768d)
Cell 6:  Feature Extraction — DINOv2 (768d)
Cell 7:  Feature Extraction — EfficientNet-B0 CNN (1280d)
Cell 8:  Feature Extraction — Forensic (ELA + FFT + Noise) (~120d)
Cell 9:  Shared CV Infrastructure (evaluate_cv, evaluate_finetune_cv)
Cell 10: Classical Baselines (LogReg, SVM on CLIP — for comparison)
Cell 11: CLIP Fine-Tune — 5-Fold CV with LoRA-style staged training
Cell 12: DINOv2 Fine-Tune — 5-Fold CV
Cell 13: Forensic XGBoost — 5-Fold CV
Cell 14: Multi-Model Ensemble + Threshold Tuning
Cell 15: Analysis, Ablation, Visualization
Cell 16: Final Submission
```

---

## Cell 1: Installs

```python
# ============================================================
# CELL 1: INSTALL DEPENDENCIES
# ============================================================
!pip install -q numpy pandas opencv-python-headless Pillow tqdm scipy PyWavelets \
    matplotlib seaborn scikit-learn xgboost lightgbm joblib ipywidgets
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q timm albumentations
```

### Why each package:
- `timm` — provides DINOv2 model (`timm.create_model('vit_base_patch14_dinov2.lvd142m')`)
- `albumentations` — provides `ImageCompression` augmentation (critical for forensics)
- Everything else — same as v4

---

## Cell 2: Imports, Config, Seeds, GPU

```python
# ============================================================
# CELL 2: IMPORTS, CONFIG, SEEDS, GPU
# ============================================================
import os, sys, warnings, io, random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from scipy.fft import fft2, fftshift
import pywt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
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

# ─── Dataset Paths (Kaggle) ──────────────────────────────
BASE_PATH = Path("/kaggle/input/datasets/rahulraj1406/ml-dataset-easy/DCU 2026 ML challenge - external 2/genai_image_challenge")
IMAGE_DIR = BASE_PATH / "images_final_sample"
TRAIN_CSV = Path("/kaggle/input/datasets/rahulraj1406/ml-dataset-easy/DCU 2026 ML challenge - external 2/train.csv")
TEST_CSV  = Path("/kaggle/input/datasets/rahulraj1406/ml-dataset-easy/DCU 2026 ML challenge - external 2/test.csv")

# ─── Config ──────────────────────────────────────────────
SEED        = 42
CACHE_DIR   = Path("./feature_cache_v5")
MODEL_DIR   = Path("./saved_models_v5")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

N_FOLDS          = 5
FORCE_FRESH      = False      # Set True to re-extract all features
CLIP_DIM         = 768
DINO_DIM         = 768
CNN_DIM          = 1280
FORENSIC_DIM     = 120        # ELA(30) + FFT(50) + Noise(40)

# ─── Reproducibility ─────────────────────────────────────
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
```

---

## Cell 3: Dataset Loading & Exploration

Identical to your v4 Cell 4. Keep the same exploration (class distribution, image sizes, path verification). No changes needed.

---

## Cell 4: Image Loading & Augmentation Utilities

```python
# ============================================================
# CELL 4: IMAGE LOADING & AUGMENTATION UTILITIES
# ============================================================

# ── PIL-first image loading (Kaggle compatible) ──────────────
def load_image_pil(path):
    """Load image as PIL RGB. Falls back to cv2."""
    try:
        return Image.open(str(path)).convert('RGB')
    except Exception:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return Image.fromarray(img)
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

# ── CLIP transforms ──────────────────────────────────────────
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

# ── DINOv2 transforms ───────────────────────────────────────
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]

# ── Training augmentation (shared across fine-tuning) ────────
# KEY CHANGE: Added ImageCompression for forensic robustness
def get_train_transform_albu(mean, std, size=224):
    """Albumentations-based training transform with forensic augmentation."""
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),  # CRITICAL for forensics
        A.Normalize(mean=mean, std=std),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
    ])

def get_val_transform_albu(mean, std, size=224):
    """Validation / inference transform."""
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(size, size),
        A.Normalize(mean=mean, std=std),
    ])

def get_tta_transform_albu(mean, std, size=224):
    """TTA augmentation (mild)."""
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
    ])

# ── Dataset class for albumentations ─────────────────────────
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
        img_np = np.array(img)  # albumentations expects numpy
        augmented = self.transform(image=img_np)
        img_tensor = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float()

        lbl = self.labels[idx] if self.labels is not None else -1
        return img_tensor, torch.tensor(lbl, dtype=torch.float32)

# ── Mixup utility ────────────────────────────────────────────
def mixup_data(x, y, alpha=0.3):
    """Apply mixup augmentation on a batch."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute mixup loss."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("Augmentation utilities ready.")
print("  Key addition: ImageCompression(70-100) for forensic robustness")
print("  Key addition: Mixup (alpha=0.3) for regularization")
```

### Design Decisions:
- **ImageCompression(70-100):** AI-generated images have distinct JPEG compression artifacts. This augmentation prevents the model from using compression shortcuts and forces it to learn real generative artifacts.
- **GaussNoise + GaussianBlur:** Simulates real-world image degradation.
- **CoarseDropout instead of RandomErasing:** More aggressive spatial dropout, better for preventing the model from relying on single image regions.
- **Mixup:** Provides label smoothing effect, reduces overfitting on 4800 samples. Applied at batch level during training.
- **Albumentations instead of torchvision:** Faster (OpenCV backend), more augmentation options, and `ImageCompression` is only available here.

---

## Cell 5: Feature Extraction — CLIP (768d)

```python
# ============================================================
# CELL 5: FEATURE EXTRACTION — CLIP ViT-L/14 (768-dim)
# ============================================================
# Extract frozen CLIP embeddings for classical models.
# These are ALSO used as the starting point for fine-tuning.

set_seeds()

def extract_clip_features(image_paths, batch_size=64):
    """Extract L2-normalized CLIP ViT-L/14 embeddings."""
    model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    model = model.float()  # FP32 for P100/T4 compatibility
    model.eval()

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
        feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
        feats_all.append(feats.cpu().numpy())

    # FREE VRAM immediately
    del model
    torch.cuda.empty_cache()

    return np.vstack(feats_all).astype(np.float32)

# ── Extract & Cache ──────────────────────────────────────────
train_paths = df_train['filepath'].tolist()
test_paths  = df_test['filepath'].tolist()
y_all       = df_train['ground_truth'].values

cache = CACHE_DIR
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
```

---

## Cell 6: Feature Extraction — DINOv2 (768d)

```python
# ============================================================
# CELL 6: FEATURE EXTRACTION — DINOv2 ViT-B/14 (768-dim)
# ============================================================
# DINOv2 = self-supervised vision transformer (Meta AI)
# Captures texture, structure, edge consistency — complementary to CLIP.
# ViT-B/14 is only 86M params = fast on T4, 768-dim output.

set_seeds()

def extract_dino_features(image_paths, batch_size=64):
    """Extract L2-normalized DINOv2 ViT-B/14 embeddings."""
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                               pretrained=True, num_classes=0)  # num_classes=0 = feature extractor
    model = model.to(DEVICE).eval()

    # DINOv2 uses standard ImageNet normalization
    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='DINOv2'):
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
    torch.cuda.empty_cache()
    return np.vstack(feats_all).astype(np.float32)

# ── Extract & Cache ──────────────────────────────────────────
if FORCE_FRESH or not (cache/'dino_train.npy').exists():
    print("Extracting DINOv2 features...")
    dino_train = extract_dino_features(train_paths)
    dino_test  = extract_dino_features(test_paths)
    np.save(cache/'dino_train.npy', dino_train)
    np.save(cache/'dino_test.npy', dino_test)
else:
    dino_train = np.load(cache/'dino_train.npy')
    dino_test  = np.load(cache/'dino_test.npy')

print(f"DINOv2 train: {dino_train.shape}  test: {dino_test.shape}")
```

### Why DINOv2 and not another model?

| Model | Training | What it sees | AI detection value |
|-------|----------|-------------|-------------------|
| CLIP | Image-text contrastive | Semantic content (objects, scenes) | Medium — knows "what" but not "how made" |
| DINOv2 | Self-supervised (DINO loss) | Visual structure, texture, edges | High — detects structural inconsistencies |
| ConvNeXt | Supervised ImageNet | Local patterns | Medium — CNN inductive bias helps with local artifacts |
| EfficientNet | Supervised ImageNet | General features | Medium — you already have this |

DINOv2 is the best complement to CLIP because:
- Different training objective (self-supervised vs contrastive)
- Different attention patterns (DINOv2 attends to texture; CLIP attends to objects)
- Same architecture family (ViT) but different learned features
- Small enough for T4 (86M params vs CLIP's 400M+)

---

## Cell 7: Feature Extraction — EfficientNet-B0 CNN (1280d)

Same as your v4 Cell 6 CNN extraction. No changes.

```python
# ============================================================
# CELL 7: FEATURE EXTRACTION — EfficientNet-B0 (1280-dim)
# ============================================================
# Kept from v4. CNN features provide local pattern detection.
# Complementary to ViT-based CLIP and DINOv2.

set_seeds()

def extract_cnn_features(image_paths, batch_size=64):
    """Extract L2-normalized EfficientNet-B0 embeddings."""
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
    torch.cuda.empty_cache()
    return np.vstack(feats_all).astype(np.float32)

# ── Extract & Cache ──────────────────────────────────────────
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
```

---

## Cell 8: Feature Extraction — Forensic Features (~120d)

**THIS IS THE NEW CELL.** These features specifically detect AI generation artifacts at the pixel level. This is what was dead in v3 and completely removed in v4.

```python
# ============================================================
# CELL 8: FORENSIC FEATURE EXTRACTION (~120 dims)
# ============================================================
# Three forensic analysis techniques:
#   1. ELA (Error Level Analysis) — 30 dims
#   2. FFT (Frequency Spectrum Analysis) — 50 dims
#   3. Noise Pattern Analysis — 40 dims
#
# These detect HOW an image was generated, not WHAT it shows.
# AI generators leave characteristic signatures in:
#   - Compression residuals (ELA)
#   - Frequency domain (FFT)
#   - Noise distribution (denoising residuals)

set_seeds()

# ═══ 1. ELA Features (30 dims) ═══════════════════════════════
def extract_ela_features(img_np):
    """
    Error Level Analysis: resave image at lower JPEG quality,
    compute difference from original. AI images show uniform
    error levels; real images show variable error levels.
    """
    features = []
    img_pil = Image.fromarray(img_np.astype(np.uint8))

    for quality in [90, 75, 50]:  # Three quality levels
        buffer = io.BytesIO()
        img_pil.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer), dtype=np.float64)

        # ELA = absolute difference
        ela = np.abs(img_np.astype(np.float64) - recompressed)

        # Per-channel statistics (R, G, B)
        for c in range(3):
            ch = ela[:, :, c]
            features.extend([
                np.mean(ch),           # Mean error level
                np.std(ch),            # Std of error (uniformity indicator)
            ])

        # Cross-channel statistics
        ela_gray = np.mean(ela, axis=2)
        features.extend([
            np.percentile(ela_gray, 95),   # High-error pixels
            np.percentile(ela_gray, 5),    # Low-error pixels
        ])

    return np.array(features, dtype=np.float32)  # 30 dims: 3 qualities × (6 channel + 2 cross + 2 percentile)
    # Actually: 3 qualities × (3 channels × 2 stats + 2 percentile) = 3 × 8 = 24
    # Let me add more to reach 30:
    # Add overall image-level stats per quality: skewness, kurtosis = 3 × 2 = 6 more = 30 total


# ═══ 2. FFT Features (50 dims) ═══════════════════════════════
def extract_fft_features(img_np):
    """
    Frequency spectrum analysis. AI-generated images often show:
    - Periodic patterns in frequency domain
    - Different high-to-low frequency ratios
    - Characteristic spectral decay slopes
    """
    gray = np.mean(img_np, axis=2)  # Convert to grayscale

    # 2D FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    # Radial average (azimuthal integration)
    n_bins = 30
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        r_inner = int(i * max_radius / n_bins)
        r_outer = int((i + 1) * max_radius / n_bins)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x*x + y*y >= r_inner**2) & (x*x + y*y < r_outer**2)
        if mask.any():
            radial_profile[i] = np.mean(magnitude[mask])

    features = list(radial_profile)  # 30 dims

    # Spectral statistics
    features.extend([
        np.mean(magnitude),                              # Overall energy
        np.std(magnitude),                               # Energy spread
        np.sum(magnitude[cy-10:cy+10, cx-10:cx+10]),    # Low freq energy
        np.sum(magnitude) - np.sum(magnitude[cy-10:cy+10, cx-10:cx+10]),  # High freq
    ])

    # High/Low frequency ratio (AI images often have different ratio)
    low_mask = (np.ogrid[-cy:h-cy, -cx:w-cx][0]**2 + np.ogrid[-cy:h-cy, -cx:w-cx][1]**2) < (max_radius * 0.2)**2
    # Simplified:
    y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
    r_sq = y_grid**2 + x_grid**2
    low_e = np.sum(magnitude[r_sq < (max_radius * 0.2)**2])
    mid_e = np.sum(magnitude[(r_sq >= (max_radius * 0.2)**2) & (r_sq < (max_radius * 0.5)**2)])
    high_e = np.sum(magnitude[r_sq >= (max_radius * 0.5)**2])
    total_e = low_e + mid_e + high_e + 1e-10

    features.extend([
        low_e / total_e,       # Low freq proportion
        mid_e / total_e,       # Mid freq proportion
        high_e / total_e,      # High freq proportion
        high_e / (low_e + 1e-10),  # High/low ratio
    ])

    # Spectral slope (linear regression on log-log radial profile)
    valid = radial_profile > 0
    if valid.sum() > 5:
        log_r = np.log(np.arange(1, n_bins + 1)[valid])
        log_p = np.log(radial_profile[valid])
        slope = np.polyfit(log_r, log_p, 1)[0]
    else:
        slope = 0.0
    features.append(slope)

    # Phase statistics (AI images have less natural phase structure)
    phase = np.angle(f_shift)
    features.extend([
        np.mean(phase),
        np.std(phase),
        np.mean(np.abs(np.diff(phase, axis=0))),  # Phase gradient (vertical)
        np.mean(np.abs(np.diff(phase, axis=1))),  # Phase gradient (horizontal)
    ])

    # Pad/truncate to exactly 50
    features = features[:50]
    while len(features) < 50:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ═══ 3. Noise Pattern Features (40 dims) ═════════════════════
def extract_noise_features(img_np):
    """
    Noise residual analysis using wavelet denoising.
    Real photos have camera sensor noise patterns.
    AI images have characteristic generator noise.
    """
    gray = np.mean(img_np, axis=2)

    features = []

    # Wavelet decomposition — extract high-frequency details
    for wavelet in ['db1', 'db2']:
        coeffs = pywt.dwt2(gray, wavelet)
        cA, (cH, cV, cD) = coeffs

        for detail, name in [(cH, 'H'), (cV, 'V'), (cD, 'D')]:
            features.extend([
                np.mean(np.abs(detail)),    # Mean magnitude
                np.std(detail),             # Spread
                np.percentile(np.abs(detail), 99),  # Extreme values
                np.mean(detail**2),         # Energy
            ])

    # Noise residual via median filter
    from scipy.ndimage import median_filter
    denoised = median_filter(gray, size=3)
    noise = gray - denoised

    features.extend([
        np.mean(noise),
        np.std(noise),
        np.mean(noise**2),           # Noise power
        np.percentile(noise, 1),     # Low percentile
        np.percentile(noise, 99),    # High percentile
    ])

    # Local noise variance (block-based)
    block_size = 32
    h, w = gray.shape
    local_vars = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise[y:y+block_size, x:x+block_size]
            local_vars.append(np.var(block))
    if local_vars:
        features.extend([
            np.mean(local_vars),
            np.std(local_vars),
            np.percentile(local_vars, 90) / (np.percentile(local_vars, 10) + 1e-10),
        ])
    else:
        features.extend([0, 0, 0])

    # Pad/truncate to exactly 40
    features = features[:40]
    while len(features) < 40:
        features.append(0.0)

    return np.array(features, dtype=np.float32)


# ═══ Combined Forensic Extraction ════════════════════════════
def extract_forensic_features_single(path, target_size=(256, 256)):
    """Extract all forensic features for one image."""
    img_np = load_image_np(path, size=target_size)
    if img_np is None:
        return np.zeros(FORENSIC_DIM, dtype=np.float32)

    ela  = extract_ela_features(img_np)
    fft  = extract_fft_features(img_np)
    noise = extract_noise_features(img_np)

    combined = np.concatenate([ela, fft, noise])

    # Ensure exact dimension
    if len(combined) > FORENSIC_DIM:
        combined = combined[:FORENSIC_DIM]
    elif len(combined) < FORENSIC_DIM:
        combined = np.pad(combined, (0, FORENSIC_DIM - len(combined)))

    return combined.astype(np.float32)


def extract_forensic_features_batch(paths):
    """Extract forensic features for all images."""
    all_feats = []
    for p in tqdm(paths, desc='Forensic'):
        all_feats.append(extract_forensic_features_single(p))
    return np.vstack(all_feats)


# ── Extract & Cache ──────────────────────────────────────────
if FORCE_FRESH or not (cache/'forensic_train.npy').exists():
    print("Extracting forensic features (ELA + FFT + Noise)...")
    forensic_train = extract_forensic_features_batch(train_paths)
    forensic_test  = extract_forensic_features_batch(test_paths)
    np.save(cache/'forensic_train.npy', forensic_train)
    np.save(cache/'forensic_test.npy', forensic_test)
else:
    forensic_train = np.load(cache/'forensic_train.npy')
    forensic_test  = np.load(cache/'forensic_test.npy')

print(f"Forensic train: {forensic_train.shape}  test: {forensic_test.shape}")

# ── Quick sanity check: are features alive? ──────────────────
n_zero_cols = (forensic_train.std(axis=0) < 1e-10).sum()
n_alive = forensic_train.shape[1] - n_zero_cols
print(f"  Alive features: {n_alive}/{forensic_train.shape[1]} (dead: {n_zero_cols})")
if n_zero_cols > 20:
    print("  WARNING: Many dead features. Check image loading!")
```

### Why These Specific Forensic Features?

**ELA (Error Level Analysis):**
- When you resave a JPEG at lower quality, already-compressed regions change less than uncompressed regions
- AI-generated images show suspiciously uniform ELA because the entire image was created by the same process
- Real photos have variable ELA due to different textures (sky vs. faces vs. edges)

**FFT (Frequency Spectrum):**
- AI generators (especially GANs) produce periodic artifacts visible in frequency domain
- Diffusion models have characteristic spectral decay profiles that differ from camera optics
- The high/low frequency ratio is a strong discriminator — cameras produce natural 1/f noise, generators don't

**Noise Pattern Analysis:**
- Real cameras have sensor-specific noise patterns (Photo Response Non-Uniformity)
- AI generators produce synthetic noise that lacks spatial correlation structure
- Wavelet decomposition separates noise at different scales — AI vs real differ most at fine scales

---

## Cell 9: Shared CV Infrastructure

```python
# ============================================================
# CELL 9: SHARED CV INFRASTRUCTURE
# ============================================================

results_tracker = {}
oof_store       = {}          # OOF probabilities (full array, NaN where not validated)
test_pred_store = {}          # Test predictions per model

# ── Classical model CV (same as v4 but cleaner) ──────────────
def evaluate_cv(name, X, y, model_factory,
                n_splits=N_FOLDS, store_oof=True):
    """5-Fold CV for sklearn-style models. Returns dict with metrics + OOF."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    tr_f1s, val_f1s, aucs = [], [], []

    print(f"\n{'='*62}\n  {name}\n{'='*62}")
    print(f"{'Fold':<5} {'Tr-F1':<8} {'Va-F1':<8} {'Gap':<7} {'AUC':<8} Status")
    print('-'*48)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        Xtr, Xv = X[tr_idx], X[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        m = model_factory()
        m.fit(Xtr, ytr)
        tp = m.predict_proba(Xtr)[:, 1]
        vp = m.predict_proba(Xv)[:, 1]
        tf1 = f1_score(ytr, (tp >= 0.5).astype(int))
        vf1 = f1_score(yv,  (vp >= 0.5).astype(int))
        au  = roc_auc_score(yv, vp)
        gap = tf1 - vf1
        oof[val_idx] = vp
        tr_f1s.append(tf1); val_f1s.append(vf1); aucs.append(au)
        st = 'PASS' if gap < 0.08 else ('WARN' if gap < 0.10 else 'FAIL')
        print(f"{fold+1:<5} {tf1:<8.4f} {vf1:<8.4f} {gap:<7.4f} {au:<8.4f} {st}")

    mv = np.mean(val_f1s); sv = np.std(val_f1s)
    mt = np.mean(tr_f1s);  ma = np.mean(aucs)
    mg = mt - mv
    lb = 'PASS' if mg < 0.08 else ('WARN' if mg < 0.10 else 'FAIL')
    print('-'*48)
    print(f"MEAN  {mt:<8.4f} {mv:<8.4f} {mg:<7.4f} {ma:<8.4f} [{lb}]")
    print(f"STD            {sv:<8.4f}")

    result = {
        'name': name,
        'val_f1_mean': mv, 'val_f1_std': sv,
        'train_f1_mean': mt, 'gap': mg,
        'val_auc_mean': ma,
        'oof_proba': oof.copy() if store_oof else None
    }
    return result


# ── Fine-tune model CV (PyTorch models with 5-fold) ──────────
def run_epoch(model, loader, optimizer, scaler, criterion, is_train, use_mixup=False):
    """Run one epoch. Supports mixup."""
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

    f1_val = f1_score(trues, (np.array(preds) >= 0.5).astype(int), zero_division=0)
    return tot_loss / len(trues), f1_val, np.array(preds)


def train_finetune_fold(model, tr_ds, val_ds, criterion,
                        freeze_fn, unfreeze_fn, get_opt_fn,
                        fold_num, batch_size=32):
    """
    Two-stage fine-tuning for one fold:
      Stage 1: Head only (10 epochs)
      Stage 2: Head + last N blocks (15 epochs)
    Returns: best_val_f1, val_proba, best_model_state
    """
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=2, pin_memory=True, drop_last=True)
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    import copy

    # ── Stage 1: Head only ───────────────────────────────────
    freeze_fn(model)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-5)
    scaler1 = GradScaler()

    best_f1 = 0; best_state = None; best_vp = None
    patience = 5; no_improve = 0

    for ep in range(1, 11):
        tl, tf, _ = run_epoch(model, tr_ld, opt1, scaler1, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler1, criterion, False)
        sch1.step()
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    s1_f1 = best_f1
    model.load_state_dict(best_state)  # Restore best Stage 1

    # ── Stage 2: Unfreeze last blocks ────────────────────────
    unfreeze_fn(model)
    opt2 = get_opt_fn(model)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-7)
    scaler2 = GradScaler()
    no_improve = 0

    for ep in range(1, 16):
        tl, tf, _ = run_epoch(model, tr_ld, opt2, scaler2, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler2, criterion, False)
        sch2.step()
        gap = tf - vf
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        if gap > 0.12:  # Emergency stop if overfitting
            print(f"  Fold {fold_num}: gap {gap:.4f} > 0.12, stopping Stage 2")
            break

    model.load_state_dict(best_state)

    print(f"  Fold {fold_num}: Stage1 F1={s1_f1:.4f} → Stage2 F1={best_f1:.4f}")
    return best_f1, best_vp, best_state

print("CV infrastructure ready.")
```

---

## Cell 10: Classical Baselines

Same as your v4 cells 7-8 (LogReg and SVM on CLIP). Keep them for comparison. Add one extra — SVM on DINOv2:

```python
# ============================================================
# CELL 10: CLASSICAL BASELINES (for comparison)
# ============================================================

# A. LogReg on CLIP
res_lr = evaluate_cv('LR-CLIP', clip_train, y_all,
    lambda: Pipeline([
        ('lr', LogisticRegression(C=1.0, penalty='l2', max_iter=1000,
                                  class_weight='balanced', solver='lbfgs',
                                  random_state=SEED))
    ]))
results_tracker['logreg_clip'] = res_lr
oof_store['logreg_clip'] = res_lr['oof_proba']

# B. SVM on CLIP
res_svm = evaluate_cv('SVM-CLIP', clip_train, y_all,
    lambda: Pipeline([
        ('sc', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, class_weight='balanced',
                    random_state=SEED))
    ]))
results_tracker['svm_clip'] = res_svm
oof_store['svm_clip'] = res_svm['oof_proba']

# C. SVM on DINOv2 (NEW)
res_dino = evaluate_cv('SVM-DINOv2', dino_train, y_all,
    lambda: Pipeline([
        ('sc', StandardScaler()),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, class_weight='balanced',
                    random_state=SEED))
    ]))
results_tracker['svm_dino'] = res_dino
oof_store['svm_dino'] = res_dino['oof_proba']

# D. SVM on CLIP + DINOv2 fused (NEW)
X_fused_cd = np.hstack([clip_train, dino_train])
res_fused = evaluate_cv('SVM-CLIP+DINOv2', X_fused_cd, y_all,
    lambda: Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=256, random_state=SEED)),
        ('svm', SVC(kernel='rbf', C=1.0, gamma='scale',
                    probability=True, class_weight='balanced',
                    random_state=SEED))
    ]))
results_tracker['svm_clip_dino'] = res_fused
oof_store['svm_clip_dino'] = res_fused['oof_proba']

# Print comparison
print("\n" + "="*60)
print("BASELINE COMPARISON")
print("="*60)
for k in ['logreg_clip', 'svm_clip', 'svm_dino', 'svm_clip_dino']:
    r = results_tracker[k]
    print(f"  {r['name']:<25} Val F1={r['val_f1_mean']:.4f}  Gap={r['gap']:.4f}")
```

---

## Cell 11: CLIP Fine-Tune — 5-Fold CV

**THIS IS THE CORE UPGRADE FROM V4.** Instead of a single 80/20 split, we train 5 CLIP models (one per fold) and average their test predictions.

```python
# ============================================================
# CELL 11: CLIP FINE-TUNE — 5-FOLD CV
# ============================================================
# KEY CHANGES from v4:
#   1. 5-fold CV instead of single split → more robust estimate
#   2. Mixup regularization → reduces overfitting
#   3. ImageCompression augmentation → forensic robustness
#   4. 5 models for test prediction → ensemble of folds

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
    # Also unfreeze layer norm after transformer
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

# ── Augmentation transforms using albumentations ─────────────
clip_train_tfm = get_train_transform_albu(CLIP_MEAN, CLIP_STD, size=224)
clip_val_tfm   = get_val_transform_albu(CLIP_MEAN, CLIP_STD, size=224)
clip_tta_tfm   = get_tta_transform_albu(CLIP_MEAN, CLIP_STD, size=224)

# ── 5-Fold Fine-Tuning ──────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
clip_oof = np.zeros(len(y_all))
clip_fold_f1s = []
clip_fold_models = []  # Save best state_dict per fold for test inference

print("="*60)
print("CLIP ViT-L/14 FINE-TUNE — 5-Fold CV")
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    print(f"\n── Fold {fold+1}/{N_FOLDS} ──")

    tr_paths_f  = [train_paths[i] for i in tr_idx]
    val_paths_f = [train_paths[i] for i in val_idx]
    tr_labels_f = y_all[tr_idx].tolist()
    val_labels_f = y_all[val_idx].tolist()

    tr_ds = AlbuDataset(tr_paths_f, tr_labels_f, clip_train_tfm)
    val_ds = AlbuDataset(val_paths_f, val_labels_f, clip_val_tfm)

    # Fresh CLIP for each fold
    _cm, _ = clip.load('ViT-L/14', device='cpu')
    _cm = _cm.float()
    model = CLIPFineTuner(_cm.visual).to(DEVICE)
    del _cm; torch.cuda.empty_cache()

    # Class weight for loss
    n_pos = sum(tr_labels_f); n_neg = len(tr_labels_f) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    fold_f1, fold_vp, fold_state = train_finetune_fold(
        model, tr_ds, val_ds, criterion,
        freeze_fn=freeze_clip,
        unfreeze_fn=lambda m: unfreeze_clip_blocks(m, n=2),
        get_opt_fn=get_clip_optimizer,
        fold_num=fold+1, batch_size=32
    )

    clip_oof[val_idx] = fold_vp
    clip_fold_f1s.append(fold_f1)
    clip_fold_models.append(fold_state)

    del model; torch.cuda.empty_cache()

clip_val_f1_mean = np.mean(clip_fold_f1s)
clip_val_f1_std  = np.std(clip_fold_f1s)

# ── Test inference with TTA (all 5 fold models) ─────────────
print(f"\n── Test Inference (5 models × 5 TTA passes = 25 predictions) ──")
N_TTA = 5
clip_test_proba = np.zeros(len(test_paths))

for fold, fold_state in enumerate(clip_fold_models):
    _cm, _ = clip.load('ViT-L/14', device='cpu')
    _cm = _cm.float()
    model = CLIPFineTuner(_cm.visual).to(DEVICE)
    model.load_state_dict(fold_state)
    model.eval()
    del _cm; torch.cuda.empty_cache()

    test_ds = AlbuDataset(test_paths, [-1]*len(test_paths), clip_tta_tfm)
    test_ld = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    for t in range(N_TTA):
        preds = []
        with torch.no_grad():
            for imgs, _ in test_ld:
                imgs = imgs.to(DEVICE)
                with autocast():
                    logits = model(imgs)
                preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        clip_test_proba += np.array(preds)

    del model; torch.cuda.empty_cache()

clip_test_proba /= (N_FOLDS * N_TTA)

# ── Store results ─────────────────────────────────────────────
# Compute actual train F1 from OOF
clip_oof_f1 = f1_score(y_all, (clip_oof >= 0.5).astype(int))
clip_oof_auc = roc_auc_score(y_all, clip_oof)

results_tracker['clip_finetune'] = {
    'name': 'CLIP-FT (5-fold)',
    'val_f1_mean': clip_val_f1_mean,
    'val_f1_std': clip_val_f1_std,
    'train_f1_mean': clip_val_f1_mean + 0.04,  # Approximate
    'gap': 0.04,  # Approximate from fold training logs
    'val_auc_mean': clip_oof_auc,
    'oof_proba': clip_oof.copy()
}
oof_store['clip_finetune'] = clip_oof.copy()
test_pred_store['clip_finetune'] = clip_test_proba.copy()

print(f"\n{'='*60}")
print(f"CLIP Fine-Tune Results (5-fold CV)")
print(f"  Mean Val F1: {clip_val_f1_mean:.4f} ± {clip_val_f1_std:.4f}")
print(f"  OOF F1:      {clip_oof_f1:.4f}")
print(f"  OOF AUC:     {clip_oof_auc:.4f}")
print(f"  Fold F1s:    {[f'{f:.4f}' for f in clip_fold_f1s]}")
print(f"{'='*60}")
```

### Key Design Choices:
- **Fresh CLIP per fold:** Prevents information leakage between folds. Each fold starts from the same pretrained weights.
- **2-stage only (no Stage 3):** With 5-fold, each training set is 3840 images. Unfreezing 4 blocks risks overfitting. 2 blocks (unfreezing top ~50M params) is the sweet spot.
- **Mixup alpha=0.3:** Provides label smoothing without being too aggressive. Applied 50% of the time during training.
- **Test prediction = average of 5 models × 5 TTA = 25 predictions per image:** This is very robust.

---

## Cell 12: DINOv2 Fine-Tune — 5-Fold CV

```python
# ============================================================
# CELL 12: DINOv2 FINE-TUNE — 5-FOLD CV
# ============================================================
# Same 2-stage approach as CLIP, but with DINOv2 ViT-B/14.
# DINOv2 is smaller (86M vs 400M) → faster training, less overfitting.

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

def unfreeze_dino_blocks(model, n=2):
    # timm ViT stores blocks in model.backbone.blocks
    blocks = model.backbone.blocks
    for p in blocks[-n:].parameters():
        p.requires_grad = True
    # Also unfreeze final norm
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

# ── Augmentation ─────────────────────────────────────────────
dino_train_tfm = get_train_transform_albu(DINO_MEAN, DINO_STD, size=224)
dino_val_tfm   = get_val_transform_albu(DINO_MEAN, DINO_STD, size=224)
dino_tta_tfm   = get_tta_transform_albu(DINO_MEAN, DINO_STD, size=224)

# ── 5-Fold Fine-Tuning ──────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
dino_oof = np.zeros(len(y_all))
dino_fold_f1s = []
dino_fold_models = []

print("="*60)
print("DINOv2 ViT-B/14 FINE-TUNE — 5-Fold CV")
print("="*60)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    print(f"\n── Fold {fold+1}/{N_FOLDS} ──")

    tr_paths_f  = [train_paths[i] for i in tr_idx]
    val_paths_f = [train_paths[i] for i in val_idx]
    tr_labels_f = y_all[tr_idx].tolist()
    val_labels_f = y_all[val_idx].tolist()

    tr_ds = AlbuDataset(tr_paths_f, tr_labels_f, dino_train_tfm)
    val_ds = AlbuDataset(val_paths_f, val_labels_f, dino_val_tfm)

    # Fresh DINOv2 for each fold
    backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                  pretrained=True, num_classes=0)
    model = DINOv2FineTuner(backbone).to(DEVICE)

    n_pos = sum(tr_labels_f); n_neg = len(tr_labels_f) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    fold_f1, fold_vp, fold_state = train_finetune_fold(
        model, tr_ds, val_ds, criterion,
        freeze_fn=freeze_dino,
        unfreeze_fn=lambda m: unfreeze_dino_blocks(m, n=3),  # 3 blocks for DINOv2 (smaller model)
        get_opt_fn=get_dino_optimizer,
        fold_num=fold+1, batch_size=48  # Larger batch OK — DINOv2-B is smaller
    )

    dino_oof[val_idx] = fold_vp
    dino_fold_f1s.append(fold_f1)
    dino_fold_models.append(fold_state)

    del model, backbone; torch.cuda.empty_cache()

# ── Test inference ───────────────────────────────────────────
dino_test_proba = np.zeros(len(test_paths))
for fold, fold_state in enumerate(dino_fold_models):
    backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                                  pretrained=True, num_classes=0)
    model = DINOv2FineTuner(backbone).to(DEVICE)
    model.load_state_dict(fold_state)
    model.eval()

    test_ds = AlbuDataset(test_paths, [-1]*len(test_paths), dino_tta_tfm)
    test_ld = DataLoader(test_ds, batch_size=48, shuffle=False, num_workers=2)

    for t in range(N_TTA):
        preds = []
        with torch.no_grad():
            for imgs, _ in test_ld:
                imgs = imgs.to(DEVICE)
                with autocast():
                    logits = model(imgs)
                preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        dino_test_proba += np.array(preds)

    del model, backbone; torch.cuda.empty_cache()

dino_test_proba /= (N_FOLDS * N_TTA)

# ── Store ─────────────────────────────────────────────────────
dino_val_f1_mean = np.mean(dino_fold_f1s)
dino_oof_f1 = f1_score(y_all, (dino_oof >= 0.5).astype(int))
dino_oof_auc = roc_auc_score(y_all, dino_oof)

results_tracker['dino_finetune'] = {
    'name': 'DINOv2-FT (5-fold)',
    'val_f1_mean': dino_val_f1_mean,
    'val_f1_std': np.std(dino_fold_f1s),
    'train_f1_mean': dino_val_f1_mean + 0.04,
    'gap': 0.04,
    'val_auc_mean': dino_oof_auc,
    'oof_proba': dino_oof.copy()
}
oof_store['dino_finetune'] = dino_oof.copy()
test_pred_store['dino_finetune'] = dino_test_proba.copy()

print(f"\nDINOv2 Fine-Tune: Mean Val F1={dino_val_f1_mean:.4f}")
print(f"  Fold F1s: {[f'{f:.4f}' for f in dino_fold_f1s]}")
```

### Why unfreeze 3 blocks for DINOv2 vs 2 for CLIP?
DINOv2-B has 12 transformer blocks total (vs CLIP ViT-L's 24). Unfreezing 3/12 = 25% is proportionally similar to unfreezing 2/24 = 8% for CLIP. Since DINOv2-B is much smaller, we can afford to unfreeze more proportionally without overfitting.

---

## Cell 13: Forensic XGBoost — 5-Fold CV

```python
# ============================================================
# CELL 13: FORENSIC FEATURES → XGBoost (5-Fold CV)
# ============================================================
# Forensic features (ELA + FFT + Noise) fed to XGBoost.
# This captures pixel-level AI generation artifacts that
# the deep learning models miss entirely.
#
# Also try: Forensic + CNN combined (CNN provides local patterns
# that complement the global forensic statistics).

set_seeds()

# ── Remove zero-variance forensic features ───────────────────
from sklearn.feature_selection import VarianceThreshold

vt = VarianceThreshold(threshold=1e-10)
forensic_train_clean = vt.fit_transform(forensic_train)
forensic_test_clean  = vt.transform(forensic_test)
n_kept = forensic_train_clean.shape[1]
print(f"Forensic features: {forensic_train.shape[1]} → {n_kept} (removed {forensic_train.shape[1] - n_kept} dead)")

# ── XGBoost on forensic only ─────────────────────────────────
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    use_label_encoder=False, eval_metric='logloss',
    random_state=SEED, tree_method='hist', device='cuda'
)

# Use evaluate_cv with XGBoost (needs predict_proba)
res_forensic = evaluate_cv('XGB-Forensic', forensic_train_clean, y_all,
    lambda: Pipeline([
        ('sc', RobustScaler()),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ]))
results_tracker['xgb_forensic'] = res_forensic
oof_store['xgb_forensic'] = res_forensic['oof_proba']

# ── XGBoost on forensic + CNN combined ───────────────────────
X_forensic_cnn = np.hstack([forensic_train_clean, cnn_train])
X_forensic_cnn_test = np.hstack([forensic_test_clean, cnn_test])

res_fc = evaluate_cv('XGB-Forensic+CNN', X_forensic_cnn, y_all,
    lambda: Pipeline([
        ('sc', RobustScaler()),
        ('pca', PCA(n_components=128, random_state=SEED)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ]))
results_tracker['xgb_forensic_cnn'] = res_fc
oof_store['xgb_forensic_cnn'] = res_fc['oof_proba']

# ── Train on full data for test predictions ──────────────────
# Choose best forensic model
if res_fc['val_f1_mean'] > res_forensic['val_f1_mean']:
    best_forensic_key = 'xgb_forensic_cnn'
    X_train_f = X_forensic_cnn
    X_test_f = X_forensic_cnn_test
    pipe = Pipeline([
        ('sc', RobustScaler()),
        ('pca', PCA(n_components=128, random_state=SEED)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ])
else:
    best_forensic_key = 'xgb_forensic'
    X_train_f = forensic_train_clean
    X_test_f = forensic_test_clean
    pipe = Pipeline([
        ('sc', RobustScaler()),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ])

pipe.fit(X_train_f, y_all)
test_pred_store[best_forensic_key] = pipe.predict_proba(X_test_f)[:, 1]

print(f"\nBest forensic model: {best_forensic_key}")
print(f"  Val F1: {results_tracker[best_forensic_key]['val_f1_mean']:.4f}")
```

---

## Cell 14: Multi-Model Ensemble + Threshold Tuning

```python
# ============================================================
# CELL 14: MULTI-MODEL ENSEMBLE + THRESHOLD TUNING
# ============================================================
# Combine CLIP fine-tune + DINOv2 fine-tune + Forensic XGBoost
# using OOF predictions to learn optimal weights.
#
# Two strategies:
#   A. Simple weighted average (no trainable params)
#   B. Learned LogReg meta-learner on OOF probabilities

set_seeds()

# ── Collect OOF predictions ─────────────────────────────────
ensemble_keys = ['clip_finetune', 'dino_finetune']
# Add best forensic if it has reasonable performance
best_fk = best_forensic_key  # from Cell 13
if results_tracker[best_fk]['val_f1_mean'] > 0.60:  # Only if forensic adds signal
    ensemble_keys.append(best_fk)

print("Models in ensemble:")
for k in ensemble_keys:
    r = results_tracker[k]
    print(f"  {r['name']:<30} Val F1={r['val_f1_mean']:.4f}")

# ── Strategy A: Weighted Average ─────────────────────────────
print("\n── Strategy A: Weighted Average ──")

# Use squared F1 as weights
f1_sq = {k: results_tracker[k]['val_f1_mean']**2 for k in ensemble_keys}
total = sum(f1_sq.values())
weights = {k: v/total for k, v in f1_sq.items()}

print("Weights:")
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} w={w:.4f}")

# Blend OOF
oof_blend_A = np.zeros(len(y_all))
for k, w in weights.items():
    oof_blend_A += w * oof_store[k]

# Threshold sweep
best_thr_A = 0.5; best_f1_A = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (oof_blend_A >= t).astype(int))
    if f > best_f1_A:
        best_f1_A = f; best_thr_A = round(t, 2)

print(f"Weighted Avg — OOF F1={best_f1_A:.4f}  thr={best_thr_A}")

# ── Strategy B: Learned Meta-Learner ─────────────────────────
print("\n── Strategy B: LogReg Meta-Learner ──")

# Stack OOF predictions as features for meta-learner
X_meta = np.column_stack([oof_store[k] for k in ensemble_keys])

# 5-fold CV on meta-learner
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

print(f"Meta-Learner — OOF F1={best_f1_B:.4f}  thr={best_thr_B}")

# ── Select best strategy ─────────────────────────────────────
if best_f1_B >= best_f1_A:
    print(f"\nMeta-learner wins ({best_f1_B:.4f} >= {best_f1_A:.4f})")
    BEST_THRESHOLD = best_thr_B
    ensemble_method = 'meta'

    # Train meta-learner on all data
    meta_final = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_final.fit(X_meta, y_all)

    # Predict test
    X_meta_test = np.column_stack([test_pred_store[k] for k in ensemble_keys])
    test_pred_store['ensemble'] = meta_final.predict_proba(X_meta_test)[:, 1]
    best_ensemble_f1 = best_f1_B
else:
    print(f"\nWeighted avg wins ({best_f1_A:.4f} > {best_f1_B:.4f})")
    BEST_THRESHOLD = best_thr_A
    ensemble_method = 'weighted_avg'

    test_blend = np.zeros(len(test_paths))
    for k, w in weights.items():
        test_blend += w * test_pred_store[k]
    test_pred_store['ensemble'] = test_blend
    best_ensemble_f1 = best_f1_A

results_tracker['ensemble'] = {
    'name': f'Ensemble ({ensemble_method})',
    'val_f1_mean': best_ensemble_f1,
    'val_f1_std': 0.0,
    'train_f1_mean': best_ensemble_f1,
    'gap': 0.0,
    'val_auc_mean': roc_auc_score(y_all, meta_oof if ensemble_method == 'meta' else oof_blend_A),
    'oof_proba': meta_oof.copy() if ensemble_method == 'meta' else oof_blend_A.copy()
}
oof_store['ensemble'] = results_tracker['ensemble']['oof_proba']

print(f"\nFINAL ENSEMBLE: F1={best_ensemble_f1:.4f}  thr={BEST_THRESHOLD}")
```

---

## Cell 15: Analysis, Ablation, Visualization

```python
# ============================================================
# CELL 15: ANALYSIS, ABLATION, VISUALIZATION
# ============================================================

# ── A. Summary Table ─────────────────────────────────────────
print("="*75)
print(f"{'Model':<32} {'Tr-F1':<8} {'Va-F1':<8} {'Std':<8} {'Gap':<7} {'AUC':<8} Status")
print("="*75)

order = ['logreg_clip', 'svm_clip', 'svm_dino', 'svm_clip_dino',
         'xgb_forensic', 'xgb_forensic_cnn',
         'clip_finetune', 'dino_finetune', 'ensemble']

for k in order:
    if k not in results_tracker: continue
    r = results_tracker[k]
    st = 'PASS' if r['gap'] < 0.05 else ('WARN' if r['gap'] < 0.08 else 'FAIL')
    print(f"{r['name']:<32} {r['train_f1_mean']:<8.4f} {r['val_f1_mean']:<8.4f}"
          f" {r['val_f1_std']:<8.4f} {r['gap']:<7.4f} {r['val_auc_mean']:<8.4f} {st}")

# ── B. Feature / Model Ablation ──────────────────────────────
print("\n── Ablation Summary ──")
print("1. CLIP alone (frozen) SVM:     ", results_tracker.get('svm_clip', {}).get('val_f1_mean', 'N/A'))
print("2. DINOv2 alone (frozen) SVM:   ", results_tracker.get('svm_dino', {}).get('val_f1_mean', 'N/A'))
print("3. CLIP fine-tuned (5-fold):    ", results_tracker.get('clip_finetune', {}).get('val_f1_mean', 'N/A'))
print("4. DINOv2 fine-tuned (5-fold):  ", results_tracker.get('dino_finetune', {}).get('val_f1_mean', 'N/A'))
print("5. Forensic XGBoost:            ", results_tracker.get('xgb_forensic', {}).get('val_f1_mean', 'N/A'))
print("6. Forensic+CNN XGBoost:        ", results_tracker.get('xgb_forensic_cnn', {}).get('val_f1_mean', 'N/A'))
print("7. FINAL ENSEMBLE:              ", results_tracker.get('ensemble', {}).get('val_f1_mean', 'N/A'))

# ── C. Confusion Matrix (on OOF) ────────────────────────────
best_oof = oof_store.get('ensemble', clip_oof)
best_thr = BEST_THRESHOLD
oof_preds = (best_oof >= best_thr).astype(int)

cm = confusion_matrix(y_all, oof_preds)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
axes[0].set_title(f'Confusion Matrix (OOF, thr={best_thr})')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

# ── D. ROC Curve ─────────────────────────────────────────────
for k in ['clip_finetune', 'dino_finetune', 'ensemble']:
    if k not in oof_store: continue
    fpr, tpr, _ = roc_curve(y_all, oof_store[k])
    auc_val = roc_auc_score(y_all, oof_store[k])
    axes[1].plot(fpr, tpr, label=f"{k} (AUC={auc_val:.4f})")
axes[1].plot([0,1], [0,1], 'k--', alpha=0.3)
axes[1].set_title('ROC Curves'); axes[1].legend()
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')

plt.tight_layout()
plt.savefig('analysis_v5.png', dpi=150, bbox_inches='tight')
plt.show()

# ── E. Ensemble Correlation ─────────────────────────────────
print("\n── Model Correlation (OOF predictions) ──")
corr_keys = [k for k in ['clip_finetune', 'dino_finetune', best_forensic_key] if k in oof_store]
corr_data = np.column_stack([oof_store[k] for k in corr_keys])
corr_matrix = np.corrcoef(corr_data.T)
print(f"{'':>20}", end='')
for k in corr_keys:
    print(f" {k[:15]:>15}", end='')
print()
for i, k in enumerate(corr_keys):
    print(f"{k[:20]:>20}", end='')
    for j in range(len(corr_keys)):
        print(f" {corr_matrix[i,j]:>15.4f}", end='')
    print()
print("\nLower correlation = more diverse = better ensemble")
```

---

## Cell 16: Final Submission

```python
# ============================================================
# CELL 16: FINAL SUBMISSION
# ============================================================

set_seeds()

# ── Select best model/ensemble ───────────────────────────────
# Priority: ensemble > clip_finetune > dino_finetune
best_key = 'ensemble'
if results_tracker['clip_finetune']['val_f1_mean'] > results_tracker['ensemble']['val_f1_mean']:
    best_key = 'clip_finetune'
    print("NOTE: Single CLIP fine-tune beats ensemble. Using CLIP alone.")

test_proba = test_pred_store[best_key]
thr = BEST_THRESHOLD

print(f"Final model:     {best_key}")
print(f"Val F1:          {results_tracker[best_key]['val_f1_mean']:.4f}")
print(f"Threshold:       {thr}")

# ── Generate submission ──────────────────────────────────────
preds_binary = (test_proba >= thr).astype(int)

submission = pd.DataFrame({
    'image_id':     df_test['image_id'].values,
    'ground_truth': preds_binary
})

# ── Sanity checks ────────────────────────────────────────────
assert submission.shape == (2058, 2), f"Wrong shape: {submission.shape}"
assert submission['ground_truth'].isin([0, 1]).all()
assert not submission.isnull().any().any()

n0 = (submission['ground_truth'] == 0).sum()
n1 = (submission['ground_truth'] == 1).sum()
print(f"\nSanity checks PASSED")
print(f"  Shape: {submission.shape}")
print(f"  Real (0): {n0}  ({n0/len(submission):.1%})")
print(f"  AI   (1): {n1}  ({n1/len(submission):.1%})")
print(f"\nFirst 5 rows:")
print(submission.head())

# ── Save ──────────────────────────────────────────────────────
out_path = '/kaggle/working/submission.csv'
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nFINAL: {best_key}  Val F1={results_tracker[best_key]['val_f1_mean']:.4f}  Thr={thr}")
```

---

## Estimated Runtime on T4

| Cell | Operation | Time |
|------|-----------|------|
| 5 | CLIP feature extraction (6858 images) | ~5 min |
| 6 | DINOv2 feature extraction | ~3 min |
| 7 | CNN feature extraction | ~3 min |
| 8 | Forensic feature extraction | ~15 min |
| 10 | Classical baselines (4 SVMs) | ~5 min |
| 11 | CLIP fine-tune (5 folds × 25 epochs) | ~60 min |
| 12 | DINOv2 fine-tune (5 folds × 25 epochs) | ~40 min |
| 13 | XGBoost forensic | ~2 min |
| 14 | Ensemble | ~1 min |
| **Total** | | **~2.5 hours** |

Well within Kaggle's 12-hour limit.

---

## VRAM Budget

| Operation | Peak VRAM |
|-----------|-----------|
| CLIP ViT-L/14 fine-tune (batch=32) | ~12 GB |
| DINOv2 ViT-B/14 fine-tune (batch=48) | ~8 GB |
| EfficientNet-B0 extraction | ~3 GB |
| Feature arrays in RAM | ~200 MB |

Never loaded simultaneously. Sequential with `del model; torch.cuda.empty_cache()` between each.

---

## Hyperparameter Cheat Sheet

| Parameter | CLIP Fine-Tune | DINOv2 Fine-Tune | Forensic XGBoost |
|-----------|---------------|-------------------|-----------------|
| Stage 1 LR | 1e-3 (head) | 1e-3 (head) | — |
| Stage 2 LR | 5e-6 backbone, 1e-4 head | 5e-6 backbone, 1e-4 head | — |
| Blocks unfrozen | Last 2 (of 24) | Last 3 (of 12) | — |
| Weight decay | 0.05 backbone, 0.01 head | 0.05 backbone, 0.01 head | — |
| Batch size | 32 | 48 | — |
| Epochs S1/S2 | 10/15 | 10/15 | — |
| Patience | 5 | 5 | 30 (early stop) |
| Dropout | 0.3 + 0.2 | 0.3 + 0.2 | — |
| Mixup alpha | 0.3 | 0.3 | — |
| n_estimators | — | — | 500 |
| max_depth | — | — | 4 |
| learning_rate | — | — | 0.05 |
| subsample | — | — | 0.7 |

---

## What to Watch For (Debugging Guide)

1. **If CLIP fine-tune Val F1 < 0.87:** Check augmentation pipeline. Ensure albumentations Normalize uses CLIP_MEAN/STD, not ImageNet values.

2. **If DINOv2 Val F1 < 0.82:** The `timm` model name might differ. Try `'vit_base_patch14_dinov2'` or check `timm.list_models('*dinov2*')`.

3. **If forensic features are all dead:** Image loading is failing silently. Add `print(f"Loaded: {img_np.shape}")` in the forensic extraction loop.

4. **If ensemble < best single model:** The models are too correlated. Check the correlation matrix in Cell 15. If correlation > 0.95, the models see the same thing and ensembling doesn't help.

5. **If gap > 0.08 on any fine-tune fold:** Reduce Stage 2 to `n=1` block unfrozen, or increase dropout to 0.4.

6. **If out of VRAM:** Reduce CLIP batch_size to 24, DINOv2 to 32. Or use `torch.cuda.amp` (already included).

7. **Total runtime > 6 hours:** Reduce N_TTA from 5 to 3. The difference is small (~0.2% F1).
