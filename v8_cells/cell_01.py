# ============================================================
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
from scipy.stats import rankdata
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
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def print_vram():
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated()) / 1e9
        used = torch.cuda.memory_allocated() / 1e9
        print(f"  VRAM: {used:.1f} GB used, {free:.1f} GB free")

def ensure_vram(min_gb=4.0):
    gpu_cleanup()
    if torch.cuda.is_available():
        free = (torch.cuda.get_device_properties(0).total_mem
                - torch.cuda.memory_allocated()) / 1e9
        if free < min_gb:
            print(f"WARNING: Only {free:.1f} GB free, need {min_gb} GB")
        print_vram()

# ─── Path Detection (Renku / Kaggle) ────────────────────────
def detect_paths():
    candidates = [
        # Renku
        (Path("../DCU 2026 ML challenge - external 2"),
         "genai_image_challenge/images_final_sample", "train.csv", "test.csv"),
        # Kaggle dataset-easy
        (Path("/kaggle/input/datasets/rahulraj1406/dataset-easy"),
         "genai_image_challenge/images_final_sample", "train.csv", "test.csv"),
        # Kaggle ml-dataset-easy
        (Path("/kaggle/input/datasets/rahulraj1406/ml-dataset-easy/DCU 2026 ML challenge - external 2"),
         "genai_image_challenge/images_final_sample", "train.csv", "test.csv"),
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

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
DINO_MEAN = [0.485, 0.456, 0.406]
DINO_STD  = [0.229, 0.224, 0.225]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD  = [0.5, 0.5, 0.5]

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
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"\nConfig: N_FOLDS={N_FOLDS}, N_TTA={N_TTA}")
print(f"Models: SigLIP({SIGLIP_DIM}d) + CLIP({CLIP_DIM}d) + DINOv2-B({DINO_DIM}d) + CNN({CNN_DIM}d) + Forensic({FORENSIC_DIM}d)")