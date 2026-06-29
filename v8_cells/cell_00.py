# ============================================================
# VERSION 8: ULTIMATE AI IMAGE DETECTION
# ============================================================
# Architecture: SigLIP + DINOv2-B + CLIP ViT-L/14 + Forensic XGB
# Strategy: 3 fine-tuned vision models + 1 XGB -> 4-model ensemble
# Key insight: V5 beat V7 because DINOv2-B > DINOv2-L on 4800 samples
# V8 adds SigLIP (V7's best, 0.9002) to V5's winning DINOv2-B + CLIP
# ============================================================

!pip install -q numpy pandas opencv-python-headless Pillow tqdm scipy PyWavelets \
    matplotlib seaborn scikit-learn xgboost lightgbm joblib ipywidgets
!pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu118
!pip install -q git+https://github.com/openai/CLIP.git
!pip install -q timm albumentations transformers
print("All packages installed.")