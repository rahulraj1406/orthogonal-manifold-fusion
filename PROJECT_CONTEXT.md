---
name: AI Image Detection Project — Full Status
description: Binary classifier (Real/AI), Kaggle GPU, 7-layer features, stacking ensemble, F1 metric, overfitting fixes in progress
type: project
---

# Project: AI vs Real Image Detection

**Goal:** Classify images as Real (0) or AI-Generated (1), optimized for F1 score.

## Dataset
- Train: 4800 images (2485 Real, 2315 AI)
- Test: 2058 images (no labels)
- Train CSV columns: `image_id`, `ground_truth` (NOT `filename`, NOT `label`)
- Test CSV columns: `image_id` only
- Submission format: `image_id, ground_truth`
- Kaggle paths:
  - `BASE_PATH = /kaggle/input/datasets/rahulraj1406/dataset-easy/genai_image_challenge`
  - `IMAGE_DIR = .../images_final_sample`
  - `TRAIN_CSV = /kaggle/input/datasets/rahulraj1406/dataset-easy/train.csv`
  - `TEST_CSV = /kaggle/input/datasets/rahulraj1406/dataset-easy/test.csv`

## Current Notebook: `version-3.ipynb` (45 cells)
- Execution: Kaggle GPU (Tesla T4 or A100)
- Also have older files: `ai_image_detection_multilayer (1).ipynb` (original), `ai_image_detection_multilayer_v2.ipynb` (intermediate)

## Feature Pipeline (1440 dims → 1039 active → 150 PCA)
| Layer | Dims | Status |
|-------|------|--------|
| L1 FFT Frequency | 186 | DEAD (all zeros — image loading via cv2 was failing) |
| L2 Pixel Forensics/ELA | 80 | DEAD |
| L3 EXIF Metadata | 40 | 10/40 alive, near-random F1 (0.54) |
| L4 CLIP ViT-L/14 | 768 | WORKING — provides ~72% importance |
| L5 EfficientNet-B0 CNN | 256 | WORKING — provides ~28% importance |
| L6 Depth/Geometry | 30 | DEAD |
| L7 Texture LBP/GLCM | 80 | DEAD |

**Why 5 layers are dead:** Image loading functions used cv2.imread which fails on Kaggle for these images. Fixed PIL-first loading in Cell 7, but cached features from the broken run are still being used. Set `FORCE_FRESH = True` in Cell 25 to re-extract with fixed loading.

## Processing Pipeline (current)
1. Extract 1440 features (7 layers)
2. `keep_cols` removes zero-variance → ~1039 features
3. RobustScaler
4. PCA(n_components=150) → 150 features
5. Stacking ensemble: XGBoost + LightGBM + RF → LogisticRegression meta
6. F1-optimal threshold tuning
7. TTA (5-pass noise augmentation) on test
8. Final model retrained on full train+val combined

## Models (current hyperparameters)
- XGB: n_estimators=300, max_depth=4, lr=0.05, subsample=0.7, colsample=0.6, reg_alpha=1.0, reg_lambda=5.0, min_child_weight=5, gamma=0.3
- LGB: same as XGB
- RF: n_estimators=300, max_depth=10, min_samples_leaf=10, min_samples_split=20, max_features=sqrt
- Meta: LogisticRegression(C=1.0)

## Overfitting History
| Version | Train F1 | Val F1 | Gap | Features |
|---------|----------|--------|-----|----------|
| v3 original | 0.983 | 0.807 | 0.176 | 1440 raw (416 dead) |
| v3 + keep_cols | 0.983 | 0.807 | 0.176 | 1039 (dead removed, still overfit) |
| v3 + keep_cols + PCA | TBD | TBD | TBD | 150 PCA components |

## Key Ablation Results (from latest run)
- All features: F1=0.8464
- CNN+CLIP only: F1=0.8514 (BEST — extra features hurt)
- No CLIP: F1=0.6875 (CLIP is dominant)
- Only CNN: F1=0.6533
- Only metadata: F1=0.5432 (near-random, no leakage)
- FFT+Texture: F1=0.0000 (completely dead)

## What's Been Done
- Fixed `filename` → `image_id` column name
- Fixed image loading (PIL-first for Kaggle compatibility)
- Added zero-variance feature removal (`keep_cols`)
- Added PCA dimensionality reduction (1039 → 150)
- Strengthened regularization (max_depth 6→4, added gamma, reg_alpha/lambda)
- Fixed train vs val evaluation (honest comparison using retrained models)
- Added feature importance by layer
- Added feature ablation study (experiments A-I)
- Added learning curves
- Suppressed LightGBM warnings in learning curves
- Fixed predict_image() to apply keep_cols + PCA pipeline
- Saved pca.pkl as model artifact

## Next Steps / TODO
1. **Run PCA version** on Kaggle (re-run from Cell 27) — check if learning curve gap drops
2. **If still overfitting:** try `FORCE_FRESH = True` to re-extract features with fixed PIL loading (may revive L1, L2, L6, L7)
3. **If still overfitting:** reduce PCA components (150 → 64), or try simpler model (just LogReg on CLIP embeddings)
4. **Submission:** verify submission.csv format matches competition expectations
