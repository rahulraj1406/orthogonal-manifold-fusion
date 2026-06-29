# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Binary image classification project: detect AI-generated images (1) vs real images (0). Optimized for **F1 score**. Runs on **Kaggle GPU notebooks** (Tesla T4 / A100).

## Key Files

- **`version-3.ipynb`** — Active notebook (45 cells). All development happens here.
- **`PROJECT_CONTEXT.md`** — Detailed project state, overfitting history, ablation results, next steps.
- **`submission.csv`** — Kaggle submission output (`image_id`, `ground_truth`).

## Dataset Details

- Train: 4800 images, Test: 2058 images
- Train CSV columns: `image_id`, `ground_truth` — **NOT** `filename` or `label`
- Test CSV columns: `image_id` only
- Kaggle paths:
  - `TRAIN_CSV = /kaggle/input/datasets/rahulraj1406/dataset-easy/train.csv`
  - `TEST_CSV = /kaggle/input/datasets/rahulraj1406/dataset-easy/test.csv`
  - `IMAGE_DIR = /kaggle/input/datasets/rahulraj1406/dataset-easy/genai_image_challenge/images_final_sample`

## Architecture

**Feature pipeline:** 7 extraction layers (1440 dims) → zero-variance removal (`keep_cols`) → RobustScaler → PCA(150) → Stacking Ensemble → F1-threshold tuning → TTA inference.

Feature layers: L1 FFT (186), L2 ELA/Forensics (80), L3 EXIF Metadata (40), L4 CLIP ViT-L/14 (768), L5 EfficientNet-B0 (256), L6 Geometry (30), L7 Texture (80). Currently only L4 CLIP and L5 CNN produce non-zero features; L1/L2/L6/L7 are dead due to cached features from a broken cv2 image loading run. Image loading was fixed to PIL-first in Cell 7.

**Ensemble:** XGBoost + LightGBM + RandomForest (Level 1) → LogisticRegression meta-learner (Level 2).

## Notebook Cell Map (version-3.ipynb)

- Cells 1-3: Setup, installs, config/paths/seeds
- Cell 5: Dataset loading (reads CSV, builds filepaths)
- Cell 7: Image loading utilities (PIL-first, cv2 fallback)
- Cells 9-22: Feature extraction layers 1-7
- Cell 24: Full extraction pipeline + caching
- Cell 25: Train/val split + run extraction + `keep_cols` computation
- Cell 27: Scaling + PCA + stacking ensemble training
- Cell 29: F1-optimal threshold sweep
- Cell 31: Evaluation plots (confusion matrix, ROC, train vs val metrics)
- Cell 33: Feature importance by layer
- Cell 35: Feature ablation study (experiments A-I)
- Cell 37: Learning curves (overfitting check)
- Cell 39: Final model retrain on full train+val
- Cell 41: Test inference with TTA + submission.csv generation
- Cell 43: Save model artifacts + single-image predict function

## Critical Constraints

- **Notebook JSON editing**: When modifying `.ipynb` cells via Python, write cell source as a list of line-strings (each ending with `\n`). Never use triple-quoted strings — `\n` inside print() becomes a literal newline, breaking Python syntax. Always `compile()` each modified cell to verify.
- **Feature cache**: Extracted features are cached to `./feature_cache/*.npy`. Set `FORCE_FRESH = True` in Cell 25 to re-extract after changing image loading or feature code.
- **Overfitting is the primary risk**: 4800 samples with high-dimensional features. Always check learning curve gap (train F1 vs val F1). Target gap < 0.05.
- **No metadata leakage**: Ablation confirmed metadata-only F1 is 0.54 (near-random). Model relies on CLIP (~72%) + CNN (~28%).
