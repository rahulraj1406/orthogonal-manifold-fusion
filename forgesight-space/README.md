---
title: ForgeSight
emoji: 🔍
colorFrom: red
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# ForgeSight — AI Image Forensics

Multi-modal ensemble AI-image detector: fine-tuned **DINOv2-Base** (518px), **SigLIP-So400m**
(224px), **CLIP ViT-L/14** (224px), plus live **ELA / FFT / Wavelet** forensic heuristics and a
DINOv2 gradient-saliency map — all shown live for every uploaded image.

Built for the DCU 2026 Adv. ML Challenge ("Orthogonal Manifold Fusion").

Demo mode: uses fold-0 of each fine-tuned backbone (weights in a private companion repo). The
paper's full system uses a 5-fold ensemble with isotonic calibration + hill-climbed weights
(F1 = 0.9325, AUC = 0.9783); this demo simplifies to DL×0.80 + Forensic×0.20 @ threshold 0.47 for
fast, free-tier CPU inference.
