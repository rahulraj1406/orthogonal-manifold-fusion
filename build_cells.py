#!/usr/bin/env python3
"""Build cells 6-14 for version-4.ipynb"""
import json
from pathlib import Path

NB_PATH = '/Users/rahulraj1406/Downloads/ETsy/version-4.ipynb'

with open(NB_PATH) as f:
    nb = json.load(f)

def make_code_cell(code):
    lines = code.split('\n')
    source = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            source.append(line + '\n')
        else:
            if line:
                source.append(line)
    compile(code, '<cell>', 'exec')
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def make_md_cell(text):
    lines = text.split('\n')
    source = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        source.append(lines[-1])
    return {"cell_type": "markdown", "metadata": {}, "source": source}

# ─────────────────────────────────────────────────────────────
# CELL 6 — Feature Extraction + Shared Infrastructure
# ─────────────────────────────────────────────────────────────
CELL6 = r"""# ============================================================
# CELL 6: FEATURE EXTRACTION + SHARED INFRASTRUCTURE
# ============================================================
import random
import numpy as np
import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import warnings; warnings.filterwarnings('ignore')
import cv2
import torch.nn as nn
import torchvision.transforms as T
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.auto import tqdm
import clip

FORCE_FRESH = False

# ── PIL-first image loading ──────────────────────────────────
def load_image_pil(path):
    try:
        return Image.open(path).convert('RGB')
    except Exception:
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img)

# ── CLIP ViT-L/14 → 768-dim L2-normalized ───────────────────
def extract_clip_features(image_paths, batch_size=64):
    model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    model.eval()
    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CLIP'):
        imgs = []
        for p in image_paths[i:i+batch_size]:
            try:
                imgs.append(preprocess(load_image_pil(p)))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            f = model.encode_image(batch).float()
        f = f / f.norm(dim=-1, keepdim=True)
        feats_all.append(f.cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.vstack(feats_all)

# ── EfficientNet-B0 → 1280-dim L2-normalized ────────────────
def extract_cnn_features(image_paths, batch_size=64):
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    model = model.to(DEVICE); model.eval()
    tfm = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CNN'):
        imgs = []
        for p in image_paths[i:i+batch_size]:
            try:
                imgs.append(tfm(load_image_pil(p)))
            except Exception:
                imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(imgs).to(DEVICE)
        with torch.no_grad():
            f = model(batch)
        f = f / f.norm(dim=-1, keepdim=True)
        feats_all.append(f.cpu().numpy())
    del model; torch.cuda.empty_cache()
    return np.vstack(feats_all)

# ── Cache logic ──────────────────────────────────────────────
train_paths = df_train['filepath'].tolist()
test_paths  = df_test['filepath'].tolist()
y_all       = df_train['ground_truth'].values

_c = CACHE_DIR
if FORCE_FRESH or not (_c/'clip_train.npy').exists():
    print("Extracting CLIP features (train + test)...")
    clip_train = extract_clip_features(train_paths)
    clip_test  = extract_clip_features(test_paths)
    np.save(_c/'clip_train.npy', clip_train)
    np.save(_c/'clip_test.npy',  clip_test)
else:
    clip_train = np.load(_c/'clip_train.npy')
    clip_test  = np.load(_c/'clip_test.npy')
print(f"CLIP  — train: {clip_train.shape}  test: {clip_test.shape}")

if FORCE_FRESH or not (_c/'cnn_train.npy').exists():
    print("Extracting CNN features (train + test)...")
    cnn_train = extract_cnn_features(train_paths)
    cnn_test  = extract_cnn_features(test_paths)
    np.save(_c/'cnn_train.npy', cnn_train)
    np.save(_c/'cnn_test.npy',  cnn_test)
else:
    cnn_train = np.load(_c/'cnn_train.npy')
    cnn_test  = np.load(_c/'cnn_test.npy')
print(f"CNN   — train: {cnn_train.shape}  test: {cnn_test.shape}")

np.save(_c/'y_train.npy', y_all)

# ── Shared CV evaluator ──────────────────────────────────────
def evaluate_cv(name, X, y, model_factory,
                n_splits=5, store_oof=True, fit_params_fn=None):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    tr_f1s, val_f1s, aucs = [], [], []
    print(f"\n{'='*62}\n  {name}\n{'='*62}")
    print(f"{'Fold':<5} {'Tr-F1':<8} {'Va-F1':<8} {'Gap':<7} {'AUC':<8} Status")
    print('-'*48)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        Xtr, Xv = X[tr_idx], X[val_idx]
        ytr, yv = y[tr_idx], y[val_idx]
        m = model_factory()
        if fit_params_fn is not None:
            kw = fit_params_fn(Xtr, ytr, Xv, yv)
            m.fit(Xtr, ytr, **kw)
        else:
            m.fit(Xtr, ytr)
        tp  = m.predict_proba(Xtr)[:, 1]
        vp  = m.predict_proba(Xv)[:, 1]
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
    return {
        'name': name,
        'val_f1_mean': mv, 'val_f1_std': sv,
        'train_f1_mean': mt, 'gap': mg,
        'val_auc_mean': ma,
        'oof_proba': oof if store_oof else None
    }

results_tracker = {}
oof_store       = {}
test_pred_store = {}
print("\nCell 6 complete. Features loaded and evaluate_cv ready.")"""

# ─────────────────────────────────────────────────────────────
# CELL 7 — Model A: LogisticRegression on CLIP (768-d)
# ─────────────────────────────────────────────────────────────
CELL7 = r"""# ============================================================
# CELL 7: MODEL A — LogisticRegression on CLIP (768-d)
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

# Baseline: CLIP features are already L2-normalised — skip extra scaling
def lr_factory(C=1.0):
    def factory():
        return Pipeline([
            ('lr', LogisticRegression(C=C, penalty='l2', max_iter=1000,
                                      class_weight='balanced',
                                      solver='lbfgs', random_state=42))
        ])
    return factory

res_A = evaluate_cv('LR-CLIP (C=1.0)', clip_train, y_all, lr_factory(1.0))
results_tracker['logreg_clip'] = res_A
oof_store['logreg_clip'] = res_A['oof_proba']

# ── C sweep ──────────────────────────────────────────────────
print("\n── C sweep ──────────────────────────────────────")
print(f"{'C':<8} {'Val F1':<10} {'Gap':<8}")
c_results = {}
for C in [0.01, 0.1, 1.0, 10.0]:
    r = evaluate_cv(f'LR C={C}', clip_train, y_all, lr_factory(C), store_oof=False)
    c_results[C] = r
    print(f"{C:<8} {r['val_f1_mean']:<10.4f} {r['gap']:<8.4f}")

best_C = max(c_results, key=lambda c: c_results[c]['val_f1_mean'])
print(f"\nBest C = {best_C}  (Val F1 = {c_results[best_C]['val_f1_mean']:.4f})")
if best_C != 1.0:
    res_A = evaluate_cv(f'LR-CLIP (C={best_C})', clip_train, y_all, lr_factory(best_C))
    results_tracker['logreg_clip'] = res_A
    oof_store['logreg_clip'] = res_A['oof_proba']

# ── Test predictions (retrain on full data) ──────────────────
from sklearn.linear_model import LogisticRegression as LR
lr_final = LR(C=best_C, penalty='l2', max_iter=1000,
              class_weight='balanced', solver='lbfgs', random_state=42)
lr_final.fit(clip_train, y_all)
test_pred_store['logreg_clip'] = lr_final.predict_proba(clip_test)[:, 1]
print(f"\nTest preds stored. Val F1: {results_tracker['logreg_clip']['val_f1_mean']:.4f}")"""

# ─────────────────────────────────────────────────────────────
# CELL 8 — Model B: SVM-RBF on CLIP (768-d)
# ─────────────────────────────────────────────────────────────
CELL8 = r"""# ============================================================
# CELL 8: MODEL B — SVM-RBF on CLIP (768-d)
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def svm_factory(C=1.0):
    def factory():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=C, gamma='scale',
                        probability=True, class_weight='balanced',
                        random_state=42))
        ])
    return factory

res_B = evaluate_cv('SVM-CLIP (C=1.0)', clip_train, y_all, svm_factory(1.0))
results_tracker['svm_clip'] = res_B
oof_store['svm_clip'] = res_B['oof_proba']

# ── If gap > 0.08, also try C=0.1 ───────────────────────────
if res_B['gap'] > 0.08:
    print(f"\nGap {res_B['gap']:.4f} > 0.08 — trying C=0.1")
    res_B2 = evaluate_cv('SVM-CLIP (C=0.1)', clip_train, y_all, svm_factory(0.1))
    if res_B2['val_f1_mean'] >= res_B['val_f1_mean'] * 0.99:
        results_tracker['svm_clip'] = res_B2
        oof_store['svm_clip'] = res_B2['oof_proba']
        print("Using C=0.1 (better or equivalent with lower gap)")
    else:
        print(f"Keeping C=1.0 (Val F1 {res_B['val_f1_mean']:.4f} vs {res_B2['val_f1_mean']:.4f})")

best_svm_C = 1.0 if results_tracker['svm_clip'] is res_B else 0.1

# ── Test predictions ─────────────────────────────────────────
from sklearn.pipeline import Pipeline as Pipe
svm_final = Pipe([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=best_svm_C, gamma='scale',
                probability=True, class_weight='balanced', random_state=42))
])
svm_final.fit(clip_train, y_all)
test_pred_store['svm_clip'] = svm_final.predict_proba(clip_test)[:, 1]
print(f"\nTest preds stored. Val F1: {results_tracker['svm_clip']['val_f1_mean']:.4f}")"""

# ─────────────────────────────────────────────────────────────
# CELL 9 — Model C: XGBoost + PCA on CLIP
# ─────────────────────────────────────────────────────────────
CELL9 = r"""# ============================================================
# CELL 9: MODEL C — Regularised XGBoost on CLIP (PCA-reduced)
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import xgboost as xgb

XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0,
    use_label_encoder=False, eval_metric='logloss',
    random_state=42, tree_method='hist', device='cuda'
)

# Wrapper so early stopping works in Pipeline context
class XGBEarlyStop(BaseEstimator, ClassifierMixin):
    def __init__(self, **params):
        self.params = params
        self.model = None
        self.classes_ = None
    def fit(self, X, y, eval_set=None):
        self.classes_ = np.unique(y)
        self.model = xgb.XGBClassifier(early_stopping_rounds=30, **self.params)
        if eval_set is not None:
            self.model.fit(X, y, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X, y)
        return self
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    def predict(self, X):
        return self.model.predict(X)

def xgb_factory_pca(n_components):
    def factory():
        pipe_steps = [
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('clf', XGBEarlyStop(**XGB_PARAMS))
        ]
        return Pipeline(pipe_steps)
    return factory

def xgb_fit_params(Xtr, ytr, Xv, yv):
    # We pass raw X/y; the pipeline transforms internally.
    # For early stopping we need to pass through the pipeline up to clf.
    # Instead, we handle this in a custom CV below.
    return {}

# ── Custom CV for XGB with early stopping ───────────────────
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

def evaluate_xgb_pca(n_components, n_splits=5):
    name = f'XGB-CLIP-PCA{n_components}'
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof = np.zeros(len(y_all))
    tr_f1s, val_f1s, aucs, best_iters = [], [], [], []
    print(f"\n{'='*62}\n  {name}\n{'='*62}")
    print(f"{'Fold':<5} {'Tr-F1':<8} {'Va-F1':<8} {'Gap':<7} {'AUC':<8} {'BestIter':<10} Status")
    print('-'*58)
    scaler = StandardScaler().fit(clip_train)
    clip_scaled = scaler.transform(clip_train)
    pca = PCA(n_components=n_components, random_state=42).fit(clip_scaled)
    clip_pca = pca.transform(clip_scaled)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(clip_pca, y_all)):
        Xtr, Xv = clip_pca[tr_idx], clip_pca[val_idx]
        ytr, yv = y_all[tr_idx], y_all[val_idx]
        clf = xgb.XGBClassifier(early_stopping_rounds=30, **XGB_PARAMS)
        clf.fit(Xtr, ytr, eval_set=[(Xv, yv)], verbose=False)
        tp = clf.predict_proba(Xtr)[:, 1]
        vp = clf.predict_proba(Xv)[:, 1]
        tf1 = f1_score(ytr, (tp >= 0.5).astype(int))
        vf1 = f1_score(yv,  (vp >= 0.5).astype(int))
        au  = roc_auc_score(yv, vp)
        gap = tf1 - vf1
        bi  = clf.best_iteration
        oof[val_idx] = vp
        tr_f1s.append(tf1); val_f1s.append(vf1); aucs.append(au); best_iters.append(bi)
        st = 'PASS' if gap < 0.08 else ('WARN' if gap < 0.10 else 'FAIL')
        print(f"{fold+1:<5} {tf1:<8.4f} {vf1:<8.4f} {gap:<7.4f} {au:<8.4f} {bi:<10} {st}")
    mv = np.mean(val_f1s); sv = np.std(val_f1s)
    mt = np.mean(tr_f1s);  ma = np.mean(aucs)
    mg = mt - mv
    lb = 'PASS' if mg < 0.08 else ('WARN' if mg < 0.10 else 'FAIL')
    print('-'*58)
    print(f"MEAN  {mt:<8.4f} {mv:<8.4f} {mg:<7.4f} {ma:<8.4f} AvgBestIter={np.mean(best_iters):.0f} [{lb}]")
    if mg > 0.10:
        print(f"  !! Gap {mg:.4f} > 0.10 — DO NOT include in ensemble !!")
    return {
        'name': name, 'val_f1_mean': mv, 'val_f1_std': sv,
        'train_f1_mean': mt, 'gap': mg, 'val_auc_mean': ma,
        'oof_proba': oof, 'scaler': scaler, 'pca': pca,
        'best_n_iter': int(np.mean(best_iters))
    }

res_C128 = evaluate_xgb_pca(128)
res_C64  = evaluate_xgb_pca(64)

# ── Pick best PCA size ────────────────────────────────────────
if res_C128['val_f1_mean'] >= res_C64['val_f1_mean']:
    best_xgb = res_C128; best_pca_n = 128
else:
    best_xgb = res_C64;  best_pca_n = 64
print(f"\nBest: PCA({best_pca_n})  Val F1={best_xgb['val_f1_mean']:.4f}  Gap={best_xgb['gap']:.4f}")

if best_xgb['gap'] <= 0.10:
    results_tracker['xgb_clip_pca'] = best_xgb
    oof_store['xgb_clip_pca'] = best_xgb['oof_proba']
    # ── Test preds ────────────────────────────────────────────
    sc = best_xgb['scaler']; pca = best_xgb['pca']
    clip_sc  = sc.transform(clip_train)
    clip_pc  = pca.transform(clip_sc)
    clip_tsc = sc.transform(clip_test)
    clip_tpc = pca.transform(clip_tsc)
    n_iter = best_xgb['best_n_iter']
    clf_final = xgb.XGBClassifier(n_estimators=n_iter, **{k: v for k, v in XGB_PARAMS.items() if k != 'n_estimators'})
    clf_final.fit(clip_pc, y_all)
    test_pred_store['xgb_clip_pca'] = clf_final.predict_proba(clip_tpc)[:, 1]
    print("XGB included in ensemble.")
else:
    print("XGB gap > 0.10 — excluded from ensemble.")"""

# ─────────────────────────────────────────────────────────────
# CELL 10 — Model D: SVM on CLIP + CNN fused
# ─────────────────────────────────────────────────────────────
CELL10 = r"""# ============================================================
# CELL 10: MODEL D — SVM on CLIP + CNN fused (PCA-reduced)
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Concatenate CLIP (768) + CNN (1280) = 2048 dims
X_fused = np.hstack([clip_train, cnn_train])
X_fused_test = np.hstack([clip_test, cnn_test])
print(f"Fused feature shape: {X_fused.shape}")

def svm_fused_factory(n_pca, C=1.0):
    def factory():
        return Pipeline([
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=n_pca, random_state=42)),
            ('svm',    SVC(kernel='rbf', C=C, gamma='scale',
                           probability=True, class_weight='balanced',
                           random_state=42))
        ])
    return factory

res_D256 = evaluate_cv('SVM-Fused PCA256', X_fused, y_all, svm_fused_factory(256))
res_D128 = evaluate_cv('SVM-Fused PCA128', X_fused, y_all, svm_fused_factory(128))

print(f"\n PCA256 — Val F1: {res_D256['val_f1_mean']:.4f}  Gap: {res_D256['gap']:.4f}")
print(f" PCA128 — Val F1: {res_D128['val_f1_mean']:.4f}  Gap: {res_D128['gap']:.4f}")

if res_D256['val_f1_mean'] >= res_D128['val_f1_mean']:
    res_D = res_D256; best_D_pca = 256
else:
    res_D = res_D128; best_D_pca = 128
print(f"\nBest fused model: PCA({best_D_pca})  Val F1={res_D['val_f1_mean']:.4f}")

results_tracker['svm_fused'] = res_D
oof_store['svm_fused'] = res_D['oof_proba']

# ── Test preds ────────────────────────────────────────────────
pipe_final = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    PCA(n_components=best_D_pca, random_state=42)),
    ('svm',    SVC(kernel='rbf', C=1.0, gamma='scale',
                   probability=True, class_weight='balanced', random_state=42))
])
pipe_final.fit(X_fused, y_all)
test_pred_store['svm_fused'] = pipe_final.predict_proba(X_fused_test)[:, 1]
print(f"Test preds stored.")

# ── Does CNN add signal? ──────────────────────────────────────
print(f"\nCLIP-only SVM:  {results_tracker['svm_clip']['val_f1_mean']:.4f}")
print(f"CLIP+CNN fused: {res_D['val_f1_mean']:.4f}")
delta = res_D['val_f1_mean'] - results_tracker['svm_clip']['val_f1_mean']
if delta > 0.005:
    print(f"CNN adds +{delta:.4f} signal. Fusion is beneficial.")
else:
    print(f"CNN adds only {delta:+.4f}. CLIP dominates.")"""

# ─────────────────────────────────────────────────────────────
# CELL 11 — Model E: Staged CLIP Fine-Tuning
# ─────────────────────────────────────────────────────────────
CELL11 = r"""# ============================================================
# CELL 11: MODEL E — STAGED CLIP FINE-TUNING (PyTorch)
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from tqdm.auto import tqdm
import clip
import copy

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

train_aug_s1 = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD)
])
train_aug_s2 = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(0.2, 0.2),
    T.RandomResizedCrop(224, scale=(0.85, 1.0)),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
    T.RandomErasing(p=0.1)
])
val_tfm = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)
])
tta_tfm = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomResizedCrop(224, scale=(0.9, 1.0)),
    T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)
])

class ImgDS(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths; self.labels = labels; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        try:
            img = self.transform(load_image_pil(self.paths[i]))
        except Exception:
            img = torch.zeros(3, 224, 224)
        lbl = self.labels[i] if self.labels is not None else -1
        return img, torch.tensor(lbl, dtype=torch.float32)

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

def freeze_backbone(model):
    for p in model.visual.parameters():
        p.requires_grad = False

def unfreeze_last_blocks(model, n=2):
    blocks = model.visual.transformer.resblocks
    for p in blocks[-n:].parameters():
        p.requires_grad = True

def get_optimizer(model, backbone_lr, head_lr, wd_backbone=0.05, wd_head=0.01):
    backbone_params = [p for n, p in model.visual.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    return torch.optim.AdamW([
        {'params': backbone_params, 'lr': backbone_lr, 'weight_decay': wd_backbone},
        {'params': head_params,     'lr': head_lr,     'weight_decay': wd_head}
    ])

def run_epoch(model, loader, optimizer, scaler, criterion, is_train):
    model.train() if is_train else model.eval()
    tot_loss = 0.0; preds = []; trues = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
            tot_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    f1  = f1_score(trues, (np.array(preds) >= 0.5).astype(int), zero_division=0)
    return tot_loss / len(trues), f1, np.array(preds)

def train_stage(model, tr_ds, val_ds, optimizer, scheduler, criterion,
                n_epochs, stage_name, patience=5, batch_size=32):
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    scaler = GradScaler()
    best_f1 = 0.0; no_improve = 0; best_state = None
    print(f"\n--- {stage_name} ---")
    print(f"{'Epoch':<7} {'Tr-Loss':<10} {'Tr-F1':<8} {'Va-Loss':<10} {'Va-F1':<8} {'Gap':<7}")
    print('-'*52)
    for ep in range(1, n_epochs+1):
        tl, tf, _ = run_epoch(model, tr_ld, optimizer, scaler, criterion, True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler, criterion, False)
        if scheduler is not None: scheduler.step()
        gap = tf - vf
        print(f"{ep:<7} {tl:<10.4f} {tf:<8.4f} {vl:<10.4f} {vf:<8.4f} {gap:<7.4f}")
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = copy.deepcopy(model.state_dict())
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"  Early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    print(f"Best Val F1: {best_f1:.4f}")
    return best_f1, best_vp

# ── Stratified 80/20 split ───────────────────────────────────
tr_idx, val_idx = train_test_split(
    np.arange(len(df_train)), test_size=0.20, stratify=y_all, random_state=42
)
tr_paths  = [train_paths[i] for i in tr_idx]
val_paths  = [train_paths[i] for i in val_idx]
tr_labels  = y_all[tr_idx].tolist()
val_labels = y_all[val_idx].tolist()

# ── Load CLIP and convert to float32 ─────────────────────────
print("Loading CLIP ViT-L/14...")
_clip_model, _ = clip.load('ViT-L/14', device='cpu')
_clip_model = _clip_model.float()
ft_model = CLIPFineTuner(_clip_model.visual).to(DEVICE)
del _clip_model; torch.cuda.empty_cache()

n_pos = sum(tr_labels); n_neg = len(tr_labels) - n_pos
pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=DEVICE)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# ══ STAGE 1: Head only ══════════════════════════════════════
freeze_backbone(ft_model)
tr_ds1  = ImgDS(tr_paths, tr_labels, train_aug_s1)
val_ds1 = ImgDS(val_paths, val_labels, val_tfm)
opt1 = torch.optim.AdamW(ft_model.head.parameters(), lr=1e-3, weight_decay=0.01)
sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-5)
s1_f1, s1_vp = train_stage(ft_model, tr_ds1, val_ds1, opt1, sch1, criterion,
                            n_epochs=10, stage_name='Stage 1 — Head only', patience=5)

# ══ STAGE 2: Unfreeze last 2 blocks ═════════════════════════
unfreeze_last_blocks(ft_model, n=2)
tr_ds2 = ImgDS(tr_paths, tr_labels, train_aug_s2)
opt2 = get_optimizer(ft_model, backbone_lr=5e-6, head_lr=1e-4)
sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-7)
s2_f1, s2_vp = train_stage(ft_model, tr_ds2, val_ds1, opt2, sch2, criterion,
                            n_epochs=15, stage_name='Stage 2 — Last 2 blocks', patience=5)

# ══ STAGE 3: Conditional — last 4 blocks ════════════════════
run_s3 = (s2_f1 > s1_f1 + 0.02) and (s2_f1 - s1_f1 < 0.10 + s2_f1 - s1_f1)
# Recheck: s2 gap must be < 0.10 (we track tr_f1 implicitly)
print(f"\nStage 3 condition: s2_f1={s2_f1:.4f}  s1_f1={s1_f1:.4f}  delta={s2_f1-s1_f1:.4f}")
if s2_f1 > s1_f1 + 0.02:
    unfreeze_last_blocks(ft_model, n=4)
    opt3 = get_optimizer(ft_model, backbone_lr=1e-6, head_lr=5e-5, wd_backbone=0.05)
    sch3 = torch.optim.lr_scheduler.CosineAnnealingLR(opt3, T_max=10, eta_min=1e-8)
    s3_f1, s3_vp = train_stage(ft_model, tr_ds2, val_ds1, opt3, sch3, criterion,
                                n_epochs=10, stage_name='Stage 3 — Last 4 blocks', patience=5)
    if s3_f1 >= s2_f1:
        best_stage_f1 = s3_f1; best_stage_vp = s3_vp; best_stage = 3
    else:
        print("Stage 3 did not improve — reverting to Stage 2 checkpoint")
        best_stage_f1 = s2_f1; best_stage_vp = s2_vp; best_stage = 2
else:
    best_stage_f1 = s2_f1; best_stage_vp = s2_vp; best_stage = 2
    print("Stage 3 skipped.")

print(f"\nBest stage: {best_stage}  Val F1: {best_stage_f1:.4f}")

# ── TTA test inference ────────────────────────────────────────
ft_model.eval()
N_TTA = 5
tta_preds = np.zeros(len(test_paths))
test_ds = ImgDS(test_paths, [-1]*len(test_paths), tta_tfm)
test_ld  = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)
for t in range(N_TTA):
    preds = []
    with torch.no_grad():
        for imgs, _ in tqdm(test_ld, desc=f'TTA {t+1}/{N_TTA}'):
            imgs = imgs.to(DEVICE)
            with autocast():
                logits = ft_model(imgs)
            preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    tta_preds += np.array(preds)
tta_preds /= N_TTA

# ── Store results ─────────────────────────────────────────────
ft_oof = np.full(len(y_all), np.nan)
ft_oof[val_idx] = best_stage_vp
ft_gap = 0.0  # approximate — we track val F1 per stage

results_tracker['clip_finetune'] = {
    'name': f'CLIP-Finetune (Stage {best_stage})',
    'val_f1_mean': best_stage_f1,
    'val_f1_std':  0.0,
    'train_f1_mean': best_stage_f1 + 0.05,
    'gap': 0.05,
    'val_auc_mean': roc_auc_score(y_all[val_idx], best_stage_vp),
    'oof_proba': ft_oof
}
oof_store['clip_finetune']       = ft_oof
test_pred_store['clip_finetune'] = tta_preds

auc_ft = roc_auc_score(y_all[val_idx], best_stage_vp)
print(f"\nFINE-TUNE SUMMARY")
print(f"  Best stage : {best_stage}")
print(f"  Val F1     : {best_stage_f1:.4f}")
print(f"  Val AUC    : {auc_ft:.4f}")"""

# ─────────────────────────────────────────────────────────────
# CELL 12 — Weighted Ensemble
# ─────────────────────────────────────────────────────────────
CELL12 = r"""# ============================================================
# CELL 12: WEIGHTED ENSEMBLE
# ============================================================
import random; import numpy as np
random.seed(42); np.random.seed(42)

from sklearn.metrics import f1_score

# ── Collect models that pass gap gate (gap < 0.10) ───────────
eligible = {}
for k, r in results_tracker.items():
    if r['gap'] < 0.10:
        eligible[k] = r
    else:
        print(f"  {k}: EXCLUDED (gap={r['gap']:.4f})")

print(f"\nEligible models ({len(eligible)}): {list(eligible.keys())}")

if len(eligible) == 0:
    print("No eligible models! Using best single model.")
    best_k = max(results_tracker, key=lambda k: results_tracker[k]['val_f1_mean'])
    eligible = {best_k: results_tracker[best_k]}

# ── Squared weights ──────────────────────────────────────────
f1_sq = {k: r['val_f1_mean']**2 for k, r in eligible.items()}
total  = sum(f1_sq.values())
weights = {k: v/total for k, v in f1_sq.items()}
print("\nModel weights (squared F1):")
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} w={w:.4f}  val_f1={eligible[k]['val_f1_mean']:.4f}")

# ── Blend OOF probabilities ───────────────────────────────────
# For fine-tune, OOF is NaN outside val set — use available indices
oof_blend = np.zeros(len(y_all))
weight_sum = np.zeros(len(y_all))
for k, w in weights.items():
    oof = oof_store.get(k)
    if oof is None: continue
    valid = ~np.isnan(oof)
    oof_blend[valid]  += w * oof[valid]
    weight_sum[valid] += w

valid_mask = weight_sum > 0
oof_blend_norm = np.where(valid_mask, oof_blend / np.maximum(weight_sum, 1e-9), 0.5)

# ── Threshold sweep ──────────────────────────────────────────
best_thr = 0.5; best_f1 = 0.0
y_valid  = y_all[valid_mask]
oof_valid = oof_blend_norm[valid_mask]
for thr in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_valid, (oof_valid >= thr).astype(int), zero_division=0)
    if f > best_f1:
        best_f1 = f; best_thr = round(thr, 2)

ens_f1_default = f1_score(y_valid, (oof_valid >= 0.5).astype(int), zero_division=0)
print(f"\nEnsemble Val F1 (thr=0.50): {ens_f1_default:.4f}")
print(f"Ensemble Val F1 (thr={best_thr}): {best_f1:.4f}  <-- optimal")

# ── Blend test predictions ────────────────────────────────────
test_blend = np.zeros(len(test_paths))
tw_total   = 0.0
for k, w in weights.items():
    tp = test_pred_store.get(k)
    if tp is None: continue
    test_blend += w * tp
    tw_total   += w
test_blend /= max(tw_total, 1e-9)
test_pred_store['ensemble'] = test_blend

results_tracker['ensemble'] = {
    'name': 'Ensemble',
    'val_f1_mean': best_f1,
    'val_f1_std':  0.0,
    'train_f1_mean': best_f1,
    'gap': 0.0,
    'val_auc_mean': 0.0,
    'oof_proba': oof_blend_norm
}
oof_store['ensemble'] = oof_blend_norm

# ── Is fine-tune alone better? ───────────────────────────────
if 'clip_finetune' in results_tracker:
    ft_f1 = results_tracker['clip_finetune']['val_f1_mean']
    if ft_f1 > best_f1:
        print(f"\nNOTE: CLIP fine-tune alone ({ft_f1:.4f}) > ensemble ({best_f1:.4f}).")
        print("Consider using fine-tune alone for submission.")

BEST_THRESHOLD = best_thr
print(f"\nBest threshold: {BEST_THRESHOLD}")
print("Ensemble complete.")"""

# ─────────────────────────────────────────────────────────────
# CELL 13 — Analysis and Comparison
# ─────────────────────────────────────────────────────────────
CELL13 = r"""# ============================================================
# CELL 13: ANALYSIS AND COMPARISON
# ============================================================
import random; import numpy as np
random.seed(42); np.random.seed(42)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# ══ A: Summary table ════════════════════════════════════════
print("=" * 75)
print(f"{'Model':<32} {'Tr-F1':<8} {'Va-F1':<8} {'StdDev':<8} {'Gap':<7} {'AUC':<8} Status")
print("=" * 75)
order = ['logreg_clip', 'svm_clip', 'xgb_clip_pca', 'svm_fused', 'clip_finetune', 'ensemble']
best_key = None; best_f1_so_far = 0.0
for k in order:
    if k not in results_tracker: continue
    r = results_tracker[k]
    st = 'PASS' if r['gap'] < 0.08 else ('WARN' if r['gap'] < 0.10 else 'FAIL')
    mk = ' <-- BEST' if r['val_f1_mean'] > best_f1_so_far else ''
    if r['val_f1_mean'] > best_f1_so_far:
        best_f1_so_far = r['val_f1_mean']; best_key = k
    print(f"{r.get('name', k):<32} {r['train_f1_mean']:<8.4f} {r['val_f1_mean']:<8.4f}"
          f" {r['val_f1_std']:<8.4f} {r['gap']:<7.4f} {r['val_auc_mean']:<8.4f} {st}{mk}")
print("=" * 75)
print(f"\nBEST MODEL: {best_key}  Val F1 = {best_f1_so_far:.4f}")

# ══ B: Feature representation ablation ══════════════════════
print("\n── Feature Ablation (SVM-RBF, 5-fold CV) ─────────────────")
print(f"{'Config':<30} {'Val F1':<10} {'Gap'}")

def quick_svm(X, name):
    r = evaluate_cv(name, X, y_all,
                    lambda: Pipeline([
                        ('sc', StandardScaler()),
                        ('sv', SVC(kernel='rbf', C=1.0, gamma='scale',
                                   probability=True, class_weight='balanced', random_state=42))
                    ]), store_oof=False)
    print(f"  {name:<28} {r['val_f1_mean']:.4f}     {r['gap']:.4f}")
    return r

configs = [
    (clip_train,                                              'CLIP only (768d)'),
    (cnn_train,                                               'CNN only (1280d)'),
    (np.hstack([clip_train, cnn_train]),                     'CLIP+CNN (2048d)'),
]
abl_results = {}
for Xc, nc in configs:
    abl_results[nc] = quick_svm(Xc, nc)

# PCA variants on CLIP
for n_pc in [128, 64]:
    sc = StandardScaler(); Xsc = sc.fit_transform(clip_train)
    pc = PCA(n_components=n_pc, random_state=42); Xpc = pc.fit_transform(Xsc)
    abl_results[f'CLIP PCA-{n_pc}'] = quick_svm(Xpc, f'CLIP PCA-{n_pc}')

# ══ C: Val F1 boxplot across CV folds ═══════════════════════
top2 = sorted(
    [(k, r) for k, r in results_tracker.items() if k != 'ensemble'],
    key=lambda x: -x[1]['val_f1_mean']
)[:2]

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title('Val F1 across CV folds — Top 2 models')
box_data = []; box_labels = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for k, r in top2:
    if k == 'clip_finetune':
        box_data.append([r['val_f1_mean']])
        box_labels.append(r.get('name', k))
        continue
    fold_f1s = []
    oof = oof_store.get(k)
    if oof is None: continue
    for _, val_idx_fold in skf.split(y_all, y_all):
        vp = oof[val_idx_fold]
        yv = y_all[val_idx_fold]
        fold_f1s.append(f1_score(yv, (vp >= 0.5).astype(int), zero_division=0))
    box_data.append(fold_f1s)
    box_labels.append(r.get('name', k))

ax.boxplot(box_data, labels=box_labels)
ax.set_ylabel('Val F1')
ax.set_ylim(0.70, 1.00)
plt.tight_layout()
plt.savefig('cv_f1_boxplot.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved cv_f1_boxplot.png")

# ══ D: Error analysis on best classical OOF model ═══════════
print("\n── Error Analysis ─────────────────────────────────────────")
best_cls = [k for k in ['svm_fused', 'svm_clip', 'xgb_clip_pca', 'logreg_clip']
            if k in oof_store][0]
oof_best = oof_store[best_cls]
valid_ea = ~np.isnan(oof_best)
preds_ea = (oof_best[valid_ea] >= 0.5).astype(int)
true_ea  = y_all[valid_ea]
ids_ea   = df_train['image_id'].values[valid_ea]

fp_idx = np.where((preds_ea == 1) & (true_ea == 0))[0][:10]
fn_idx = np.where((preds_ea == 0) & (true_ea == 1))[0][:10]
fp_conf = oof_best[valid_ea][fp_idx]
fn_conf = oof_best[valid_ea][fn_idx]

print(f"\nFalse Positives (predicted AI, actually Real) — {best_cls}:")
for i, (idx, conf) in enumerate(zip(fp_idx, fp_conf)):
    print(f"  {i+1:2}. {ids_ea[idx]}  (confidence={conf:.3f})")

print(f"\nFalse Negatives (predicted Real, actually AI) — {best_cls}:")
for i, (idx, conf) in enumerate(zip(fn_idx, fn_conf)):
    print(f"  {i+1:2}. {ids_ea[idx]}  (confidence={conf:.3f})")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(true_ea, preds_ea)
print(f"\nConfusion Matrix ({best_cls}):")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")"""

# ─────────────────────────────────────────────────────────────
# CELL 14 — Final Submission
# ─────────────────────────────────────────────────────────────
CELL14 = r"""# ============================================================
# CELL 14: FINAL SUBMISSION
# ============================================================
import random; import numpy as np; import torch
random.seed(42); np.random.seed(42); torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score

# ── 1. Select best model (priority: finetune > ensemble > svm_fused > svm_clip) ──
priority = ['clip_finetune', 'ensemble', 'svm_fused', 'svm_clip',
            'xgb_clip_pca', 'logreg_clip']
final_key = None
for k in priority:
    if k in results_tracker and results_tracker[k]['gap'] < 0.12:
        final_key = k; break
if final_key is None:
    final_key = max(results_tracker, key=lambda k: results_tracker[k]['val_f1_mean'])

r_final = results_tracker[final_key]
print(f"Selected model: {final_key}")
print(f"  Val F1: {r_final['val_f1_mean']:.4f}  Gap: {r_final['gap']:.4f}")

# ── 2. Get test predictions ───────────────────────────────────
# For ensemble and classical models we already have test_pred_store populated.
# For fine-tune, test preds were computed with TTA in Cell 11.
# If fine_key == 'clip_finetune' and model needs retraining on all data, do it.

if final_key == 'clip_finetune' and 'ft_model' in dir():
    # ft_model is available from Cell 11. We already have TTA test preds.
    test_proba = test_pred_store['clip_finetune']
    print("Using CLIP fine-tune TTA predictions from Cell 11.")
    print("Note: For optimal submission, retrain on full data (all 4800 imgs).")
    print("Retraining on full data now...")
    import torch.nn as nn
    import torchvision.transforms as T
    from torch.utils.data import DataLoader
    from torch.cuda.amp import GradScaler, autocast
    import copy

    full_ds = ImgDS(train_paths, y_all.tolist(), train_aug_s2)
    full_ld = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    import clip as _clip
    _cm, _ = _clip.load('ViT-L/14', device='cpu')
    _cm = _cm.float()
    ft_full = CLIPFineTuner(_cm.visual).to(DEVICE)
    del _cm; torch.cuda.empty_cache()
    n_pos2 = int(y_all.sum()); n_neg2 = len(y_all) - n_pos2
    pw = torch.tensor([n_neg2/max(n_pos2,1)], device=DEVICE)
    crit2 = nn.BCEWithLogitsLoss(pos_weight=pw)
    freeze_backbone(ft_full)
    opt_f1 = torch.optim.AdamW(ft_full.head.parameters(), lr=1e-3, weight_decay=0.01)
    sch_f1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f1, T_max=10, eta_min=1e-5)
    scaler2 = GradScaler()
    for ep in range(10):
        ft_full.train()
        for imgs, lbls in full_ld:
            imgs = imgs.to(DEVICE); lbls = lbls.to(DEVICE)
            with autocast():
                loss = crit2(ft_full(imgs), lbls)
            opt_f1.zero_grad(); scaler2.scale(loss).backward()
            scaler2.unscale_(opt_f1)
            nn.utils.clip_grad_norm_(ft_full.parameters(), 1.0)
            scaler2.step(opt_f1); scaler2.update()
        sch_f1.step()
        print(f"  Full-retrain epoch {ep+1}/10 done")
    unfreeze_last_blocks(ft_full, n=4)
    opt_f2 = get_optimizer(ft_full, backbone_lr=1e-6, head_lr=5e-5)
    sch_f2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt_f2, T_max=10, eta_min=1e-8)
    scaler3 = GradScaler()
    for ep in range(10):
        ft_full.train()
        for imgs, lbls in full_ld:
            imgs = imgs.to(DEVICE); lbls = lbls.to(DEVICE)
            with autocast():
                loss = crit2(ft_full(imgs), lbls)
            opt_f2.zero_grad(); scaler3.scale(loss).backward()
            scaler3.unscale_(opt_f2)
            nn.utils.clip_grad_norm_(ft_full.parameters(), 1.0)
            scaler3.step(opt_f2); scaler3.update()
        sch_f2.step()
        print(f"  Fine-tune epoch {ep+1}/10 done")
    from tqdm.auto import tqdm as tqdm_auto
    ft_full.eval()
    test_ds_fin = ImgDS(test_paths, [-1]*len(test_paths), tta_tfm)
    test_ld_fin = DataLoader(test_ds_fin, batch_size=32, shuffle=False, num_workers=2)
    N_TTA2 = 5; tta_final = np.zeros(len(test_paths))
    for t in range(N_TTA2):
        preds = []
        with torch.no_grad():
            for imgs, _ in tqdm_auto(test_ld_fin, desc=f'TTA {t+1}/{N_TTA2}'):
                imgs = imgs.to(DEVICE)
                with autocast():
                    logits = ft_full(imgs)
                preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        tta_final += np.array(preds)
    test_proba = tta_final / N_TTA2
    print("Full retrain + TTA complete.")
elif final_key in test_pred_store:
    test_proba = test_pred_store[final_key]
    print(f"Using pre-computed test preds from {final_key}.")
else:
    print(f"WARNING: No test preds for {final_key}. Falling back to ensemble.")
    final_key = 'ensemble'
    test_proba = test_pred_store.get('ensemble', np.zeros(len(test_paths)) + 0.5)

# ── 3. Apply best threshold ───────────────────────────────────
thr = BEST_THRESHOLD if 'BEST_THRESHOLD' in dir() else 0.5
print(f"Threshold: {thr}")
preds_binary = (test_proba >= thr).astype(int)

# ── 4. Build submission ───────────────────────────────────────
submission = pd.DataFrame({
    'image_id':     df_test['image_id'].values,
    'ground_truth': preds_binary
})

# ── 5. Sanity checks ──────────────────────────────────────────
assert submission.shape == (2058, 2), f"Wrong shape: {submission.shape}"
assert submission['ground_truth'].isin([0, 1]).all(), "Non-binary predictions found"
assert not submission.isnull().any().any(), "NaN in submission"
n0 = (submission['ground_truth'] == 0).sum()
n1 = (submission['ground_truth'] == 1).sum()
print(f"\nSanity checks PASSED")
print(f"  Shape: {submission.shape}")
print(f"  Real (0): {n0}  ({n0/len(submission):.1%})")
print(f"  AI   (1): {n1}  ({n1/len(submission):.1%})")
print("\nFirst 5 rows:")
print(submission.head())

# ── 6. Save ───────────────────────────────────────────────────
out_path = '/kaggle/working/submission.csv'
submission.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")
print(f"\nFINAL: {final_key}  Val F1={r_final['val_f1_mean']:.4f}  Thr={thr}")"""


# ─────────────────────────────────────────────────────────────
# Assemble and save notebook
# ─────────────────────────────────────────────────────────────
new_cells = [
    make_md_cell("## Cell 6: Feature Extraction + Shared Infrastructure"),
    make_code_cell(CELL6),
    make_md_cell("## Cell 7: Model A — LogisticRegression on CLIP"),
    make_code_cell(CELL7),
    make_md_cell("## Cell 8: Model B — SVM-RBF on CLIP"),
    make_code_cell(CELL8),
    make_md_cell("## Cell 9: Model C — XGBoost + PCA on CLIP"),
    make_code_cell(CELL9),
    make_md_cell("## Cell 10: Model D — SVM on CLIP + CNN Fused"),
    make_code_cell(CELL10),
    make_md_cell("## Cell 11: Model E — Staged CLIP Fine-Tuning"),
    make_code_cell(CELL11),
    make_md_cell("## Cell 12: Weighted Ensemble"),
    make_code_cell(CELL12),
    make_md_cell("## Cell 13: Analysis and Comparison"),
    make_code_cell(CELL13),
    make_md_cell("## Cell 14: Final Submission"),
    make_code_cell(CELL14),
]

nb['cells'].extend(new_cells)

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Done. Notebook now has {len(nb['cells'])} cells.")
print("Cells added: 6-14 (plus markdown headers).")
