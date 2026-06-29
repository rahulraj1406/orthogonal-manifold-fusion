# ============================================================
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
    make_clip_model, clip_fold_models,
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
print_vram()