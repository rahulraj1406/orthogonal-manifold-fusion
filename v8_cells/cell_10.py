# ============================================================
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
    base = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    model = SigLIPFineTuner(base.vision_model, embed_dim=1152)
    del base
    return model

siglip_train_tfm = get_train_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)
siglip_val_tfm   = get_val_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)
siglip_tta_tfm   = get_tta_transform(SIGLIP_MEAN, SIGLIP_STD, size=224)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
siglip_oof = np.zeros(len(y_all))
siglip_fold_f1s = []
siglip_fold_models = []

print("=" * 60)
print("SigLIP So400m FINE-TUNE — 5-Fold CV")
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
    make_siglip_model, siglip_fold_models,
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
print_vram()