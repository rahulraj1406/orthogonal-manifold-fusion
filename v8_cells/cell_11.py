# ============================================================
# CELL 12: DINOv2 ViT-B/14 FINE-TUNE — 5-Fold CV
# ============================================================
# This was the STAR of V5 (Val F1=0.9048). Keeping exact same setup.
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
    make_dino_model, dino_fold_models,
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
print_vram()