# ============================================================
# CELL 14: FORENSIC + CNN -> XGBoost (5-Fold CV)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
set_seeds()

# Remove dead forensic features
vt = VarianceThreshold(threshold=1e-10)
forensic_train_clean = vt.fit_transform(forensic_train)
forensic_test_clean  = vt.transform(forensic_test)
print(f"Forensic: {forensic_train.shape[1]} -> {forensic_train_clean.shape[1]} features")

# Combine forensic + CNN
X_combo_train = np.hstack([forensic_train_clean, cnn_train])
X_combo_test  = np.hstack([forensic_test_clean, cnn_test])

XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=5,
    reg_alpha=0.1, reg_lambda=1.0, gamma=0.1,
    use_label_encoder=False, eval_metric='logloss',
    random_state=SEED, tree_method='hist', device='cuda'
)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
xgb_oof = np.zeros(len(y_all))
xgb_fold_f1s = []

print("=" * 60)
print("XGB-Forensic+CNN — 5-Fold CV")
print("=" * 60)
print(f"{'Fold':<5} {'Tr-F1':<8} {'Va-F1':<8} {'Gap':<7} {'AUC':<8}")
print('-' * 40)

for fold, (tr_idx, val_idx) in enumerate(skf.split(y_all, y_all)):
    Xtr = X_combo_train[tr_idx]; Xv = X_combo_train[val_idx]
    ytr = y_all[tr_idx]; yv = y_all[val_idx]

    pipe = Pipeline([
        ('sc', RobustScaler()),
        ('pca', PCA(n_components=128, random_state=SEED)),
        ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
    ])
    pipe.fit(Xtr, ytr)
    tp = pipe.predict_proba(Xtr)[:, 1]
    vp = pipe.predict_proba(Xv)[:, 1]
    tf = f1_score(ytr, (tp >= 0.5).astype(int))
    vf = f1_score(yv, (vp >= 0.5).astype(int))
    au = roc_auc_score(yv, vp)
    gap = tf - vf
    xgb_oof[val_idx] = vp
    xgb_fold_f1s.append(vf)
    print(f"{fold+1:<5} {tf:<8.4f} {vf:<8.4f} {gap:<7.4f} {au:<8.4f}")

xgb_val_f1 = np.mean(xgb_fold_f1s)
xgb_oof_auc = roc_auc_score(y_all, xgb_oof)
print('-' * 40)
print(f"MEAN  {'':8} {xgb_val_f1:<8.4f}")

# Train on all data for test predictions
pipe_final = Pipeline([
    ('sc', RobustScaler()),
    ('pca', PCA(n_components=128, random_state=SEED)),
    ('xgb', xgb.XGBClassifier(**XGB_PARAMS))
])
pipe_final.fit(X_combo_train, y_all)
xgb_test_proba = pipe_final.predict_proba(X_combo_test)[:, 1]

results_tracker['xgb_combo'] = {
    'name': 'XGB-Forensic+CNN', 'val_f1_mean': xgb_val_f1,
    'val_f1_std': np.std(xgb_fold_f1s), 'val_auc_mean': xgb_oof_auc,
    'oof_proba': xgb_oof.copy()
}
oof_store['xgb_combo'] = xgb_oof.copy()
test_pred_store['xgb_combo'] = xgb_test_proba.copy()

print(f"\nXGB Forensic+CNN: Val F1={xgb_val_f1:.4f}, OOF AUC={xgb_oof_auc:.4f}")