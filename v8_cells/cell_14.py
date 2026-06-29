# ============================================================
# CELL 15: 4-MODEL ENSEMBLE + THRESHOLD TUNING
# ============================================================
set_seeds()

ensemble_keys = ['siglip_finetune', 'dino_finetune', 'clip_finetune']
# Add XGB if it adds signal
if results_tracker['xgb_combo']['val_f1_mean'] > 0.55:
    ensemble_keys.append('xgb_combo')

print("=" * 60)
print("ENSEMBLE COMPOSITION")
print("=" * 60)
for k in ensemble_keys:
    r = results_tracker[k]
    print(f"  {r['name']:<30} | Val F1: {r['val_f1_mean']:.4f}")

# ── Strategy A: Weighted Average (F1-squared weights) ───────
print("\n-- Strategy A: Weighted Average --")
f1_sq = {k: results_tracker[k]['val_f1_mean']**2 for k in ensemble_keys}
total_w = sum(f1_sq.values())
weights = {k: v/total_w for k, v in f1_sq.items()}
for k, w in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"  {k:<30} w={w:.4f}")

oof_blend_A = np.zeros(len(y_all))
for k, w in weights.items():
    oof_blend_A += w * oof_store[k]

best_thr_A = 0.5; best_f1_A = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (oof_blend_A >= t).astype(int))
    if f > best_f1_A:
        best_f1_A = f; best_thr_A = round(t, 2)
print(f"Result A: OOF F1={best_f1_A:.4f} @ Thr={best_thr_A}")

# ── Strategy B: LogReg Meta-Learner ─────────────────────────
print("\n-- Strategy B: LogReg Meta-Learner --")
X_meta = np.column_stack([oof_store[k] for k in ensemble_keys])
skf_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
meta_oof = np.zeros(len(y_all))
for fold, (tr_idx, val_idx) in enumerate(skf_meta.split(y_all, y_all)):
    meta_lr = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_lr.fit(X_meta[tr_idx], y_all[tr_idx])
    meta_oof[val_idx] = meta_lr.predict_proba(X_meta[val_idx])[:, 1]

best_thr_B = 0.5; best_f1_B = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (meta_oof >= t).astype(int))
    if f > best_f1_B:
        best_f1_B = f; best_thr_B = round(t, 2)
print(f"Result B: OOF F1={best_f1_B:.4f} @ Thr={best_thr_B}")

# ── Strategy C: Rank-Based Average ──────────────────────────
print("\n-- Strategy C: Rank-Based Average --")
rank_blend = np.zeros(len(y_all))
for k in ensemble_keys:
    ranks = rankdata(oof_store[k]) / len(y_all)
    rank_blend += weights[k] * ranks

best_thr_C = 0.5; best_f1_C = 0
for t in np.arange(0.30, 0.71, 0.01):
    f = f1_score(y_all, (rank_blend >= t).astype(int))
    if f > best_f1_C:
        best_f1_C = f; best_thr_C = round(t, 2)
print(f"Result C: OOF F1={best_f1_C:.4f} @ Thr={best_thr_C}")

# ── Select best ──────────────────────────────────────────────
strategies = {'weighted_avg': (best_f1_A, best_thr_A),
              'meta': (best_f1_B, best_thr_B),
              'rank_avg': (best_f1_C, best_thr_C)}
best_strat = max(strategies, key=lambda k: strategies[k][0])
best_ensemble_f1, BEST_THRESHOLD = strategies[best_strat]

print(f"\n{'='*60}")
print(f"WINNER: {best_strat} (F1={best_ensemble_f1:.4f})")

# Generate test predictions
if best_strat == 'meta':
    meta_final = LogisticRegression(C=1.0, random_state=SEED, max_iter=1000)
    meta_final.fit(X_meta, y_all)
    X_meta_test = np.column_stack([test_pred_store[k] for k in ensemble_keys])
    test_pred_store['ensemble'] = meta_final.predict_proba(X_meta_test)[:, 1]
    final_oof = meta_oof
elif best_strat == 'rank_avg':
    test_blend = np.zeros(len(test_paths))
    for k in ensemble_keys:
        test_blend += weights[k] * (rankdata(test_pred_store[k]) / len(test_paths))
    test_pred_store['ensemble'] = test_blend
    final_oof = rank_blend
else:
    test_blend = np.zeros(len(test_paths))
    for k, w in weights.items():
        test_blend += w * test_pred_store[k]
    test_pred_store['ensemble'] = test_blend
    final_oof = oof_blend_A

results_tracker['ensemble'] = {
    'name': f'Ensemble ({best_strat})',
    'val_f1_mean': best_ensemble_f1,
    'val_f1_std': 0.0,
    'val_auc_mean': roc_auc_score(y_all, final_oof),
    'oof_proba': final_oof.copy()
}
oof_store['ensemble'] = final_oof.copy()

print(f"FINAL ENSEMBLE: {best_strat} | F1={best_ensemble_f1:.4f} | Thr={BEST_THRESHOLD}")
print(f"{'='*60}")