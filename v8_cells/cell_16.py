# ============================================================
# CELL 17: FINAL SUBMISSION
# ============================================================
set_seeds()

# Use ensemble predictions
test_proba = test_pred_store['ensemble']
thr = BEST_THRESHOLD

# Check if best single model beats ensemble
best_single_key = max(['siglip_finetune', 'dino_finetune', 'clip_finetune'],
                      key=lambda k: results_tracker[k]['val_f1_mean'])
best_single_f1 = results_tracker[best_single_key]['val_f1_mean']

if best_single_f1 > best_ensemble_f1:
    print(f"NOTE: {best_single_key} ({best_single_f1:.4f}) > ensemble ({best_ensemble_f1:.4f})")
    print(f"Using {best_single_key} for submission instead.")
    test_proba = test_pred_store[best_single_key]
    single_oof = oof_store[best_single_key]
    thr = 0.5
    best_f1_s = 0
    for t in np.arange(0.30, 0.71, 0.01):
        f = f1_score(y_all, (single_oof >= t).astype(int))
        if f > best_f1_s:
            best_f1_s = f; thr = round(t, 2)
    final_model_name = best_single_key
    final_f1 = best_f1_s
else:
    final_model_name = 'ensemble'
    final_f1 = best_ensemble_f1

print(f"{'='*60}")
print(f"FINAL MODEL: {final_model_name}")
print(f"Val F1: {final_f1:.4f}")
print(f"Threshold: {thr}")
print(f"{'='*60}")

preds_binary = (test_proba >= thr).astype(int)
submission = pd.DataFrame({
    'image_id':     df_test['image_id'].values,
    'ground_truth': preds_binary
})

# Sanity checks
assert submission.shape == (len(df_test), 2), f"Wrong shape: {submission.shape}"
assert submission['ground_truth'].isin([0, 1]).all()
assert not submission.isnull().any().any()

n0 = (submission['ground_truth'] == 0).sum()
n1 = (submission['ground_truth'] == 1).sum()
print(f"\nSanity checks PASSED")
print(f"  Real (0): {n0} ({n0/len(submission):.1%})")
print(f"  AI   (1): {n1} ({n1/len(submission):.1%})")
print(f"\nFirst 5 rows:")
print(submission.head())

# Save — try multiple output locations
saved = False
for out_path in ['/kaggle/working/submission.csv',
                 '/home/jovyan/work/data/submission.csv',
                 './submission.csv']:
    try:
        d = os.path.dirname(out_path)
        if d:
            os.makedirs(d, exist_ok=True)
        submission.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")
        saved = True
        break
    except Exception:
        continue

if not saved:
    submission.to_csv('submission.csv', index=False)
    print("\nSaved: submission.csv")

print(f"\nFINAL: {final_model_name} | Val F1={final_f1:.4f} | Thr={thr}")