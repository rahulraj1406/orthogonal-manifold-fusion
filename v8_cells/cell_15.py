# ============================================================
# CELL 16: ANALYSIS & VISUALIZATION
# ============================================================

print("=" * 75)
print(f"{'Model':<32} {'Va-F1':<8} {'AUC':<8}")
print("=" * 75)
for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'xgb_combo', 'ensemble']:
    if k not in results_tracker:
        continue
    r = results_tracker[k]
    print(f"  {r['name']:<30} {r['val_f1_mean']:<8.4f} {r['val_auc_mean']:<8.4f}")

# Model Correlation
print("\nModel Correlation:")
corr_keys = [k for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'xgb_combo']
             if k in oof_store]
corr_data = np.column_stack([oof_store[k] for k in corr_keys])
corr_matrix = np.corrcoef(corr_data.T)
labels = [k[:15] for k in corr_keys]
print(f"{'':>18}", end='')
for l in labels:
    print(f" {l:>15}", end='')
print()
for i, l in enumerate(labels):
    print(f"{l:>18}", end='')
    for j in range(len(labels)):
        print(f" {corr_matrix[i,j]:>15.4f}", end='')
    print()
print("\nLower correlation = more diverse = better ensemble")

# Ablation
print("\n--- ABLATION ---")
for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'xgb_combo', 'ensemble']:
    if k in results_tracker:
        print(f"  {results_tracker[k]['name']:<32} {results_tracker[k]['val_f1_mean']:.4f}")

# Confusion matrix + ROC
best_oof = oof_store.get('ensemble', siglip_oof)
oof_preds = (best_oof >= BEST_THRESHOLD).astype(int)
cm = confusion_matrix(y_all, oof_preds)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Real', 'AI'], yticklabels=['Real', 'AI'])
axes[0].set_title(f'Confusion Matrix (OOF, thr={BEST_THRESHOLD})')
axes[0].set_ylabel('True'); axes[0].set_xlabel('Predicted')

for k in ['siglip_finetune', 'dino_finetune', 'clip_finetune', 'ensemble']:
    if k not in oof_store:
        continue
    fpr, tpr, _ = roc_curve(y_all, oof_store[k])
    auc_val = roc_auc_score(y_all, oof_store[k])
    axes[1].plot(fpr, tpr, label=f"{k[:20]} (AUC={auc_val:.4f})")
axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[1].set_title('ROC Curves'); axes[1].legend()
axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')

plt.tight_layout()
plt.savefig('analysis_v8.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nFINAL ENSEMBLE F1: {best_ensemble_f1:.4f}")