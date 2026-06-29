# ============================================================
# CELL 3: DATASET LOADING
# ============================================================
set_seeds()

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)

df_train['filepath'] = df_train['image_id'].apply(lambda x: str(IMAGE_DIR / x))
df_test['filepath']  = df_test['image_id'].apply(lambda x: str(IMAGE_DIR / x))

print(f"Train: {len(df_train)} images, Test: {len(df_test)} images")
print(f"Class balance: Real={sum(df_train['ground_truth']==0)} ({sum(df_train['ground_truth']==0)/len(df_train):.1%}), "
      f"AI={sum(df_train['ground_truth']==1)} ({sum(df_train['ground_truth']==1)/len(df_train):.1%})")

# Verify paths
found = sum(Path(p).exists() for p in df_train['filepath'][:50])
print(f"Path check: {found}/50 train images found")
assert found >= 45, "Too many missing images!"

train_paths = df_train['filepath'].tolist()
test_paths  = df_test['filepath'].tolist()
y_all       = df_train['ground_truth'].values

print(f"Ready: {len(train_paths)} train, {len(test_paths)} test")