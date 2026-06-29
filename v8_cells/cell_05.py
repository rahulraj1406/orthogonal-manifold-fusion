# ============================================================
# CELL 6: FEATURE EXTRACTION — CLIP ViT-L/14 (768-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(4.0)
set_seeds()

def extract_clip_features(image_paths, batch_size=64):
    model, preprocess = clip.load('ViT-L/14', device=DEVICE)
    model = model.float().eval()

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CLIP'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is not None:
                batch_imgs.append(preprocess(img))
            else:
                batch_imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

if FORCE_FRESH or not (cache/'clip_train.npy').exists():
    print("Extracting CLIP features...")
    clip_train = extract_clip_features(train_paths)
    clip_test  = extract_clip_features(test_paths)
    np.save(cache/'clip_train.npy', clip_train)
    np.save(cache/'clip_test.npy', clip_test)
else:
    clip_train = np.load(cache/'clip_train.npy')
    clip_test  = np.load(cache/'clip_test.npy')

print(f"CLIP train: {clip_train.shape}  test: {clip_test.shape}")
print_vram()