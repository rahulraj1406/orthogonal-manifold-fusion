# ============================================================
# CELL 7: FEATURE EXTRACTION — DINOv2 ViT-B/14 (768-dim)
# ============================================================
# Using DINOv2-BASE (not Large!) — V5 proved B >> L on 4800 samples
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(2.0)
set_seeds()

def extract_dino_features(image_paths, batch_size=64):
    model = timm.create_model('vit_base_patch14_dinov2.lvd142m',
                               pretrained=True, num_classes=0)
    model = model.to(DEVICE).eval()

    tfm = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(DINO_MEAN, DINO_STD),
    ])

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='DINOv2-B'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is not None:
                batch_imgs.append(tfm(img))
            else:
                batch_imgs.append(torch.zeros(3, 224, 224))
        batch = torch.stack(batch_imgs).to(DEVICE)
        with torch.no_grad():
            feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

if FORCE_FRESH or not (cache/'dino_train.npy').exists():
    print("Extracting DINOv2-B features...")
    dino_train = extract_dino_features(train_paths)
    dino_test  = extract_dino_features(test_paths)
    np.save(cache/'dino_train.npy', dino_train)
    np.save(cache/'dino_test.npy', dino_test)
else:
    dino_train = np.load(cache/'dino_train.npy')
    dino_test  = np.load(cache/'dino_test.npy')

print(f"DINOv2-B train: {dino_train.shape}  test: {dino_test.shape}")
print_vram()