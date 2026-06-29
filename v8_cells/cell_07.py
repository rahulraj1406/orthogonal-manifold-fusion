# ============================================================
# CELL 8: FEATURE EXTRACTION — EfficientNet-B0 (1280-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(2.0)
set_seeds()

def extract_cnn_features(image_paths, batch_size=128):
    try:
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    except Exception:
        from torchvision.models import efficientnet_b0
        model = efficientnet_b0(pretrained=True)
    model.classifier = nn.Identity()
    model = model.to(DEVICE).eval()

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='CNN'):
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

if FORCE_FRESH or not (cache/'cnn_train.npy').exists():
    print("Extracting CNN features...")
    cnn_train = extract_cnn_features(train_paths)
    cnn_test  = extract_cnn_features(test_paths)
    np.save(cache/'cnn_train.npy', cnn_train)
    np.save(cache/'cnn_test.npy', cnn_test)
else:
    cnn_train = np.load(cache/'cnn_train.npy')
    cnn_test  = np.load(cache/'cnn_test.npy')

print(f"CNN train: {cnn_train.shape}  test: {cnn_test.shape}")
print_vram()