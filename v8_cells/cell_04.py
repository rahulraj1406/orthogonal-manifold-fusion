# ============================================================
# CELL 5: FEATURE EXTRACTION — SigLIP So400m (1152-dim)
# ============================================================
import torch, gc
torch.cuda.empty_cache()
gc.collect()
ensure_vram(4.0)
set_seeds()

from transformers import SiglipModel, SiglipProcessor

def extract_siglip_features(image_paths, batch_size=32):
    print("Loading SigLIP So400m-patch14-224...")
    model = SiglipModel.from_pretrained("google/siglip-so400m-patch14-224")
    processor = SiglipProcessor.from_pretrained("google/siglip-so400m-patch14-224")
    model = model.to(DEVICE).eval()
    vision_model = model.vision_model

    feats_all = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc='SigLIP'):
        batch_imgs = []
        for p in image_paths[i:i+batch_size]:
            img = load_image_pil(p)
            if img is None:
                img = Image.new('RGB', (224, 224), (128, 128, 128))
            batch_imgs.append(img)
        inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
        pixel_values = inputs['pixel_values'].to(DEVICE)
        with torch.no_grad():
            outputs = vision_model(pixel_values=pixel_values)
            feats = outputs.pooler_output.float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        feats_all.append(feats.cpu().numpy())

    del model, vision_model, processor
    gpu_cleanup()
    return np.vstack(feats_all).astype(np.float32)

cache = CACHE_DIR
if FORCE_FRESH or not (cache/'siglip_train.npy').exists():
    print("Extracting SigLIP features...")
    siglip_train = extract_siglip_features(train_paths)
    siglip_test  = extract_siglip_features(test_paths)
    np.save(cache/'siglip_train.npy', siglip_train)
    np.save(cache/'siglip_test.npy', siglip_test)
else:
    siglip_train = np.load(cache/'siglip_train.npy')
    siglip_test  = np.load(cache/'siglip_test.npy')

print(f"SigLIP train: {siglip_train.shape}  test: {siglip_test.shape}")
print_vram()