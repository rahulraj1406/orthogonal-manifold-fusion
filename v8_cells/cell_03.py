# ============================================================
# CELL 4: IMAGE LOADING & AUGMENTATION UTILITIES
# ============================================================

def load_image_pil(path):
    """Load image as PIL RGB."""
    try:
        return Image.open(str(path)).convert('RGB')
    except Exception:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception:
            pass
    return None

def load_image_np(path, size=None):
    """Load image as numpy array."""
    img = load_image_pil(path)
    if img is None:
        return None
    if size:
        img = img.resize(size, Image.LANCZOS)
    return np.array(img, dtype=np.float64)

# ── Augmentation transforms (albumentations) ────────────────
def get_train_transform(mean, std, size=224):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
        A.GaussNoise(var_limit=(5, 30), p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        A.Normalize(mean=mean, std=std),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
    ])

def get_val_transform(mean, std, size=224):
    return A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(size, size),
        A.Normalize(mean=mean, std=std),
    ])

def get_tta_transform(mean, std, size=224):
    return A.Compose([
        A.RandomResizedCrop(size, size, scale=(0.9, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=mean, std=std),
    ])

class AlbuDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_pil(self.paths[idx])
        if img is None:
            img = Image.new('RGB', (224, 224), (128, 128, 128))
        img_np = np.array(img)
        augmented = self.transform(image=img_np)
        img_tensor = torch.from_numpy(augmented['image'].transpose(2, 0, 1)).float()
        lbl = self.labels[idx] if self.labels is not None else -1
        return img_tensor, torch.tensor(lbl, dtype=torch.float32)

def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

print("Augmentation utilities ready.")