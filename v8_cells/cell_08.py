# ============================================================
# CELL 9: FORENSIC FEATURE EXTRACTION (~114 dims)
# ============================================================
set_seeds()

def extract_ela_features(img_np):
    features = []
    img_pil = Image.fromarray(img_np.astype(np.uint8))
    for quality in [90, 75, 50]:
        buffer = io.BytesIO()
        img_pil.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        recompressed = np.array(Image.open(buffer), dtype=np.float64)
        ela = np.abs(img_np - recompressed)
        for c in range(3):
            ch = ela[:, :, c]
            features.extend([np.mean(ch), np.std(ch)])
        ela_gray = np.mean(ela, axis=2)
        features.extend([np.percentile(ela_gray, 95), np.percentile(ela_gray, 5)])
    return np.array(features, dtype=np.float32)  # 24 dims

def extract_fft_features(img_np):
    gray = np.mean(img_np, axis=2)
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)

    n_bins = 30
    radial_profile = np.zeros(n_bins)
    for i in range(n_bins):
        r_inner = int(i * max_radius / n_bins)
        r_outer = int((i + 1) * max_radius / n_bins)
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = (x*x + y*y >= r_inner**2) & (x*x + y*y < r_outer**2)
        if mask.any():
            radial_profile[i] = np.mean(magnitude[mask])

    features = list(radial_profile)  # 30
    features.extend([np.mean(magnitude), np.std(magnitude),
                     np.sum(magnitude[cy-10:cy+10, cx-10:cx+10]),
                     np.sum(magnitude) - np.sum(magnitude[cy-10:cy+10, cx-10:cx+10])])

    y_grid, x_grid = np.ogrid[-cy:h-cy, -cx:w-cx]
    r_sq = y_grid**2 + x_grid**2
    low_e = np.sum(magnitude[r_sq < (max_radius * 0.2)**2])
    mid_e = np.sum(magnitude[(r_sq >= (max_radius * 0.2)**2) & (r_sq < (max_radius * 0.5)**2)])
    high_e = np.sum(magnitude[r_sq >= (max_radius * 0.5)**2])
    total_e = low_e + mid_e + high_e + 1e-10
    features.extend([low_e/total_e, mid_e/total_e, high_e/total_e, high_e/(low_e+1e-10)])

    valid = radial_profile > 0
    slope = np.polyfit(np.log(np.arange(1, n_bins+1)[valid]),
                       np.log(radial_profile[valid]), 1)[0] if valid.sum() > 5 else 0.0
    features.append(slope)

    phase = np.angle(f_shift)
    features.extend([np.mean(phase), np.std(phase),
                     np.mean(np.abs(np.diff(phase, axis=0))),
                     np.mean(np.abs(np.diff(phase, axis=1)))])
    features = features[:50]
    while len(features) < 50:
        features.append(0.0)
    return np.array(features, dtype=np.float32)

def extract_noise_features(img_np):
    gray = np.mean(img_np, axis=2)
    features = []
    for wavelet in ['db1', 'db2']:
        coeffs = pywt.dwt2(gray, wavelet)
        cA, (cH, cV, cD) = coeffs
        for detail in [cH, cV, cD]:
            features.extend([np.mean(np.abs(detail)), np.std(detail),
                             np.percentile(np.abs(detail), 99), np.mean(detail**2)])
    denoised = median_filter(gray, size=3)
    noise = gray - denoised
    features.extend([np.mean(noise), np.std(noise), np.mean(noise**2),
                     np.percentile(noise, 1), np.percentile(noise, 99)])
    block_size = 32
    h, w = gray.shape
    local_vars = []
    for yy in range(0, h - block_size, block_size):
        for xx in range(0, w - block_size, block_size):
            local_vars.append(np.var(noise[yy:yy+block_size, xx:xx+block_size]))
    if local_vars:
        features.extend([np.mean(local_vars), np.std(local_vars),
                         np.percentile(local_vars, 90) / (np.percentile(local_vars, 10) + 1e-10)])
    else:
        features.extend([0, 0, 0])
    features = features[:40]
    while len(features) < 40:
        features.append(0.0)
    return np.array(features, dtype=np.float32)

def extract_forensic_single(path, target_size=(256, 256)):
    img_np = load_image_np(path, size=target_size)
    if img_np is None:
        return np.zeros(FORENSIC_DIM, dtype=np.float32)
    ela = extract_ela_features(img_np)
    fft_f = extract_fft_features(img_np)
    noise = extract_noise_features(img_np)
    combined = np.concatenate([ela, fft_f, noise])
    if len(combined) > FORENSIC_DIM:
        combined = combined[:FORENSIC_DIM]
    elif len(combined) < FORENSIC_DIM:
        combined = np.pad(combined, (0, FORENSIC_DIM - len(combined)))
    return combined.astype(np.float32)

def extract_forensic_batch(paths):
    return np.vstack([extract_forensic_single(p) for p in tqdm(paths, desc='Forensic')])

if FORCE_FRESH or not (cache/'forensic_train.npy').exists():
    print("Extracting forensic features...")
    forensic_train = extract_forensic_batch(train_paths)
    forensic_test  = extract_forensic_batch(test_paths)
    np.save(cache/'forensic_train.npy', forensic_train)
    np.save(cache/'forensic_test.npy', forensic_test)
else:
    forensic_train = np.load(cache/'forensic_train.npy')
    forensic_test  = np.load(cache/'forensic_test.npy')

n_alive = (forensic_train.std(axis=0) > 1e-10).sum()
print(f"Forensic train: {forensic_train.shape}  Alive: {n_alive}/{forensic_train.shape[1]}")