# ============================================================
# CELL 10: SHARED CV + FINE-TUNE INFRASTRUCTURE
# ============================================================

results_tracker = {}
oof_store       = {}
test_pred_store = {}

def run_epoch(model, loader, optimizer, scaler, criterion, is_train, use_mixup=False):
    model.train() if is_train else model.eval()
    tot_loss = 0.0; preds = []; trues = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE); labels = labels.to(DEVICE)
            if is_train and use_mixup and random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.3)
                with autocast():
                    logits = model(imgs)
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                with autocast():
                    logits = model(imgs)
                    loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            tot_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds.extend(probs.tolist())
            trues.extend(labels.cpu().numpy().tolist())
    f1 = f1_score(trues, (np.array(preds) >= 0.5).astype(int), zero_division=0)
    return tot_loss / len(trues), f1, np.array(preds)

def save_state_cpu(model):
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}

def train_finetune_fold(model, tr_ds, val_ds, criterion,
                        freeze_fn, unfreeze_fn, get_opt_fn,
                        fold_num, batch_size=16):
    tr_ld = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                       num_workers=4, pin_memory=True, drop_last=True)
    va_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)

    # Stage 1: Head only
    freeze_fn(model)
    opt1 = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=1e-3, weight_decay=0.01)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=10, eta_min=1e-5)
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    best_f1 = 0; best_state = None; best_vp = None
    patience = 5; no_improve = 0

    for ep in range(1, 11):
        tl, tf, _ = run_epoch(model, tr_ld, opt1, scaler, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler, criterion, False)
        sch1.step()
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = save_state_cpu(model)
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    s1_f1 = best_f1
    model.load_state_dict(best_state)

    # Stage 2: Unfreeze last blocks
    unfreeze_fn(model)
    opt2 = get_opt_fn(model)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=15, eta_min=1e-7)
    scaler2 = GradScaler(enabled=(DEVICE == "cuda"))
    no_improve = 0

    for ep in range(1, 16):
        tl, tf, _ = run_epoch(model, tr_ld, opt2, scaler2, criterion, True, use_mixup=True)
        vl, vf, vp = run_epoch(model, va_ld, None, scaler2, criterion, False)
        sch2.step()
        gap = tf - vf
        if vf > best_f1:
            best_f1 = vf; no_improve = 0
            best_state = save_state_cpu(model)
            best_vp = vp.copy()
        else:
            no_improve += 1
        if no_improve >= patience:
            break
        if gap > 0.12:
            print(f"  Fold {fold_num}: gap {gap:.4f} > 0.12, stopping Stage 2")
            break

    model.load_state_dict(best_state)
    print(f"   Fold {fold_num}: Stage1 F1={s1_f1:.4f} -> Stage2 F1={best_f1:.4f}")
    return best_f1, best_vp, best_state

def run_test_tta(model_init_fn, fold_models, test_paths,
                 tta_tfm, batch_size=16, n_tta=N_TTA):
    test_proba = np.zeros(len(test_paths))
    for fold_idx, fold_state in enumerate(fold_models):
        print(f"   Inference Fold {fold_idx+1}...")
        model = model_init_fn()
        model.load_state_dict(fold_state)
        model = model.to(DEVICE).eval()

        test_ds = AlbuDataset(test_paths, [-1]*len(test_paths), tta_tfm)
        test_ld = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

        for t in range(n_tta):
            preds = []
            with torch.no_grad():
                for imgs, _ in test_ld:
                    imgs = imgs.to(DEVICE)
                    with autocast():
                        logits = model(imgs)
                    preds.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            test_proba += np.array(preds)

        del model
        gpu_cleanup()

    test_proba /= (len(fold_models) * n_tta)
    return test_proba

print("CV + Fine-tune infrastructure ready.")