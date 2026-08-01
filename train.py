import torch
import torch.nn as nn
import numpy as np

def angular_error_batch(pred, target):
    pred_norm   = pred   / pred.norm(dim=1, keepdim=True).clamp(min=1e-6)
    target_norm = target / target.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos_sim = (pred_norm * target_norm).sum(dim=1).clamp(-1, 1)
    return torch.acos(cos_sim) * (180 / np.pi)

def train(model, train_loader, test_loader, optimizer, criterion, scheduler, device, epochs=50):
    best_val_err = float('inf')
    history = {
        'train_loss': [], 'val_loss': [],
        'train_ang_err': [], 'val_ang_err': [],
        'acc_5deg': [], 'acc_10deg': []
    }

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss, train_ang_errs = 0, []

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            pred = model(imgs)
            loss = criterion(pred, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_loss += loss.item()
            train_ang_errs.append(angular_error_batch(pred.detach(), labels.detach()))

        # ── Validate ──
        model.eval()
        val_loss, val_ang_errs = 0, []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs)
                val_loss += criterion(pred, labels).item()
                val_ang_errs.append(angular_error_batch(pred, labels))

        # ── Metrics ──
        train_ang = torch.cat(train_ang_errs)
        val_ang   = torch.cat(val_ang_errs)

        m = {
            'train_loss'    : train_loss / len(train_loader),
            'val_loss'      : val_loss   / len(test_loader),
            'train_ang_err' : train_ang.mean().item(),
            'val_ang_err'   : val_ang.mean().item(),
            'acc_5deg'      : (val_ang < 5).float().mean().item()  * 100,
            'acc_10deg'     : (val_ang < 10).float().mean().item() * 100,
            'val_median'    : val_ang.median().item(),
            'val_std'       : val_ang.std().item(),
        }

        for key in history:
            history[key].append(m[key])

        # ✅ Step scheduler
        scheduler.step(m['val_ang_err'])

        print(
            f"Epoch [{epoch+1:02d}/{epochs}] "
            f"| Train Loss: {m['train_loss']:.4f} "
            f"| Val Loss: {m['val_loss']:.4f} "
            f"| Train Err: {m['train_ang_err']:.2f}° "
            f"| Val Err: {m['val_ang_err']:.2f}° "
            f"| Acc@5°: {m['acc_5deg']:.1f}% "
            f"| Acc@10°: {m['acc_10deg']:.1f}% "
            f"| Median: {m['val_median']:.2f}° "
            f"| Std: {m['val_std']:.2f}°"
        )

        # ✅ Save best model
        if m['val_ang_err'] < best_val_err:
            best_val_err = m['val_ang_err']
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  ✅ Best model saved! Val Error: {best_val_err:.2f}°")

    return history