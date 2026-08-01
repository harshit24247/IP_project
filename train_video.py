import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split, ConcatDataset
from video_dataset import WebcamGazeDataset, MPIIGazeDataset
from video_model import GazeNetVideo

WEBCAM_DIR  = r'C:\Users\ASUS\eye_movement_detection\video_dataset'
MPIIGAZE_DIR = r'C:\Users\ASUS\eye_movement_detection\MPIIGaze\Data\Normalized'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}")

def angular_error_batch(pred, target):
    pred_norm   = pred   / pred.norm(dim=1, keepdim=True).clamp(min=1e-6)
    target_norm = target / target.norm(dim=1, keepdim=True).clamp(min=1e-6)
    cos_sim = (pred_norm * target_norm).sum(dim=1).clamp(-1, 1)
    return torch.acos(cos_sim) * (180 / np.pi)

# Load both datasets
webcam_data  = WebcamGazeDataset(WEBCAM_DIR, sequence_len=8)
mpiigaze_data = MPIIGazeDataset(MPIIGAZE_DIR, sequence_len=8)

print(f"\n✅ Webcam dataset:   {len(webcam_data)} sequences")
print(f"✅ MPIIGaze dataset: {len(mpiigaze_data)} sequences")

# Combine
combined = ConcatDataset([webcam_data, mpiigaze_data])
print(f"✅ Total combined:   {len(combined)} sequences\n")

# Split 80/20
train_size = int(0.8 * len(combined))
test_size  = len(combined) - train_size
train_set, test_set = random_split(combined, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=64, shuffle=True,  num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False, num_workers=0)

# Model — start fresh with pretrained backbone
model     = GazeNetVideo(pretrained_path='best_model.pth').to(device)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
)
criterion = nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

best_val_err = float('inf')
EPOCHS       = 50

for epoch in range(EPOCHS):
    if epoch == 10:
        model.unfreeze_backbone()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train
    model.train()
    train_loss, train_errs = 0, []
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        pred = model(seqs)
        loss = criterion(pred, labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        train_loss += loss.item()
        train_errs.append(angular_error_batch(pred.detach(), labels.detach()))

    # Validate
    model.eval()
    val_loss, val_errs = 0, []
    with torch.no_grad():
        for seqs, labels in test_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            pred = model(seqs)
            val_loss += criterion(pred, labels).item()
            val_errs.append(angular_error_batch(pred, labels))

    # Metrics
    train_ang = torch.cat(train_errs)
    val_ang   = torch.cat(val_errs)
    t_err  = train_ang.mean().item()
    v_err  = val_ang.mean().item()
    acc5   = (val_ang < 5).float().mean().item()  * 100
    acc10  = (val_ang < 10).float().mean().item() * 100

    scheduler.step(v_err)

    print(
        f"Epoch [{epoch+1:02d}/{EPOCHS}] "
        f"| Train Loss: {train_loss/len(train_loader):.4f} "
        f"| Val Loss: {val_loss/len(test_loader):.4f} "
        f"| Train Err: {t_err:.2f}° "
        f"| Val Err: {v_err:.2f}° "
        f"| Acc@5°: {acc5:.1f}% "
        f"| Acc@10°: {acc10:.1f}%"
    )

    if v_err < best_val_err:
        best_val_err = v_err
        torch.save(model.state_dict(), 'best_video_model.pth')
        print(f"  ✅ Best model saved! Val Error: {best_val_err:.2f}°")

print(f"\n🏆 Done! Best Val Error: {best_val_err:.2f}°")