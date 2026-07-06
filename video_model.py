import torch
import torch.nn as nn
from model import GazeNet

class GazeNetVideo(nn.Module):
    def __init__(self, pretrained_path='best_model.pth'):
        super().__init__()
        base = GazeNet()
        base.load_state_dict(torch.load(pretrained_path, map_location='cpu', weights_only=False))
        self.backbone = base.backbone

        for param in self.backbone.parameters():
            param.requires_grad = False

        # ✅ AdaptiveAvgPool to force output to (batch, 1280)
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.lstm = nn.LSTM(
            input_size=1280, hidden_size=256,
            num_layers=2, batch_first=True, dropout=0.3
        )
        self.regressor = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("✅ Backbone unfrozen!")

    def forward(self, x):
        batch, seq_len = x.shape[:2]
        x = x.view(batch * seq_len, *x.shape[2:])

        # ✅ Extract features and pool to fixed size
        features = self.backbone(x)           # (batch*seq, 1280, H, W)
        features = self.pool(features)        # (batch*seq, 1280, 1, 1)
        features = features.view(batch, seq_len, -1)  # (batch, seq, 1280)

        lstm_out, _ = self.lstm(features)
        return self.regressor(lstm_out[:, -1, :])