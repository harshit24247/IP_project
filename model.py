import torch.nn as nn
import torchvision.models as models

class GazeNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = models.mobilenet_v2(weights=None)
        
        # Adapt for grayscale (1 channel)
        mobilenet.features[0][0] = nn.Conv2d(
            1, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        self.backbone  = mobilenet.features
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.regressor(self.backbone(x))