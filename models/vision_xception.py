import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False, **kwargs)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class MiniXception(nn.Module):
    def __init__(self, in_channels=1, feat_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            SeparableConv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            SeparableConv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            SeparableConv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, feat_dim)
    def forward(self, x):
        # Input: (B, C, H, W)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Example usage:
# model = MiniXception(in_channels=1, feat_dim=256)
