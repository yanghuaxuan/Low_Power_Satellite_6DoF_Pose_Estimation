# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")  # hides most PIL/PNG warnings
warnings.filterwarnings("ignore", message="libpng warning")           # specifically targets the libpng spam
warnings.filterwarnings("ignore", message=".*eXIf: duplicate.*")

# Silence libpng via environment (most effective)
import os
os.environ["LIBPNG_NO_WARNINGS"] = "1"

class SimpleConvBlock(nn.Module):
    """Basic convolutional block: Conv → BN → ReLU → optional MaxPool"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class RGBEventPoseNet(nn.Module):
    """
    Simple from-scratch CNN for 6D pose regression.
    Input: RGB (3,H,W) + Event (1,H,W) → Output: 7D pose [x,y,z, qx,qy,qz,qw]
    """
    def __init__(self, input_size=(720, 800), num_conv_layers=5, base_channels=32):
        super().__init__()
        h, w = input_size

        # RGB branch (3 input channels)
        self.rgb_conv = nn.Sequential(
            SimpleConvBlock(3,   base_channels,     pool=True),   # → /2
            SimpleConvBlock(base_channels,   base_channels*2,   pool=True),
            SimpleConvBlock(base_channels*2, base_channels*4,   pool=True),
            SimpleConvBlock(base_channels*4, base_channels*8,   pool=True),
            SimpleConvBlock(base_channels*8, base_channels*16,  pool=True),  # → /32
        )

        # Event branch (1 input channel)
        self.event_conv = nn.Sequential(
            SimpleConvBlock(1,   base_channels,     pool=True),
            SimpleConvBlock(base_channels,   base_channels*2,   pool=True),
            SimpleConvBlock(base_channels*2, base_channels*4,   pool=True),
            SimpleConvBlock(base_channels*4, base_channels*8,   pool=True),
            SimpleConvBlock(base_channels*8, base_channels*16,  pool=True),
        )

        # After 5 poolings: spatial size ≈ H/32 × W/32
        feat_h = h // 32
        feat_w = w // 32
        feat_dim = base_channels * 16 * 2  # RGB + Event concatenated

        # Fusion & regression head
        self.pos_head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 3),  # [tx, ty, tz]
        )

        # Fusion & regression head
        self.rot_head = nn.Sequential(
            nn.Linear(feat_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(512, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),  
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            nn.Linear(16, 4),  # [qx, qy, qz, qw]
        )



    def forward(self, rgb, event):
        # Feature extraction
        rgb_feat = self.rgb_conv(rgb)      # [B, C*16, H/32, W/32]
        event_feat = self.event_conv(event)  # [B, C*16, H/32, W/32]

        # Global average pooling + concatenate
        rgb_feat = F.adaptive_avg_pool2d(rgb_feat, (1, 1)).view(rgb_feat.size(0), -1)
        event_feat = F.adaptive_avg_pool2d(event_feat, (1, 1)).view(event_feat.size(0), -1)
        fused = torch.cat([rgb_feat, event_feat], dim=1)  # [B, feat_dim]

        # Regression head
        # pose = self.fusion(fused)

        # Optional: normalize quaternion part (qx,qy,qz,qw)
        # quat = pose[:, 3:]
        # quat_norm = torch.norm(quat, p=2, dim=1, keepdim=True) + 1e-8
        # quat_normalized = quat / quat_norm
        # quat_normalized = F.normalize(quat, p=2, dim=1)
        # quat_normalized = torch.clamp(quat_normalized, -1.0, 1.0)
        # pose = torch.cat([pose[:, :3], quat_normalized], dim=1)


        pos = self.pos_head(fused)
        rot = self.rot_head(fused)
        rot = F.normalize(rot, p=2, dim=1)
        rot = torch.clamp(rot, -1.0, 1.0)
        pose = torch.cat([pos, rot], dim=1)
        

        # final 7D output: [tx, ty, tz, qx, qy, qz, qw]
        return pose


# Quick test / usage example
if __name__ == "__main__":
    model = RGBEventPoseNet(input_size=(720, 800))
    rgb   = torch.randn(2, 3, 720, 800)
    event = torch.randn(2, 1, 720, 800)
    out = model(rgb, event)
    print("Output shape:", out.shape)          # should be [2, 7]
    print("Sample output:\n", out)