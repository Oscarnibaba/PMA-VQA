import torch
from torch import nn
import torch.nn.functional as F

class SimpleVQAHead(nn.Module):
    def __init__(self, c4_dims, num_answers=9, hidden=1024, dropout=0.1):
        super().__init__()
        self._needs_text = False
        self.mlp = nn.Sequential(
            nn.Linear(c4_dims, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_answers),
        )

    def forward(self, x_c4):
        # 全局平均池化 + MLP分类
        x = F.adaptive_avg_pool2d(x_c4, 1).flatten(1)  # (B, C)
        return self.mlp(x)  # (B, num_answers)

class ProgressiveFeatureFusionVQAHead(nn.Module):
    def __init__(self, embed_dim, num_answers=9, hidden=1024, dropout=0.1):
        super().__init__()
        self._needs_text = False
        self.c1_proj = nn.Conv2d(embed_dim, hidden, 1)
        self.c2_proj = nn.Conv2d(2 * embed_dim, hidden, 1)
        self.c3_proj = nn.Conv2d(4 * embed_dim, hidden, 1)
        self.c4_proj = nn.Conv2d(8 * embed_dim, hidden, 1)

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden * 4, hidden * 2, 3, padding=1),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden * 2, hidden, 3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_answers)
        )

    def forward(self, features):
        c1, c2, c3, c4 = features

        c1_resized = F.interpolate(self.c1_proj(c1), size=c4.shape[-2:], mode='bilinear', align_corners=False)
        c2_resized = F.interpolate(self.c2_proj(c2), size=c4.shape[-2:], mode='bilinear', align_corners=False)
        c3_resized = F.interpolate(self.c3_proj(c3), size=c4.shape[-2:], mode='bilinear', align_corners=False)
        c4_proj = self.c4_proj(c4)

        fused = torch.cat([c1_resized, c2_resized, c3_resized, c4_proj], dim=1)
        fused = self.fusion_conv(fused)

        return self.classifier(fused)