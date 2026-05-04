"""
dl_model.py
===========
ResNet-style CNN with Squeeze-and-Excitation (SE) channel attention.

SE blocks let the network learn WHICH feature channels matter most for each
gait class — which is highly effective for structured plantar pressure maps
where heel/forefoot/midfoot signals each carry distinct discriminative information.

Architecture flow:
  Stem (1→64, stride-2)
  → SE-ResStage  64→ 64  (×2 blocks, stride-1)
  → SE-ResStage  64→128  (×2 blocks, stride-2)
  → SE-ResStage 128→256  (×2 blocks, stride-2)
  → SE-ResStage 256→512  (×3 blocks, stride-2)
  → AdaptiveAvgPool 1×1
  → Classifier (512 → 256 → 128 → num_classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Squeeze-and-Excitation block
# ─────────────────────────────────────────────────────────────────────────────
class SEBlock(nn.Module):
    """Channel attention: Squeeze → Excite → Scale."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


# ─────────────────────────────────────────────────────────────────────────────
# Residual block with BatchNorm + SE attention
# ─────────────────────────────────────────────────────────────────────────────
class SEResBlock(nn.Module):
    """
    Standard residual block with an SE channel-attention gate on the output.
    Pre-activation (BN→ReLU→Conv) gives slightly better gradient flow.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 se_reduction: int = 16):
        super().__init__()
        # Main path
        self.path = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
        )
        self.se = SEBlock(out_ch, se_reduction)

        # Projection shortcut when dimensions change
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        return self.se(self.path(x)) + self.shortcut(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────
class PressureCNN(nn.Module):
    """
    PressureCNN — SE-ResNet for plantar pressure gait classification.

    Input : (B, 1, 128, 48)  — normalised to [0, 1]
    Output: (B, num_classes) — raw logits
    """

    def __init__(self, num_classes: int = 6, se_reduction: int = 16,
                 dropout: float = 0.3):
        super().__init__()

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),   # 32 × 12
        )

        # ── Residual stages ───────────────────────────────────────────────────
        self.stage1 = self._make_stage( 64,  64,  n=2, stride=1, r=se_reduction)
        self.stage2 = self._make_stage( 64, 128,  n=2, stride=2, r=se_reduction)
        self.stage3 = self._make_stage(128, 256,  n=2, stride=2, r=se_reduction)
        self.stage4 = self._make_stage(256, 512,  n=3, stride=2, r=se_reduction)

        # ── Global average pool ───────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # ── Classifier head ───────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    # ── helpers ───────────────────────────────────────────────────────────────
    def _make_stage(self, in_ch, out_ch, n, stride, r):
        layers = [SEResBlock(in_ch, out_ch, stride=stride, se_reduction=r)]
        for _ in range(1, n):
            layers.append(SEResBlock(out_ch, out_ch, stride=1, se_reduction=r))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.pool(x)
        x = self.head(x)
        return x
