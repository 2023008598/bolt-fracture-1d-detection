import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 基础残差块 (1D)
# ─────────────────────────────────────────────
class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        pad = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size,
                               stride=stride, padding=pad, dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ─────────────────────────────────────────────
# 多尺度特征提取模块
# ─────────────────────────────────────────────
class MultiScaleBlock(nn.Module):
    """并行使用不同卷积核捕获不同频率特征"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        branch_ch = out_ch // 4
        self.b1 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 3, padding=1, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.b2 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 7, padding=3, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.b3 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 15, padding=7, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.b4 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch, 31, padding=15, bias=False),
            nn.BatchNorm1d(branch_ch), nn.ReLU()
        )
        self.proj = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch), nn.ReLU()
        )

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.proj(out)


# ─────────────────────────────────────────────
# 通道注意力 (SE Block)
# ─────────────────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, ch // reduction),
            nn.ReLU(),
            nn.Linear(ch // reduction, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        w = x.mean(dim=2)           # global avg pool -> (B, C)
        w = self.fc(w).unsqueeze(2) # (B, C, 1)
        return x * w


# ─────────────────────────────────────────────
# 主模型: ResNet1D + 多尺度 + SE注意力
# ─────────────────────────────────────────────
class BoltNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # 输入层: 多尺度特征提取
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # 多尺度模块
        self.ms = MultiScaleBlock(64, 128)

        # ResNet 主干
        self.layer1 = nn.Sequential(
            ResBlock1D(128, 128),
            SEBlock(128),
            ResBlock1D(128, 128),
            SEBlock(128),
        )
        self.pool1 = nn.MaxPool1d(2)

        self.layer2 = nn.Sequential(
            ResBlock1D(128, 256, stride=1),
            SEBlock(256),
            ResBlock1D(256, 256, dilation=2),
            SEBlock(256),
        )
        self.pool2 = nn.MaxPool1d(2)

        self.layer3 = nn.Sequential(
            ResBlock1D(256, 512, stride=1),
            SEBlock(512),
            ResBlock1D(512, 512, dilation=2),
            SEBlock(512),
        )

        # 全局池化 (同时用 avg + max，拼接)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.gmp = nn.AdaptiveMaxPool1d(1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.stem(x)        # (B, 64, L/4)
        x = self.ms(x)          # (B, 128, L/4)
        x = self.layer1(x)
        x = self.pool1(x)
        x = self.layer2(x)
        x = self.pool2(x)
        x = self.layer3(x)

        avg = self.gap(x).squeeze(-1)   # (B, 512)
        mx  = self.gmp(x).squeeze(-1)   # (B, 512)
        feat = torch.cat([avg, mx], dim=1)  # (B, 1024)

        return self.classifier(feat)
