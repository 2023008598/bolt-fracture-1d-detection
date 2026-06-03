import os
import numpy as np
import torch
from torch.utils.data import Dataset


class BoltDataset(Dataset):
    """
    螺栓断裂检测数据集
    Yes (正常) -> label 0
    No  (断裂) -> label 1
    """

    def __init__(self, yes_dir, no_dir, signal_length=7000, augment=False):
        self.signal_length = signal_length
        self.augment = augment
        self.data = []
        self.labels = []

        self._load_dir(yes_dir, label=0)
        self._load_dir(no_dir, label=1)

        self.data = np.array(self.data, dtype=np.float32)   # (N, L)
        self.labels = np.array(self.labels, dtype=np.int64)

        print(f"数据集加载完成: 共 {len(self.data)} 个样本, 信号长度 {self.data.shape[1]}")
        print(f"  正常(Yes): {(self.labels==0).sum()}  断裂(No): {(self.labels==1).sum()}")

    def _load_dir(self, directory, label):
        files = sorted(os.listdir(directory))
        txt_files = [f for f in files if f.endswith('.txt')]
        print(f"  加载 {'Yes' if label==0 else 'No'} ({len(txt_files)} 个文件)...")
        for i, fname in enumerate(txt_files):
            path = os.path.join(directory, fname)
            signal = self._load_signal(path)
            self.data.append(signal)
            self.labels.append(label)
            if (i + 1) % 500 == 0:
                print(f"    {i+1}/{len(txt_files)}")

    def _load_signal(self, path):
        # 比 np.loadtxt 快 5-10x
        signal = np.fromiter(
            (float(line.split()[1]) for line in open(path) if line.strip()),
            dtype=np.float32
        )

        # 统一长度
        if len(signal) >= self.signal_length:
            signal = signal[:self.signal_length]
        else:
            signal = np.pad(signal, (0, self.signal_length - len(signal)), mode='edge')

        # Z-score 标准化
        mu, sigma = signal.mean(), signal.std()
        signal = (signal - mu) / (sigma + 1e-8)

        return signal.astype(np.float32)

    def _augment(self, signal):
        """训练时数据增强"""
        # 随机加高斯噪声
        # 注释掉高斯噪声，因为它会影响模型对微小断裂波形特征的判断
        #if np.random.rand() < 0.5:
        #    signal = signal + np.random.normal(0, 0.02, signal.shape).astype(np.float32)
        # 随机幅度缩放
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            signal = signal * scale
        # 随机时间偏移
        if np.random.rand() < 0.5:
            shift = np.random.randint(-200, 200)
            signal = np.roll(signal, shift)
        return signal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal = self.data[idx].copy()
        label = self.labels[idx]

        if self.augment:
            signal = self._augment(signal)

        # shape: (1, L) — 单通道一维信号
        x = torch.from_numpy(signal).unsqueeze(0)
        y = torch.tensor(label, dtype=torch.long)
        return x, y
