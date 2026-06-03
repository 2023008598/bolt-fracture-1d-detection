"""
单文件预测脚本
用法: python predict.py <txt文件路径>
"""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model import BoltNet

SIGNAL_LENGTH = 7000
MODEL_PATH    = os.path.join(os.path.dirname(__file__), 'best_model.pth')
LABELS        = {0: '正常 (Yes)', 1: '断裂 (No)'}


def load_signal(path, length=SIGNAL_LENGTH):
    raw    = np.loadtxt(path)
    signal = raw[:, 1].astype(np.float32)
    if len(signal) >= length:
        signal = signal[:length]
    else:
        signal = np.pad(signal, (0, length - len(signal)), mode='edge')
    mu, sigma = signal.mean(), signal.std()
    signal = (signal - mu) / (sigma + 1e-8)
    return signal


def predict(file_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BoltNet(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    signal = load_signal(file_path)
    x = torch.from_numpy(signal).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,L)

    with torch.no_grad():
        out  = model(x)
        prob = torch.softmax(out, dim=1)[0]
        pred = out.argmax(dim=1).item()

    print(f"文件: {file_path}")
    print(f"预测结果: {LABELS[pred]}")
    print(f"置信度 — 正常: {prob[0]*100:.1f}%  断裂: {prob[1]*100:.1f}%")
    return pred


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("用法: python predict.py <txt文件路径>")
        sys.exit(1)
    predict(sys.argv[1])
