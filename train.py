"""
螺栓断裂检测 — 训练脚本
用法: python train.py
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
# 新增：绘图和评估所需库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import numpy as np
# 确保能 import 同目录模块
sys.path.insert(0, os.path.dirname(__file__))
from dataset import BoltDataset
from model import BoltNet

# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────
YES_DIR       = r"F:\shiyan1-bolt_detection\data\Yes"
NO_DIR        = r"F:\shiyan1-bolt_detection\data\No"
SAVE_PATH     = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model.pth')
SIGNAL_LENGTH = 7000
BATCH_SIZE    = 32
EPOCHS        = 50
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
TRAIN_RATIO   = 0.8
# 新增：绘图字体设置（解决中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 新增：创建结果保存目录
os.makedirs('experiment_results', exist_ok=True)

# ─────────────────────────────────────────────
# 数据
# ─────────────────────────────────────────────
print("=" * 50)
print("加载数据集...")
full_dataset = BoltDataset(YES_DIR, NO_DIR, signal_length=SIGNAL_LENGTH, augment=False)
n_train = int(TRAIN_RATIO * len(full_dataset))
n_test  = len(full_dataset) - n_train
train_set, test_set = random_split(
    full_dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(42)
)
# 训练集开启增强
train_set.dataset.augment = False   # random_split 共享 dataset，用 wrapper 更安全
# 用 Subset + 自定义 collate 或直接在 Dataset 里按 idx 判断
# 简单做法: 单独建一个 augment=True 的训练集
train_dataset = BoltDataset(YES_DIR, NO_DIR, signal_length=SIGNAL_LENGTH, augment=True)
# 用相同的 indices 切分
train_dataset_sub = torch.utils.data.Subset(train_dataset, train_set.indices)
train_loader = DataLoader(train_dataset_sub, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_set, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=True)
print(f"训练集: {len(train_dataset_sub)}  测试集: {len(test_set)}")

# ─────────────────────────────────────────────
# 模型 / 优化器
# ─────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
model     = BoltNet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
# 余弦退火学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ─────────────────────────────────────────────
# 训练循环
# ─────────────────────────────────────────────
best_acc = 0.0
# 新增：记录训练过程的真实数据
train_loss_list = []
train_acc_list = []
val_acc_list = []
lr_list = []
print("=" * 50)
print("开始训练...")
print("=" * 50)

for epoch in range(1, EPOCHS + 1):
    # ── 训练 ──
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(x)
        loss = criterion(out, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += x.size(0)
    train_loss = total_loss / total
    train_acc  = 100.0 * correct / total
    # 新增：保存每轮训练的真实数据
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    lr_list.append(scheduler.get_last_lr()[0])

    # ── 验证 ──
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out  = model(x)
            pred = out.argmax(dim=1)
            val_correct += (pred == y).sum().item()
            val_total   += x.size(0)
    val_acc = 100.0 * val_correct / val_total
    # 新增：保存每轮验证的真实准确率
    val_acc_list.append(val_acc)

    scheduler.step()
    print(f"Epoch [{epoch:3d}/{EPOCHS}]  "
          f"Loss: {train_loss:.4f}  "
          f"Train Acc: {train_acc:.2f}%  "
          f"Val Acc: {val_acc:.2f}%  "
          f"LR: {scheduler.get_last_lr()[0]:.6f}")
    # 保存最优模型
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ 保存最优模型 (Val Acc: {best_acc:.2f}%)")

# ─────────────────────────────────────────────
# 新增核心部分：训练完成后，用真实数据生成所有结果图+指标文件
# ─────────────────────────────────────────────
def generate_all_plots():
    print("=" * 50)
    print("开始生成真实实验结果图...")
    # 加载最优模型做最终评估
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()
    y_true = []
    y_pred = []
    y_score = []  # 用于ROC曲线的预测概率
    val_loss_list = []  # 计算验证集损失

    # 遍历测试集，获取真实标签、预测标签、预测概率、验证损失
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss_list.append(loss.item() * x.size(0))
            # 保存标签和概率
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            y_score.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    # 计算验证集总损失
    val_loss = sum(val_loss_list) / len(test_set)

    # 图1：训练/验证损失+准确率曲线（双子图）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)
    epochs_range = np.arange(1, EPOCHS+1)
    # 损失曲线
    ax1.plot(epochs_range, train_loss_list, label='训练损失', color='blue', linewidth=1.5)
    ax1.axhline(y=val_loss, label='最终验证损失', color='red', linestyle='--', linewidth=1.5)
    ax1.set_title('训练损失曲线', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.legend()
    ax1.grid(alpha=0.3)
    # 准确率曲线
    ax2.plot(epochs_range, train_acc_list, label='训练准确率', color='blue', linewidth=1.5)
    ax2.plot(epochs_range, val_acc_list, label='验证准确率', color='orange', linewidth=1.5)
    ax2.axhline(y=best_acc, label=f'最优验证准确率({best_acc:.2f}%)', color='red', linestyle='--', linewidth=1.5)
    ax2.set_title('训练/验证准确率曲线', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('准确率 (%)', fontsize=12)
    ax2.legend()
    ax2.grid(alpha=0.3)
    plt.savefig(os.path.join('experiment_results', '【真实】训练曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 图2：混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5), tight_layout=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常(Yes)', '断裂(No)'],
                yticklabels=['正常(Yes)', '断裂(No)'])
    plt.title('混淆矩阵', fontsize=14)
    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.savefig(os.path.join('experiment_results', '【真实】混淆矩阵.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 图3：ROC曲线（带AUC值）
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6), tight_layout=True)
    plt.plot(fpr, tpr, color='darkorange', linewidth=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linewidth=2, linestyle='--')
    plt.title('ROC曲线', fontsize=14)
    plt.xlabel('假正例率(FPR)', fontsize=12)
    plt.ylabel('真正例率(TPR)', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join('experiment_results', '【真实】ROC曲线.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 图4：真实样本波形对比图（从测试集取正常/断裂各1个）
    # 找到测试集中正常和断裂的样本索引
    normal_idx = [i for i, lab in enumerate(y_true) if lab == 0][0]
    broken_idx = [i for i, lab in enumerate(y_true) if lab == 1][0]
    # 获取原始信号
    normal_sig = test_set[normal_idx][0].squeeze().cpu().numpy()
    broken_sig = test_set[broken_idx][0].squeeze().cpu().numpy()
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), tight_layout=True, sharex=True)
    ax1.plot(normal_sig, color='green', linewidth=0.7)
    ax1.set_title('正常螺栓信号（测试集真实样本）', fontsize=14)
    ax1.set_ylabel('标准化幅值', fontsize=12)
    ax1.grid(alpha=0.3)
    ax2.plot(broken_sig, color='red', linewidth=0.7)
    ax2.set_title('断裂螺栓信号（测试集真实样本）', fontsize=14)
    ax2.set_xlabel('时间步（共7000点）', fontsize=12)
    ax2.set_ylabel('标准化幅值', fontsize=12)
    ax2.grid(alpha=0.3)
    plt.savefig(os.path.join('experiment_results', '【真实】螺栓信号对比图.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 生成分类报告（精确率、召回率、F1-score）
    cls_report = classification_report(y_true, y_pred, target_names=['正常(Yes)', '断裂(No)'], digits=4)
    with open(os.path.join('experiment_results', '【真实】分类报告.txt'), 'w', encoding='utf-8') as f:
        f.write(f"最优验证准确率：{best_acc:.4f}%\n")
        f.write(f"最终验证损失：{val_loss:.4f}\n")
        f.write("="*50 + "\n")
        f.write("分类报告\n")
        f.write("="*50 + "\n")
        f.write(cls_report)

    # 打印生成结果提示
    print("✅ 所有真实结果图已生成，保存至：experiment_results 目录")
    print(f"📊 最优验证准确率：{best_acc:.2f}%")
    print("📁 生成文件包含：训练曲线、混淆矩阵、ROC曲线、信号对比图、分类报告")

# 调用绘图函数
generate_all_plots()

# 原有训练完成提示
print("=" * 50)
print(f"训练完成！最优验证准确率: {best_acc:.2f}%")
print(f"模型已保存至: {SAVE_PATH}")