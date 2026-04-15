#!/usr/bin/env python3
"""
ODConv K-Fold Cross-Validation Training Script
Author: Taymaz

This script:
- Implements K-Fold cross-validation for comprehensive evaluation
- Trains ODNet with ODConv2d on any dataset (no pre-splitting needed)
- Generates extensive statistics and visualizations for each fold
- Saves aggregated results and best model
- All results organized by dataset name
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Union, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import (
    classification_report, roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    precision_recall_fscore_support, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import multiprocessing


# -------------------------------
# ODConv2d Implementation
# -------------------------------
class ODConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int], str] = 1,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            bias: bool = True,
            K: int = 4,
            reduction: int = 4,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        assert in_channels % groups == 0, "in_channels must be divisible by groups"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kh, kw)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
        self.groups = groups
        self.K = K

        in_per_group = in_channels // groups

        self.weight = nn.Parameter(torch.empty(K, out_channels, in_per_group, kh, kw))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        hidden = max(8, in_channels // reduction)
        self.attention_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
        )
        self.head_K = nn.Linear(hidden, K)
        self.head_o = nn.Linear(hidden, out_channels)
        self.head_i = nn.Linear(hidden, in_per_group)
        self.head_h = nn.Linear(hidden, kh)
        self.head_w = nn.Linear(hidden, kw)
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            nn.init.kaiming_normal_(self.weight[k], mode="fan_out", nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        for m in self.attention_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
        for head in [self.head_K, self.head_o, self.head_i, self.head_h, self.head_w]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _compute_dynamic_weight(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.shape
        h = self.attention_mlp(x)
        aK = F.softmax(self.head_K(h), dim=1)
        aO = torch.sigmoid(self.head_o(h))
        aI = torch.sigmoid(self.head_i(h))
        aH = torch.sigmoid(self.head_h(h))
        aW = torch.sigmoid(self.head_w(h))
        aH = aH / (aH.sum(dim=1, keepdim=True) + 1e-6)
        aW = aW / (aW.sum(dim=1, keepdim=True) + 1e-6)
        alpha = (
                aK[:, :, None, None, None, None]
                * aO[:, None, :, None, None, None]
                * aI[:, None, None, :, None, None]
                * aH[:, None, None, None, :, None]
                * aW[:, None, None, None, None, :]
        )
        weight_dyn = torch.einsum("bkoihw,koihw->boihw", alpha, self.weight)
        return weight_dyn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        in_per_group = self.in_channels // self.groups
        kh, kw = self.kernel_size
        w_dyn = self._compute_dynamic_weight(x)
        if self.groups > 1:
            x = x.view(B, self.groups, in_per_group, H, W)
            x = x.reshape(1, B * self.groups * in_per_group, H, W)
            w_dyn = w_dyn.unsqueeze(2).repeat(1, 1, self.groups, 1, 1, 1)
            w_dyn = w_dyn.reshape(B * self.groups * self.out_channels, in_per_group, kh, kw)
            bias = self.bias.repeat(B * self.groups) if self.bias is not None else None
            out = F.conv2d(
                x,
                w_dyn,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=B * self.groups,
            )
            _, _, Ho, Wo = out.shape
            out = out.view(B, self.groups, self.out_channels, Ho, Wo).sum(dim=1)
        else:
            x = x.reshape(1, B * self.in_channels, H, W)
            w_dyn = w_dyn.reshape(B * self.out_channels, self.in_channels, kh, kw)
            bias = self.bias.repeat(B) if self.bias is not None else None
            out = F.conv2d(
                x,
                w_dyn,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=B,
            )
            _, _, Ho, Wo = out.shape
            out = out.view(B, self.out_channels, Ho, Wo)
        return out


# -------------------------------
# ODConvBlock + ODNet Architecture
# -------------------------------
class ODConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1, K=4, reduction=4, groups=1):
        super().__init__()
        self.conv1 = ODConv2d(cin, cout, 3, stride=stride, padding=1, K=K, reduction=reduction, groups=groups)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = ODConv2d(cout, cout, 3, stride=1, padding=1, K=K, reduction=reduction, groups=groups)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.down = None
        if stride != 1 or cin != cout:
            self.down = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.act(out + identity)
        return out


class ODNet(nn.Module):
    def __init__(self, num_classes=2, width=64, K=4, reduction=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ODConvBlock(width, width, stride=1, K=K, reduction=reduction),
            ODConvBlock(width, width, stride=1, K=K, reduction=reduction),
        )
        self.layer2 = nn.Sequential(
            ODConvBlock(width, width * 2, stride=2, K=K, reduction=reduction),
            ODConvBlock(width * 2, width * 2, stride=1, K=K, reduction=reduction),
        )
        self.layer3 = nn.Sequential(
            ODConvBlock(width * 2, width * 4, stride=2, K=K, reduction=reduction),
            ODConvBlock(width * 4, width * 4, stride=1, K=K, reduction=reduction),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.head(x)
        return x


def train_one_fold(model, train_loader, val_loader, device, epochs, lr, wd):

    """Train model for one fold"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val_acc = 0.0
    best_model_state = None
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    for epoch in range(epochs):
        # Training
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * imgs.size(0)
            total += labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()

        train_acc = 100.0 * correct / total
        train_loss = loss_sum / total

        # Validation
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                total += labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()

        val_acc = 100.0 * correct / total
        sched.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(
                f"    Epoch {epoch + 1:03d}: loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

    return best_model_state, best_val_acc, history


def evaluate_model(model, data_loader, device, class_names):
    """Comprehensive model evaluation"""
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if len(class_names) == 2:
                y_prob.extend(probs[:, 1].cpu().numpy())
            else:
                y_prob.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Calculate comprehensive metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    metrics['precision_per_class'] = precision
    metrics['recall_per_class'] = recall
    metrics['f1_per_class'] = f1
    metrics['support_per_class'] = support

    # Averaged metrics
    metrics['precision_macro'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
    metrics['recall_macro'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
    metrics['f1_macro'] = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]

    metrics['precision_weighted'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[0]
    metrics['recall_weighted'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[1]
    metrics['f1_weighted'] = precision_recall_fscore_support(y_true, y_pred, average='weighted')[2]

    # Additional metrics
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    # ROC and PR curves (for binary classification)
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        metrics['roc_auc'] = auc(fpr, tpr)
        metrics['fpr'] = fpr
        metrics['tpr'] = tpr

        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
        metrics['precision_curve'] = precision_curve
        metrics['recall_curve'] = recall_curve

    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    metrics['y_prob'] = y_prob

    return metrics


def plot_fold_results(fold_idx, history, metrics, class_names, results_dir):
    """Generate comprehensive plots for a single fold"""
    fold_dir = os.path.join(results_dir, f'fold_{fold_idx}')
    os.makedirs(fold_dir, exist_ok=True)

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training Loss - Fold {fold_idx}', fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Accuracy - Fold {fold_idx}', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Confusion Matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = metrics['confusion_matrix']
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title(f'Confusion Matrix (Counts) - Fold {fold_idx}', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title(f'Confusion Matrix (Normalized) - Fold {fold_idx}', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-class metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(class_names))
    width = 0.25

    ax.bar(x - width, metrics['precision_per_class'], width, label='Precision', color='skyblue')
    ax.bar(x, metrics['recall_per_class'], width, label='Recall', color='lightcoral')
    ax.bar(x + width, metrics['f1_per_class'], width, label='F1-Score', color='lightgreen')

    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Metrics - Fold {fold_idx}', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. ROC and PR curves (binary only)
    if len(class_names) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # ROC
        axes[0].plot(metrics['fpr'], metrics['tpr'], 'b-', lw=2,
                     label=f'AUC = {metrics["roc_auc"]:.4f}')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'ROC Curve - Fold {fold_idx}', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # PR
        axes[1].plot(metrics['recall_curve'], metrics['precision_curve'], 'b-', lw=2,
                     label=f'AP = {metrics["avg_precision"]:.4f}')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'Precision-Recall Curve - Fold {fold_idx}', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(fold_dir, 'roc_pr_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_aggregate_results(all_fold_metrics, class_names, results_dir):
    """Generate aggregate plots across all folds"""
    n_folds = len(all_fold_metrics)

    # 1. Accuracy across folds
    fig, ax = plt.subplots(figsize=(10, 6))
    fold_nums = list(range(1, n_folds + 1))
    accuracies = [m['accuracy'] * 100 for m in all_fold_metrics]

    ax.plot(fold_nums, accuracies, 'bo-', markersize=10, linewidth=2)
    ax.axhline(np.mean(accuracies), color='r', linestyle='--', linewidth=2,
               label=f'Mean = {np.mean(accuracies):.2f}%')
    ax.fill_between(fold_nums,
                    np.mean(accuracies) - np.std(accuracies),
                    np.mean(accuracies) + np.std(accuracies),
                    alpha=0.2, color='red')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy Across Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(fold_nums)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'accuracy_across_folds.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Average confusion matrix
    avg_cm = np.mean([m['confusion_matrix'] for m in all_fold_metrics], axis=0)
    std_cm = np.std([m['confusion_matrix'] for m in all_fold_metrics], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='Blues', ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title('Average Confusion Matrix (Counts)', fontweight='bold')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    avg_cm_norm = avg_cm / avg_cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(avg_cm_norm, annot=True, fmt='.2%', cmap='Blues', ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title('Average Confusion Matrix (Normalized)', fontweight='bold')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'average_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Metrics distribution
    metrics_to_plot = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted']
    metric_names = ['Accuracy', 'Balanced Accuracy', 'F1 (Macro)', 'F1 (Weighted)']

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(metrics_to_plot))

    data = [[m[metric] for m in all_fold_metrics] for metric in metrics_to_plot]
    bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True,
                    showmeans=True, meanline=True)

    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')

    ax.set_xticks(positions)
    ax.set_xticklabels(metric_names, rotation=15, ha='right')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Distribution of Metrics Across Folds', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Per-class performance across folds
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(class_names))
    width = 0.25

    avg_precision = np.mean([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    avg_recall = np.mean([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1_per_class'] for m in all_fold_metrics], axis=0)

    std_precision = np.std([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    std_recall = np.std([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    std_f1 = np.std([m['f1_per_class'] for m in all_fold_metrics], axis=0)

    ax.bar(x - width, avg_precision, width, yerr=std_precision, label='Precision',
           color='skyblue', capsize=5)
    ax.bar(x, avg_recall, width, yerr=std_recall, label='Recall',
           color='lightcoral', capsize=5)
    ax.bar(x + width, avg_f1, width, yerr=std_f1, label='F1-Score',
           color='lightgreen', capsize=5)

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Average Per-Class Metrics Across All Folds', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'average_per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 5. ROC curves (binary only)
    if len(class_names) == 2 and 'roc_auc' in all_fold_metrics[0]:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Individual ROC curves
        for i, m in enumerate(all_fold_metrics):
            axes[0].plot(m['fpr'], m['tpr'], alpha=0.3,
                         label=f'Fold {i + 1} (AUC={m["roc_auc"]:.3f})')

        mean_fpr = np.linspace(0, 1, 100)
        tprs = []
        for m in all_fold_metrics:
            tprs.append(np.interp(mean_fpr, m['fpr'], m['tpr']))
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)

        axes[0].plot(mean_fpr, mean_tpr, 'b-', linewidth=3,
                     label=f'Mean (AUC={np.mean([m["roc_auc"] for m in all_fold_metrics]):.3f})')
        axes[0].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                             alpha=0.2, color='blue')
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1)
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curves - All Folds', fontweight='bold')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # Mean ROC only
        axes[1].plot(mean_fpr, mean_tpr, 'b-', linewidth=3,
                     label=f'Mean ROC (AUC={np.mean([m["roc_auc"] for m in all_fold_metrics]):.4f}±{np.std([m["roc_auc"] for m in all_fold_metrics]):.4f})')
        axes[1].fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr,
                             alpha=0.2, color='blue', label='±1 std')
        axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Mean ROC Curve', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'aggregate_roc_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # 6. Average Classification Report as Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Calculate average metrics
    avg_precision = np.mean([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    avg_recall = np.mean([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1_per_class'] for m in all_fold_metrics], axis=0)
    avg_support = np.mean([m['support_per_class'] for m in all_fold_metrics], axis=0)

    std_precision = np.std([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    std_recall = np.std([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    std_f1 = np.std([m['f1_per_class'] for m in all_fold_metrics], axis=0)

    # Create table data
    table_data = []
    table_data.append(['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])

    for i, cls in enumerate(class_names):
        table_data.append([
            cls,
            f'{avg_precision[i]:.4f}±{std_precision[i]:.4f}',
            f'{avg_recall[i]:.4f}±{std_recall[i]:.4f}',
            f'{avg_f1[i]:.4f}±{std_f1[i]:.4f}',
            f'{avg_support[i]:.0f}'
        ])

    # Add macro average
    table_data.append(['', '', '', '', ''])
    table_data.append([
        'Macro Avg',
        f'{np.mean(avg_precision):.4f}±{np.mean(std_precision):.4f}',
        f'{np.mean(avg_recall):.4f}±{np.mean(std_recall):.4f}',
        f'{np.mean(avg_f1):.4f}±{np.mean(std_f1):.4f}',
        f'{np.sum(avg_support):.0f}'
    ])

    # Add weighted average
    total_support = np.sum(avg_support)
    weighted_precision = np.sum(avg_precision * avg_support) / total_support
    weighted_recall = np.sum(avg_recall * avg_support) / total_support
    weighted_f1 = np.sum(avg_f1 * avg_support) / total_support

    table_data.append([
        'Weighted Avg',
        f'{weighted_precision:.4f}±{np.mean(std_precision):.4f}',
        f'{weighted_recall:.4f}±{np.mean(std_recall):.4f}',
        f'{weighted_f1:.4f}±{np.mean(std_f1):.4f}',
        f'{total_support:.0f}'
    ])

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style average rows
    for i in range(5):
        table[(len(class_names) + 2, i)].set_facecolor('#E3F2FD')
        table[(len(class_names) + 3, i)].set_facecolor('#E3F2FD')

    plt.title('Average Classification Report Across All Folds',
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(results_dir, 'average_classification_report.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def save_aggregate_report(all_fold_metrics, class_names, results_dir, dataset_name):
    """Save comprehensive text report"""
    report_path = os.path.join(results_dir, f'{dataset_name}_complete_report.txt')

    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"K-FOLD CROSS-VALIDATION RESULTS - {dataset_name}\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Number of Folds: {len(all_fold_metrics)}\n")
        f.write(f"Classes: {class_names}\n\n")

        # Aggregate statistics
        f.write("=" * 80 + "\n")
        f.write("AGGREGATE STATISTICS ACROSS ALL FOLDS\n")
        f.write("=" * 80 + "\n\n")

        metrics_to_report = {
            'Accuracy': 'accuracy',
            'Balanced Accuracy': 'balanced_accuracy',
            'Precision (Macro)': 'precision_macro',
            'Recall (Macro)': 'recall_macro',
            'F1-Score (Macro)': 'f1_macro',
            'Precision (Weighted)': 'precision_weighted',
            'Recall (Weighted)': 'recall_weighted',
            'F1-Score (Weighted)': 'f1_weighted',
            'Matthews Correlation Coefficient': 'mcc',
            'Cohen\'s Kappa': 'kappa',
        }

        for name, key in metrics_to_report.items():
            values = [m[key] for m in all_fold_metrics]
            f.write(f"{name:40s}: {np.mean(values):.4f} ± {np.std(values):.4f}\n")

        if 'roc_auc' in all_fold_metrics[0]:
            auc_values = [m['roc_auc'] for m in all_fold_metrics]
            f.write(f"{'ROC AUC':40s}: {np.mean(auc_values):.4f} ± {np.std(auc_values):.4f}\n")

            ap_values = [m['avg_precision'] for m in all_fold_metrics]
            f.write(f"{'Average Precision':40s}: {np.mean(ap_values):.4f} ± {np.std(ap_values):.4f}\n")

        # AVERAGE CLASSIFICATION REPORT
        f.write("\n" + "=" * 80 + "\n")
        f.write("AVERAGE CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        # Calculate average metrics per class
        avg_precision = np.mean([m['precision_per_class'] for m in all_fold_metrics], axis=0)
        avg_recall = np.mean([m['recall_per_class'] for m in all_fold_metrics], axis=0)
        avg_f1 = np.mean([m['f1_per_class'] for m in all_fold_metrics], axis=0)
        avg_support = np.mean([m['support_per_class'] for m in all_fold_metrics], axis=0)

        std_precision = np.std([m['precision_per_class'] for m in all_fold_metrics], axis=0)
        std_recall = np.std([m['recall_per_class'] for m in all_fold_metrics], axis=0)
        std_f1 = np.std([m['f1_per_class'] for m in all_fold_metrics], axis=0)

        # Header
        f.write(f"{'':20s} {'Precision':>20s} {'Recall':>20s} {'F1-Score':>20s} {'Support':>12s}\n")
        f.write("-" * 80 + "\n")

        # Per-class results
        for i, cls in enumerate(class_names):
            f.write(f"{cls:20s} ")
            f.write(f"{avg_precision[i]:>8.4f}±{std_precision[i]:<.4f}   ")
            f.write(f"{avg_recall[i]:>8.4f}±{std_recall[i]:<.4f}   ")
            f.write(f"{avg_f1[i]:>8.4f}±{std_f1[i]:<.4f}   ")
            f.write(f"{avg_support[i]:>10.1f}\n")

        f.write("\n")

        # Macro average
        f.write(f"{'Macro Avg':20s} ")
        f.write(f"{np.mean(avg_precision):>8.4f}±{np.mean(std_precision):<.4f}   ")
        f.write(f"{np.mean(avg_recall):>8.4f}±{np.mean(std_recall):<.4f}   ")
        f.write(f"{np.mean(avg_f1):>8.4f}±{np.mean(std_f1):<.4f}   ")
        f.write(f"{np.sum(avg_support):>10.1f}\n")

        # Weighted average
        total_support = np.sum(avg_support)
        weighted_precision = np.sum(avg_precision * avg_support) / total_support
        weighted_recall = np.sum(avg_recall * avg_support) / total_support
        weighted_f1 = np.sum(avg_f1 * avg_support) / total_support

        f.write(f"{'Weighted Avg':20s} ")
        f.write(f"{weighted_precision:>8.4f}±{np.mean(std_precision):<.4f}   ")
        f.write(f"{weighted_recall:>8.4f}±{np.mean(std_recall):<.4f}   ")
        f.write(f"{weighted_f1:>8.4f}±{np.mean(std_f1):<.4f}   ")
        f.write(f"{total_support:>10.1f}\n")

        # Per-class statistics
        f.write("\n" + "=" * 80 + "\n")
        f.write("PER-CLASS STATISTICS (AVERAGED ACROSS FOLDS)\n")
        f.write("=" * 80 + "\n\n")

        for i, cls in enumerate(class_names):
            f.write(f"Class: {cls}\n")
            f.write("-" * 40 + "\n")

            precision_vals = [m['precision_per_class'][i] for m in all_fold_metrics]
            recall_vals = [m['recall_per_class'][i] for m in all_fold_metrics]
            f1_vals = [m['f1_per_class'][i] for m in all_fold_metrics]
            support_vals = [m['support_per_class'][i] for m in all_fold_metrics]

            f.write(f"  Precision: {np.mean(precision_vals):.4f} ± {np.std(precision_vals):.4f}\n")
            f.write(f"  Recall:    {np.mean(recall_vals):.4f} ± {np.std(recall_vals):.4f}\n")
            f.write(f"  F1-Score:  {np.mean(f1_vals):.4f} ± {np.std(f1_vals):.4f}\n")
            f.write(f"  Support:   {np.mean(support_vals):.1f} ± {np.std(support_vals):.1f}\n\n")

        # Individual fold results
        f.write("=" * 80 + "\n")
        f.write("INDIVIDUAL FOLD RESULTS\n")
        f.write("=" * 80 + "\n\n")

        for i, m in enumerate(all_fold_metrics):
            f.write(f"Fold {i + 1}:\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Accuracy:           {m['accuracy']:.4f}\n")
            f.write(f"  Balanced Accuracy:  {m['balanced_accuracy']:.4f}\n")
            f.write(f"  F1 (Macro):         {m['f1_macro']:.4f}\n")
            f.write(f"  F1 (Weighted):      {m['f1_weighted']:.4f}\n")
            if 'roc_auc' in m:
                f.write(f"  ROC AUC:            {m['roc_auc']:.4f}\n")
            f.write("\n")

        # Average confusion matrix
        f.write("=" * 80 + "\n")
        f.write("AVERAGE CONFUSION MATRIX\n")
        f.write("=" * 80 + "\n\n")

        avg_cm = np.mean([m['confusion_matrix'] for m in all_fold_metrics], axis=0)
        std_cm = np.std([m['confusion_matrix'] for m in all_fold_metrics], axis=0)

        f.write("Mean ± Std:\n")
        f.write(f"{'':15s} ")
        for cls in class_names:
            f.write(f"{cls:>15s} ")
        f.write("\n")

        for i, cls in enumerate(class_names):
            f.write(f"{cls:15s} ")
            for j in range(len(class_names)):
                f.write(f"{avg_cm[i, j]:>7.1f}±{std_cm[i, j]:>4.1f}   ")
            f.write("\n")

    print(f"\n✓ Complete report saved: {report_path}")


def train_kfold(data_dir, dataset_name, n_splits=5, epochs=30, batch_size=16,
                lr=3e-4, wd=0.05, width=64, K=4, reduction=4):
    """Main K-Fold training function"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # Setup directories
    results_dir = f"Results_{dataset_name}"
    checkpoint_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data transforms
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load full dataset
    full_dataset = ImageFolder(root=data_dir)
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"Dataset: {dataset_name}")
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"K-Fold splits: {n_splits}\n")

    # K-Fold split
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_fold_metrics = []
    best_overall_acc = 0.0
    best_model_state = None
    best_fold = 0

    print("=" * 80)
    print("STARTING K-FOLD CROSS-VALIDATION")
    print("=" * 80 + "\n")

    for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(full_dataset), 1):
        print(f"\n{'=' * 80}")
        print(f"FOLD {fold_idx}/{n_splits}")
        print(f"{'=' * 80}")
        print(f"Train samples: {len(train_ids)}, Validation samples: {len(val_ids)}\n")

        # Create datasets for this fold
        train_dataset = ImageFolder(root=data_dir, transform=train_tf)
        val_dataset = ImageFolder(root=data_dir, transform=val_tf)

        train_subset = Subset(train_dataset, train_ids)
        val_subset = Subset(val_dataset, val_ids)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)

        # Create and train model
        model = ODNet(num_classes=num_classes, width=width, K=K, reduction=reduction).to(device)

        best_model_state_fold, best_val_acc, history = train_one_fold(
            model, train_loader, val_loader, device, epochs, lr, wd
        )

        # Load best model from this fold
        model.load_state_dict(best_model_state_fold)

        # Evaluate
        fold_metrics = evaluate_model(model, val_loader, device, class_names)
        all_fold_metrics.append(fold_metrics)

        # Plot fold results
        plot_fold_results(fold_idx, history, fold_metrics, class_names, results_dir)

        # Save fold checkpoint
        fold_checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_{dataset_name}.pth')
        torch.save(best_model_state_fold, fold_checkpoint_path)

        # Track best model
        if fold_metrics['accuracy'] > best_overall_acc:
            best_overall_acc = fold_metrics['accuracy']
            best_model_state = best_model_state_fold
            best_fold = fold_idx

        print(f"\n✓ Fold {fold_idx} complete:")
        print(f"  Validation Accuracy: {fold_metrics['accuracy'] * 100:.2f}%")
        print(f"  F1 (Macro): {fold_metrics['f1_macro']:.4f}")
        if 'roc_auc' in fold_metrics:
            print(f"  ROC AUC: {fold_metrics['roc_auc']:.4f}")

    # Save best overall model
    best_model_path = os.path.join(checkpoint_dir, f'Best_{dataset_name}.pth')
    torch.save(best_model_state, best_model_path)

    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nBest model from Fold {best_fold} with accuracy: {best_overall_acc * 100:.2f}%")
    print(f"Best model saved: {best_model_path}\n")

    # Generate aggregate visualizations
    print("Generating aggregate visualizations...")
    plot_aggregate_results(all_fold_metrics, class_names, results_dir)

    # Save comprehensive report
    save_aggregate_report(all_fold_metrics, class_names, results_dir, dataset_name)

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    accuracies = [m['accuracy'] for m in all_fold_metrics]
    f1_scores = [m['f1_macro'] for m in all_fold_metrics]

    print(f"\nAccuracy:        {np.mean(accuracies) * 100:.2f}% ± {np.std(accuracies) * 100:.2f}%")
    print(f"F1 (Macro):      {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

    if 'roc_auc' in all_fold_metrics[0]:
        auc_scores = [m['roc_auc'] for m in all_fold_metrics]
        print(f"ROC AUC:         {np.mean(auc_scores):.4f} ± {np.std(auc_scores):.4f}")

    # Print average classification report
    print("\n" + "=" * 80)
    print("AVERAGE CLASSIFICATION REPORT")
    print("=" * 80)

    avg_precision = np.mean([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    avg_recall = np.mean([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    avg_f1 = np.mean([m['f1_per_class'] for m in all_fold_metrics], axis=0)
    avg_support = np.mean([m['support_per_class'] for m in all_fold_metrics], axis=0)

    std_precision = np.std([m['precision_per_class'] for m in all_fold_metrics], axis=0)
    std_recall = np.std([m['recall_per_class'] for m in all_fold_metrics], axis=0)
    std_f1 = np.std([m['f1_per_class'] for m in all_fold_metrics], axis=0)

    print(f"\n{'':20s} {'Precision':>20s} {'Recall':>20s} {'F1-Score':>20s} {'Support':>12s}")
    print("-" * 80)

    for i, cls in enumerate(class_names):
        print(f"{cls:20s} "
              f"{avg_precision[i]:>8.4f}±{std_precision[i]:<.4f}   "
              f"{avg_recall[i]:>8.4f}±{std_recall[i]:<.4f}   "
              f"{avg_f1[i]:>8.4f}±{std_f1[i]:<.4f}   "
              f"{avg_support[i]:>10.1f}")

    print()
    print(f"{'Macro Avg':20s} "
          f"{np.mean(avg_precision):>8.4f}±{np.mean(std_precision):<.4f}   "
          f"{np.mean(avg_recall):>8.4f}±{np.mean(std_recall):<.4f}   "
          f"{np.mean(avg_f1):>8.4f}±{np.mean(std_f1):<.4f}   "
          f"{np.sum(avg_support):>10.1f}")

    total_support = np.sum(avg_support)
    weighted_precision = np.sum(avg_precision * avg_support) / total_support
    weighted_recall = np.sum(avg_recall * avg_support) / total_support
    weighted_f1 = np.sum(avg_f1 * avg_support) / total_support

    print(f"{'Weighted Avg':20s} "
          f"{weighted_precision:>8.4f}±{np.mean(std_precision):<.4f}   "
          f"{weighted_recall:>8.4f}±{np.mean(std_recall):<.4f}   "
          f"{weighted_f1:>8.4f}±{np.mean(std_f1):<.4f}   "
          f"{total_support:>10.1f}")

    print("\n" + "=" * 80)
    print(f"All results saved in: {results_dir}/")
    print("=" * 80)


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # Configuration
    multiprocessing.freeze_support()
    DATA_DIR = r"d:\Project\dataset\RichaAllByFolder\RichaSK"  # Your dataset path
    DATASET_NAME = "RichaSK"  # Your dataset name

    # Check if path exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ Dataset path not found: {DATA_DIR}")
    else:
        print("=" * 80)
        print("ODConv K-Fold Cross-Validation Training")
        print("=" * 80)
        print(f"✅ Dataset path: {DATA_DIR}")
        print(f"✅ Dataset name: {DATASET_NAME}")
        print("=" * 80 + "\n")

        # Run K-Fold training
        train_kfold(
            data_dir=DATA_DIR,
            dataset_name=DATASET_NAME,
            n_splits=20,  # Number of folds
            epochs=100,  # Epochs per fold
            batch_size=16,  # Batch size
            lr=3e-4,  # Learning rate
            wd=0.05,  # Weight decay
            width=64,  # Model width
            K=4,  # ODConv K parameter
            reduction=4  # ODConv reduction parameter
        )