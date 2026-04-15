#!/usr/bin/env python3
"""
Omni-Dimensional Dynamic Convolution (ODConv2d) for Image Classification
Author: Taymaz (updated with separate train/test folders)

This script:
- Implements simplified ODConv2d (Omni-Dimensional Dynamic Convolution)
- Builds ODNet CNN architecture
- Trains from separate train and test folders
- Automatically splits train data into train/val sets
- Saves the best model checkpoint
- Generates comprehensive classification reports and plots
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from typing import Union, Tuple

from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve, \
    average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


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


# -------------------------------
# Training Function (Separate Train/Test Folders)
# -------------------------------
def train_imagefolder(train_dir, test_dir, epochs=20, batch_size=16, lr=3e-4, wd=0.05, width=64, K=4, reduction=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data augmentation for training
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # No augmentation for validation/test
    test_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load train and test datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_tf)
    test_dataset = ImageFolder(root=test_dir, transform=test_tf)

    # Split train into train + validation (85% train, 15% val)
    n_train_total = len(train_dataset)
    n_train = int(0.85 * n_train_total)
    n_val = n_train_total - n_train

    trainset, valset = random_split(train_dataset, [n_train, n_val])

    # Update validation set to use test transforms
    valset.dataset = ImageFolder(root=train_dir, transform=test_tf)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ODNet(num_classes=len(train_dataset.classes), width=width, K=K, reduction=reduction).to(device)
    class_names = train_dataset.classes
    print(f"Detected classes: {class_names}")
    print(f"Training samples: {n_train}, Validation samples: {n_val}, Test samples: {len(test_dataset)}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = 0.0
    os.makedirs("results_old/checkpoints", exist_ok=True)

    train_losses = []
    train_accs = []
    val_accs = []

    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60)

    for epoch in range(epochs):
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

        # validation
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

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"results_old/checkpoints/best_odconv_idr0042.pth")
            print(
                f"Epoch {epoch + 1:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}% ⭐ NEW BEST")
        else:
            print(
                f"Epoch {epoch + 1:03d}: train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, val_acc={val_acc:.2f}%")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best validation accuracy: {best_acc:.2f}%")
    print("=" * 60)
    print("Model saved to checkpoints/best_odconv_idr0042.pth\n")

    # -------------------------------
    # Load best model for testing
    # -------------------------------
    model.load_state_dict(torch.load("results_old/checkpoints/best_odconv_idr0042.pth"))
    model.eval()

    y_true, y_pred, y_prob = [], [], []

    print("Evaluating on test set...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            if len(class_names) == 2:
                y_prob.extend(probs[:, 1].cpu().numpy())  # for binary classification
            else:
                y_prob.extend(probs.cpu().numpy())  # for multi-class

    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Create results directory
    results_dir = "results_IDR0042"
    os.makedirs(results_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. TRAINING CURVES
    # -------------------------------------------------------------
    print("Generating training curves...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(range(1, epochs + 1), train_losses, 'b-', linewidth=2, label='Training Loss')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training Loss over Epochs', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Accuracy curve
    axes[1].plot(range(1, epochs + 1), train_accs, 'b-', linewidth=2, label='Training Accuracy')
    axes[1].plot(range(1, epochs + 1), val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{results_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 2. CLASSIFICATION REPORT
    # -------------------------------------------------------------
    print("Generating classification report...")
    text_report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Test Set)")
    print("=" * 60)
    print(text_report)

    # -------------------------------------------------------------
    # 3. CONFUSION MATRIX
    # -------------------------------------------------------------
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = confusion_matrix(y_true, y_pred, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_xlabel('Predicted Labels', fontsize=12)
    axes[0].set_ylabel('True Labels', fontsize=12)
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_xlabel('Predicted Labels', fontsize=12)
    axes[1].set_ylabel('True Labels', fontsize=12)
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 4. ROC CURVE & AUC (for binary classification)
    # -------------------------------------------------------------
    if len(class_names) == 2:
        print("Generating ROC curve...")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

        # -------------------------------------------------------------
        # 5. PRECISION-RECALL CURVE
        # -------------------------------------------------------------
        print("Generating precision-recall curve...")
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR Curve (AP = {avg_precision:.4f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{results_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    # -------------------------------------------------------------
    # 6. PER-CLASS METRICS BAR CHART
    # -------------------------------------------------------------
    print("Generating per-class metrics...")
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')

    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f"{results_dir}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.close()

    # -------------------------------------------------------------
    # 7. SAVE ALL METRICS TO TEXT FILE
    # -------------------------------------------------------------
    print("Saving metrics to file...")
    with open(f"{results_dir}/classification_report.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("ODConv Classification Report - IDR0042 Dataset\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Model: ODNet (ODConv2d)\n")
        f.write(f"Classes: {class_names}\n")
        f.write(f"Training samples: {n_train}\n")
        f.write(f"Validation samples: {n_val}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Best validation accuracy: {best_acc:.2f}%\n\n")

        f.write("=" * 60 + "\n")
        f.write("CLASSIFICATION REPORT (Test Set)\n")
        f.write("=" * 60 + "\n")
        f.write(text_report)

        if len(class_names) == 2:
            f.write(f"\n\nROC AUC Score: {roc_auc:.4f}\n")
            f.write(f"Average Precision Score: {avg_precision:.4f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("CONFUSION MATRIX\n")
        f.write("=" * 60 + "\n")
        f.write(f"{cm}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("=" * 60 + "\n")
        for i, cls in enumerate(class_names):
            f.write(f"{cls}:\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-Score: {f1[i]:.4f}\n")
            f.write(f"  Support: {support[i]}\n\n")

    # -------------------------------------------------------------
    # 8. SUMMARY VISUALIZATION
    # -------------------------------------------------------------
    print("Generating summary visualization...")
    test_accuracy = 100.0 * np.sum(y_true == y_pred) / len(y_true)

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle('ODConv Model Performance Summary - IDR0042 Dataset',
                 fontsize=16, fontweight='bold', y=0.98)

    # Training curves
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(range(1, epochs + 1), train_accs, 'b-', linewidth=2, label='Training', marker='o', markersize=3)
    ax1.plot(range(1, epochs + 1), val_accs, 'r-', linewidth=2, label='Validation', marker='s', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Confusion matrix
    ax2 = fig.add_subplot(gs[1, 0])
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', cbar=True, ax=ax2,
                xticklabels=class_names, yticklabels=class_names)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Confusion Matrix')

    # Per-class metrics
    ax3 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(class_names))
    width = 0.25
    ax3.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax3.bar(x, recall, width, label='Recall', color='lightcoral')
    ax3.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    ax3.set_ylabel('Score')
    ax3.set_title('Per-Class Metrics')
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names)
    ax3.legend()
    ax3.set_ylim([0, 1.05])
    ax3.grid(True, alpha=0.3, axis='y')

    # ROC curve (if binary)
    if len(class_names) == 2:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.4f}')
        ax4.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        ax4.set_xlabel('False Positive Rate')
        ax4.set_ylabel('True Positive Rate')
        ax4.set_title('ROC Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(recall, precision, color='blue', lw=2, label=f'AP = {avg_precision:.4f}')
        ax5.set_xlabel('Recall')
        ax5.set_ylabel('Precision')
        ax5.set_title('Precision-Recall Curve')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    else:
        # Summary statistics for multi-class
        ax4 = fig.add_subplot(gs[2, :])
        metrics_text = f"Test Accuracy: {test_accuracy:.2f}%\n"
        metrics_text += f"Macro Avg F1: {np.mean(f1):.4f}\n"
        metrics_text += f"Weighted Avg F1: {np.average(f1, weights=support):.4f}"
        ax4.text(0.5, 0.5, metrics_text, ha='center', va='center',
                 fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')

    plt.savefig(f"{results_dir}/summary_visualization.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "=" * 60)
    print("Results saved successfully!")
    print("=" * 60)
    print(f"Results directory: {results_dir}/")
    print("Generated files:")
    print(f"  • training_curves.png")
    print(f"  • confusion_matrix.png")
    print(f"  • per_class_metrics.png")
    if len(class_names) == 2:
        print(f"  • roc_curve.png")
        print(f"  • precision_recall_curve.png")
    print(f"  • summary_visualization.png")
    print(f"  • classification_report.txt")
    print(f"\nTest Accuracy: {test_accuracy:.2f}%")
    print("=" * 60 + "\n")


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    base_path = "d://Project/dataset/idr0042/By_Taymaz"
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training dataset path not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset path not found: {test_path}")

    print("=" * 60)
    print("ODConv Training Script - IDR0042 Dataset")
    print("=" * 60)
    print(f"✅ Training dataset path found: {train_path}")
    print(f"✅ Test dataset path found: {test_path}")
    print("=" * 60 + "\n")

    train_imagefolder(
        train_dir=train_path,
        test_dir=test_path,
        epochs=50,
        batch_size=32,
        lr=3e-4,
        wd=0.05,
        width=64,
        K=4,
        reduction=4
    )