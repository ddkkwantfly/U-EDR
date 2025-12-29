# train_multiple_enn.py
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma
from torch.utils.data import DataLoader, TensorDataset

from data_loader import (
    SSVEPConfig,
    list_mat_files,
    split_subjects,
    build_multi_subject_arrays,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# EEGNet with Dirichlet Evidence Head
# ============================================================
class EEGNetENN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        F1: int = 4,
        D: int = 1,
        dropout: float = 0.5,
        kernel_length: int = 32,
    ):
        super().__init__()
        F2 = F1 * D

        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length),
            padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16),
            padding=(0, 8), groups=F1 * D, bias=False
        )
        self.sep_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat_dim = self.forward_features(dummy).shape[1]

        self.evidence = nn.Linear(feat_dim, n_classes)

    def forward_features(self, x):
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = F.elu(self.bn3(self.sep_point(self.sep_depth(x))))
        x = self.pool2(x)
        x = self.drop2(x)
        return x.flatten(1)

    def forward(self, x):
        feat = self.forward_features(x)
        evidence = F.softplus(self.evidence(feat))
        alpha = evidence + 1.0
        return alpha


# ============================================================
# EDL / ENN Loss
# ============================================================
def edl_loss(alpha, target, kl_weight=0.001):
    """
    alpha: [B, K]
    target: [B]
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)

    y = F.one_hot(target, num_classes=K).float()

    # expected cross-entropy
    ece = torch.sum(
        y * (digamma(S) - digamma(alpha)),
        dim=1
    )

    # KL to uniform Dirichlet
    beta = torch.ones_like(alpha)
    kl = (
        torch.lgamma(S)
        - torch.sum(torch.lgamma(alpha), dim=1)
        - torch.lgamma(torch.tensor(K, device=alpha.device))
        + torch.sum((alpha - beta) * (digamma(alpha) - digamma(S)), dim=1)
    )

    return torch.mean(ece + kl_weight * kl)


# ============================================================
# Training utilities
# ============================================================
@torch.no_grad()
def eval_ic_acc(model, loader):
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        alpha = model(xb)
        pred = torch.argmax(alpha, dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / total


# ============================================================
# Main
# ============================================================
@dataclass
class Args:
    mat_dir: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\mat"
    save_path: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\checkpoints\eegnet_enn_best.pt"

    id_classes: Tuple[int, ...] = (0,1,2,3,4,5,6,7)

    fs: int = 256
    onset_1idx: int = 39
    start_s: float = 1.0
    win_s: float = 2.0

    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    patience: int = 25

    kl_weight: float = 0.001
    seed: int = 0


def main(a: Args):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), 7, 1, 2, seed=a.seed)

    cfg = SSVEPConfig(
        fs=a.fs,
        onset_1idx=a.onset_1idx,
        start_s=a.start_s,
        win_s=a.win_s,
    )

    # -------------------------
    # Build datasets
    # -------------------------
    Xtr, ytr, _ = build_multi_subject_arrays(
        [mats[i] for i in splits["train"]],
        cfg, split_name="train"
    )
    Xva, yva, _ = build_multi_subject_arrays(
        [mats[i] for i in splits["val"]],
        cfg, split_name="val"
    )

    id_map = {c: i for i, c in enumerate(a.id_classes)}

    def filter_id(X, y):
        m = np.isin(y, a.id_classes)
        X = X[m]
        y = np.array([id_map[int(t)] for t in y[m]], dtype=np.int64)
        return X, y

    Xtr, ytr = filter_id(Xtr, ytr)
    Xva, yva = filter_id(Xva, yva)

    # to torch
    Xtr = torch.from_numpy(Xtr).float().unsqueeze(1)
    ytr = torch.from_numpy(ytr)
    Xva = torch.from_numpy(Xva).float().unsqueeze(1)
    yva = torch.from_numpy(yva)

    tr_loader = DataLoader(
        TensorDataset(Xtr, ytr),
        batch_size=a.batch_size,
        shuffle=True
    )
    va_loader = DataLoader(
        TensorDataset(Xva, yva),
        batch_size=a.batch_size,
        shuffle=False
    )

    # -------------------------
    # Model
    # -------------------------
    model = EEGNetENN(
        n_channels=Xtr.shape[2],
        n_samples=Xtr.shape[3],
        n_classes=len(a.id_classes),
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best_val = 0.0
    patience = 0

    # -------------------------
    # Training loop
    # -------------------------
    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            opt.zero_grad()
            alpha = model(xb)
            loss = edl_loss(alpha, yb, kl_weight=a.kl_weight)
            loss.backward()
            opt.step()

        val_acc = eval_ic_acc(model, va_loader)

        print(f"Epoch {ep:03d} | val acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            patience = 0
            torch.save(model.state_dict(), a.save_path)
        else:
            patience += 1
            if patience >= a.patience:
                print(f"Early stop at epoch {ep}")
                break

    print(f"Best val acc: {best_val:.3f}")
    print(f"Saved ENN checkpoint to: {a.save_path}")


if __name__ == "__main__":
    main(Args())
