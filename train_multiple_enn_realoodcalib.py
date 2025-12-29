from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Literal

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
# Model: EEGNet + evidence head
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
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
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
        evidence = F.softplus(self.evidence(feat))  # >=0
        alpha = evidence + 1.0
        return alpha


# ============================================================
# Losses
# ============================================================
def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1))
    alpha: [B,K]
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    beta = torch.ones_like(alpha)

    # NOTE: torch.lgamma(torch.tensor(K)) is scalar; keep on device
    lgammaK = torch.lgamma(torch.tensor(float(K), device=alpha.device))

    kl = (
        torch.lgamma(S)
        - torch.sum(torch.lgamma(alpha), dim=1)
        - lgammaK
        + torch.sum((alpha - beta) * (digamma(alpha) - digamma(S)), dim=1)
    )
    return kl.mean()


def edl_id_loss(alpha: torch.Tensor, target: torch.Tensor, kl_weight: float = 1e-3) -> torch.Tensor:
    """
    ID classification loss: expected CE + kl_weight * KL(Dir(alpha)||Dir(1))
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, num_classes=K).float()

    ece = torch.sum(y * (digamma(S) - digamma(alpha)), dim=1)  # expected CE
    kl = dirichlet_kl_to_uniform(alpha)
    return ece.mean() + kl_weight * kl


def ood_low_evidence_loss(alpha: torch.Tensor, mode: Literal["kl", "smean"] = "kl") -> torch.Tensor:
    """
    OOD calibration loss: force low evidence.
      - "kl": pull Dir(alpha) toward uniform prior Dir(1)
      - "smean": minimize S = sum(alpha)
    """
    if mode == "kl":
        return dirichlet_kl_to_uniform(alpha)
    if mode == "smean":
        return alpha.sum(dim=1).mean()
    raise ValueError(mode)


# ============================================================
# Metrics
# ============================================================
@torch.no_grad()
def ic_acc(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        alpha = model(xb)
        pred = torch.argmax(alpha, dim=1)
        correct += (pred == yb).sum().item()
        total += len(yb)
    return correct / max(total, 1)


# ============================================================
# Config
# ============================================================
@dataclass
class Args:
    mat_dir: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\mat"
    save_path: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\checkpoints\eegnet_enn_realoodcalib_best.pt"

    # Train on these as ID classes (IC commands)
    id_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    # Use these as REAL OOD for evidence collapse calibration (optional)
    use_real_ood: bool = True
    real_ood_classes: Tuple[int, ...] = (8, 9, 10, 11)

    # window / preprocess
    fs: int = 256
    onset_1idx: int = 39
    start_s: float = 1.0
    win_s: float = 2.0

    # train setup
    seed: int = 0
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    patience: int = 25

    # EDL
    kl_weight_id: float = 1e-3

    # real OOD calibration
    ood_ratio: float = 0.20      # fraction of each batch used for OOD calib (sampled from OOD pool)
    lambda_ood: float = 0.5
    ood_loss_mode: str = "kl"    # "kl" or "smean"

    # EEGNet params
    F1: int = 4
    D: int = 1
    dropout: float = 0.5
    kernel_length: int = 32


def main(a: Args):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), 7, 1, 2, seed=a.seed)

    cfg = SSVEPConfig(fs=a.fs, onset_1idx=a.onset_1idx, start_s=a.start_s, win_s=a.win_s)

    # Build arrays (raw labels are original class indices)
    Xtr_all, ytr_all, _ = build_multi_subject_arrays([mats[i] for i in splits["train"]], cfg, split_name="train")
    Xva_all, yva_all, _ = build_multi_subject_arrays([mats[i] for i in splits["val"]], cfg, split_name="val")
    Xte_all, yte_all, _ = build_multi_subject_arrays([mats[i] for i in splits["test"]], cfg, split_name="test")

    # Map ID classes to [0..K-1]
    id_map = {c: i for i, c in enumerate(a.id_classes)}

    def filter_id(X, y):
        m = np.isin(y, a.id_classes)
        X = X[m]
        y = np.array([id_map[int(t)] for t in y[m]], dtype=np.int64)
        return X, y

    def filter_ood(X, y):
        m = np.isin(y, a.real_ood_classes)
        return X[m], y[m]

    # ID train/val/test
    Xtr, ytr = filter_id(Xtr_all, ytr_all)
    Xva, yva = filter_id(Xva_all, yva_all)
    Xte, yte = filter_id(Xte_all, yte_all)

    # Optional REAL OOD pool from TRAIN split only
    if a.use_real_ood:
        Xood_pool, _ = filter_ood(Xtr_all, ytr_all)
        if len(Xood_pool) == 0:
            raise RuntimeError("use_real_ood=True but no OOD samples found in TRAIN split. Check real_ood_classes.")
    else:
        Xood_pool = None

    # to torch [N,1,C,T]
    Xtr_t = torch.from_numpy(Xtr).float().unsqueeze(1)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).float().unsqueeze(1)
    yva_t = torch.from_numpy(yva)
    Xte_t = torch.from_numpy(Xte).float().unsqueeze(1)
    yte_t = torch.from_numpy(yte)

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=a.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=a.batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=a.batch_size, shuffle=False)

    # OOD pool loader (sample batches)
    if a.use_real_ood:
        Xood_t = torch.from_numpy(Xood_pool).float().unsqueeze(1)
        ood_loader = DataLoader(TensorDataset(Xood_t), batch_size=a.batch_size, shuffle=True, drop_last=True)
        ood_iter = iter(ood_loader)

    # Model
    model = EEGNetENN(
        n_channels=Xtr.shape[1],
        n_samples=Xtr.shape[2],
        n_classes=len(a.id_classes),
        F1=a.F1, D=a.D, dropout=a.dropout, kernel_length=a.kernel_length
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    best_val = 0.0
    bad = 0

    for ep in range(1, a.epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()

            # ID loss
            alpha_id = model(xb)
            loss_id = edl_id_loss(alpha_id, yb, kl_weight=a.kl_weight_id)

            # REAL OOD calibration loss
            if a.use_real_ood and a.ood_ratio > 0:
                try:
                    (xood_batch,) = next(ood_iter)
                except StopIteration:
                    ood_iter = iter(ood_loader)
                    (xood_batch,) = next(ood_iter)

                xood_batch = xood_batch.to(DEVICE)
                m = int(round(xb.size(0) * a.ood_ratio))
                if m > 0:
                    xood = xood_batch[:m]
                    alpha_ood = model(xood)
                    loss_ood = ood_low_evidence_loss(alpha_ood, mode=a.ood_loss_mode)  # force collapse
                    loss = loss_id + a.lambda_ood * loss_ood
                else:
                    loss_ood = torch.tensor(0.0, device=DEVICE)
                    loss = loss_id
            else:
                loss_ood = torch.tensor(0.0, device=DEVICE)
                loss = loss_id

            loss.backward()
            opt.step()

        val_acc = ic_acc(model, va_loader)

        if ep == 1 or ep % 10 == 0:
            tr_acc = ic_acc(model, tr_loader)
            te_acc = ic_acc(model, te_loader)
            print(
                f"Epoch {ep:03d} | "
                f"train acc {tr_acc:.3f} | val acc {val_acc:.3f} | test(ID) acc {te_acc:.3f} | "
                f"use_real_ood {a.use_real_ood} ood_ratio {a.ood_ratio:.2f} lambda_ood {a.lambda_ood:.2f} mode {a.ood_loss_mode}"
            )
        else:
            print(f"Epoch {ep:03d} | val acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save(model.state_dict(), a.save_path)
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stop at epoch {ep} (best val {best_val:.3f})")
                break

    print(f"Best val acc: {best_val:.3f}")
    print(f"Saved checkpoint to: {a.save_path}")


if __name__ == "__main__":
    args = Args()
    main(args)
