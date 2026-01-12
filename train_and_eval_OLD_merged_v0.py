from __future__ import annotations
"""
Merged (single-file) reproduction of your **OLD** pipeline:
- Train: EEGNetENN + EDL ID loss + (optional) REAL-OOD evidence-collapse calibration sampled from TRAIN split only.
- Eval: ID accuracy on (test+val) by default; OOD set from TEST only; scores: evidence_ood / maha_min / fusion_geo.

This file is a strict merge of:
  - your old training script
  - your old eval_id_acc_and_ood.py script

No protocol changes are introduced; only refactoring into one file.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import digamma
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

from data_loader import (
    SSVEPConfig,
    list_mat_files,
    split_subjects,
    build_multi_subject_arrays,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model: EEGNet + evidence head (MUST match between train/eval)
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
# Losses (EDL)
# ============================================================
def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """
    KL(Dir(alpha) || Dir(1))
    alpha: [B,K]
    """
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    beta = torch.ones_like(alpha)

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
# Metrics utils
# ============================================================
def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    return float(np.min(fpr[idx])) if len(idx) > 0 else float("nan")


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
# Eval feature + geometry fusion (same as old eval script)
# ============================================================
@torch.no_grad()
def extract_feats(model: EEGNetENN, X: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    feats = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        fb = model.forward_features(xb).cpu().numpy()
        feats.append(fb)
    return np.concatenate(feats, axis=0)


def fit_maha_shared(Ftr: np.ndarray, ytr: np.ndarray, K: int, shrink: float = 1e-2):
    D = Ftr.shape[1]
    mu = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mu[k] = Ftr[ytr == k].mean(axis=0)

    Xc = Ftr - mu[ytr]
    cov = (Xc.T @ Xc) / max(len(Ftr) - 1, 1)
    cov = cov + shrink * np.eye(D, dtype=np.float32)
    inv_cov = np.linalg.inv(cov).astype(np.float32)
    return mu, inv_cov


def maha_d2_all(Fx: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    N, D = Fx.shape
    K = mu.shape[0]
    d2 = np.empty((N, K), dtype=np.float32)
    for k in range(K):
        diff = Fx - mu[k][None, :]
        d2[:, k] = np.sum((diff @ inv_cov) * diff, axis=1)
    return d2


def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=axis, keepdims=True) + 1e-12)


def geo_weights_from_d2_softmax(d2: np.ndarray, tau: float = 10.0) -> np.ndarray:
    return softmax_np(-d2 / tau, axis=1)


def geo_weights_from_d2_inv(d2: np.ndarray, eps: float = 1e-6, p: float = 1.0) -> np.ndarray:
    w = 1.0 / (d2 + eps) ** p
    w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    return w


@torch.no_grad()
def score_batch_classwise(
    model: EEGNetENN,
    X: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    tau: float = 10.0,
    use_inv: bool = False,
    inv_p: float = 1.0,
    bs: int = 256
) -> Dict[str, np.ndarray]:
    model.eval()
    out = {"evidence_ood": [], "maha_min": [], "fusion_geo": []}

    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        alpha = model(xb)                 # [B,K]
        S = alpha.sum(dim=1)              # [B]
        evidence = (alpha - 1.0)          # [B,K]

        feat = model.forward_features(xb).cpu().numpy()
        d2_all = maha_d2_all(feat, mu, inv_cov)      # [B,K]
        d2_min = d2_all.min(axis=1)                  # [B]

        if use_inv:
            w = geo_weights_from_d2_inv(d2_all, eps=1e-6, p=inv_p)
        else:
            w = geo_weights_from_d2_softmax(d2_all, tau=tau)

        e = evidence.cpu().numpy().astype(np.float32)
        e_w = e * w
        S_geo = np.sum(e_w + 1.0, axis=1).astype(np.float32)

        out["evidence_ood"].append((-(S)).cpu().numpy())
        out["maha_min"].append(d2_min.astype(np.float32))
        out["fusion_geo"].append((-S_geo).astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


@torch.no_grad()
def predict_id_probs_and_labels(
    model: EEGNetENN,
    X: np.ndarray,
    bs: int = 256
):
    model.eval()
    probs = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        alpha = model(xb)
        p = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)
        probs.append(p.cpu().numpy())
    prob = np.concatenate(probs, axis=0)
    pred = np.argmax(prob, axis=1).astype(np.int64)
    return prob.astype(np.float32), pred


# ============================================================
# Single Args for train+eval (defaults match old scripts)
# ============================================================
@dataclass
class Args:
    # paths
    mat_dir: str = r"D:\UM_Project\U-EDR\mat"
    save_path: str = r"D:\UM_Project\U-EDR\checkpoints\eegnet_enn_realoodcalib_best.pt"

    # split protocol (old: 7/1/2, seed=0)
    split_seed: int = 0
    n_train_subjects: int = 7
    n_val_subjects: int = 1
    n_test_subjects: int = 2

    # classes
    id_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)

    # training-time REAL OOD calibration (old: optional; default True; uses TRAIN split only)
    use_real_ood: bool = True
    real_ood_classes: Tuple[int, ...] = (8, 9, 10, 11)

    # eval-time OOD set (old: from TEST only)
    ood_classes: Tuple[int, ...] = (8, 9, 10, 11)

    # window / preprocess (old: single start)
    fs: int = 256
    onset_1idx: int = 39
    start_s: float = 1.0
    win_s: float = 2.0
    start_s_list: Tuple[float, ...] = (1.0, 1.5, 2.0)

    # train setup
    seed: int = 0
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    patience: int = 25

    # EDL
    kl_weight_id: float = 1e-3

    # real OOD calibration
    # 这里经过测试，这里的选择非常非常的影响最后的具体数值。是重要额调参内容。
    ood_ratio: float = 0.05
    lambda_ood: float = 0.2
    ood_loss_mode: str = "kl"   # "kl" or "smean"

    # EEGNet params
    F1: int = 4
    D: int = 1
    dropout: float = 0.5
    kernel_length: int = 32

    # eval setup (old)
    test_plus_val_as_id: bool = True
    shrink: float = 1e-2
    eval_batch_size: int = 256
    use_inv_weights: bool = False
    tau: float = 10.0
    inv_p: float = 1.0


def train(a: Args) -> None:
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), a.n_train_subjects, a.n_val_subjects, a.n_test_subjects, seed=a.split_seed)

    cfg = SSVEPConfig(fs=a.fs, onset_1idx=a.onset_1idx, start_s=a.start_s, win_s=a.win_s)

    Xtr_all, ytr_all, _ = build_multi_subject_arrays_multistart(
        [mats[i] for i in splits["train"]], cfg, "train", a.start_s_list
    )
    Xva_all, yva_all, _ = build_multi_subject_arrays_multistart(
        [mats[i] for i in splits["val"]], cfg, "val", a.start_s_list
    )
    Xte_all, yte_all, _ = build_multi_subject_arrays_multistart(
        [mats[i] for i in splits["test"]], cfg, "test", a.start_s_list
    )


    id_map = {c: i for i, c in enumerate(a.id_classes)}

    def filter_id(X, y):
        m = np.isin(y, a.id_classes)
        X = X[m]
        y = np.array([id_map[int(t)] for t in y[m]], dtype=np.int64)
        return X, y

    def filter_ood(X, y):
        m = np.isin(y, a.real_ood_classes)
        return X[m], y[m]

    Xtr, ytr = filter_id(Xtr_all, ytr_all)
    Xva, yva = filter_id(Xva_all, yva_all)
    Xte, yte = filter_id(Xte_all, yte_all)

    if a.use_real_ood:
        Xood_pool, _ = filter_ood(Xtr_all, ytr_all)
        if len(Xood_pool) == 0:
            raise RuntimeError("use_real_ood=True but no OOD samples found in TRAIN split. Check real_ood_classes.")
    else:
        Xood_pool = None

    Xtr_t = torch.from_numpy(Xtr).float().unsqueeze(1)
    ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(Xva).float().unsqueeze(1)
    yva_t = torch.from_numpy(yva)
    Xte_t = torch.from_numpy(Xte).float().unsqueeze(1)
    yte_t = torch.from_numpy(yte)

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=a.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=a.batch_size, shuffle=False)
    te_loader = DataLoader(TensorDataset(Xte_t, yte_t), batch_size=a.batch_size, shuffle=False)

    if a.use_real_ood:
        Xood_t = torch.from_numpy(Xood_pool).float().unsqueeze(1)
        ood_loader = DataLoader(TensorDataset(Xood_t), batch_size=a.batch_size, shuffle=True, drop_last=True)
        ood_iter = iter(ood_loader)

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

            alpha_id = model(xb)
            loss_id = edl_id_loss(alpha_id, yb, kl_weight=a.kl_weight_id)

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
                    loss_ood = ood_low_evidence_loss(alpha_ood, mode=a.ood_loss_mode)
                    loss = loss_id + a.lambda_ood * loss_ood
                else:
                    loss = loss_id
            else:
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


def evaluate(a: Args) -> None:
    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), a.n_train_subjects, a.n_val_subjects, a.n_test_subjects, seed=a.split_seed)
    cfg = SSVEPConfig(fs=a.fs, onset_1idx=a.onset_1idx, start_s=a.start_s, win_s=a.win_s)

    Xtr_all, ytr_all, _ = build_multi_subject_arrays([mats[i] for i in splits["train"]], cfg, split_name="train")
    Xva_all, yva_all, _ = build_multi_subject_arrays([mats[i] for i in splits["val"]], cfg, split_name="val")
    Xte_all, yte_all, _ = build_multi_subject_arrays([mats[i] for i in splits["test"]], cfg, split_name="test")

    id_map = {c: i for i, c in enumerate(a.id_classes)}
    mask_tr = np.isin(ytr_all, a.id_classes)
    Xtr = Xtr_all[mask_tr]
    ytr = np.array([id_map[int(t)] for t in ytr_all[mask_tr]], dtype=np.int64)

    if a.test_plus_val_as_id:
        Xid_src = np.concatenate([Xte_all, Xva_all], axis=0)
        yid_src = np.concatenate([yte_all, yva_all], axis=0)
    else:
        Xid_src = Xte_all
        yid_src = yte_all

    mask_id = np.isin(yid_src, a.id_classes)
    X_id = Xid_src[mask_id]
    y_id = np.array([id_map[int(t)] for t in yid_src[mask_id]], dtype=np.int64)

    mask_ood = np.isin(yte_all, a.ood_classes)
    X_ood = Xte_all[mask_ood]

    print(f"Train ID samples: {len(Xtr)}")
    print(f"Test  ID samples: {len(X_id)} (test+val={a.test_plus_val_as_id})")
    print(f"Test OOD samples: {len(X_ood)}")

    C, L = Xtr.shape[1], Xtr.shape[2]
    K = len(a.id_classes)
    model = EEGNetENN(C, L, K, F1=a.F1, D=a.D, dropout=a.dropout, kernel_length=a.kernel_length).to(DEVICE)
    model.load_state_dict(torch.load(a.save_path, map_location="cpu"), strict=True)
    model.eval()

    _, y_pred = predict_id_probs_and_labels(model, X_id, bs=a.eval_batch_size)
    acc = accuracy_score(y_id, y_pred)
    print(f"\n[ID classification] acc={acc:.4f}  (K={K})")

    Ftr = extract_feats(model, Xtr, bs=a.eval_batch_size)
    mu, inv_cov = fit_maha_shared(Ftr, ytr, K, shrink=a.shrink)

    sid = score_batch_classwise(
        model, X_id, mu, inv_cov,
        tau=a.tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.eval_batch_size
    )
    sod = score_batch_classwise(
        model, X_ood, mu, inv_cov,
        tau=a.tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.eval_batch_size
    )

    y_true = np.concatenate([np.zeros(len(X_id)), np.ones(len(X_ood))])

    s_id = sid["evidence_ood"]
    s_ood = sod["evidence_ood"]
    print(f"-S mean ID : {s_id.mean():.4f}  [5,25,50,75,95]={np.percentile(s_id,[5,25,50,75,95]).round(4)}")
    print(f"-S mean OOD: {s_ood.mean():.4f}  [5,25,50,75,95]={np.percentile(s_ood,[5,25,50,75,95]).round(4)}")

    d_id = sid["maha_min"]; d_ood = sod["maha_min"]
    print(f"dmin ID  : [5,25,50,75,95]={np.percentile(d_id,[5,25,50,75,95]).round(4)}")
    print(f"dmin OOD : [5,25,50,75,95]={np.percentile(d_ood,[5,25,50,75,95]).round(4)}")

    for key in ["evidence_ood", "maha_min", "fusion_geo"]:
        score = np.concatenate([sid[key], sod[key]])
        au = roc_auc_score(y_true, score)
        print(f"{key:>12s} | AUROC={au:.4f}")
        for tpr_t in [0.80, 0.90, 0.95]:
            fpr_t = fpr_at_tpr(y_true, score, tpr_t)
            print(f"  TPR@{tpr_t:.2f} -> FPR={fpr_t:.4f}")

    print(f"[fusion_geo params] use_inv={a.use_inv_weights} tau={a.tau} inv_p={a.inv_p} shrink={a.shrink}")
    print("Done.")

def build_multi_subject_arrays_multistart(mats, cfg, split_name: str, start_s_list):
    Xs, ys, sids = [], [], []
    for s in start_s_list:
        cfg_s = SSVEPConfig(
            fs=cfg.fs,
            onset_1idx=cfg.onset_1idx,
            start_s=float(s),
            win_s=cfg.win_s,
            bandpass_low=getattr(cfg, "bandpass_low", None),
            bandpass_high=getattr(cfg, "bandpass_high", None),
            bandpass_order=getattr(cfg, "bandpass_order", None),
            per_channel_zscore=getattr(cfg, "per_channel_zscore", False),
        )
        X, y, sid = build_multi_subject_arrays(mats, cfg_s, split_name=split_name)
        Xs.append(X); ys.append(y); sids.append(sid)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    sid = np.concatenate(sids, axis=0)
    return X, y, sid




def main():
    a = Args()
    train(a)
    evaluate(a)


if __name__ == "__main__":
    main()
