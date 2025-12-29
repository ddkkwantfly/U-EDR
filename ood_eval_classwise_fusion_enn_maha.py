from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

from data_loader import (
    SSVEPConfig,
    list_mat_files,
    split_subjects,
    build_multi_subject_arrays,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model must match training
# ============================================================
class EEGNetENN(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes, F1=4, D=1, dropout=0.5, kernel_length=32):
        super().__init__()
        F2 = F1 * D
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout)

        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
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
# Metrics
# ============================================================
def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    return float(np.min(fpr[idx])) if len(idx) > 0 else float("nan")


# ============================================================
# Feature extraction
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


# ============================================================
# Mahalanobis fitting (shared covariance)
# ============================================================
def fit_maha_shared(Ftr: np.ndarray, ytr: np.ndarray, K: int, shrink: float = 1e-2):
    """
    Returns class means mu[K,D] and inv_cov[D,D].
    """
    D = Ftr.shape[1]
    mu = np.zeros((K, D), dtype=np.float32)
    for k in range(K):
        mu[k] = Ftr[ytr == k].mean(axis=0)

    # pooled covariance
    Xc = Ftr - mu[ytr]
    cov = (Xc.T @ Xc) / max(len(Ftr) - 1, 1)
    cov = cov + shrink * np.eye(D, dtype=np.float32)
    inv_cov = np.linalg.inv(cov).astype(np.float32)
    return mu, inv_cov


def maha_d2_all(Fx: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    """
    Fx [N,D], mu [K,D], inv_cov [D,D] -> d2 [N,K] (squared Mahalanobis per class)
    """
    N, D = Fx.shape
    K = mu.shape[0]
    d2 = np.empty((N, K), dtype=np.float32)
    for k in range(K):
        diff = Fx - mu[k][None, :]
        d2[:, k] = np.sum((diff @ inv_cov) * diff, axis=1)
    return d2


# ============================================================
# Geometry weights
# ============================================================
def softmax_np(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / (np.sum(ez, axis=axis, keepdims=True) + 1e-12)


def geo_weights_from_d2_softmax(d2: np.ndarray, tau: float = 10.0) -> np.ndarray:
    """
    w = softmax(-d2 / tau)
    """
    return softmax_np(-d2 / tau, axis=1)


def geo_weights_from_d2_inv(d2: np.ndarray, eps: float = 1e-6, p: float = 1.0) -> np.ndarray:
    """
    w_k ∝ 1/(d2_k + eps)^p
    """
    w = 1.0 / (d2 + eps) ** p
    w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    return w


# ============================================================
# Scoring
# ============================================================
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
    """
    OOD scores (higher => more OOD):
      evidence_ood = -S  where S=sum(alpha)
      maha_min     = min_k d2_k   (geometry-only reference)
      fusion_geo   = -S_geo where evidence per class is weighted by geometry:
                     e = alpha-1
                     e' = e * w
                     alpha' = e' + 1
                     S_geo = sum(alpha')
    """
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

        # weights
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


# ============================================================
# Main
# ============================================================
@dataclass
class Args:
    # mat_dir: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\mat"
    mat_dir: str = r"D:\UM_Project\U-EDR\mat"
    # ckpt_path: str = r"D:\UM_Project\U-EDR\12JFPM_SSVEP-master\data\checkpoints\eegnet_enn_realoodcalib_best.pt"
    ckpt_path: str = r"D:\UM_Project\U-EDR\checkpoints\eegnet_enn_realoodcalib_best.pt"

    id_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    ood_classes: Tuple[int, ...] = (8, 9, 10, 11)

    fs: int = 256
    onset_1idx: int = 39
    start_s: float = 1.0
    win_s: float = 2.0

    # eval setup
    test_plus_val_as_id: bool = True  # match你现在的做法：ID=val+test
    shrink: float = 1e-2
    batch_size: int = 256

    # classwise fusion params
    use_inv_weights: bool = False     # False=softmax(-d2/tau) (推荐)
    tau: float = 10.0
    inv_p: float = 1.0               # only used if use_inv_weights=True


def main(a: Args):
    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), 7, 1, 2, seed=0)
    cfg = SSVEPConfig(fs=a.fs, onset_1idx=a.onset_1idx, start_s=a.start_s, win_s=a.win_s)

    Xtr_all, ytr_all, _ = build_multi_subject_arrays([mats[i] for i in splits["train"]], cfg, split_name="train")
    Xva_all, yva_all, _ = build_multi_subject_arrays([mats[i] for i in splits["val"]], cfg, split_name="val")
    Xte_all, yte_all, _ = build_multi_subject_arrays([mats[i] for i in splits["test"]], cfg, split_name="test")

    # Train ID set for fitting mu, Sigma
    id_map = {c: i for i, c in enumerate(a.id_classes)}
    mask_tr = np.isin(ytr_all, a.id_classes)
    Xtr = Xtr_all[mask_tr]
    ytr = np.array([id_map[int(t)] for t in ytr_all[mask_tr]], dtype=np.int64)

    # ID eval set
    if a.test_plus_val_as_id:
        Xid_src = np.concatenate([Xte_all, Xva_all], axis=0)
        yid_src = np.concatenate([yte_all, yva_all], axis=0)
    else:
        Xid_src = Xte_all
        yid_src = yte_all

    mask_id = np.isin(yid_src, a.id_classes)
    mask_ood = np.isin(yte_all, a.ood_classes)  # OOD from TEST only (你也可以改成val+test)
    X_id = Xid_src[mask_id]
    X_ood = Xte_all[mask_ood]

    print(f"Train ID samples: {len(Xtr)}")
    print(f"Test  ID samples: {len(X_id)} (test+val={a.test_plus_val_as_id})")
    print(f"Test OOD samples: {len(X_ood)}")

    # model
    C, L = Xtr.shape[1], Xtr.shape[2]
    K = len(a.id_classes)
    model = EEGNetENN(C, L, K).to(DEVICE)
    model.load_state_dict(torch.load(a.ckpt_path, map_location="cpu"), strict=True)
    model.eval()

    # fit Mahalanobis on TRAIN-ID features
    Ftr = extract_feats(model, Xtr, bs=a.batch_size)
    mu, inv_cov = fit_maha_shared(Ftr, ytr, K, shrink=a.shrink)

    # score
    sid = score_batch_classwise(
        model, X_id, mu, inv_cov,
        tau=a.tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.batch_size
    )
    sod = score_batch_classwise(
        model, X_ood, mu, inv_cov,
        tau=a.tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.batch_size
    )

    y_true = np.concatenate([np.zeros(len(X_id)), np.ones(len(X_ood))])

    # Some descriptive stats
    s_id = sid["evidence_ood"]
    s_ood = sod["evidence_ood"]
    print(f"-S mean ID : {s_id.mean():.4f}  [5,25,50,75,95]={np.percentile(s_id,[5,25,50,75,95]).round(4)}")
    print(f"-S mean OOD: {s_ood.mean():.4f}  [5,25,50,75,95]={np.percentile(s_ood,[5,25,50,75,95]).round(4)}")

    d_id = sid["maha_min"]; d_ood = sod["maha_min"]
    print(f"dmin ID  : [5,25,50,75,95]={np.percentile(d_id,[5,25,50,75,95]).round(4)}")
    print(f"dmin OOD : [5,25,50,75,95]={np.percentile(d_ood,[5,25,50,75,95]).round(4)}")

    # Metrics
    for key in ["evidence_ood", "maha_min", "fusion_geo"]:
        score = np.concatenate([sid[key], sod[key]])
        au = roc_auc_score(y_true, score)
        print(f"{key:>12s} | AUROC={au:.4f}")
        for tpr_t in [0.80, 0.90, 0.95]:
            fpr_t = fpr_at_tpr(y_true, score, tpr_t)
            print(f"  TPR@{tpr_t:.2f} -> FPR={fpr_t:.4f}")

    print(f"[fusion_geo params] use_inv={a.use_inv_weights} tau={a.tau} inv_p={a.inv_p}")
    print("Done.")


if __name__ == "__main__":
    main(Args())
