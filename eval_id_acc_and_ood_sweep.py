from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

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
# ROC helper
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
    Fx [N,D], mu [K,D], inv_cov [D,D] -> d2 [N,K]
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
    return softmax_np(-d2 / tau, axis=1)


def geo_weights_from_d2_inv(d2: np.ndarray, eps: float = 1e-6, p: float = 1.0) -> np.ndarray:
    w = 1.0 / (d2 + eps) ** p
    w = w / (np.sum(w, axis=1, keepdims=True) + 1e-12)
    return w


# ============================================================
# Scoring (your original design)
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
      maha_min     = min_k d2_k
      fusion_geo   = -S_geo where e=(alpha-1), weights from d2, e'=e*w, alpha'=e'+1
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
# ID classification (Dirichlet mean)
# ============================================================
@torch.no_grad()
def predict_id_labels(model: EEGNetENN, X: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        alpha = model(xb)  # [B,K]
        p = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)
        preds.append(torch.argmax(p, dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(np.int64)


# ============================================================
# Args
# ============================================================
@dataclass
class Args:
    mat_dir: str = r"D:\UM_Project\U-EDR\mat"
    ckpt_path: str = r"D:\UM_Project\U-EDR\checkpoints\eegnet_enn_realoodcalib_best.pt"

    id_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    ood_classes: Tuple[int, ...] = (8, 9, 10, 11)

    fs: int = 256
    onset_1idx: int = 39
    start_s: float = 1.0
    win_s: float = 2.0

    # evaluation setup (keep exactly as your first script)
    test_plus_val_as_id: bool = True      # ID eval = test+val
    ood_from_test_only: bool = True       # OOD only from test (like your first script)

    batch_size: int = 256

    # sweep space
    seeds: Tuple[int, ...] = (0, 1, 2, 3, 4)
    taus: Tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)
    shrinks: Tuple[float, ...] = (0.001, 0.01, 0.05, 0.1)

    # fusion variant
    use_inv_weights: bool = False
    inv_p: float = 1.0


def evaluate_one(seed: int, tau: float, shrink: float, a: Args) -> Dict[str, Any]:
    mats = list_mat_files(a.mat_dir)

    # subject split seed = seed
    splits = split_subjects(len(mats), 7, 1, 2, seed=seed)
    cfg = SSVEPConfig(fs=a.fs, onset_1idx=a.onset_1idx, start_s=a.start_s, win_s=a.win_s)

    Xtr_all, ytr_all, _ = build_multi_subject_arrays([mats[i] for i in splits["train"]], cfg, split_name="train")
    Xva_all, yva_all, _ = build_multi_subject_arrays([mats[i] for i in splits["val"]], cfg, split_name="val")
    Xte_all, yte_all, _ = build_multi_subject_arrays([mats[i] for i in splits["test"]], cfg, split_name="test")

    # Train ID set
    id_map = {c: i for i, c in enumerate(a.id_classes)}
    mask_tr = np.isin(ytr_all, a.id_classes)
    Xtr = Xtr_all[mask_tr]
    ytr = np.array([id_map[int(t)] for t in ytr_all[mask_tr]], dtype=np.int64)

    # ID eval set (test + val)
    if a.test_plus_val_as_id:
        Xid_src = np.concatenate([Xte_all, Xva_all], axis=0)
        yid_src = np.concatenate([yte_all, yva_all], axis=0)
    else:
        Xid_src = Xte_all
        yid_src = yte_all

    mask_id = np.isin(yid_src, a.id_classes)
    X_id = Xid_src[mask_id]
    y_id = np.array([id_map[int(t)] for t in yid_src[mask_id]], dtype=np.int64)

    # OOD set
    if a.ood_from_test_only:
        mask_ood = np.isin(yte_all, a.ood_classes)
        X_ood = Xte_all[mask_ood]
    else:
        # (optional) allow OOD from test+val too
        yood_src = yid_src
        Xood_src = Xid_src
        mask_ood = np.isin(yood_src, a.ood_classes)
        X_ood = Xood_src[mask_ood]

    # model
    C, L = Xtr.shape[1], Xtr.shape[2]
    K = len(a.id_classes)
    model = EEGNetENN(C, L, K).to(DEVICE)
    model.load_state_dict(torch.load(a.ckpt_path, map_location="cpu"), strict=True)
    model.eval()

    # ID acc
    y_pred = predict_id_labels(model, X_id, bs=a.batch_size)
    id_acc = float(accuracy_score(y_id, y_pred))

    # fit maha on TRAIN-ID feats
    Ftr = extract_feats(model, Xtr, bs=a.batch_size)
    mu, inv_cov = fit_maha_shared(Ftr, ytr, K, shrink=shrink)

    # score ID / OOD
    sid = score_batch_classwise(
        model, X_id, mu, inv_cov,
        tau=tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.batch_size
    )
    sod = score_batch_classwise(
        model, X_ood, mu, inv_cov,
        tau=tau, use_inv=a.use_inv_weights, inv_p=a.inv_p, bs=a.batch_size
    )

    y_true = np.concatenate([np.zeros(len(X_id)), np.ones(len(X_ood))]).astype(np.int64)

    def pack_metrics(key: str) -> Dict[str, float]:
        score = np.concatenate([sid[key], sod[key]])
        au = float(roc_auc_score(y_true, score))
        f80 = fpr_at_tpr(y_true, score, 0.80)
        f90 = fpr_at_tpr(y_true, score, 0.90)
        f95 = fpr_at_tpr(y_true, score, 0.95)
        return {
            f"{key}_auroc": au,
            f"{key}_fpr@tpr80": float(f80),
            f"{key}_fpr@tpr90": float(f90),
            f"{key}_fpr@tpr95": float(f95),
        }

    out = {
        "seed": seed,
        "tau": tau,
        "shrink": shrink,
        "n_train_id": int(len(Xtr)),
        "n_test_id": int(len(X_id)),
        "n_test_ood": int(len(X_ood)),
        "id_acc": id_acc,
    }
    out.update(pack_metrics("evidence_ood"))
    out.update(pack_metrics("maha_min"))
    out.update(pack_metrics("fusion_geo"))
    return out


def main(a: Args):
    rows: List[Dict[str, Any]] = []
    for seed in a.seeds:
        for shrink in a.shrinks:
            for tau in a.taus:
                r = evaluate_one(seed=seed, tau=tau, shrink=shrink, a=a)
                rows.append(r)
                print(
                    f"[seed={seed}] shrink={shrink:g} tau={tau:g} | "
                    f"acc={r['id_acc']:.4f} | "
                    f"fusion AUROC={r['fusion_geo_auroc']:.4f} "
                    f"FPR@TPR0.90={r['fusion_geo_fpr@tpr90']:.4f}"
                )

    df = pd.DataFrame(rows)

    # aggregate across seeds
    group_cols = ["shrink", "tau"]
    agg = df.groupby(group_cols).agg(
        id_acc_mean=("id_acc", "mean"),
        id_acc_std=("id_acc", "std"),
        fusion_auroc_mean=("fusion_geo_auroc", "mean"),
        fusion_auroc_std=("fusion_geo_auroc", "std"),
        fusion_fpr90_mean=("fusion_geo_fpr@tpr90", "mean"),
        fusion_fpr90_std=("fusion_geo_fpr@tpr90", "std"),
        fusion_fpr80_mean=("fusion_geo_fpr@tpr80", "mean"),
        fusion_fpr95_mean=("fusion_geo_fpr@tpr95", "mean"),
        maha_auroc_mean=("maha_min_auroc", "mean"),
        evid_auroc_mean=("evidence_ood_auroc", "mean"),
    ).reset_index()

    # choose best: minimize mean FPR@TPR=0.90, tie-break by higher AUROC
    agg_sorted = agg.sort_values(
        by=["fusion_fpr90_mean", "fusion_auroc_mean"],
        ascending=[True, False]
    )

    print("\n================= Sweep summary (mean±std over seeds) =================")
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 160)
    print(agg_sorted.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    best = agg_sorted.iloc[0].to_dict()
    print("\n================= Best config (by min fusion FPR@TPR0.90) =================")
    print(
        f"BEST: shrink={best['shrink']:.6g}, tau={best['tau']:.6g} | "
        f"fusion FPR@TPR0.90={best['fusion_fpr90_mean']:.4f}±{best['fusion_fpr90_std']:.4f} | "
        f"fusion AUROC={best['fusion_auroc_mean']:.4f}±{best['fusion_auroc_std']:.4f} | "
        f"ID acc={best['id_acc_mean']:.4f}±{best['id_acc_std']:.4f}"
    )

    # optional: save csv
    out_csv = "sweep_results.csv"
    df.to_csv(out_csv, index=False)
    agg_sorted.to_csv("sweep_results_summary.csv", index=False)
    print(f"\nSaved: {out_csv} and sweep_results_summary.csv")


if __name__ == "__main__":
    main(Args())
