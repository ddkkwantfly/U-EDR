# train_eval_route1_v3_bnfreeze.py
# ============================================================
# Route-1 v3: 6/2/2 subject split + low pseudo-OOD (<= real calib-OOD count) + BN-freeze after warmup
#
# Changes vs v1:
#  1) Default subject split: train/val/test = 6/2/2 (val is now 2 subjects -> more reliable)
#  2) Pseudo-OOD is LIMITED:
#       - per training batch: m_pseudo <= m_real
#       - in test: n_pseudo_test <= n_test_calib_ood
#  3) Pseudo mode default: "mix" (less "cheating" than pure timeperm)
#  4) Reports BOTH:
#       - Real-OOD only (calib + unseen)
#       - Pseudo-OOD only
#       - ALL-OOD (real + pseudo)  [for completeness, not main claim]
#
# Keep protocol clean (no leakage):
#   - calib OOD pool comes ONLY from TRAIN subjects
#   - test metrics computed ONLY on TEST subjects
# ============================================================

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Sequence, Optional, Literal

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
    load_mat_eeg,
    bandpass_filter,
    split_trials,
    extract_windows_multi_start_single_subject,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Model
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.depthwise(x)))
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = F.elu(self.bn3(x))
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        evidence = F.softplus(self.evidence(feat))
        alpha = evidence + 1.0
        return alpha


# ============================================================
# EDL losses
# ============================================================
def dirichlet_kl_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
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
    K = alpha.size(1)
    S = alpha.sum(dim=1, keepdim=True)
    y = F.one_hot(target, num_classes=K).float()
    ece = torch.sum(y * (digamma(S) - digamma(alpha)), dim=1)
    kl = dirichlet_kl_to_uniform(alpha)
    return ece.mean() + kl_weight * kl


def ood_low_evidence_loss(alpha: torch.Tensor, mode: Literal["kl", "smean"] = "kl") -> torch.Tensor:
    if mode == "kl":
        return dirichlet_kl_to_uniform(alpha)
    if mode == "smean":
        return alpha.sum(dim=1).mean()
    raise ValueError(mode)


# ============================================================
# Metrics
# ============================================================
def fpr_at_tpr(y_true: np.ndarray, y_score: np.ndarray, tpr_target: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    idx = np.where(tpr >= tpr_target)[0]
    return float(np.min(fpr[idx])) if len(idx) > 0 else float("nan")


@torch.no_grad()
def id_acc_from_alpha(model: nn.Module, X: np.ndarray, y: np.ndarray, bs: int = 256) -> float:
    model.eval()
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        alpha = model(xb)
        p = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-12)
        preds.append(torch.argmax(p, dim=1).cpu().numpy())
    yhat = np.concatenate(preds, axis=0) if preds else np.zeros((0,), dtype=np.int64)
    return float(accuracy_score(y, yhat)) if len(yhat) else float("nan")


# ============================================================
# Pseudo OOD generator (safe axes)
# ============================================================
def generate_pseudo_ood(
    X_id: np.ndarray,
    n: int,
    rng: np.random.Generator,
    mode: Literal["noise", "timeperm", "chanperm", "mix"] = "mix",
    noise_scale: float = 0.6,
) -> np.ndarray:
    assert X_id.ndim == 3, f"Expected [N,C,T], got {X_id.shape}"
    N, C, T = X_id.shape
    if N == 0 or n <= 0:
        return np.zeros((0, C, T), dtype=np.float32)

    idx = rng.integers(0, N, size=n)
    X = X_id[idx].copy().astype(np.float32)

    ch_std = X.std(axis=(0, 2), keepdims=True).astype(np.float32) + 1e-6  # [1,C,1]

    def do_noise(x):
        eps = rng.standard_normal(size=x.shape).astype(np.float32)
        return x + (noise_scale * ch_std) * eps

    def do_timeperm(x):
        out = x.copy()
        for i in range(out.shape[0]):
            perm = rng.permutation(T)
            out[i] = out[i][:, perm]
        return out

    def do_chanperm(x):
        out = x.copy()
        for i in range(out.shape[0]):
            perm = rng.permutation(C)
            out[i] = out[i][perm, :]
        return out

    if mode == "noise":
        return do_noise(X)
    if mode == "timeperm":
        return do_timeperm(X)
    if mode == "chanperm":
        return do_chanperm(X)

    # mix
    out = X.copy()
    choices = rng.integers(0, 3, size=n)  # 0=noise,1=timeperm,2=chanperm
    for i in range(n):
        xi = out[i:i+1]
        if choices[i] == 0:
            out[i:i+1] = do_noise(xi)
        elif choices[i] == 1:
            out[i:i+1] = do_timeperm(xi)
        else:
            out[i:i+1] = do_chanperm(xi)
    return out.astype(np.float32)


# ============================================================
# Multi-subject arrays with multi-start windows
# ============================================================
def build_multi_subject_arrays_multi_start(
    mat_paths: Sequence[str | Path],
    cfg: SSVEPConfig,
    mat_key: str = "eeg",
    subject_ids: Optional[Sequence[int]] = None,
    trial_seed: int = 0,
    n_train_trials: int = 10,
    n_val_trials: int = 2,
    n_test_trials: Optional[int] = None,
    split_name: str = "train",
    start_s_list: Sequence[float] = (1.0, 1.5, 2.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if split_name not in ("train", "val", "test"):
        raise ValueError("split_name must be train/val/test")
    if subject_ids is None:
        subject_ids = list(range(len(mat_paths)))
    if len(subject_ids) != len(mat_paths):
        raise ValueError("subject_ids length must match mat_paths")

    X_all, y_all, sid_all = [], [], []

    for sid, mp in zip(subject_ids, mat_paths):
        eeg = load_mat_eeg(mp, key=mat_key)
        eeg = bandpass_filter(
            eeg, fs=cfg.fs,
            low=cfg.bandpass_low, high=cfg.bandpass_high,
            order=cfg.bandpass_order,
            axis=2,
        )
        K, C, T, R = eeg.shape
        splits = split_trials(R, n_train=n_train_trials, n_val=n_val_trials, n_test=n_test_trials, seed=trial_seed)
        trial_ids = splits[split_name]

        X, y = extract_windows_multi_start_single_subject(
            eeg, cfg, trial_ids, start_s_list=start_s_list, win_s=cfg.win_s
        )
        X_all.append(X)
        y_all.append(y)
        sid_all.append(np.full((len(y),), sid, dtype=np.int64))

    X_all = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_all, axis=0).astype(np.int64, copy=False)
    sid_all = np.concatenate(sid_all, axis=0).astype(np.int64, copy=False)
    return X_all, y_all, sid_all


# ============================================================
# Mahalanobis + classwise geo fusion scoring (your strong variant)
# ============================================================
@torch.no_grad()
def extract_feats(model: EEGNetENN, X: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    feats = []
    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        fb = model.forward_features(xb).cpu().numpy()
        feats.append(fb)
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1), dtype=np.float32)


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


@torch.no_grad()
def score_batch_classwise(
    model: EEGNetENN,
    X: np.ndarray,
    mu: np.ndarray,
    inv_cov: np.ndarray,
    tau: float = 10.0,
    bs: int = 256
) -> Dict[str, np.ndarray]:
    model.eval()
    out = {"evidence_ood": [], "maha_min": [], "fusion_geo": []}

    for i in range(0, len(X), bs):
        xb = torch.from_numpy(X[i:i+bs]).float().to(DEVICE).unsqueeze(1)
        alpha = model(xb)
        S = alpha.sum(dim=1)
        evidence = (alpha - 1.0)

        feat = model.forward_features(xb).cpu().numpy()
        d2_all = maha_d2_all(feat, mu, inv_cov)
        d2_min = d2_all.min(axis=1)

        w = geo_weights_from_d2_softmax(d2_all, tau=tau)

        e = evidence.cpu().numpy().astype(np.float32)
        e_w = e * w
        S_geo = np.sum(e_w + 1.0, axis=1).astype(np.float32)

        out["evidence_ood"].append((-(S)).cpu().numpy())
        out["maha_min"].append(d2_min.astype(np.float32))
        out["fusion_geo"].append((-S_geo).astype(np.float32))

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


# ============================================================
# Args
# ============================================================
@dataclass
class Args:
    mat_dir: str = r"D:\UM_Project\U-EDR\mat"
    ckpt_dir: str = r"D:\UM_Project\U-EDR\checkpoints"

    id_classes: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7)
    calib_ood_classes: Tuple[int, ...] = (8, 9)
    unseen_ood_classes: Tuple[int, ...] = (10, 11)

    # splits
    subject_seed: int = 0
    trial_seed: int = 0
    n_train_subjects: int = 6
    n_val_subjects: int = 2
    n_test_subjects: int = 2

    # trial split
    n_train_trials: int = 10
    n_val_trials: int = 2
    n_test_trials: Optional[int] = None

    # window/preprocess
    fs: int = 256
    onset_1idx: int = 39
    bandpass_low: float = 6.0
    bandpass_high: float = 90.0
    start_s_list: Tuple[float, ...] = (1.0, 1.5, 2.0)
    win_s: float = 2.0
    per_channel_zscore: bool = True

    # training
    seed: int = 0
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    patience: int = 30

    # warmup+ramp
    warmup_epochs: int = 30
    ramp_epochs: int = 30


    # BN freeze (recommended when mixing ID + OOD / pseudo)
    freeze_bn_after_warmup: bool = True

    # EDL
    kl_weight_id: float = 1e-3

    # OOD calibration (low pseudo, capped)
    real_ood_ratio: float = 0.05
    pseudo_ood_ratio: float = 0.03  # will be capped by m_real
    lambda_real_ood: float = 0.20
    lambda_pseudo_ood: float = 0.10
    ood_loss_mode: str = "kl"

    pseudo_mode_train: str = "mix"
    pseudo_noise_scale_train: float = 0.6

    # model hyper
    F1: int = 4
    D: int = 1
    dropout: float = 0.5
    kernel_length: int = 32

    # geometry eval
    maha_shrink: float = 1e-2
    geo_tau: float = 20.0

    # test pseudo OOD (capped by #calib-ood test samples)
    pseudo_mode_test: str = "mix"
    pseudo_noise_scale_test: float = 0.6



# ============================================================
# BN-freeze helper
# ============================================================
def freeze_batchnorm_stats(model: nn.Module) -> None:
    """
    Freeze BatchNorm running statistics by putting only BN layers into eval() mode.
    This prevents mixed ID+OOD batches from corrupting BN running mean/var.
    Dropout and other layers stay in train() mode.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()

# ============================================================
# Main
# ============================================================
def main(a: Args):
    # seeds
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    mats = list_mat_files(a.mat_dir)
    splits = split_subjects(len(mats), a.n_train_subjects, a.n_val_subjects, a.n_test_subjects, seed=a.subject_seed)

    # hard sanity (no subject leakage)
    tr_set, va_set, te_set = set(splits["train"].tolist()), set(splits["val"].tolist()), set(splits["test"].tolist())
    assert tr_set.isdisjoint(va_set) and tr_set.isdisjoint(te_set) and va_set.isdisjoint(te_set), "Subject split overlaps!"

    train_mats = [mats[i] for i in splits["train"]]
    val_mats   = [mats[i] for i in splits["val"]]
    test_mats  = [mats[i] for i in splits["test"]]

    cfg = SSVEPConfig(
        fs=a.fs,
        onset_1idx=a.onset_1idx,
        bandpass_low=a.bandpass_low,
        bandpass_high=a.bandpass_high,
        start_s=1.0,
        win_s=a.win_s,
        per_channel_zscore=a.per_channel_zscore,
    )

    Xtr_all, ytr_all, _ = build_multi_subject_arrays_multi_start(
        train_mats, cfg, mat_key="eeg",
        subject_ids=splits["train"].tolist(),
        trial_seed=a.trial_seed,
        n_train_trials=a.n_train_trials, n_val_trials=a.n_val_trials, n_test_trials=a.n_test_trials,
        split_name="train",
        start_s_list=a.start_s_list,
    )
    Xva_all, yva_all, _ = build_multi_subject_arrays_multi_start(
        val_mats, cfg, mat_key="eeg",
        subject_ids=splits["val"].tolist(),
        trial_seed=a.trial_seed,
        n_train_trials=a.n_train_trials, n_val_trials=a.n_val_trials, n_test_trials=a.n_test_trials,
        split_name="val",
        start_s_list=a.start_s_list,
    )
    Xte_all, yte_all, _ = build_multi_subject_arrays_multi_start(
        test_mats, cfg, mat_key="eeg",
        subject_ids=splits["test"].tolist(),
        trial_seed=a.trial_seed,
        n_train_trials=a.n_train_trials, n_val_trials=a.n_val_trials, n_test_trials=a.n_test_trials,
        split_name="test",
        start_s_list=a.start_s_list,
    )

    id_map = {c: i for i, c in enumerate(a.id_classes)}
    K = len(a.id_classes)

    def filter_id(X, y):
        m = np.isin(y, a.id_classes)
        Xo = X[m].astype(np.float32, copy=False)
        yo = np.array([id_map[int(t)] for t in y[m]], dtype=np.int64)
        return Xo, yo

    def filter_classes(X, y, classes: Tuple[int, ...]):
        m = np.isin(y, classes)
        return X[m].astype(np.float32, copy=False), y[m].astype(np.int64, copy=False)

    # ID sets
    Xtr_id, ytr_id = filter_id(Xtr_all, ytr_all)
    Xva_id, yva_id = filter_id(Xva_all, yva_all)
    Xte_id, yte_id = filter_id(Xte_all, yte_all)

    # calib OOD pool from TRAIN subjects only
    Xood_pool, _ = filter_classes(Xtr_all, ytr_all, a.calib_ood_classes)
    if len(Xood_pool) == 0:
        raise RuntimeError(f"No calib OOD samples in TRAIN for calib_ood_classes={a.calib_ood_classes}.")

    # eval OOD from TEST subjects
    Xte_calib_ood, _ = filter_classes(Xte_all, yte_all, a.calib_ood_classes)
    Xte_unseen_ood, _ = filter_classes(Xte_all, yte_all, a.unseen_ood_classes)

    print("============== DATA ==============")
    print(f"Subject split: train={sorted(tr_set)} val={sorted(va_set)} test={sorted(te_set)}")
    print(f"Train ID: {len(Xtr_id)} | Val ID: {len(Xva_id)} | Test ID: {len(Xte_id)}")
    print(f"Train calib-OOD pool: {len(Xood_pool)}")
    print(f"Test calib-OOD: {len(Xte_calib_ood)} | Test unseen-OOD: {len(Xte_unseen_ood)}")
    print(f"Multi-start: {a.start_s_list}, win_s={a.win_s}s")

    # loaders
    Xtr_t = torch.from_numpy(Xtr_id).float().unsqueeze(1)
    ytr_t = torch.from_numpy(ytr_id).long()
    Xva_t = torch.from_numpy(Xva_id).float().unsqueeze(1)
    yva_t = torch.from_numpy(yva_id).long()

    tr_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=a.batch_size, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=a.batch_size, shuffle=False)

    # model
    C, T = Xtr_id.shape[1], Xtr_id.shape[2]
    model = EEGNetENN(C, T, K, F1=a.F1, D=a.D, dropout=a.dropout, kernel_length=a.kernel_length).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # save path
    ckpt_dir = Path(a.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"eegnet_enn_route1_v3_bnfreeze_seed{a.seed}_subj{a.subject_seed}_trial{a.trial_seed}.pt\\"
    save_path = ckpt_dir / ckpt_name

    def ramp_factor(epoch: int) -> float:
        if epoch <= a.warmup_epochs:
            return 0.0
        if a.ramp_epochs <= 0:
            return 1.0
        t = (epoch - a.warmup_epochs) / float(a.ramp_epochs)
        return float(np.clip(t, 0.0, 1.0))

    best_val = -1.0
    bad = 0

    for ep in range(1, a.epochs + 1):
        model.train()
        rf = ramp_factor(ep)
        if a.freeze_bn_after_warmup and rf > 0.0:
            freeze_batchnorm_stats(model)

        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            opt.zero_grad()

            alpha_id = model(xb)
            loss = edl_id_loss(alpha_id, yb, kl_weight=a.kl_weight_id)

            if rf > 0.0:
                B = xb.size(0)
                m_real = int(round(B * a.real_ood_ratio))
                m_pseudo = int(round(B * a.pseudo_ood_ratio))
                # IMPORTANT: cap pseudo by real (your request)
                m_pseudo = min(m_pseudo, m_real) if m_real > 0 else 0

                if m_real > 0 and a.lambda_real_ood > 0:
                    idx = rng.integers(0, len(Xood_pool), size=m_real)
                    xood_real = torch.from_numpy(Xood_pool[idx]).float().to(DEVICE).unsqueeze(1)
                    alpha_real = model(xood_real)
                    loss_real = ood_low_evidence_loss(alpha_real, mode=a.ood_loss_mode)
                    loss = loss + rf * a.lambda_real_ood * loss_real

                if m_pseudo > 0 and a.lambda_pseudo_ood > 0:
                    Xpseudo = generate_pseudo_ood(
                        X_id=Xtr_id,
                        n=m_pseudo,
                        rng=rng,
                        mode=a.pseudo_mode_train,
                        noise_scale=a.pseudo_noise_scale_train,
                    )
                    xood_pseudo = torch.from_numpy(Xpseudo).float().to(DEVICE).unsqueeze(1)
                    alpha_pseudo = model(xood_pseudo)
                    loss_pseudo = ood_low_evidence_loss(alpha_pseudo, mode=a.ood_loss_mode)
                    loss = loss + rf * a.lambda_pseudo_ood * loss_pseudo

            loss.backward()
            opt.step()

        val_acc = id_acc_from_alpha(model, Xva_id, yva_id, bs=256)
        if ep == 1 or ep % 10 == 0:
            tr_acc = id_acc_from_alpha(model, Xtr_id, ytr_id, bs=256)
            print(f"Epoch {ep:03d} | rf={rf:.2f} | train_acc={tr_acc:.3f} | val_acc={val_acc:.3f}")
        else:
            print(f"Epoch {ep:03d} | rf={rf:.2f} | val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            bad = 0
            torch.save(model.state_dict(), str(save_path))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"Early stop at epoch {ep} (best val {best_val:.3f})")
                break

    print(f"\nBest val acc: {best_val:.3f}")
    print(f"Saved checkpoint to: {save_path}")

    # -------------------------
    # EVAL (TEST subjects only)
    # -------------------------
    model.load_state_dict(torch.load(str(save_path), map_location="cpu"), strict=True)
    model.eval()

    print("\n==================== EVAL (TEST subjects only) ====================")
    print(f"Train ID samples: {len(Xtr_id)}")
    print(f"Val   ID samples: {len(Xva_id)}")
    print(f"Test  ID samples: {len(Xte_id)}")
    print(f"Test calib-OOD samples (classes={a.calib_ood_classes}): {len(Xte_calib_ood)}")
    print(f"Test unseen-OOD samples (classes={a.unseen_ood_classes}): {len(Xte_unseen_ood)}")

    # test pseudo OOD capped by calib-ood test count (your request)
    n_pseudo_test = min(len(Xte_calib_ood), len(Xte_id))  # also cannot exceed #ID
    Xte_pseudo_ood = generate_pseudo_ood(
        X_id=Xte_id,
        n=n_pseudo_test,
        rng=rng,
        mode=a.pseudo_mode_test,
        noise_scale=a.pseudo_noise_scale_test,
    )
    print(f"Test pseudo-OOD samples (mode={a.pseudo_mode_test}): {len(Xte_pseudo_ood)}")

    id_acc = id_acc_from_alpha(model, Xte_id, yte_id, bs=256)
    print(f"\n[ID classification] acc={id_acc:.4f}  (K={K})")

    # geometry stats from TRAIN-ID only
    Ftr = extract_feats(model, Xtr_id, bs=256)
    mu, inv_cov = fit_maha_shared(Ftr, ytr_id, K=K, shrink=a.maha_shrink)

    sid = score_batch_classwise(model, Xte_id, mu, inv_cov, tau=a.geo_tau, bs=256)
    scal = score_batch_classwise(model, Xte_calib_ood, mu, inv_cov, tau=a.geo_tau, bs=256) if len(Xte_calib_ood) else None
    sun  = score_batch_classwise(model, Xte_unseen_ood, mu, inv_cov, tau=a.geo_tau, bs=256) if len(Xte_unseen_ood) else None
    sp   = score_batch_classwise(model, Xte_pseudo_ood, mu, inv_cov, tau=a.geo_tau, bs=256) if len(Xte_pseudo_ood) else None

    def report_ood(name: str, sod: Dict[str, np.ndarray]):
        print(f"\n--- OOD set: {name} ---")
        y_true = np.concatenate([np.zeros(len(Xte_id)), np.ones(len(sod["evidence_ood"]))]).astype(np.int64)
        for key in ["evidence_ood", "maha_min", "fusion_geo"]:
            score = np.concatenate([sid[key], sod[key]])
            au = roc_auc_score(y_true, score)
            print(f"{key:>12s} | AUROC={au:.4f}")
            for tpr_t in [0.80, 0.90, 0.95]:
                fpr_t = fpr_at_tpr(y_true, score, tpr_t)
                print(f"  TPR@{tpr_t:.2f} -> FPR={fpr_t:.4f}")

    # real OOD sets
    if scal is not None:
        report_ood("calib_real_ood (seen in train as OOD-calib)", scal)
    if sun is not None:
        report_ood("unseen_real_ood (never used in training)", sun)

    # real-only combined (recommended main claim)
    real_parts = [d for d in [scal, sun] if d is not None]
    if real_parts:
        sod_real = {k: np.concatenate([d[k] for d in real_parts], axis=0) for k in ["evidence_ood", "maha_min", "fusion_geo"]}
        report_ood("REAL_OOD_ONLY (calib + unseen)", sod_real)

    # pseudo only (diagnostic)
    if sp is not None:
        report_ood("pseudo_ood (generated, capped)", sp)

    # all OOD (for completeness)
    parts = [d for d in [scal, sun, sp] if d is not None]
    if parts:
        sod_all = {k: np.concatenate([d[k] for d in parts], axis=0) for k in ["evidence_ood", "maha_min", "fusion_geo"]}
        report_ood("ALL_OOD (real + pseudo)", sod_all)

    print(f"\n[geo fusion params] tau={a.geo_tau} maha_shrink={a.maha_shrink}")
    print("Done.")


if __name__ == "__main__":
    main(Args())
