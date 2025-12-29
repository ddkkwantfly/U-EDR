# data_loader.py
# ============================================================
# SSVEP .mat loader + preprocessing + trial-level windowing
# For eeg tensor shaped: [targets, channels, time, trials] = [12, 8, 1114, 15]
#
# Core design:
# - within-subject first (one .mat file = one subject)
# - split by TRIAL index to avoid leakage
# - windowing uses onset index + start_s + win_s
#
# Output:
# - torch Dataset that yields (x, y) with x shape [1, C, L] (EEGNet-friendly)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

import torch
from torch.utils.data import Dataset, DataLoader


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class SSVEPConfig:
    fs: int = 256
    onset_1idx: int = 39         # dataset statement: onset at 39th sample (1-index)
    n_classes: int = 12

    # preprocessing
    bandpass_low: float = 6.0
    bandpass_high: float = 90.0
    bandpass_order: int = 4

    # windowing
    start_s: float = 1.0         # recommended stable start from your grid-search
    win_s: float = 3.0           # e.g. 3.0s for strong baseline, later use 1.0s
    per_channel_zscore: bool = True


# -------------------------
# IO
# -------------------------
def load_mat_eeg(mat_path: str | Path, key: str = "eeg") -> np.ndarray:
    """
    Load EEG from a .mat file.

    Expected shape: [targets, channels, time, trials]
    """
    mat_path = Path(mat_path)
    m = sio.loadmat(str(mat_path))
    if key not in m:
        keys = [k for k in m.keys() if not k.startswith("__")]
        raise KeyError(f"Key '{key}' not found in {mat_path}. Available keys: {keys}")

    eeg = np.asarray(m[key])
    if eeg.ndim != 4:
        raise ValueError(f"Expected eeg to be 4D [K,C,T,R], got shape={eeg.shape} in {mat_path}")

    return eeg.astype(np.float32, copy=False)


# -------------------------
# Preprocess
# -------------------------
def bandpass_filter(
    x: np.ndarray,
    fs: int,
    low: float,
    high: float,
    order: int = 4,
    axis: int = -1,
) -> np.ndarray:
    """
    Bandpass filter along the given axis (time axis).

    For your eeg shaped [K, C, T, R], use axis=2 (T dimension).
    """
    nyq = 0.5 * fs
    if not (0 < low < high < nyq):
        raise ValueError(f"Invalid bandpass: low={low}, high={high}, nyq={nyq}")

    b, a = butter(order, [low / nyq, high / nyq], btype="band")

    # filtfilt requires length along 'axis' > padlen. For safety:
    n = x.shape[axis]
    padlen = 3 * (max(len(a), len(b)) - 1)
    if n <= padlen:
        raise ValueError(
            f"Signal length along axis={axis} is {n}, but filtfilt padlen is {padlen}. "
            f"Choose the correct time axis or use longer signals."
        )

    y = filtfilt(b, a, x, axis=axis)
    return y.astype(np.float32, copy=False)



def zscore_channels(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Per-sample, per-channel z-score.
    x shape: [C, L] or [N, C, L]
    """
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


# -------------------------
# Windowing utilities
# -------------------------
def onset_0idx(cfg: SSVEPConfig) -> int:
    return cfg.onset_1idx - 1


def slice_window_indices(cfg: SSVEPConfig, start_s: Optional[float] = None, win_s: Optional[float] = None) -> Tuple[int, int]:
    """
    Convert (start_s, win_s) to [start_idx, end_idx) in sample indices,
    using cfg.onset as a reference anchor: start = onset + start_s*fs.
    """
    start_s = cfg.start_s if start_s is None else start_s
    win_s = cfg.win_s if win_s is None else win_s

    start = onset_0idx(cfg) + int(round(start_s * cfg.fs))
    L = int(round(win_s * cfg.fs))
    end = start + L
    return start, end


def extract_windows_single_subject(
    eeg: np.ndarray,
    cfg: SSVEPConfig,
    trial_ids: Sequence[int],
    start_s: Optional[float] = None,
    win_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract ONE window per (class, trial) for a single subject, with TRIAL-level selection.

    eeg: [K, C, T, R]
    trial_ids: list of trial indices used for this split (e.g., train_trials)
    returns:
      X: [N, C, L]
      y: [N]
    """
    K, C, T, R = eeg.shape
    if K != cfg.n_classes:
        raise ValueError(f"Expected {cfg.n_classes} classes, got {K}")
    if max(trial_ids) >= R:
        raise ValueError(f"trial_ids contains {max(trial_ids)} but trials={R}")

    s, e = slice_window_indices(cfg, start_s=start_s, win_s=win_s)
    if e > T:
        raise ValueError(f"Window exceeds available length: end={e} > T={T}. "
                         f"Try smaller win_s or smaller start_s.")

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for k in range(K):
        for r in trial_ids:
            x = eeg[k, :, s:e, r]  # [C, L]
            if cfg.per_channel_zscore:
                x = zscore_channels(x)
            X_list.append(x.astype(np.float32, copy=False))
            y_list.append(k)

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)  # [N, C, L]
    y = np.asarray(y_list, dtype=np.int64)
    return X, y

def extract_windows_multi_start_single_subject(
    eeg: np.ndarray,
    cfg: SSVEPConfig,
    trial_ids: Sequence[int],
    start_s_list: Sequence[float],
    win_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Like extract_windows_single_subject, but extract multiple windows per trial.

    eeg: [K, C, T, R]
    returns X: [N, C, L], y: [N]
    """
    K, C, T, R = eeg.shape
    win_s = cfg.win_s if win_s is None else win_s
    L = int(round(win_s * cfg.fs))

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    for k in range(K):
        for r in trial_ids:
            for start_s in start_s_list:
                s = onset_0idx(cfg) + int(round(start_s * cfg.fs))
                e = s + L
                if e > T:
                    continue
                x = eeg[k, :, s:e, r]
                if cfg.per_channel_zscore:
                    x = zscore_channels(x)
                X_list.append(x.astype(np.float32, copy=False))
                y_list.append(k)

    X = np.stack(X_list, axis=0).astype(np.float32, copy=False)
    y = np.asarray(y_list, dtype=np.int64)
    return X, y


def split_trials(
    n_trials: int,
    n_train: int = 10,
    n_val: int = 2,
    n_test: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Split trial indices (same split applied across all classes in a subject).
    Default for R=15: train=10, val=2, test=3.
    """
    if n_test is None:
        n_test = n_trials - n_train - n_val
    if n_train + n_val + n_test != n_trials:
        raise ValueError("n_train+n_val+n_test must equal n_trials")

    rng = np.random.RandomState(seed)
    idx = np.arange(n_trials)
    rng.shuffle(idx)

    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


# -------------------------
# PyTorch Dataset & Dataloaders
# -------------------------
class EEGWindowDataset(Dataset):
    """
    X: [N, C, L] float32
    y: [N] int64
    returns x as [1, C, L] for EEGNet-style models
    """
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"X must be [N,C,L], got {X.shape}")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(f"y must be [N], got {y.shape} for X {X.shape}")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, idx: int):
        x = self.X[idx].unsqueeze(0)  # [1, C, L]
        y = self.y[idx]
        return x, y


def build_subject_datasets(
    mat_path: str | Path,
    cfg: SSVEPConfig,
    mat_key: str = "eeg",
    seed: int = 0,
    n_train_trials: int = 10,
    n_val_trials: int = 2,
    n_test_trials: Optional[int] = None,
) -> Tuple[EEGWindowDataset, EEGWindowDataset, EEGWindowDataset, Dict[str, np.ndarray]]:
    """
    Full pipeline for ONE subject:
      load -> bandpass -> trial split -> window extraction -> datasets
    """
    eeg = load_mat_eeg(mat_path, key=mat_key)  # [K,C,T,R]

    # bandpass before windowing (recommended)
    eeg = bandpass_filter(
    eeg,
    fs=cfg.fs,
    low=cfg.bandpass_low,
    high=cfg.bandpass_high,
    order=cfg.bandpass_order,
    axis=2,   # <-- time dimension for [K, C, T, R]
    )

    K, C, T, R = eeg.shape
    splits = split_trials(
        n_trials=R,
        n_train=n_train_trials,
        n_val=n_val_trials,
        n_test=n_test_trials,
        seed=seed,
    )

    # Xtr, ytr = extract_windows_single_subject(eeg, cfg, splits["train"])
    # Xva, yva = extract_windows_single_subject(eeg, cfg, splits["val"])
    # Xte, yte = extract_windows_single_subject(eeg, cfg, splits["test"])
    # 建议先用 2s 窗 + 多起点
    cfg = cfg  # 不用改这行，只说明
    start_s_list = [1.0, 1.5, 2.0]  # 你也可以先 [1.0, 2.0]

    Xtr, ytr = extract_windows_multi_start_single_subject(eeg, cfg, splits["train"], start_s_list=start_s_list)
    Xva, yva = extract_windows_multi_start_single_subject(eeg, cfg, splits["val"], start_s_list=start_s_list)
    Xte, yte = extract_windows_multi_start_single_subject(eeg, cfg, splits["test"], start_s_list=start_s_list)

    ds_tr = EEGWindowDataset(Xtr, ytr)
    ds_va = EEGWindowDataset(Xva, yva)
    ds_te = EEGWindowDataset(Xte, yte)

    return ds_tr, ds_va, ds_te, splits


def build_subject_loaders(
    mat_path: str | Path,
    cfg: SSVEPConfig,
    mat_key: str = "eeg",
    seed: int = 0,
    n_train_trials: int = 10,
    n_val_trials: int = 2,
    n_test_trials: Optional[int] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """
    Convenience wrapper returning DataLoaders.
    """
    ds_tr, ds_va, ds_te, splits = build_subject_datasets(
        mat_path=mat_path,
        cfg=cfg,
        mat_key=mat_key,
        seed=seed,
        n_train_trials=n_train_trials,
        n_val_trials=n_val_trials,
        n_test_trials=n_test_trials,
    )

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    val_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)
    test_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    return train_loader, val_loader, test_loader, splits


# -------------------------
# Quick sanity check runner
# -------------------------
if __name__ == "__main__":
    # Example usage:
    cfg = SSVEPConfig(
        fs=256,
        onset_1idx=39,
        bandpass_low=6.0,
        bandpass_high=90.0,
        start_s=1.0,
        win_s=3.0,
        per_channel_zscore=True,
    )

    mat_path = "s1.mat"  # <-- change
    tr_loader, va_loader, te_loader, splits = build_subject_loaders(
        mat_path, cfg, mat_key="eeg", seed=0, batch_size=32
    )

    xb, yb = next(iter(tr_loader))
    print("splits:", {k: v.tolist() for k, v in splits.items()})
    print("batch x:", xb.shape, xb.dtype)  # [B, 1, C, L]
    print("batch y:", yb.shape, yb.dtype)



from glob import glob

def list_mat_files(root: str | Path, pattern: str = "*.mat") -> List[Path]:
    root = Path(root)
    files = sorted([Path(p) for p in glob(str(root / pattern))])
    if not files:
        raise FileNotFoundError(f"No .mat files found in {root} with pattern {pattern}")
    return files


def split_subjects(
    n_subjects: int,
    n_train: int,
    n_val: int,
    n_test: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if n_test is None:
        n_test = n_subjects - n_train - n_val
    if n_train + n_val + n_test != n_subjects:
        raise ValueError("n_train+n_val+n_test must equal n_subjects")

    rng = np.random.RandomState(seed)
    idx = np.arange(n_subjects)
    rng.shuffle(idx)
    return {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }


def build_multi_subject_arrays(
    mat_paths: Sequence[str | Path],
    cfg: SSVEPConfig,
    mat_key: str = "eeg",
    subject_ids: Optional[Sequence[int]] = None,
    # within-subject trial split (still used inside each subject to avoid leakage)
    trial_seed: int = 0,
    n_train_trials: int = 10,
    n_val_trials: int = 2,
    n_test_trials: Optional[int] = None,
    split_name: str = "train",  # "train" | "val" | "test"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For a list of subjects, create pooled arrays for one split.
    Returns:
      X: [N, C, L]
      y: [N]
      sid: [N]  (subject id per sample)
    """
    if split_name not in ("train", "val", "test"):
        raise ValueError("split_name must be train/val/test")

    if subject_ids is None:
        subject_ids = list(range(len(mat_paths)))
    if len(subject_ids) != len(mat_paths):
        raise ValueError("subject_ids length must match mat_paths")

    X_all, y_all, sid_all = [], [], []

    for sid, mp in zip(subject_ids, mat_paths):
        eeg = load_mat_eeg(mp, key=mat_key)          # [K,C,T,R]
        eeg = bandpass_filter(
            eeg, fs=cfg.fs,
            low=cfg.bandpass_low, high=cfg.bandpass_high,
            order=cfg.bandpass_order,
            axis=2,                                 # <-- time axis
        )

        K, C, T, R = eeg.shape
        splits = split_trials(R, n_train=n_train_trials, n_val=n_val_trials, n_test=n_test_trials, seed=trial_seed)

        trial_ids = splits[split_name]
        X, y = extract_windows_single_subject(eeg, cfg, trial_ids)
        X_all.append(X)
        y_all.append(y)
        sid_all.append(np.full((len(y),), sid, dtype=np.int64))

    X_all = np.concatenate(X_all, axis=0).astype(np.float32, copy=False)
    y_all = np.concatenate(y_all, axis=0).astype(np.int64, copy=False)
    sid_all = np.concatenate(sid_all, axis=0).astype(np.int64, copy=False)
    return X_all, y_all, sid_all


class EEGWindowDatasetWithSubject(Dataset):
    """
    Like EEGWindowDataset but also returns subject_id.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, sid: np.ndarray):
        if X.ndim != 3:
            raise ValueError(f"X must be [N,C,L], got {X.shape}")
        if y.ndim != 1 or sid.ndim != 1 or len(X) != len(y) or len(y) != len(sid):
            raise ValueError("X, y, sid lengths mismatch")

        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.sid = torch.from_numpy(sid).long()

    def __len__(self) -> int:
        return self.y.numel()

    def __getitem__(self, idx: int):
        x = self.X[idx].unsqueeze(0)  # [1,C,L]
        return x, self.y[idx], self.sid[idx]


def build_multi_subject_loaders(
    mat_dir: str | Path,
    cfg: SSVEPConfig,
    pattern: str = "*.mat",
    mat_key: str = "eeg",
    # subject split
    n_train_subjects: int = 7,
    n_val_subjects: int = 1,
    n_test_subjects: Optional[int] = None,
    subject_seed: int = 0,
    # within subject trial split
    trial_seed: int = 0,
    n_train_trials: int = 10,
    n_val_trials: int = 2,
    n_test_trials: Optional[int] = None,
    # loader
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    id_classes: Optional[Sequence[int]] = None,

) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, List[Path]]]:
    """
    Cross-subject split: subjects in train/val/test are disjoint.
    Inside each subject, we still split trials (optional but safe).
    """
    mats = list_mat_files(mat_dir, pattern=pattern)
    S = len(mats)

    subj_split = split_subjects(S, n_train_subjects, n_val_subjects, n_test_subjects, seed=subject_seed)
    train_mats = [mats[i] for i in subj_split["train"]]
    val_mats   = [mats[i] for i in subj_split["val"]]
    test_mats  = [mats[i] for i in subj_split["test"]]

    Xtr, ytr, sidtr = build_multi_subject_arrays(
        train_mats, cfg, mat_key=mat_key,
        subject_ids=subj_split["train"].tolist(),
        trial_seed=trial_seed,
        n_train_trials=n_train_trials, n_val_trials=n_val_trials, n_test_trials=n_test_trials,
        split_name="train",
    )
    Xva, yva, sidva = build_multi_subject_arrays(
        val_mats, cfg, mat_key=mat_key,
        subject_ids=subj_split["val"].tolist(),
        trial_seed=trial_seed,
        n_train_trials=n_train_trials, n_val_trials=n_val_trials, n_test_trials=n_test_trials,
        split_name="val",
    )
    Xte, yte, sidte = build_multi_subject_arrays(
        test_mats, cfg, mat_key=mat_key,
        subject_ids=subj_split["test"].tolist(),
        trial_seed=trial_seed,
        n_train_trials=n_train_trials, n_val_trials=n_val_trials, n_test_trials=n_test_trials,
        split_name="test",
    )

    if id_classes is not None:
      Xtr, ytr, sidtr = select_classes_and_remap_with_sid(Xtr, ytr, sidtr, id_classes)
      Xva, yva, sidva = select_classes_and_remap_with_sid(Xva, yva, sidva, id_classes)
      Xte, yte, sidte = select_classes_and_remap_with_sid(Xte, yte, sidte, id_classes)

    ds_tr = EEGWindowDatasetWithSubject(Xtr, ytr, sidtr)
    ds_va = EEGWindowDatasetWithSubject(Xva, yva, sidva)
    ds_te = EEGWindowDatasetWithSubject(Xte, yte, sidte)

    tr_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=pin_memory)
    va_loader = DataLoader(ds_va, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    te_loader = DataLoader(ds_te, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    info = {"train": train_mats, "val": val_mats, "test": test_mats}
    return tr_loader, va_loader, te_loader, info


def select_classes_and_remap(
    X: np.ndarray,
    y: np.ndarray,
    class_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only samples with y in class_ids, and remap labels to 0..len(class_ids)-1.
    """
    class_ids = list(class_ids)
    class_to_new = {c: i for i, c in enumerate(class_ids)}

    mask = np.isin(y, class_ids)
    X2 = X[mask]
    y2_old = y[mask]
    y2 = np.array([class_to_new[int(t)] for t in y2_old], dtype=np.int64)
    return X2.astype(np.float32, copy=False), y2


def select_classes_and_remap_with_sid(
    X: np.ndarray,
    y: np.ndarray,
    sid: np.ndarray,
    class_ids: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    class_ids = list(class_ids)
    class_to_new = {c: i for i, c in enumerate(class_ids)}

    mask = np.isin(y, class_ids)
    X2 = X[mask]
    y2_old = y[mask]
    sid2 = sid[mask]
    y2 = np.array([class_to_new[int(t)] for t in y2_old], dtype=np.int64)
    return X2.astype(np.float32, copy=False), y2, sid2.astype(np.int64, copy=False)
