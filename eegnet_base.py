# train_eegnet.py
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import SSVEPConfig, build_subject_loaders

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EEGNet(nn.Module):
    """
    EEGNet-like model for input [B, 1, C, T]
    """
    def __init__(
        self,
        n_channels: int,
        n_samples: int,
        n_classes: int,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.5,
        kernel_length: int = 64,
    ):
        super().__init__()
        F2 = F1 * D

        # Temporal conv
        self.conv1 = nn.Conv2d(1, F1, kernel_size=(1, kernel_length),
                               padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Depthwise spatial conv
        self.depthwise = nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1),
                                   groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.drop1 = nn.Dropout(dropout)

        # Separable conv
        self.sep_depth = nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, 16),
                                   padding=(0, 8), groups=F1 * D, bias=False)
        self.sep_point = nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8))
        self.drop2 = nn.Dropout(dropout)

        # Infer feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_samples)
            feat = self._features(dummy)
            feat_dim = feat.shape[1]

        self.classifier = nn.Linear(feat_dim, n_classes)

    def _features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)

        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.sep_depth(x)
        x = self.sep_point(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x.flatten(1)

    def forward(self, x):
        feat = self._features(x)
        return self.classifier(feat)


@torch.no_grad()
def eval_acc(model: nn.Module, loader, criterion):
    model.eval()
    total, correct = 0, 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += float(loss) * y.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.size(0)
    return loss_sum / total, correct / total


def train_one_subject(
    mat_path: str,
    epochs: int = 200,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 0,
):
    # dataloader config: your current best CCA stable setting
    cfg = SSVEPConfig(
        start_s=1.0,
        win_s=2.0,
        bandpass_low=6.0,
        bandpass_high=90.0,
        per_channel_zscore=True,
    )

    train_loader, val_loader, test_loader, splits = build_subject_loaders(
        mat_path,
        cfg,
        mat_key="eeg",
        seed=seed,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
    )

    # infer dimensions from one batch
    xb, yb = next(iter(train_loader))
    _, _, C, T = xb.shape
    n_classes = int(torch.max(yb).item() + 1)

    model = EEGNet(n_channels=C, n_samples=T, n_classes=n_classes,
                   F1=8, D=2, dropout=0.25, kernel_length=64).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # opt = torch.optim.Adam(model.parameters(), lr=lr)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)


    best_val = -1.0
    best_state = None
    patience = 30
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

        val_loss, val_acc = eval_acc(model, val_loader, criterion)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if ep == 1 or ep % 10 == 0:
            tr_loss, tr_acc = eval_acc(model, train_loader, criterion)
            print(f"Epoch {ep:03d} | train acc {tr_acc:.3f} | val acc {val_acc:.3f} | val loss {val_loss:.4f}")

        if bad >= patience:
            print(f"Early stop at epoch {ep} (best val acc {best_val:.3f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_acc = eval_acc(model, test_loader, criterion)
    print("\nSplits:", {k: v.tolist() for k, v in splits.items()})
    print(f"Best val acc: {best_val:.3f}")
    print(f"Test acc:     {test_acc:.3f}  (loss {test_loss:.4f})")

    return model


if __name__ == "__main__":
    # 改成你的文件名
    train_one_subject("s1.mat", epochs=200, lr=1e-3, batch_size=64, seed=0)
