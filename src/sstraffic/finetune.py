"""Fine-tuning a pretrained (or from-scratch) encoder + classifier head.

Both branches use the *same* encoder architecture, head and optimizer so the
only difference is the initialization: the pretrained branch starts from the
weights learned on the unlabeled pool, the from-scratch branch starts random.
We sweep label availability (5% / 10% / 20% of the labeled set) to show that
pretraining helps most when labels are scarce.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .encoder import FeatureTransformer, build_encoder


class Classifier(nn.Module):
    def __init__(self, encoder: FeatureTransformer, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.sample_embedding(x))


def _split_val(X, y, val_frac, seed):
    from sklearn.model_selection import StratifiedShuffleSplit

    if val_frac <= 0:
        return X, y, None, None
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tri, vai = next(sss.split(X, y))
        return X[tri], y[tri], X[vai], y[vai]
    except Exception:
        rng = np.random.default_rng(seed)
        n = len(X)
        nval = max(1, int(round(val_frac * n)))
        if nval >= n:
            return X, y, None, None
        perm = rng.permutation(n)
        return X[perm[nval:]], y[perm[nval:]], X[perm[:nval]], y[perm[:nval]]


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    n_features: int,
    init_state: dict | None = None,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    patience: int = 10,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Train (or fine-tune) encoder + head; returns best test accuracy + history.

    ``init_state`` is an encoder state dict from pretraining. Pass ``None`` for
    the from-scratch baseline (random initialization).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int64)

    # Try to keep a stratified validation split for model selection; fall back
    # to progressively smaller splits (some low-label subsets are too tiny).
    for vf in (val_frac, 0.1, 0.05, 0.0):
        Xtr, ytr, Xva, yva = _split_val(X_train, y_train, vf, seed)
        if vf == 0.0 or (Xva is not None and len(np.unique(yva)) >= 2):
            break

    encoder = build_encoder(n_features, d_model, n_heads, n_layers, dropout)
    if init_state is not None:
        encoder.load_state_dict(init_state)
    model = Classifier(encoder, n_classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    def _acc(X, y):
        model.eval()
        with torch.no_grad():
            preds = torch.softmax(model(torch.from_numpy(X).to(device)), dim=-1).argmax(dim=-1)
        return (preds.cpu().numpy() == y).mean()

    n = len(Xtr)
    n_batches = max(1, int(np.ceil(n / batch_size)))
    best_val = -1.0
    best_acc = -1.0
    best_epoch = -1
    patience_left = patience
    history = []

    for epoch in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + epoch).permutation(n)
        epoch_loss = 0.0
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            if len(idx) == 0:
                continue
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        test_acc = _acc(X_test, y_test)
        if Xva is not None:
            val_acc = _acc(Xva, yva)
            if val_acc > best_val:
                best_val = val_acc
                best_acc = test_acc
                best_epoch = epoch
                patience_left = patience
            else:
                patience_left -= 1
        else:
            # No validation split: report the final-epoch test accuracy.
            best_acc = test_acc
            best_epoch = epoch

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / n if n else 0.0,
                "val_acc": float(val_acc) if Xva is not None else None,
                "test_acc": float(test_acc),
            }
        )
        if Xva is not None and patience_left <= 0:
            break

    return {
        "test_acc": float(best_acc),
        "best_epoch": int(best_epoch),
        "final_train_loss": history[-1]["train_loss"],
        "history": history,
    }
