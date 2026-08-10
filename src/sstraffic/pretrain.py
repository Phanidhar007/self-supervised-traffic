"""Self-supervised pretraining on the unlabeled traffic pool.

Primary objective -- masked-feature reconstruction
    A fraction of feature dimensions is masked per sample (replaced by a
    learnable [MASK] embedding); the encoder is trained to regress the raw
    (standardized) feature values back, MSE on masked positions only.

Auxiliary objective -- masked-view contrastive (optional, ON by default)
    Two augmented views of the same flow (independent feature masking +
    Gaussian noise) must map to nearby sample embeddings (InfoNCE). This
    sharpens the sample-level representation, which is what makes the
    representation transferable at very low label budgets.

The losses are combined:  L = lambda_rec * L_mask + lambda_con * L_ntx.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import FeatureTransformer, build_encoder


def _ntx_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Standard NT-Xent (InfoNCE) over in-batch negatives."""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    z = F.normalize(z, dim=1)
    sim = (z @ z.T) / temperature
    sim = sim - torch.eye(2 * B, device=z.device) * 1e9
    labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    return F.cross_entropy(sim, labels)


def pretrain_encoder(
    X_unlabeled: np.ndarray,
    n_features: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 2,
    dropout: float = 0.1,
    epochs: int = 30,
    batch_size: int = 256,
    lr: float = 2e-3,
    mask_frac: float = 0.3,
    aug_noise: float = 0.2,
    temperature: float = 0.15,
    lambda_con: float = 1.0,
    lambda_rec: float = 1.0,
    use_contrastive: bool = True,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[FeatureTransformer, list[dict]]:
    """Pretrain on the unlabeled pool.

    Returns (encoder, history) where history is a per-epoch list of
    ``{"loss", "rec", "con"}`` (mean over batches).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_encoder(n_features, d_model, n_heads, n_layers, dropout).to(device)
    proj = torch.nn.Sequential(
        torch.nn.Linear(d_model, d_model),
        torch.nn.ReLU(),
        torch.nn.Linear(d_model, 32),
    ).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) + list(proj.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    X = torch.from_numpy(X_unlabeled.astype(np.float32))
    n = X.shape[0]
    n_batches = int(np.ceil(n / batch_size))
    rng = np.random.default_rng(seed + 1)
    history: list[dict] = []

    model.train()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        tot_loss = tot_rec = tot_con = 0.0
        n_seen = 0
        for bi in range(n_batches):
            idx = perm[bi * batch_size : (bi + 1) * batch_size]
            xb = X[idx].to(device)
            B = xb.shape[0]

            # Two augmented views: independent feature masking + noise.
            m1 = (torch.rand(B, n_features, device=device) < mask_frac)
            m2 = (torch.rand(B, n_features, device=device) < mask_frac)
            v1 = xb + torch.randn_like(xb) * aug_noise
            v2 = xb + torch.randn_like(xb) * aug_noise
            v1[m1] = 0.0
            v2[m2] = 0.0

            optimizer.zero_grad()

            t1 = model.forward(v1, m1)   # single forward per view
            t2 = model.forward(v2, m2)

            if use_contrastive:
                e1 = model.sample_embedding_from_tokens(t1)
                e2 = model.sample_embedding_from_tokens(t2)
                loss_con = _ntx_loss(proj(e1), proj(e2), temperature)
            else:
                loss_con = xb.sum() * 0.0

            preds1 = model.reg_head(model.head_norm(t1[:, 1:])).squeeze(-1)
            preds2 = model.reg_head(model.head_norm(t2[:, 1:])).squeeze(-1)
            loss_rec = 0.5 * (
                F.mse_loss(preds1[m1], xb[m1]) + F.mse_loss(preds2[m2], xb[m2])
            )

            loss = lambda_rec * loss_rec + lambda_con * loss_con
            loss.backward()
            optimizer.step()

            tot_loss += loss.item() * B
            tot_rec += loss_rec.item() * B
            tot_con += loss_con.item() * B
            n_seen += B

        scheduler.step()
        history.append(
            {
                "loss": tot_loss / n_seen,
                "rec": tot_rec / n_seen,
                "con": tot_con / n_seen,
            }
        )

    return model.cpu(), history
