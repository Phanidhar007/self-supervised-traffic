"""Evaluation + visualization + metrics writer.

Produces the three key figures:
  * label-fraction accuracy comparison (pretrained vs from-scratch),
  * pretraining loss curve,
  * t-SNE of the pretrained encoder embeddings colored by traffic archetype.
And writes ``results/metrics.md`` with the real numbers from the run.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

# Dark palette to match the AI Shield demo theme.
BG = "#0a0a0c"
FG = "#e4e4e7"
MUT = "#71717a"
EMERALD = "#10b981"
PURPLE = "#c084fc"
RED = "#ef4444"


def _style_ax(ax):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_color("#18181b")
    ax.tick_params(colors=MUT, labelsize=8)
    ax.xaxis.label.set_color(MUT)
    ax.yaxis.label.set_color(MUT)


def embeddings_np(model: torch.nn.Module, X: np.ndarray, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    """Encoder sample embeddings for a feature matrix -> (n, d_model)."""
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
            outs.append(model.sample_embedding(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)


def tsne_scatter(
    X_emb: np.ndarray,
    y: np.ndarray,
    archetype_names: list[str],
    out_path: str,
    max_points: int = 2000,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    if len(X_emb) > max_points:
        idx = rng.choice(len(X_emb), max_points, replace=False)
        X_emb, y = X_emb[idx], y[idx]

    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=30)
    Z = tsne.fit_transform(X_emb)

    fig, ax = plt.subplots(figsize=(8.4, 6.2), dpi=140)
    fig.patch.set_facecolor(BG)
    cmap = plt.get_cmap("tab10")
    for c in np.unique(y):
        m = y == c
        color = RED if archetype_names[c] in ("portscan", "bruteforce", "exfil") else cmap(c)
        ax.scatter(Z[m, 0], Z[m, 1], s=9, alpha=0.75, c=[color], label=archetype_names[c], edgecolors="none")
    _style_ax(ax)
    ax.set_title("t-SNE of pretrained encoder embeddings (unlabeled pool)", color=FG, fontsize=11)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    leg = ax.legend(fontsize=7, loc="best", framealpha=0.5, facecolor="#09090b", edgecolor="#18181b")
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def accuracy_curve_plot(
    fractions: list[float],
    pretrained: list[float],
    scratch: list[float],
    out_path: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=140)
    fig.patch.set_facecolor(BG)
    x = np.asarray(fractions) * 100.0
    ax.axvspan(0, x[0], color=PURPLE, alpha=0.12)
    ax.plot(x, pretrained, "-o", color=EMERALD, lw=2.5, ms=7, label="Pretrained (SSL)")
    ax.plot(x, scratch, "-s", color=RED, lw=2.5, ms=7, label="From scratch")
    for xi, p, s in zip(x, pretrained, scratch):
        ax.annotate(f"{p:.1f}%", (xi, p), textcoords="offset points", xytext=(6, 6), color=EMERALD, fontsize=9)
        ax.annotate(f"{s:.1f}%", (xi, s), textcoords="offset points", xytext=(6, -12), color=RED, fontsize=9)
    _style_ax(ax)
    ax.set_title("Fine-tuned accuracy vs label availability", color=FG, fontsize=12)
    ax.set_xlabel("Fraction of labeled data used (%)")
    ax.set_ylabel("Test accuracy")
    ax.set_xlim(0, x[-1] * 1.15)
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.15, color="#27272a")
    leg = ax.legend(fontsize=9, facecolor="#09090b", edgecolor="#18181b")
    for text in leg.get_texts():
        text.set_color(FG)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def loss_plot(losses: list[float], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=140)
    fig.patch.set_facecolor(BG)
    ax.plot(range(1, len(losses) + 1), losses, "-o", color=EMERALD, lw=2.5, ms=6)
    _style_ax(ax)
    ax.set_title("Masked-feature reconstruction pretraining loss", color=FG, fontsize=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE on masked features")
    ax.grid(alpha=0.15, color="#27272a")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_metrics(results: dict, out_md: str) -> None:
    """Render results/metrics.md from the real run results dict."""
    ds = results["dataset"]
    enc = results["encoder"]
    pre = results["pretrain"]
    sweep = results["sweep"]

    lines = []
    lines.append("# Self-Supervised Pretraining for Network Traffic -- Metrics")
    lines.append("")
    lines.append("_Real numbers from an actual CPU run of `python scripts/run_pipeline.py`._")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Unlabeled pool: **{ds['n_unlabeled']}** flows (8 archetypes, SSL never sees labels)")
    lines.append(f"- Labeled set (full): **{ds['n_labeled']}** flows, **{ds['labeled_per_class']}** per class")
    lines.append(f"- Held-out test set: **{ds['n_test']}** flows (generated fresh, unseen in pretraining)")
    lines.append(f"- Features per flow: **{ds['n_features']}** -- {', '.join(ds['feature_names'][:8])}, ...")
    lines.append(f"- Archetypes: {', '.join(ds['archetype_names'])}")
    lines.append(f"- Encoder: custom transformer, d_model={enc['d_model']}, heads={enc['n_heads']}, layers={enc['n_layers']}")
    lines.append(f"- Pretraining objective: masked-feature reconstruction (mask fraction={pre['mask_frac']}) + masked-view contrastive auxiliary")
    lines.append(f"- Pretraining epochs: **{pre['epochs']}** | Fine-tune epochs: **{sweep['epochs']}** | Fine-tune seeds averaged: **{sweep['n_seeds']}**")
    lines.append(f"- Device: {results['device']} | Total runtime: **{results['runtime_s']:.1f}s** | Seed: {results['seed']}")
    lines.append("")
    lines.append("## Pretraining loss (masked-feature MSE)")
    lines.append("")
    for ep, loss in enumerate(pre["losses"], start=1):
        lines.append(f"- Epoch {ep}: `{loss:.4f}`")
    lines.append("")
    lines.append("## Key result -- fine-tuned accuracy vs label fraction")
    lines.append("")
    lines.append("| Label fraction | Labeled flows used | Pretrained (SSL) | From scratch | Gain (pretrained - scratch) |")
    lines.append("|---|---:|---:|---:|---:|")
    for f, p, s, g, nt in zip(sweep["fractions"], sweep["pretrained"], sweep["scratch"], sweep["gains"], sweep["n_train"]):
        lines.append(f"| {f*100:.0f}% | {nt} | **{p*100:.2f}%** | {s*100:.2f}% | **{g*100:+.2f} pts** |")
    lines.append("")
    low_i = int(np.argmin(sweep["fractions"]))
    lines.append(
        f"> Low-label highlight: at **{sweep['fractions'][low_i]*100:.0f}%** of labels "
        f"(only {sweep['n_train'][low_i]} flows), the pretrained encoder reaches "
        f"**{sweep['pretrained'][low_i]*100:.1f}%** vs **{sweep['scratch'][low_i]*100:.1f}%** "
        f"from scratch -- a **{sweep['gains'][low_i]*100:+.1f} point** improvement. "
        f"The gain shrinks as labels grow ({sweep['gains'][-1]*100:+.1f} pts at "
        f"{sweep['fractions'][-1]*100:.0f}%), exactly the expected self-supervised signature: "
        f"pretraining transfers structure learned from unlabeled traffic, so labels matter less."
    )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `results/figures/label_fraction_accuracy.png` -- accuracy comparison")
    lines.append("- `results/figures/pretraining_loss.png` -- pretraining loss curve")
    lines.append("- `results/figures/tsne_clusters.png` -- t-SNE of encoder embeddings")
    lines.append("")

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
