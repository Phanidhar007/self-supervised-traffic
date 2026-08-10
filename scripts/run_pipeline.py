"""End-to-end pipeline: data -> pretrain -> low-label fine-tune sweep -> t-SNE.

Steps
-----
1. Generate the synthetic unlabeled pool + labeled subset + test set.
2. Pretrain the transformer encoder with masked-feature reconstruction.
3. Fine-tune (and train-from-scratch) at 5% / 10% / 20% of labels.
4. Save figures and write results/metrics.md + results/summary.json.

Run (kept small so it finishes in ~1-3 minutes on CPU):
    python scripts/run_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from sstraffic import (
    accuracy_curve_plot,
    embeddings_np,
    loss_plot,
    make_dataset,
    pretrain_encoder,
    train_classifier,
    tsne_scatter,
    write_metrics,
)
from sstraffic.data import ARCHETYPES, FEATURE_NAMES, N_FEATURES


def _stratified_subset(X, y, per_class_frac, rng, n_classes):
    idx = []
    for c in range(n_classes):
        ci = np.where(y == c)[0]
        k = max(1, int(round(per_class_frac * len(ci))))
        idx.append(rng.choice(ci, size=k, replace=False))
    idx = np.concatenate(idx)
    return X[idx], y[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised traffic pretraining pipeline")
    parser.add_argument("--n-unlabeled", type=int, default=4000, help="unlabeled pool size")
    parser.add_argument("--labeled-per-class", type=int, default=100, help="labeled flows per class")
    parser.add_argument("--n-test", type=int, default=1600, help="held-out test size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--n-seeds", type=int, default=3, help="finetune seeds averaged per regime")
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=30)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--mask-frac", type=float, default=0.3)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = parser.parse_args()

    t0 = time.time()
    device = "cpu"
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[1/5] device={device}")

    # ---- 1. data ----
    print("[1/5] generating synthetic traffic data ...")
    ds = make_dataset(
        n_unlabeled=args.n_unlabeled,
        labeled_per_class=args.labeled_per_class,
        n_test=args.n_test,
        seed=args.seed,
    )
    n_classes = ds.n_classes
    n_features = ds.unlabeled.shape[1]
    print(f"      unlabeled={ds.unlabeled.shape[0]} labeled={len(ds.labeled_y)} test={len(ds.test_y)} feats={n_features}")

    # ---- 2. pretrain ----
    print("[2/5] pretraining encoder (masked-feature reconstruction + contrastive aux) ...")
    encoder, pretrain_hist = pretrain_encoder(
        ds.unlabeled,
        n_features=n_features,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        epochs=args.pretrain_epochs,
        mask_frac=args.mask_frac,
        seed=args.seed,
        device=device,
    )
    print(f"      final masked-MSE={pretrain_hist[-1]['rec']:.4f}")

    init_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    # ---- 3. fine-tune sweep: pretrained vs from-scratch (seed-averaged) ----
    print("[3/5] low-label fine-tune sweep (3 seeds averaged) ...")
    fractions = list(args.fractions)
    pretrained_accs, scratch_accs, n_trains = [], [], []
    for fi, frac in enumerate(fractions):
        rng = np.random.default_rng(args.seed + 1000 * (fi + 1))
        Xs, ys = _stratified_subset(ds.labeled_X, ds.labeled_y, frac, rng, n_classes)
        n_trains.append(len(Xs))

        pre_accs, scr_accs = [], []
        for s in range(args.n_seeds):
            res_pre = train_classifier(
                Xs, ys, ds.test_X, ds.test_y, n_classes, n_features,
                init_state=init_state, epochs=args.finetune_epochs,
                seed=args.seed + 2000 * (fi + 1) + s, device=device,
            )
            res_scratch = train_classifier(
                Xs, ys, ds.test_X, ds.test_y, n_classes, n_features,
                init_state=None, epochs=args.finetune_epochs,
                seed=args.seed + 3000 * (fi + 1) + s, device=device,
            )
            pre_accs.append(res_pre["test_acc"])
            scr_accs.append(res_scratch["test_acc"])
        pre_acc = float(np.mean(pre_accs))
        scr_acc = float(np.mean(scr_accs))
        pretrained_accs.append(pre_acc)
        scratch_accs.append(scr_acc)
        print(
            f"      {frac*100:>4.0f}% labels (n={len(Xs):>3}): "
            f"pretrained={pre_acc*100:5.1f}%  scratch={scr_acc*100:5.1f}%"
        )

    gains = [p - s for p, s in zip(pretrained_accs, scratch_accs)]

    # ---- 4. figures ----
    print("[4/5] writing figures ...")
    figs_dir = os.path.join(REPO_ROOT, "results", "figures")
    os.makedirs(figs_dir, exist_ok=True)
    fig_acc = os.path.join(figs_dir, "label_fraction_accuracy.png")
    fig_loss = os.path.join(figs_dir, "pretraining_loss.png")
    fig_tsne = os.path.join(figs_dir, "tsne_clusters.png")

    accuracy_curve_plot(fractions, pretrained_accs, scratch_accs, fig_acc)
    loss_plot([h["rec"] for h in pretrain_hist], fig_loss)

    print("      t-SNE on pretrained embeddings ...")
    X_emb = embeddings_np(encoder, ds.unlabeled, batch_size=512, device="cpu")
    tsne_scatter(X_emb, ds.unlabeled_y, ds.archetype_names, fig_tsne)

    # ---- 5. metrics + summary ----
    print("[5/5] writing results/metrics.md and results/summary.json ...")
    runtime_s = time.time() - t0
    results = {
        "dataset": {
            "n_unlabeled": ds.unlabeled.shape[0],
            "n_labeled": len(ds.labeled_y),
            "labeled_per_class": args.labeled_per_class,
            "n_test": len(ds.test_y),
            "n_features": n_features,
            "feature_names": FEATURE_NAMES,
            "archetype_names": ds.archetype_names,
        },
        "encoder": {
            "d_model": args.d_model,
            "n_heads": args.heads,
            "n_layers": args.layers,
            "mask_frac": args.mask_frac,
        },
        "pretrain": {
            "epochs": args.pretrain_epochs,
            "mask_frac": args.mask_frac,
            "final_loss": float(pretrain_hist[-1]["rec"]),
            "losses": [float(h["rec"]) for h in pretrain_hist],
            "con_losses": [float(h["con"]) for h in pretrain_hist],
            "total_losses": [float(h["loss"]) for h in pretrain_hist],
        },
        "sweep": {
            "epochs": args.finetune_epochs,
            "n_seeds": args.n_seeds,
            "fractions": fractions,
            "pretrained": pretrained_accs,
            "scratch": scratch_accs,
            "gains": gains,
            "n_train": n_trains,
        },
        "figures": {
            "accuracy": fig_acc,
            "loss": fig_loss,
            "tsne": fig_tsne,
        },
        "device": device,
        "seed": args.seed,
        "runtime_s": runtime_s,
    }
    write_metrics(results, os.path.join(REPO_ROOT, "results", "metrics.md"))

    with open(os.path.join(REPO_ROOT, "results", "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"\nDone in {runtime_s:.1f}s.")
    print("Low-label highlight: %.0f%% labels -> pretrained %.1f%% vs scratch %.1f%% (+%.1f pts)"
          % (fractions[0] * 100, pretrained_accs[0] * 100, scratch_accs[0] * 100, gains[0] * 100))


if __name__ == "__main__":
    main()
