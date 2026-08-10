# Self-Supervised Traffic — Demo (AI Shield)

A **live results dashboard** is deployed at https://self-supervised-traffic.vercel.app (real metrics + figures, AI Shield theme). This Streamlit app is the full local version, showing the *measured*
properties of the pipeline in the AI Shield dark theme.

## Run

```bash
python scripts/run_pipeline.py   # once -- produces results/figures + results/summary.json
streamlit run demo/app.py
```

## What it shows

- **Low-label accuracy comparison** -- fine-tuned accuracy vs label fraction,
  pretrained vs from-scratch, with the low-label regime highlighted.
- **Pretraining loss curve** -- masked-feature reconstruction MSE per epoch.
- **t-SNE embedding clusters** -- encoder embeddings of the unlabeled pool,
  colored by traffic archetype (attack classes in red).
- **Stat cards** -- gain at the lowest label fraction, unlabeled pool size,
  final pretrain MSE, test-set size.
- **Metrics table** -- accuracy per label fraction with the pretraining gain.

Everything is read from `results/figures/*.png` and `results/summary.json`
produced by the real run, so the demo always reflects the measured numbers.


## 🌐 Live demo

https://self-supervised-traffic.vercel.app — real metrics + figures dashboard (AI Shield theme).
