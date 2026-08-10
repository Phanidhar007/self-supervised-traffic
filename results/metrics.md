# Self-Supervised Pretraining for Network Traffic -- Metrics

_Real numbers from an actual CPU run of `python scripts/run_pipeline.py`._

## Setup

- Unlabeled pool: **4000** flows (8 archetypes, SSL never sees labels)
- Labeled set (full): **800** flows, **100** per class
- Held-out test set: **1600** flows (generated fresh, unseen in pretraining)
- Features per flow: **16** -- pkts_in, pkts_out, bytes_in, bytes_out, duration_s, dst_port, port_entropy, syn_ratio, ...
- Archetypes: web, dns, ssh, mail, video, portscan, bruteforce, exfil
- Encoder: custom transformer, d_model=64, heads=2, layers=2
- Pretraining objective: masked-feature reconstruction (mask fraction=0.3) + masked-view contrastive auxiliary
- Pretraining epochs: **30** | Fine-tune epochs: **30** | Fine-tune seeds averaged: **3**
- Device: cpu | Total runtime: **176.1s** | Seed: 0

## Pretraining loss (masked-feature MSE)

- Epoch 1: `1.5469`
- Epoch 2: `1.0217`
- Epoch 3: `1.0140`
- Epoch 4: `0.9874`
- Epoch 5: `0.9803`
- Epoch 6: `0.9464`
- Epoch 7: `0.8875`
- Epoch 8: `0.7847`
- Epoch 9: `0.7573`
- Epoch 10: `0.6983`
- Epoch 11: `0.7134`
- Epoch 12: `0.6760`
- Epoch 13: `0.6613`
- Epoch 14: `0.6452`
- Epoch 15: `0.6078`
- Epoch 16: `0.5929`
- Epoch 17: `0.5665`
- Epoch 18: `0.5437`
- Epoch 19: `0.5362`
- Epoch 20: `0.5287`
- Epoch 21: `0.5106`
- Epoch 22: `0.4981`
- Epoch 23: `0.4891`
- Epoch 24: `0.4953`
- Epoch 25: `0.4795`
- Epoch 26: `0.4820`
- Epoch 27: `0.4870`
- Epoch 28: `0.4780`
- Epoch 29: `0.4807`
- Epoch 30: `0.4780`

## Key result -- fine-tuned accuracy vs label fraction

| Label fraction | Labeled flows used | Pretrained (SSL) | From scratch | Gain (pretrained - scratch) |
|---|---:|---:|---:|---:|
| 5% | 40 | **83.67%** | 58.73% | **+24.94 pts** |
| 10% | 80 | **87.88%** | 66.10% | **+21.77 pts** |
| 20% | 160 | **91.38%** | 85.21% | **+6.17 pts** |

> Low-label highlight: at **5%** of labels (only 40 flows), the pretrained encoder reaches **83.7%** vs **58.7%** from scratch -- a **+24.9 point** improvement. The gain shrinks as labels grow (+6.2 pts at 20%), exactly the expected self-supervised signature: pretraining transfers structure learned from unlabeled traffic, so labels matter less.

## Figures

- `results/figures/label_fraction_accuracy.png` -- accuracy comparison
- `results/figures/pretraining_loss.png` -- pretraining loss curve
- `results/figures/tsne_clusters.png` -- t-SNE of encoder embeddings
