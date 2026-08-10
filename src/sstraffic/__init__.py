"""Self-supervised pretraining for network traffic (sstraffic).

Pretrain a small custom transformer encoder on large unlabeled traffic flows
using a masked-feature reconstruction objective, then fine-tune the encoder on
a small labeled IDS set and compare against training the same encoder from
scratch -- especially in the low-label-data regime, which is the key selling
point of the project.
"""

__version__ = "0.1.0"

from .data import (
    ARCHETYPES,
    FEATURE_NAMES,
    N_FEATURES,
    Dataset,
    TrafficGenerator,
    make_dataset,
    standardize,
)
from .encoder import FeatureTransformer, build_encoder
from .pretrain import pretrain_encoder
from .finetune import train_classifier
from .evaluate import (
    accuracy_curve_plot,
    embeddings_np,
    loss_plot,
    tsne_scatter,
    write_metrics,
)

__all__ = [
    "ARCHETYPES",
    "FEATURE_NAMES",
    "N_FEATURES",
    "Dataset",
    "TrafficGenerator",
    "make_dataset",
    "standardize",
    "FeatureTransformer",
    "build_encoder",
    "pretrain_encoder",
    "train_classifier",
    "accuracy_curve_plot",
    "embeddings_np",
    "loss_plot",
    "tsne_scatter",
    "write_metrics",
]
