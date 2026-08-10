"""Synthetic network-traffic flow generator with traffic archetypes.

No public datasets (CICIDS / UNSW-NB15) are downloaded -- the spec explicitly
allows synthetic data. Flows are sampled from a handful of underlying
"traffic archetypes" (web, dns, ssh, mail, video, portscan, bruteforce, exfil)
so the unlabeled pool has real cluster structure that self-supervised
pretraining can discover and that shows up in t-SNE.

Each archetype is defined by a prototype vector of realistic flow features plus
a per-feature noise scale. Sampling is additive Gaussian noise on raw feature
units (multiplicative spread captured by per-archetype sigma), which yields
overlapping-but-clustered classes -- a realistic stand-in for real traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Realistic-ish flow features (16 dimensions).
FEATURE_NAMES = [
    "pkts_in",
    "pkts_out",
    "bytes_in",
    "bytes_out",
    "duration_s",
    "dst_port",
    "port_entropy",
    "syn_ratio",
    "fin_ratio",
    "ack_ratio",
    "protocol",
    "flow_rate_ps",
    "hour_of_day",
    "pkt_size_mean",
    "pkt_size_std",
    "ttl_mean",
]
N_FEATURES = len(FEATURE_NAMES)

# Feature-index groups used for clipping / rounding after sampling.
_COUNT_FEATS = [0, 1, 2, 3, 4, 11]          # counts / durations / rates >= 0
_FRAC_FEATS = [6, 7, 8, 9]                  # ratios in [0, 1]
_PROTO_FEAT = 10                            # discrete 0=TCP,1=UDP,2=ICMP
_PORT_FEAT = 5
_HOUR_FEAT = 12
_SIZE_FEATS = [13, 14]
_TTL_FEAT = 15


@dataclass
class Archetype:
    name: str
    is_attack: bool
    mean: np.ndarray
    sigma: np.ndarray


def _pt(values):
    return np.asarray(values, dtype=float)


def _arch(name, attack, mean, sigma):
    return Archetype(name=name, is_attack=attack, mean=_pt(mean), sigma=_pt(sigma))


# Each row: pkts_in, pkts_out, bytes_in, bytes_out, duration_s, dst_port,
#           port_entropy, syn_ratio, fin_ratio, ack_ratio, protocol,
#           flow_rate_ps, hour_of_day, pkt_size_mean, pkt_size_std, ttl_mean
ARCHETYPES = [
    _arch(
        "web", False,
        [8, 10, 3000, 1500, 1.2, 443, 0.25, 0.15, 0.10, 0.75, 0, 20, 14, 420, 180, 64],
        [1.5, 2.0, 700, 400, 0.4, 60, 0.08, 0.04, 0.03, 0.05, 0.2, 5, 2, 90, 60, 2],
    ),
    _arch(
        "dns", False,
        [2, 1.5, 160, 90, 0.02, 53, 0.18, 0.05, 0.02, 0.55, 1, 6, 16, 120, 30, 64],
        [0.6, 0.4, 40, 25, 0.008, 10, 0.07, 0.02, 0.01, 0.06, 0.3, 2, 3, 25, 8, 2],
    ),
    _arch(
        "ssh", False,
        [55, 60, 22000, 24000, 150, 22, 0.12, 0.06, 0.03, 0.90, 0, 1.2, 10, 310, 120, 64],
        [12, 14, 4000, 4500, 40, 5, 0.06, 0.02, 0.02, 0.05, 0.2, 0.3, 2, 60, 40, 2],
    ),
    _arch(
        "mail", False,
        [30, 18, 10000, 45000, 25, 25, 0.15, 0.12, 0.08, 0.70, 0, 3, 9, 480, 260, 64],
        [8, 5, 2000, 9000, 8, 5, 0.07, 0.03, 0.03, 0.06, 0.2, 0.8, 2, 90, 80, 2],
    ),
    _arch(
        "video", False,
        [240, 14, 320000, 9000, 90, 1935, 0.45, 0.08, 0.05, 0.85, 0, 55, 20, 1150, 420, 64],
        [40, 4, 60000, 2500, 25, 200, 0.10, 0.02, 0.02, 0.05, 0.2, 10, 2, 200, 120, 2],
    ),
    _arch(
        "portscan", True,
        [350, 6, 6000, 400, 0.6, 30000, 0.95, 0.96, 0.01, 0.03, 0, 600, 2, 58, 8, 64],
        [60, 2, 1500, 120, 0.2, 15000, 0.05, 0.03, 0.01, 0.02, 0.2, 120, 1, 6, 3, 2],
    ),
    _arch(
        "bruteforce", True,
        [420, 390, 95000, 65000, 40, 22, 0.20, 0.35, 0.05, 0.60, 0, 70, 3, 150, 70, 64],
        [80, 70, 15000, 12000, 12, 5, 0.08, 0.06, 0.02, 0.08, 0.2, 15, 1, 40, 25, 2],
    ),
    _arch(
        "exfil", True,
        [12, 2800, 2500, 9000000, 60, 443, 0.30, 0.10, 0.05, 0.85, 0, 220, 1, 1400, 120, 64],
        [3, 500, 800, 2000000, 15, 60, 0.08, 0.03, 0.02, 0.06, 0.2, 50, 1, 120, 50, 2],
    ),
]

ARCHETYPE_NAMES = [a.name for a in ARCHETYPES]


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / (std + 1e-8)


class TrafficGenerator:
    """Samples standardized flow vectors from the archetype prototypes."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def sample_raw(self, n: int, stratify: bool = True) -> tuple[np.ndarray, np.ndarray]:
        n_arch = len(ARCHETYPES)
        per = max(1, n // n_arch)
        parts = []
        labels = []
        for ai, arch in enumerate(ARCHETYPES):
            m = per
            if not stratify and ai == n_arch - 1:
                m = n - per * (n_arch - 1)
            X = self.rng.normal(0.0, 1.0, size=(m, N_FEATURES)) * arch.sigma + arch.mean
            parts.append(X)
            labels.append(np.full(m, ai, dtype=np.int64))
        X = np.concatenate(parts, axis=0)[:n]
        y = np.concatenate(labels, axis=0)[:n]

        X[:, _COUNT_FEATS] = np.maximum(X[:, _COUNT_FEATS], 0.0)
        X[:, _FRAC_FEATS] = np.clip(X[:, _FRAC_FEATS], 0.0, 1.0)
        X[:, _SIZE_FEATS] = np.maximum(X[:, _SIZE_FEATS], 1.0)
        X[:, _HOUR_FEAT] = np.clip(X[:, _HOUR_FEAT], 0.0, 24.0)
        X[:, _PORT_FEAT] = np.clip(X[:, _PORT_FEAT], 1.0, 65535.0)
        X[:, _PROTO_FEAT] = np.round(X[:, _PROTO_FEAT]).clip(0, 2)
        X[:, _TTL_FEAT] = np.clip(X[:, _TTL_FEAT], 0.0, 255.0)

        for f in [0, 1, 2, 3, 5, 13, 14, 15]:
            X[:, f] = np.round(X[:, f])
        return X, y

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        return self.sample_raw(n)


@dataclass
class Dataset:
    unlabeled: np.ndarray
    labeled_X: np.ndarray
    labeled_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray
    unlabeled_y: np.ndarray = None
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    archetype_names: list[str] = field(default_factory=lambda: list(ARCHETYPE_NAMES))
    n_classes: int = len(ARCHETYPES)
    pool_mean: np.ndarray = None
    pool_std: np.ndarray = None


def make_dataset(
    n_unlabeled: int = 4000,
    labeled_per_class: int = 40,
    n_test: int = 1600,
    seed: int = 0,
) -> Dataset:
    """Build the unlabeled pool, a labeled subset, and a held-out test set.

    The labeled set is stratified-sampled *from* the unlabeled pool (matching
    the real-world assumption: labels are scarce but the flows exist), while
    the test set is generated fresh and never seen in pretraining.
    Standardization statistics are fit on the unlabeled pool only.
    """
    gen = TrafficGenerator(seed=seed)
    pool_X_raw, pool_y = gen.sample(n_unlabeled)

    mean = pool_X_raw.mean(axis=0)
    std = pool_X_raw.std(axis=0) + 1e-8
    pool_X = standardize(pool_X_raw, mean, std)

    # Stratified labeled subset from the pool.
    lbl_raw = []
    lbl_y = []
    for c in range(len(ARCHETYPES)):
        idx = np.where(pool_y == c)[0]
        chosen = gen.rng.choice(idx, size=labeled_per_class, replace=False)
        lbl_raw.append(pool_X_raw[chosen])
        lbl_y.append(np.full(labeled_per_class, c, dtype=np.int64))
    labeled_X = standardize(np.concatenate(lbl_raw, axis=0), mean, std)
    labeled_y = np.concatenate(lbl_y, axis=0)

    test_raw, test_y = gen.sample(n_test)
    test_X = standardize(test_raw, mean, std)

    return Dataset(
        unlabeled=pool_X,
        unlabeled_y=pool_y,
        labeled_X=labeled_X,
        labeled_y=labeled_y,
        test_X=test_X,
        test_y=test_y,
        pool_mean=mean,
        pool_std=std,
    )
