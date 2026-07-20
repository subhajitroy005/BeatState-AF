"""Patient is the statistical unit. Paired bootstrap CI + equivalence (TOST-style)."""
from __future__ import annotations
import numpy as np


def paired_bootstrap_ci(diffs, n_boot=5000, alpha=0.05, seed=0):
    d = np.asarray(diffs, float); d = d[~np.isnan(d)]
    if d.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, d.size, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), (float(lo), float(hi))


def superiority(diffs, margin=0.0, **kw):
    """factored - baseline per patient. Superior if CI lower bound > margin."""
    mean, (lo, hi) = paired_bootstrap_ci(diffs, **kw)
    return dict(mean=mean, ci=(lo, hi), superior=bool(lo > margin))


def equivalent(diffs, margin=0.01, **kw):
    """Practically equivalent if the whole CI lies within +/- margin (TOST-style)."""
    mean, (lo, hi) = paired_bootstrap_ci(diffs, **kw)
    return dict(mean=mean, ci=(lo, hi), equivalent=bool(lo > -margin and hi < margin))
