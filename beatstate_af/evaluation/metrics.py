"""Patient-level metrics. macro-F1 is primary (never accuracy)."""
from __future__ import annotations
import numpy as np


def _prf(y, yhat):
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int)
    f1s = []
    for c in (0, 1):
        tp = np.sum((yhat == c) & (y == c))
        fp = np.sum((yhat == c) & (y != c))
        fn = np.sum((yhat != c) & (y == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    tp = np.sum((yhat == 1) & (y == 1)); fn = np.sum((yhat != 1) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0)); fp = np.sum((yhat == 1) & (y == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return dict(macro_f1=float(np.mean(f1s)), sensitivity=float(sens),
                specificity=float(spec), ppv=float(ppv))


def patient_metrics(y, yhat):
    return _prf(y, yhat)


def discordance_accuracy(y, yhat, disc):
    """Per-beat accuracy on concordant vs prespecified discordant beats."""
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int); disc = np.asarray(disc).astype(int)
    out = {}
    for name, mask in (("concordant", disc == 0), ("discordant", disc == 1)):
        out[name] = float(np.mean(yhat[mask] == y[mask])) if mask.any() else float("nan")
    return out
