"""Patient-level metrics for AF detection."""
from __future__ import annotations
import numpy as np


def _binary_f1(y, yhat, cls):
    tp = np.sum((yhat == cls) & (y == cls))
    fp = np.sum((yhat == cls) & (y != cls))
    fn = np.sum((yhat != cls) & (y == cls))
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def _prf(y, yhat):
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int)
    present = [int(c) for c in sorted(np.unique(y))]
    present_f1s = [_binary_f1(y, yhat, c) for c in present]
    af_f1 = _binary_f1(y, yhat, 1) if np.any(y == 1) else float("nan")
    tp = np.sum((yhat == 1) & (y == 1)); fn = np.sum((yhat != 1) & (y == 1))
    tn = np.sum((yhat == 0) & (y == 0)); fp = np.sum((yhat == 1) & (y == 0))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    present_macro = float(np.mean(present_f1s)) if present_f1s else float("nan")
    return dict(af_f1=float(af_f1),
                present_class_macro_f1=present_macro,
                macro_f1=present_macro,
                sensitivity=float(sens),
                specificity=float(spec),
                ppv=float(ppv),
                patient_has_af=bool(np.any(y == 1)),
                patient_has_nonaf=bool(np.any(y == 0)),
                n_af_beats=int(np.sum(y == 1)),
                n_nonaf_beats=int(np.sum(y == 0)))


def patient_metrics(y, yhat):
    return _prf(y, yhat)


def discordance_accuracy(y, yhat, disc):
    """Per-beat accuracy on concordant vs prespecified discordant beats."""
    y = np.asarray(y).astype(int); yhat = np.asarray(yhat).astype(int); disc = np.asarray(disc).astype(int)
    out = {}
    for name, mask in (("concordant", disc == 0), ("discordant", disc == 1)):
        out[name] = float(np.mean(yhat[mask] == y[mask])) if mask.any() else float("nan")
    return out
