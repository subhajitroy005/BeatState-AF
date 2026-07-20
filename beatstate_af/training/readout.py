"""Trained linear readout for the walking skeleton (logistic regression, numpy).

Only this is trained; the reservoir recurrence is fixed. Claude Code replaces the
whole (reservoir + readout) with a trained recurrent net, keeping fit/predict.
"""
from __future__ import annotations
import numpy as np


class LogisticReadout:
    def __init__(self, l2=1e-3, lr=0.5, iters=400, seed=0):
        self.l2, self.lr, self.iters, self.seed = l2, lr, iters, seed

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        self.mu = X.mean(0); self.sd = X.std(0) + 1e-8
        Xs = (X - self.mu) / self.sd
        n, d = Xs.shape
        rng = np.random.default_rng(self.seed)
        w = rng.normal(0, 0.01, d); b = 0.0
        for _ in range(self.iters):
            z = np.clip(Xs @ w + b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            gw = Xs.T @ g / n + self.l2 * w
            gb = g.mean()
            w -= self.lr * gw; b -= self.lr * gb
        self.w, self.b = w, b
        return self

    @property
    def n_params(self):
        return int(self.w.size + 1)

    def predict_proba(self, X):
        Xs = (np.asarray(X, float) - self.mu) / self.sd
        z = np.clip(Xs @ self.w + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))
