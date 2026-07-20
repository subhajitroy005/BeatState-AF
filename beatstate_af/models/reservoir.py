"""Walking-skeleton state models (pure numpy).

WHY A RESERVOIR: the skeleton's job is to de-risk the PIPELINE (data -> tokens ->
bounded recurrent state -> per-beat readout -> patient-level metric -> memory
ledger -> kill decision) with zero heavy deps, NOT to be the final model. The
recurrent weights are fixed (echo-state style); only a linear readout is trained
downstream, so the harness runs deterministically in seconds. Claude Code
replaces these with trained GRU / factored-GRU cells in PyTorch (see CLAUDE.md)
behind the SAME interface and the SAME matched-capacity + ledger contracts.

Three constructors implement the core scientific comparison:
  monolithic : one state dim D, DENSE recurrence  (D*D params).
  factored   : two states (atrial | rhythm), BLOCK-DIAGONAL recurrence
               (2*(D/2)^2 = D^2/2), features routed BY PHYSIOLOGY.
  random     : identical block structure, features routed RANDOMLY -- the
               structural-sparsity control that isolates physiological
               semantics from block-diagonal sparsity.
"""
from __future__ import annotations
import numpy as np


def _reservoir(rng, dim, in_dim, spectral_radius=0.9, in_scale=0.6):
    Wr = rng.standard_normal((dim, dim))
    radius = np.max(np.abs(np.linalg.eigvals(Wr))) if dim > 0 else 1.0
    Wr = Wr * (spectral_radius / (radius + 1e-8))
    Wi = rng.standard_normal((dim, in_dim)) * in_scale
    return Wr, Wi


class ReservoirModel:
    def __init__(self, model_id, atrial_idx, rhythm_idx, n_features,
                 atrial_state_dim, rhythm_state_dim, seed, leak=0.4, dense=False):
        self.model_id = model_id
        self.atrial_idx = np.asarray(atrial_idx, dtype=int)
        self.rhythm_idx = np.asarray(rhythm_idx, dtype=int)
        self.n_features = n_features
        self.atrial_state_dim = atrial_state_dim
        self.rhythm_state_dim = rhythm_state_dim
        self.total_state_dim = atrial_state_dim + rhythm_state_dim
        self.leak = leak
        self.dense = dense
        rng = np.random.default_rng(seed)
        if dense:
            self.Wr, self.Wi = _reservoir(rng, self.total_state_dim, n_features)
        else:
            self.WrA, self.WiA = _reservoir(rng, atrial_state_dim, len(self.atrial_idx))
            self.WrR, self.WiR = _reservoir(rng, rhythm_state_dim, len(self.rhythm_idx))
        self.reset_state()

    def reset_state(self, batch_size: int = 1) -> None:
        self.h = np.zeros(self.total_state_dim)

    def step(self, feat: np.ndarray) -> np.ndarray:
        a = self.leak
        if self.dense:
            self.h = (1 - a) * self.h + a * np.tanh(self.Wr @ self.h + self.Wi @ feat)
        else:
            hA = self.h[:self.atrial_state_dim]
            hR = self.h[self.atrial_state_dim:]
            hA = (1 - a) * hA + a * np.tanh(self.WrA @ hA + self.WiA @ feat[self.atrial_idx])
            hR = (1 - a) * hR + a * np.tanh(self.WrR @ hR + self.WiR @ feat[self.rhythm_idx])
            self.h = np.concatenate([hA, hR])
        return self.h

    def recurrent_param_count(self) -> int:
        return (self.total_state_dim ** 2 if self.dense
                else self.atrial_state_dim ** 2 + self.rhythm_state_dim ** 2)

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.total_state_dim * dtype_bytes


def build_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed):
    """Factory enforcing matched TOTAL state dimension across all three kinds."""
    half = total_state_dim // 2
    if kind == "monolithic":
        return ReservoirModel("monolithic", atrial_idx, rhythm_idx, n_features,
                              half, total_state_dim - half, seed, dense=True)
    if kind == "factored":
        return ReservoirModel("factored", atrial_idx, rhythm_idx, n_features,
                              half, total_state_dim - half, seed, dense=False)
    if kind == "random":
        rng = np.random.default_rng(seed + 9973)
        perm = rng.permutation(n_features)
        ra, rr = perm[:len(atrial_idx)], perm[len(atrial_idx):]
        return ReservoirModel("random", ra, rr, n_features,
                              half, total_state_dim - half, seed, dense=False)
    raise ValueError(f"unknown model kind: {kind}")
