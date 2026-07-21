"""Trained recurrent state models (PyTorch) behind the StreamingAFModel interface.

Three matched-capacity arms, identical to the reservoir skeleton's contract:
  gru_monolithic : one dense GRU of dim D            (recurrent params 3*D*D)
  gru_factored   : two block GRUs (atrial | rhythm), features routed BY PHYSIOLOGY
                   (recurrent params 3*(dA^2 + dR^2) ~= half of monolithic)
  gru_random     : identical block structure, features routed RANDOMLY -- the
                   structural-sparsity control.

The recurrent cell is causal with O(1) persistent state. `step()` advances that
state one beat at a time in numpy using the trained weights and is exactly the
batched `forward` recurrence (verified in tests), so the same model is a training
graph AND a streaming detector. `recurrent_param_count` counts hidden-to-hidden
gate weights only (what the block structure changes); the fusion head is tiny and
counted separately in the ledger.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def _gru_cell_np(x, h, Wih, Whh, bih, bhh):
    """One PyTorch-GRU step in numpy. x:(F,) h:(H,) -> h':(H,)."""
    H = h.shape[0]
    gi = Wih @ x + bih
    gh = Whh @ h + bhh
    r = _sigmoid(gi[:H] + gh[:H])
    z = _sigmoid(gi[H:2 * H] + gh[H:2 * H])
    n = np.tanh(gi[2 * H:] + r * gh[2 * H:])
    return ((1.0 - z) * n + z * h).astype(np.float32)


class StreamingGRU(nn.Module):
    def __init__(self, model_id, atrial_idx, rhythm_idx, n_features,
                 atrial_state_dim, rhythm_state_dim, seed, dense):
        super().__init__()
        torch.manual_seed(seed)
        self.model_id = model_id
        self.atrial_idx = np.asarray(atrial_idx, dtype=int)
        self.rhythm_idx = np.asarray(rhythm_idx, dtype=int)
        self.n_features = n_features
        self.atrial_state_dim = atrial_state_dim
        self.rhythm_state_dim = rhythm_state_dim
        self.total_state_dim = atrial_state_dim + rhythm_state_dim
        self.dense = dense
        if dense:
            self.gru = nn.GRU(n_features, self.total_state_dim, batch_first=True)
        else:
            self.gru_a = nn.GRU(len(self.atrial_idx), atrial_state_dim, batch_first=True)
            self.gru_r = nn.GRU(len(self.rhythm_idx), rhythm_state_dim, batch_first=True)
        self.head = nn.Linear(self.total_state_dim, 1)
        self.reset_state()

    # ---- training / batched inference --------------------------------------
    def forward(self, x, h0=None):
        """x:(B,T,F) -> logits:(B,T), hidden:(B,H). Causal recurrence."""
        if self.dense:
            out, hn = self.gru(x, h0)
            h_last = hn[-1]
        else:
            xa = x[:, :, self.atrial_idx]
            xr = x[:, :, self.rhythm_idx]
            h0a = h0[0] if h0 is not None else None
            h0r = h0[1] if h0 is not None else None
            oa, hna = self.gru_a(xa, h0a)
            orr, hnr = self.gru_r(xr, h0r)
            out = torch.cat([oa, orr], dim=2)
            h_last = (hna[-1], hnr[-1])
        logits = self.head(out).squeeze(-1)
        return logits, h_last

    # ---- streaming (numpy, O(1) state) -------------------------------------
    def reset_state(self, batch_size: int = 1) -> None:
        self.h = np.zeros(self.total_state_dim, dtype=np.float32)

    def _np_weights(self, gru):
        return (gru.weight_ih_l0.detach().numpy().astype(np.float32),
                gru.weight_hh_l0.detach().numpy().astype(np.float32),
                gru.bias_ih_l0.detach().numpy().astype(np.float32),
                gru.bias_hh_l0.detach().numpy().astype(np.float32))

    def step(self, feat: np.ndarray) -> np.ndarray:
        feat = np.asarray(feat, dtype=np.float32)
        if self.dense:
            W = self._np_weights(self.gru)
            self.h = _gru_cell_np(feat, self.h, *W)
        else:
            dA = self.atrial_state_dim
            hA, hR = self.h[:dA], self.h[dA:]
            hA = _gru_cell_np(feat[self.atrial_idx], hA, *self._np_weights(self.gru_a))
            hR = _gru_cell_np(feat[self.rhythm_idx], hR, *self._np_weights(self.gru_r))
            self.h = np.concatenate([hA, hR]).astype(np.float32)
        return self.h

    # ---- memory ledger ------------------------------------------------------
    def recurrent_param_count(self) -> int:
        if self.dense:
            return 3 * self.total_state_dim ** 2
        return 3 * self.atrial_state_dim ** 2 + 3 * self.rhythm_state_dim ** 2

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.total_state_dim * dtype_bytes

    def readout_param_count(self) -> int:
        return int(sum(p.numel() for p in self.head.parameters()))


def build_gru_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed,
                    routing_seed=None):
    """Factory enforcing matched TOTAL state dimension across all three arms."""
    half = total_state_dim // 2
    if kind == "gru_monolithic":
        return StreamingGRU("gru_monolithic", atrial_idx, rhythm_idx, n_features,
                            half, total_state_dim - half, seed, dense=True)
    if kind == "gru_factored":
        return StreamingGRU("gru_factored", atrial_idx, rhythm_idx, n_features,
                            half, total_state_dim - half, seed, dense=False)
    if kind == "gru_random":
        route = seed + 9973 if routing_seed is None else int(routing_seed)
        rng = np.random.default_rng(route)
        perm = rng.permutation(n_features)
        ra, rr = perm[:len(atrial_idx)], perm[len(atrial_idx):]
        return StreamingGRU("gru_random", ra, rr, n_features,
                            half, total_state_dim - half, seed, dense=False)
    raise ValueError(f"unknown gru kind: {kind}")


# ---- training / evaluation -------------------------------------------------
def _standardizer(cohort, ids):
    X = np.vstack([cohort[p]["feat"] for p in ids]).astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def fit_gru(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed,
            cohort, train_ids, hp, routing_seed=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gru_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed,
                            routing_seed=routing_seed)
    mu, sd = _standardizer(cohort, train_ids)

    epochs = int(hp.get("epochs", 25))
    lr = float(hp.get("lr", 5e-3))
    wd = float(hp.get("weight_decay", 1e-4))
    chunk = int(hp.get("chunk_len", 512))
    grad_clip = float(hp.get("grad_clip", 2.0))
    # compute-bounded pilot: cap TRAINING sequence length per patient (0 = full).
    # Evaluation always uses the full sequence; only training BPTT is truncated.
    train_max = int(hp.get("train_max_beats", 0))

    # class balance -> pos_weight for BCE
    y_all = np.concatenate([(cohort[p]["y"][:train_max] if train_max else cohort[p]["y"]) for p in train_ids])
    pos = float(y_all.sum()); neg = float(len(y_all) - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    seqs = []
    for p in train_ids:
        f = (cohort[p]["feat"].astype(np.float32) - mu) / sd
        y = cohort[p]["y"].astype(np.float32)
        if train_max:
            f, y = f[:train_max], y[:train_max]
        seqs.append((torch.from_numpy(f), torch.from_numpy(y)))

    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(seqs))
        for si in order:
            f, y = seqs[si]
            T = f.shape[0]
            if T < 2:
                continue
            h = None
            for s in range(0, T, chunk):
                xb = f[s:s + chunk].unsqueeze(0)
                yb = y[s:s + chunk].unsqueeze(0)
                opt.zero_grad()
                logits, h = model(xb, h)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                # detach carried state for truncated BPTT
                if model.dense:
                    h = h.detach().unsqueeze(0)
                else:
                    h = (h[0].detach().unsqueeze(0), h[1].detach().unsqueeze(0))
    model.eval()
    return model, mu, sd


def predict_proba_seq(model, feat, mu, sd):
    f = ((feat.astype(np.float32) - mu) / sd)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(f).unsqueeze(0))
    return torch.sigmoid(logits).squeeze(0).numpy()


def _head_params_np(model):
    w = model.head.weight.detach().numpy().astype(np.float32).reshape(-1)
    b = model.head.bias.detach().numpy().astype(np.float32).reshape(-1)
    return w, np.float32(b[0])


def proba_from_state(model, state: np.ndarray) -> float:
    w, b = _head_params_np(model)
    logit = np.float32(np.dot(w, state.astype(np.float32)) + b)
    return float(1.0 / (1.0 + np.exp(-logit)))


def predict_proba_streaming(model, feat, mu, sd):
    """Primary E01v2 inference path: normalize then call model.step(token)."""
    f = ((feat.astype(np.float32) - mu.astype(np.float32)) / sd.astype(np.float32)).astype(np.float32)
    model.reset_state()
    out = np.zeros(f.shape[0], dtype=np.float32)
    for i, token in enumerate(f):
        state = model.step(token)
        out[i] = np.float32(proba_from_state(model, state))
    return out


def fit_gru_with_validation(kind, n_features, atrial_idx, rhythm_idx, total_state_dim,
                            seed, cohort, train_ids, validation_ids, hp,
                            routing_seed=None):
    """Train once and record train loss + validation metric after each epoch."""
    from beatstate_af.evaluation.metrics import patient_metrics

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_gru_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed,
                            routing_seed=routing_seed)
    mu, sd = _standardizer(cohort, train_ids)

    epochs = int(hp.get("epochs", 6))
    lr = float(hp.get("lr", 1e-2))
    wd = float(hp.get("weight_decay", 1e-4))
    chunk = int(hp.get("chunk_len", 512))
    grad_clip = float(hp.get("grad_clip", 2.0))
    train_max = int(hp.get("train_max_beats", 0))

    y_all = np.concatenate([(cohort[p]["y"][:train_max] if train_max else cohort[p]["y"]) for p in train_ids])
    pos = float(y_all.sum())
    neg = float(len(y_all) - pos)
    pos_weight = torch.tensor([neg / pos if pos > 0 else 1.0], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    seqs = []
    for p in train_ids:
        f = (cohort[p]["feat"].astype(np.float32) - mu) / sd
        y = cohort[p]["y"].astype(np.float32)
        if train_max:
            f, y = f[:train_max], y[:train_max]
        seqs.append((torch.from_numpy(f.astype(np.float32)), torch.from_numpy(y)))

    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        order = rng.permutation(len(seqs))
        for si in order:
            f, y = seqs[si]
            T = f.shape[0]
            if T < 2:
                continue
            h = None
            for s in range(0, T, chunk):
                xb = f[s:s + chunk].unsqueeze(0)
                yb = y[s:s + chunk].unsqueeze(0)
                opt.zero_grad()
                logits, h = model(xb, h)
                loss = loss_fn(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                opt.step()
                losses.append(float(loss.detach().cpu()))
                if model.dense:
                    h = h.detach().unsqueeze(0)
                else:
                    h = (h[0].detach().unsqueeze(0), h[1].detach().unsqueeze(0))
        model.eval()
        vals = []
        for p in validation_ids:
            prob = predict_proba_seq(model, cohort[p]["feat"], mu, sd)
            yhat = (prob >= 0.5).astype(int)
            vals.append(patient_metrics(cohort[p]["y"], yhat)["present_class_macro_f1"])
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_present_class_macro_f1": float(np.nanmean(vals)) if vals else float("nan"),
        })
    return model, mu, sd, history
