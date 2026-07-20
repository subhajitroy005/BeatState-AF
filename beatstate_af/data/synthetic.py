"""Synthetic AF-like beat streams for the WALKING SKELETON ONLY.

Real research must not use this: it exists so the kill-gate pipeline runs
end-to-end with zero heavy deps. Each 'beat' carries an atrial-evidence vector
and a rhythm-evidence vector, a binary AF label, and a discordance flag.

Feature layout (n_features = 8):
  atrial_idx = [0,1,2,3] : p_wave_score, fwave_energy, morph_a, morph_b
  rhythm_idx = [4,5,6,7] : rr, drr, rr_over_median, local_var

Physiology:
  non-AF  -> P-wave present, low f-wave, regular rhythm.
  AF      -> P-wave absent,  high f-wave, irregular rhythm.
DISCORDANCE (the mechanism under test) is injected in two forms:
  type-1 (PAC-like, label non-AF): rhythm looks AF-ish, atrial looks non-AF.
  type-2 (regularised AF, label AF): rhythm looks regular, atrial looks AF.

TODO(claude-code): replace with beatstate_af.data.wfdb_io on real AFDB records.
"""
from __future__ import annotations
import numpy as np

ATRIAL_IDX = [0, 1, 2, 3]
RHYTHM_IDX = [4, 5, 6, 7]
N_FEATURES = 8


def _emit(state, rng, discord=None):
    f = np.zeros(N_FEATURES)
    af = state == 1
    # atrial channel
    p_present = 0.15 if af else 0.9
    fwave = 0.85 if af else 0.1
    # rhythm channel
    drr = 0.8 if af else 0.12
    lvar = 0.85 if af else 0.15
    if discord == 1:      # PAC-like: rhythm AF-ish, atrial non-AF (label stays non-AF)
        drr, lvar = 0.8, 0.8
    elif discord == 2:    # regularised AF: rhythm regular, atrial AF (label stays AF)
        drr, lvar = 0.15, 0.15
    f[0] = p_present + rng.normal(0, 0.12)
    f[1] = fwave + rng.normal(0, 0.12)
    f[2] = rng.normal(0, 1) + (0.4 if af else -0.4)
    f[3] = rng.normal(0, 1)
    f[4] = 0.7 + rng.normal(0, 0.15)                 # rr (normalised, weak signal)
    f[5] = drr + rng.normal(0, 0.12)
    f[6] = 1.0 + rng.normal(0, 0.1)
    f[7] = lvar + rng.normal(0, 0.12)
    return f


def make_patient(seed, n_beats=600, p_af=0.35, stay=0.985, p_disc=0.12):
    rng = np.random.default_rng(seed)
    state = 1 if rng.random() < p_af else 0
    feats, ys, disc = [], [], []
    for _ in range(n_beats):
        # sticky Markov switching -> episodes
        if rng.random() > stay:
            state = 1 - state
        d = 0
        if rng.random() < p_disc:
            d = 1 if state == 0 else 2
        feats.append(_emit(state, rng, discord=d))
        ys.append(state)
        disc.append(1 if d else 0)
    return dict(feat=np.array(feats), y=np.array(ys), disc=np.array(disc))


def make_cohort(n_patients, base_seed=0, **kw):
    return {f"P{ i:04d}": make_patient(base_seed + i, **kw) for i in range(n_patients)}
