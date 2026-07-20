"""Patient-disjoint splitting with a hard leakage assertion."""
from __future__ import annotations
import numpy as np


def patient_disjoint_split(patient_ids, train_frac=0.6, seed=0):
    ids = sorted(patient_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    k = int(round(len(ids) * train_frac))
    train, test = set(ids[:k]), set(ids[k:])
    assert train.isdisjoint(test), "LEAKAGE: patient in both splits"
    assert train and test, "empty split"
    return sorted(train), sorted(test)
