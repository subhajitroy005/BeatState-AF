"""Patient is the statistical unit. Paired bootstrap CI and one-sided tests."""
from __future__ import annotations
from collections import defaultdict
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


def noninferiority(diffs, delta=0.02, **kw):
    """compressed - reference is non-inferior when lower CI > -delta."""
    mean, (lo, hi) = paired_bootstrap_ci(diffs, **kw)
    return dict(mean=mean, ci=(lo, hi), noninferior=bool(lo > -float(delta)))


def assert_patient_level_rows(rows, patient_key="patient_id", seed_key="model_seed"):
    """Reject direct repeated seed-patient rows before patient aggregation."""
    seen = set()
    repeated = set()
    for row in rows:
        patient = row[patient_key]
        seed = row.get(seed_key, "")
        if (patient, seed) in seen:
            continue
        if patient in {p for p, _ in seen}:
            repeated.add(patient)
        seen.add((patient, seed))
    if repeated:
        raise ValueError("seed-level rows must be averaged within patient before inference")


def average_metric_by_patient(rows, metric, group_keys):
    """Average seed/routing repetitions within each patient for model summaries."""
    buckets = defaultdict(list)
    exemplar = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys) + (row["patient_id"],)
        val = row.get(metric, "")
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = float("nan")
        buckets[key].append(val)
        exemplar.setdefault(key, row)
    out = []
    for key, vals in sorted(buckets.items()):
        row = {k: v for k, v in zip(group_keys, key[:-1])}
        row["patient_id"] = key[-1]
        row[metric] = float(np.nanmean(vals))
        row["n_repetitions"] = int(np.sum(~np.isnan(np.asarray(vals, dtype=float))))
        ex = exemplar[key]
        for copy_key in ("qrs_mode", "patient_has_af", "patient_has_nonaf", "n_beats",
                         "n_af_beats", "n_nonaf_beats", "n_discordant_beats",
                         "n_concordant_beats"):
            if copy_key in ex:
                row[copy_key] = ex[copy_key]
        out.append(row)
    return out


def paired_differences(patient_rows, metric, model_a, dim_a, model_b, dim_b):
    """Return patient-level paired metric differences after aggregation."""
    lookup = {}
    for row in patient_rows:
        key = (row["model_id"], int(row["total_state_dim"]), row["patient_id"])
        lookup[key] = row
    patients_a = {p for (m, d, p) in lookup if m == model_a and d == int(dim_a)}
    patients_b = {p for (m, d, p) in lookup if m == model_b and d == int(dim_b)}
    patients = sorted(patients_a & patients_b)
    diffs = []
    for patient in patients:
        va = float(lookup[(model_a, int(dim_a), patient)][metric])
        vb = float(lookup[(model_b, int(dim_b), patient)][metric])
        if np.isfinite(va) and np.isfinite(vb):
            diffs.append(va - vb)
    return np.asarray(diffs, dtype=float), patients
