"""Real-data loader: PhysioNet AFDB -> the synthetic generator's token contract.

`load_afdb_record` emits dict(feat=(T,8) float32, y=(T,), disc=(T,), patient_id)
identical in shape/semantics to beatstate_af.data.synthetic, so nothing
downstream (models, ledger, kill gate) changes. Rhythm features derive from
DEPLOYABLE R-peaks by default (the causal detector); qrs_mode='oracle' uses the
machine `.qrs` beats as an upper-bound diagnostic only (report Delta_QRS).

Labels: AFIB rhythm -> 1, else 0. Discordance is the prespecified atrial/rhythm
disagreement subset (docs/discordance_definition.md).
"""
from __future__ import annotations
import json, os
import numpy as np

from beatstate_af.preprocessing.qrs import CausalQRSDetector
from beatstate_af.preprocessing.features import extract_features

ATRIAL_IDX = [0, 1, 2, 3]
RHYTHM_IDX = [4, 5, 6, 7]
N_FEATURES = 8


def _rhythm_segments(ann, sig_len):
    """(start_sample, end_sample, label) intervals from .atr aux_note change points."""
    samp, aux = ann.sample, ann.aux_note
    segs, cur = [], "N"
    for i in range(len(samp)):
        a = aux[i].strip().lstrip("(") if aux[i].strip() else cur
        cur = a
        start = int(samp[i])
        end = int(samp[i + 1]) if i + 1 < len(samp) else int(sig_len)
        segs.append((start, end, cur))
    if not segs:
        segs = [(0, int(sig_len), "N")]
    return segs


def _labels_and_disc(beat_samples, sqi, segs, fs, af_rhythms, disc_cfg):
    starts = np.array([s for s, _, _ in segs])
    labels = [lab for _, _, lab in segs]
    idx = np.clip(np.searchsorted(starts, beat_samples, side="right") - 1, 0, len(labels) - 1)
    beat_lab = np.array([labels[i] for i in idx])
    y = np.isin(beat_lab, list(af_rhythms)).astype(int)

    # discordance ---------------------------------------------------------
    win = float(disc_cfg.get("transition_window_s", 5.0)) * fs
    nonaf = set(disc_cfg.get("nonaf_arrhythmia", []))
    pct = float(disc_cfg.get("low_sqi_percentile", 10))

    # AFIB onset/offset boundaries
    bnds = []
    for i in range(1, len(segs)):
        prev_af = segs[i - 1][2] in af_rhythms
        cur_af = segs[i][2] in af_rhythms
        if prev_af != cur_af:
            bnds.append(segs[i][0])
    bnds = np.array(bnds) if bnds else np.array([], dtype=float)

    disc = np.zeros(len(beat_samples), dtype=int)
    if bnds.size:
        d = np.abs(beat_samples[:, None] - bnds[None, :]).min(axis=1)
        disc |= (d < win).astype(int)
    if nonaf:
        disc |= np.isin(beat_lab, list(nonaf)).astype(int)
    if sqi.size:
        thr = np.percentile(sqi, pct)
        disc |= (sqi <= thr).astype(int)
    return y, disc


def load_afdb_record(record_path, qrs_mode="deployable", fs=250, channel=0,
                     max_beats=15000, af_rhythms=("AFIB",), disc_cfg=None):
    import wfdb
    disc_cfg = disc_cfg or {}
    cap_samples = int(max_beats * fs)  # bound work; ~first `max_beats` beats
    hdr = wfdb.rdheader(record_path)
    sampto = min(hdr.sig_len, cap_samples)
    rec = wfdb.rdrecord(record_path, sampfrom=0, sampto=sampto, channels=[channel])
    sig = rec.p_signal[:, 0].astype(float)
    sig = np.nan_to_num(sig, nan=0.0)

    if qrs_mode == "oracle":
        qann = wfdb.rdann(record_path, "qrs", sampfrom=0, sampto=sampto)
        rpeaks = np.asarray(qann.sample, dtype=int)
    elif qrs_mode == "deployable":
        rpeaks = CausalQRSDetector(fs).detect(sig)
    else:
        raise ValueError(f"unknown qrs_mode: {qrs_mode}")

    feat, beat_samples, sqi = extract_features(sig, rpeaks, fs)
    if feat.shape[0] > max_beats:
        feat, beat_samples, sqi = feat[:max_beats], beat_samples[:max_beats], sqi[:max_beats]

    atr = wfdb.rdann(record_path, "atr", sampfrom=0, sampto=sampto)
    segs = _rhythm_segments(atr, sampto)
    y, disc = _labels_and_disc(beat_samples, sqi, segs, fs, af_rhythms, disc_cfg)

    patient_id = os.path.basename(record_path)
    return dict(feat=feat, y=y, disc=disc, patient_id=patient_id,
                beat_samples=beat_samples, n_rpeaks=int(len(rpeaks)))


def load_afdb_cohort(dataset_cfg, qrs_mode="deployable", cache=True):
    """Build {patient_id: dict(feat,y,disc)} for all manifest records."""
    with open(dataset_cfg["manifest"]) as f:
        manifest = json.load(f)
    cache_dir = dataset_cfg["cache_dir"]
    fs = dataset_cfg["fs_hz"]
    channel = dataset_cfg["signal_channel"]
    max_beats = dataset_cfg["per_record_max_beats"]
    af_rhythms = tuple(dataset_cfg.get("af_rhythms", ["AFIB"]))
    disc_cfg = dataset_cfg.get("discordance", {})

    npz_dir = os.path.join(cache_dir, "cache")
    os.makedirs(npz_dir, exist_ok=True)
    cohort = {}
    for rec in manifest["records"]:
        cpath = os.path.join(npz_dir, f"{rec}_{qrs_mode}_{max_beats}.npz")
        if cache and os.path.exists(cpath):
            d = np.load(cpath)
            cohort[rec] = dict(feat=d["feat"], y=d["y"], disc=d["disc"])
            continue
        rp = os.path.join(cache_dir, rec)
        out = load_afdb_record(rp, qrs_mode=qrs_mode, fs=fs, channel=channel,
                               max_beats=max_beats, af_rhythms=af_rhythms, disc_cfg=disc_cfg)
        cohort[rec] = dict(feat=out["feat"], y=out["y"], disc=out["disc"])
        if cache:
            np.savez_compressed(cpath, feat=out["feat"], y=out["y"], disc=out["disc"])
    return cohort
