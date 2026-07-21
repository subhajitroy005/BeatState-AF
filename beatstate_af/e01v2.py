"""Shared helpers for the E01v2 audit-corrected AFDB kill gate."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from beatstate_af.config import load_config
from beatstate_af.data.wfdb_io import _labels_and_disc, _rhythm_segments
from beatstate_af.preprocessing.features import (
    SignalNormalizer,
    extract_features,
    feature_memory_report,
    pooled_signal_normalizer,
)
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector, match_qrs_detections
from beatstate_af.provenance import file_sha256


ATRIAL_IDX = [0, 1, 2, 3]
RHYTHM_IDX = [4, 5, 6, 7]
N_FEATURES = 8
PRIMARY_METRIC = "present_class_macro_f1"
PROTOCOL_VERSION = "E01v2"
SPLIT_ID = "afdb_split_v2"


E01V2_RESULT_COLUMNS = [
    "run_id",
    "experiment_id",
    "protocol_version",
    "config_hash",
    "protocol_hash",
    "full_commit_sha",
    "git_tree_dirty",
    "environment_hash",
    "dataset_id",
    "qrs_mode",
    "split_id",
    "model_id",
    "total_state_dim",
    "atrial_state_dim",
    "rhythm_state_dim",
    "model_seed",
    "routing_seed",
    "patient_id",
    "patient_has_af",
    "patient_has_nonaf",
    "n_beats",
    "n_af_beats",
    "n_nonaf_beats",
    "n_discordant_beats",
    "n_concordant_beats",
    "af_f1",
    "present_class_macro_f1",
    "sensitivity",
    "specificity",
    "ppv",
    "acc_concordant",
    "acc_discordant",
    "model_weight_bytes",
    "model_bias_bytes",
    "readout_bytes",
    "normalizer_bytes",
    "filter_constant_bytes",
    "qrs_persistent_bytes",
    "qrs_scratch_bytes",
    "feature_persistent_bytes",
    "feature_scratch_bytes",
    "recurrent_state_bytes",
    "recurrent_step_activation_bytes",
    "output_bytes",
    "total_static_bytes",
    "total_persistent_bytes",
    "total_peak_inference_bytes",
    "recurrent_param_count",
    "total_model_param_count",
    "estimated_ops_per_beat",
    "status",
]


GENERATED_DIRS = ("results/e01v2", "figures/e01v2", "figures/audit")


def read_json(path: str | os.PathLike) -> dict:
    with open(path) as fh:
        return json.load(fh)


def write_json(path: str | os.PathLike, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)
        fh.write("\n")


def load_split(path: str | os.PathLike = "manifests/afdb_split_v2.json") -> dict:
    split = read_json(path)
    for key in ("train", "validation", "test"):
        split[key] = [str(x) for x in split[key]]
    return split


def qrs_mode_from_dataset_id(dataset_id: str) -> str:
    if dataset_id.endswith("_oracle"):
        return "oracle"
    if dataset_id.endswith("_deployable"):
        return "deployable"
    raise ValueError(f"cannot infer qrs mode from dataset_id={dataset_id!r}")


def load_raw_signal(record_path: str, fs: int, channel: int, max_beats: int):
    import wfdb

    cap_samples = int(max_beats * fs)
    hdr = wfdb.rdheader(record_path)
    sampto = min(hdr.sig_len, cap_samples)
    rec = wfdb.rdrecord(record_path, sampfrom=0, sampto=sampto, channels=[channel])
    sig = rec.p_signal[:, 0].astype(np.float32)
    return np.nan_to_num(sig, nan=0.0), int(sampto)


def training_signal_normalizer(dataset_cfg: dict, train_ids: Iterable[str]) -> SignalNormalizer:
    signals = []
    for rec in train_ids:
        sig, _ = load_raw_signal(
            os.path.join(dataset_cfg["cache_dir"], rec),
            fs=int(dataset_cfg["fs_hz"]),
            channel=int(dataset_cfg["signal_channel"]),
            max_beats=int(dataset_cfg["per_record_max_beats"]),
        )
        signals.append(sig)
    return pooled_signal_normalizer(signals, dtype_bytes=4)


def _detect_rpeaks(record_path: str, sig: np.ndarray, qrs_mode: str, fs: int, sampto: int):
    import wfdb

    if qrs_mode == "oracle":
        qann = wfdb.rdann(record_path, "qrs", sampfrom=0, sampto=sampto)
        return np.asarray(qann.sample, dtype=int), None
    if qrs_mode == "deployable":
        detector = StreamingQRSDetector(fs)
        return detector.detect(sig), detector
    raise ValueError(f"unknown qrs_mode: {qrs_mode}")


def qrs_reference_metrics(record_path: str, detected, fs: int, sampto: int,
                          tolerance_s: float = 0.150) -> dict:
    import wfdb

    ref = wfdb.rdann(record_path, "qrs", sampfrom=0, sampto=sampto).sample
    metrics = match_qrs_detections(ref, detected, int(tolerance_s * fs))
    return {
        "qrs_sensitivity": metrics.sensitivity,
        "qrs_ppv": metrics.ppv,
        "qrs_timing_error_samples": metrics.mean_abs_timing_error_samples,
        "qrs_matched_peaks": metrics.matched_peaks,
        "qrs_missed_peaks": metrics.missed_peaks,
        "qrs_false_peaks": metrics.false_peaks,
        "qrs_reference_peaks": metrics.reference_peaks,
        "qrs_detected_peaks": metrics.detected_peaks,
    }


def load_afdb_cohort_v2(dataset_cfg: dict, split: dict, qrs_mode: str,
                        normalizer: SignalNormalizer, cache: bool = True):
    import wfdb

    with open(dataset_cfg["manifest"]) as fh:
        manifest = json.load(fh)
    fs = int(dataset_cfg["fs_hz"])
    channel = int(dataset_cfg["signal_channel"])
    max_beats = int(dataset_cfg["per_record_max_beats"])
    af_rhythms = tuple(dataset_cfg.get("af_rhythms", ["AFIB"]))
    disc_cfg = dataset_cfg.get("discordance", {})
    cache_dir = Path(dataset_cfg["cache_dir"]) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm_tag = f"{normalizer.median:.6g}_{normalizer.scale:.6g}".encode()
    norm_hash = hashlib.sha256(norm_tag).hexdigest()[:12]

    cohort = {}
    qrs_rows = []
    for rec in manifest["records"]:
        cpath = cache_dir / f"e01v2_{rec}_{qrs_mode}_{max_beats}_{norm_hash}.npz"
        record_path = os.path.join(dataset_cfg["cache_dir"], rec)
        if cache and cpath.exists():
            d = np.load(cpath, allow_pickle=False)
            cohort[rec] = {
                "feat": d["feat"],
                "y": d["y"],
                "disc": d["disc"],
                "beat_samples": d["beat_samples"],
            }
            if "qrs_sensitivity" in d:
                row = {k: d[k].item() for k in d.files if k.startswith("qrs_")}
                row["patient_id"] = str(d["patient_id"].item()) if "patient_id" in d else rec
                qrs_rows.append(row)
            continue

        sig, sampto = load_raw_signal(record_path, fs=fs, channel=channel, max_beats=max_beats)
        rpeaks, detector = _detect_rpeaks(record_path, sig, qrs_mode, fs, sampto)
        feat, beat_samples, sqi = extract_features(sig, rpeaks, fs, normalizer=normalizer)
        if feat.shape[0] > max_beats:
            feat, beat_samples, sqi = feat[:max_beats], beat_samples[:max_beats], sqi[:max_beats]
        atr = wfdb.rdann(record_path, "atr", sampfrom=0, sampto=sampto)
        segs = _rhythm_segments(atr, sampto)
        y, disc = _labels_and_disc(beat_samples, sqi, segs, fs, af_rhythms, disc_cfg)
        cohort[rec] = {"feat": feat, "y": y, "disc": disc, "beat_samples": beat_samples}

        save = {"feat": feat, "y": y, "disc": disc, "beat_samples": beat_samples}
        if qrs_mode == "deployable":
            qrs = qrs_reference_metrics(record_path, rpeaks, fs=fs, sampto=sampto)
            qrs.update({"patient_id": rec})
            qrs_rows.append(qrs)
            save.update(qrs)
        if cache:
            np.savez_compressed(cpath, **save)

    return cohort, qrs_rows


def write_rows(path: str | os.PathLike, rows: list[dict], fieldnames: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def protocol_hash(path: str = "configs/protocol_v2.yaml") -> str:
    return file_sha256(path)


def dataset_manifest_hash(config_path: str) -> str:
    cfg = load_config(config_path)
    return file_sha256(cfg["manifest"])


def split_manifest_hash(path: str = "manifests/afdb_split_v2.json") -> str:
    return file_sha256(path)


def feature_memory(dataset_cfg: dict) -> dict[str, int]:
    return feature_memory_report(float(dataset_cfg["fs_hz"]), dtype_bytes=4)
