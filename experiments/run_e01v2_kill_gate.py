#!/usr/bin/env python3
"""Audit-corrected E01v2 AFDB kill-gate runner."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.config import load_config
from beatstate_af.e01v2 import (
    ATRIAL_IDX,
    E01V2_RESULT_COLUMNS,
    GENERATED_DIRS,
    N_FEATURES,
    PRIMARY_METRIC,
    PROTOCOL_VERSION,
    RHYTHM_IDX,
    SPLIT_ID,
    feature_memory,
    load_afdb_cohort_v2,
    load_split,
    protocol_hash,
    training_signal_normalizer,
    write_rows,
)
from beatstate_af.evaluation.metrics import discordance_accuracy, patient_metrics
from beatstate_af.memory.ledger import memory_report
from beatstate_af.models.gru import fit_gru, predict_proba_streaming
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector
from beatstate_af.provenance import (
    environment_hash,
    environment_report,
    file_sha256,
    full_git_commit,
    git_tree_dirty,
)
from beatstate_af.statistics.stats import noninferiority, paired_bootstrap_ci, superiority


NUMERIC_AGG_FIELDS = [
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
]


def _route_options(model_id: str, routing_seeds: list[int]):
    return routing_seeds if model_id == "gru_random" else [""]


def _load_selected(path: str) -> dict:
    if not Path(path).exists():
        return {"arms": {}}
    with open(path) as fh:
        return yaml.safe_load(fh) or {"arms": {}}


def _hp_for(cfg: dict, selected: dict, model_id: str, dim: int, model_seed: int, routing_seed):
    key = f"{model_id}|{dim}|{model_seed}|{routing_seed}"
    hp = dict(cfg.get("gru", {}))
    if key in selected.get("arms", {}):
        arm = selected["arms"][key]
        for k in ("epochs", "lr", "weight_decay", "chunk_len", "grad_clip", "train_max_beats"):
            hp[k] = arm[k]
    return hp


def _aggregate_patient(rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row["model_id"], int(row["total_state_dim"]), row["patient_id"])].append(row)
    out = []
    for (model_id, dim, patient), vals in sorted(buckets.items()):
        ex = vals[0]
        row = {
            "qrs_mode": ex["qrs_mode"],
            "model_id": model_id,
            "total_state_dim": dim,
            "patient_id": patient,
            "patient_has_af": ex["patient_has_af"],
            "patient_has_nonaf": ex["patient_has_nonaf"],
            "n_beats": ex["n_beats"],
            "n_af_beats": ex["n_af_beats"],
            "n_nonaf_beats": ex["n_nonaf_beats"],
            "n_discordant_beats": ex["n_discordant_beats"],
            "n_concordant_beats": ex["n_concordant_beats"],
            "n_model_seeds": len({v["model_seed"] for v in vals}),
            "n_routing_seeds": len({v["routing_seed"] for v in vals if v["routing_seed"] != ""}) or 1,
        }
        for field in NUMERIC_AGG_FIELDS:
            arr = np.asarray([float(v[field]) for v in vals if str(v[field]) != ""], dtype=float)
            row[field] = float(np.nanmean(arr)) if arr.size else float("nan")
        out.append(row)
    return out


def _model_summary(patient_rows: list[dict]) -> list[dict]:
    buckets = defaultdict(list)
    for row in patient_rows:
        buckets[(row["qrs_mode"], row["model_id"], int(row["total_state_dim"]))].append(row)
    out = []
    for (qrs_mode, model_id, dim), vals in sorted(buckets.items()):
        row = {
            "qrs_mode": qrs_mode,
            "model_id": model_id,
            "total_state_dim": dim,
            "n_unique_test_patients": len({v["patient_id"] for v in vals}),
            "mean_present_class_macro_f1": float(np.nanmean([float(v["present_class_macro_f1"]) for v in vals])),
            "mean_af_f1": float(np.nanmean([float(v["af_f1"]) for v in vals])),
            "total_peak_inference_bytes": float(np.nanmean([float(v["total_peak_inference_bytes"]) for v in vals])),
            "total_static_bytes": float(np.nanmean([float(v["total_static_bytes"]) for v in vals])),
            "total_persistent_bytes": float(np.nanmean([float(v["total_persistent_bytes"]) for v in vals])),
            "recurrent_param_count": float(np.nanmean([float(v["recurrent_param_count"]) for v in vals])),
        }
        out.append(row)
    return out


def _lookup(patient_rows: list[dict]):
    return {(r["model_id"], int(r["total_state_dim"]), r["patient_id"]): r for r in patient_rows}


def _paired(patient_rows, cfg):
    lookup = _lookup(patient_rows)
    seed = int(cfg["bootstrap_seed"])
    reps = int(cfg["bootstrap_reps"])
    metric = cfg["primary_metric"]
    comparisons = [
        ("same_budget_factored16_minus_monolithic16", "gru_factored", 16, "gru_monolithic", 16, "superiority", cfg["superiority_margin"]),
        ("same_budget_factored16_minus_random16", "gru_factored", 16, "gru_random", 16, "superiority", cfg["superiority_margin"]),
        ("same_budget_factored32_minus_monolithic32", "gru_factored", 32, "gru_monolithic", 32, "superiority", cfg["superiority_margin"]),
        ("same_budget_factored32_minus_random32", "gru_factored", 32, "gru_random", 32, "superiority", cfg["superiority_margin"]),
        ("compression_factored16_minus_monolithic32", "gru_factored", 16, "gru_monolithic", 32, "noninferiority", cfg["noninferiority_margin"]),
    ]
    rows = []
    for name, ma, da, mb, db, test, margin in comparisons:
        patients = sorted(
            {p for (m, d, p) in lookup if m == ma and d == da}
            & {p for (m, d, p) in lookup if m == mb and d == db}
        )
        diffs = []
        for patient in patients:
            va = float(lookup[(ma, da, patient)][metric])
            vb = float(lookup[(mb, db, patient)][metric])
            if np.isfinite(va) and np.isfinite(vb):
                diffs.append(va - vb)
        if test == "superiority":
            stat = superiority(diffs, margin=float(margin), n_boot=reps, seed=seed)
            success = stat["superior"]
        else:
            stat = noninferiority(diffs, delta=float(margin), n_boot=reps, seed=seed)
            success = stat["noninferior"]
        rows.append({
            "comparison": name,
            "model_a": ma,
            "dim_a": da,
            "model_b": mb,
            "dim_b": db,
            "metric": metric,
            "test": test,
            "mean_diff": stat["mean"],
            "ci_low": stat["ci"][0],
            "ci_high": stat["ci"][1],
            "success": bool(success),
            "n_unique_test_patients": len({r["patient_id"] for r in patient_rows}),
            "n_model_seeds": len(cfg["model_seeds"]),
            "n_routing_seeds": len(cfg["routing_seeds"]),
            "n_valid_paired_patients": len(diffs),
            "missing_patient_policy": "drop_unpaired_or_nan_after_seed_aggregation",
            "ci_method": "paired_patient_bootstrap_percentile",
            "bootstrap_seed": seed,
            "bootstrap_reps": reps,
            "practical_margin": margin,
        })
    return rows


def _discordance(patient_rows):
    lookup = _lookup(patient_rows)
    out = []
    for dim in (16, 32):
        patients = sorted(
            {p for (m, d, p) in lookup if m == "gru_factored" and d == dim}
            & {p for (m, d, p) in lookup if m == "gru_monolithic" and d == dim}
        )
        for patient in patients:
            f = lookup[("gru_factored", dim, patient)]
            m = lookup[("gru_monolithic", dim, patient)]
            disc = float(f["acc_discordant"]) - float(m["acc_discordant"])
            conc = float(f["acc_concordant"]) - float(m["acc_concordant"])
            out.append({
                "qrs_mode": f["qrs_mode"],
                "total_state_dim": dim,
                "patient_id": patient,
                "n_discordant_beats": f["n_discordant_beats"],
                "n_concordant_beats": f["n_concordant_beats"],
                "factored_minus_monolithic_discordant_acc": disc,
                "factored_minus_monolithic_concordant_acc": conc,
                "interaction": disc - conc if np.isfinite(disc) and np.isfinite(conc) else float("nan"),
            })
    return out


def _memory_rows(seed_rows):
    seen = {}
    for row in seed_rows:
        key = (row["qrs_mode"], row["model_id"], int(row["total_state_dim"]), row["routing_seed"])
        if key not in seen:
            seen[key] = {k: row[k] for k in E01V2_RESULT_COLUMNS if k.endswith("_bytes") or k in (
                "qrs_mode", "model_id", "total_state_dim", "routing_seed",
                "recurrent_param_count", "total_model_param_count", "estimated_ops_per_beat",
            )}
    return list(seen.values())


def _write_decision(path: Path, cfg: dict, summary: list[dict], paired_rows: list[dict]):
    p = {r["comparison"]: r for r in paired_rows}
    comp = bool(p["compression_factored16_minus_monolithic32"]["success"])
    sem16 = bool(p["same_budget_factored16_minus_random16"]["success"])
    same32 = bool(p["same_budget_factored32_minus_monolithic32"]["success"] and p["same_budget_factored32_minus_random32"]["success"])
    verdict = "CONTINUE" if (comp and sem16) or same32 else "STOP_CONFIRMED"
    by_model = {(r["model_id"], int(r["total_state_dim"])): r for r in summary}
    path.write_text(
        f"# E01v2 {cfg['qrs_mode']} decision\n\n"
        f"Primary metric: `{cfg['primary_metric']}`. Statistical unit: patient after seed aggregation.\n\n"
        f"## Model Means\n"
        f"- monolithic-32: {by_model[('gru_monolithic', 32)]['mean_present_class_macro_f1']:.4f}\n"
        f"- factored-32: {by_model[('gru_factored', 32)]['mean_present_class_macro_f1']:.4f}\n"
        f"- random-32: {by_model[('gru_random', 32)]['mean_present_class_macro_f1']:.4f}\n\n"
        f"## Routes\n"
        f"- Compression factored-16 minus monolithic-32: mean {p['compression_factored16_minus_monolithic32']['mean_diff']:+.4f}, "
        f"95% CI [{p['compression_factored16_minus_monolithic32']['ci_low']:+.4f}, {p['compression_factored16_minus_monolithic32']['ci_high']:+.4f}], "
        f"non-inferior: {p['compression_factored16_minus_monolithic32']['success']}; random control at 16: {sem16}\n"
        f"- Same-budget 32 superiority versus monolithic and random: {same32}\n\n"
        f"## VERDICT: {verdict}\n"
    )
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dirty = git_tree_dirty(ignore_paths=GENERATED_DIRS)
    if dirty and not (args.allow_dirty or cfg.get("allow_dirty", False)):
        raise SystemExit("refusing to create E01v2 result from dirty source tree")

    split = load_split(cfg["split_manifest"])
    dataset_cfg = load_config(cfg["dataset_config"])
    normalizer = training_signal_normalizer(dataset_cfg, split["train"])
    cohort, qrs_rows = load_afdb_cohort_v2(dataset_cfg, split, cfg["qrs_mode"], normalizer)
    selected = _load_selected(cfg["selected_hyperparameters"])
    run_id = uuid.uuid4().hex[:12]
    commit = full_git_commit()
    envh = environment_hash()
    phash = protocol_hash(cfg["protocol_config"])
    status = "NONCONFIRMATORY" if dirty else "PASS"
    outdir = Path(cfg["output_dir"])
    outdir.mkdir(parents=True, exist_ok=True)

    env = environment_report()
    env.update({
        "environment_hash": envh,
        "full_commit_sha": commit,
        "git_tree_dirty": dirty,
        "config_hash": cfg["_config_hash"],
        "protocol_hash": phash,
        "dataset_manifest_hash": file_sha256(dataset_cfg["manifest"]),
        "split_manifest_hash": file_sha256(cfg["split_manifest"]),
        "dependency_lock_hash": file_sha256("requirements-lock.txt"),
        "random_seed_bundle": {"model_seeds": cfg["model_seeds"], "routing_seeds": cfg["routing_seeds"], "split_seed": split["split_seed"]},
    })
    Path("results/e01v2").mkdir(parents=True, exist_ok=True)
    with open("results/e01v2/e01v2_environment.json", "w") as fh:
        json.dump(env, fh, indent=2, sort_keys=True)
        fh.write("\n")

    qrs_detector = StreamingQRSDetector(int(dataset_cfg["fs_hz"])) if cfg["qrs_mode"] == "deployable" else None
    fmem = feature_memory(dataset_cfg)
    seed_rows = []
    for model_id in cfg["models"]:
        for dim in cfg["state_dims"]:
            for model_seed in cfg["model_seeds"]:
                for routing_seed in _route_options(model_id, cfg["routing_seeds"]):
                    route_arg = None if routing_seed == "" else int(routing_seed)
                    hp = _hp_for(cfg, selected, model_id, int(dim), int(model_seed), routing_seed)
                    model, mu, sd = fit_gru(
                        model_id,
                        N_FEATURES,
                        ATRIAL_IDX,
                        RHYTHM_IDX,
                        int(dim),
                        int(model_seed),
                        cohort,
                        split["train"],
                        hp,
                        routing_seed=route_arg,
                    )
                    normalizer_bytes = int(normalizer.nbytes + mu.nbytes + sd.nbytes)
                    mem = memory_report(
                        model,
                        dtype_bytes=int(cfg["dtype_bytes"]),
                        normalizer_bytes=normalizer_bytes,
                        filter_constant_bytes=qrs_detector.filter_constant_bytes() if qrs_detector else 0,
                        qrs_persistent_bytes=qrs_detector.persistent_state_bytes() if qrs_detector else 0,
                        qrs_scratch_bytes=qrs_detector.scratch_bytes() if qrs_detector else 0,
                        feature_persistent_bytes=fmem["feature_persistent_bytes"],
                        feature_scratch_bytes=fmem["feature_scratch_bytes"],
                    )
                    for patient in split["test"]:
                        prob = predict_proba_streaming(model, cohort[patient]["feat"], mu, sd)
                        yhat = (prob >= 0.5).astype(int)
                        metrics = patient_metrics(cohort[patient]["y"], yhat)
                        disc = discordance_accuracy(cohort[patient]["y"], yhat, cohort[patient]["disc"])
                        n_beats = int(len(cohort[patient]["y"]))
                        n_disc = int(np.sum(cohort[patient]["disc"] == 1))
                        row = {
                            "run_id": run_id,
                            "experiment_id": cfg["experiment_id"],
                            "protocol_version": PROTOCOL_VERSION,
                            "config_hash": cfg["_config_hash"],
                            "protocol_hash": phash,
                            "full_commit_sha": commit,
                            "git_tree_dirty": dirty,
                            "environment_hash": envh,
                            "dataset_id": cfg["dataset_id"],
                            "qrs_mode": cfg["qrs_mode"],
                            "split_id": SPLIT_ID,
                            "model_id": model_id,
                            "total_state_dim": int(dim),
                            "atrial_state_dim": mem["atrial_state_dim"],
                            "rhythm_state_dim": mem["rhythm_state_dim"],
                            "model_seed": int(model_seed),
                            "routing_seed": routing_seed,
                            "patient_id": patient,
                            "patient_has_af": metrics["patient_has_af"],
                            "patient_has_nonaf": metrics["patient_has_nonaf"],
                            "n_beats": n_beats,
                            "n_af_beats": metrics["n_af_beats"],
                            "n_nonaf_beats": metrics["n_nonaf_beats"],
                            "n_discordant_beats": n_disc,
                            "n_concordant_beats": n_beats - n_disc,
                            "af_f1": metrics["af_f1"],
                            "present_class_macro_f1": metrics["present_class_macro_f1"],
                            "sensitivity": metrics["sensitivity"],
                            "specificity": metrics["specificity"],
                            "ppv": metrics["ppv"],
                            "acc_concordant": disc["concordant"],
                            "acc_discordant": disc["discordant"],
                            "status": status,
                        }
                        row.update(mem)
                        seed_rows.append(row)

    write_rows(outdir / "e01v2_seed_patient.csv", seed_rows, E01V2_RESULT_COLUMNS)
    patient_rows = _aggregate_patient(seed_rows)
    write_rows(outdir / "e01v2_patient_aggregated.csv", patient_rows, list(patient_rows[0].keys()))
    summary = _model_summary(patient_rows)
    write_rows(outdir / "e01v2_model_summary.csv", summary, list(summary[0].keys()))
    paired = _paired(patient_rows, cfg)
    write_rows(outdir / "e01v2_paired_tests.csv", paired, list(paired[0].keys()))
    mem_rows = _memory_rows(seed_rows)
    write_rows(outdir / "e01v2_memory_ledger.csv", mem_rows, list(mem_rows[0].keys()))
    disc_rows = _discordance(patient_rows)
    disc_fields = [
        "qrs_mode", "total_state_dim", "patient_id", "n_discordant_beats",
        "n_concordant_beats", "factored_minus_monolithic_discordant_acc",
        "factored_minus_monolithic_concordant_acc", "interaction",
    ]
    write_rows(outdir / "e01v2_discordance.csv", disc_rows, disc_fields)
    if cfg["qrs_mode"] == "deployable":
        for row in qrs_rows:
            row["qrs_persistent_bytes"] = qrs_detector.persistent_state_bytes()
            row["qrs_scratch_bytes"] = qrs_detector.scratch_bytes()
        qrs_fields = [
            "patient_id", "qrs_sensitivity", "qrs_ppv", "qrs_timing_error_samples",
            "qrs_matched_peaks", "qrs_missed_peaks", "qrs_false_peaks",
            "qrs_reference_peaks", "qrs_detected_peaks", "qrs_persistent_bytes",
            "qrs_scratch_bytes",
        ]
        write_rows(outdir / "e01v2_qrs_metrics.csv", qrs_rows, qrs_fields)
    verdict = _write_decision(outdir / "e01v2_decision.md", cfg, summary, paired)
    print(f"wrote E01v2 {cfg['qrs_mode']} outputs to {outdir} -> {verdict}")


if __name__ == "__main__":
    main()
