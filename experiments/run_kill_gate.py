#!/usr/bin/env python3
"""The KILL GATE. First runnable evidence chain, unchanged in logic across data
sources and model families.

Compares, at matched total state dimension and identical tokens/precision:
  monolithic (dense state) vs factored (physiology block-diagonal) vs random
  (block-diagonal, non-physiological assignment -- the sparsity control).

Data source (synthetic | afdb_oracle | afdb_deployable) and model family
(reservoir | gru) are selected by config; the comparison, aggregation, statistics
and verdict are identical. Synthetic proves the harness; the meaningful verdict is
GRU on afdb_deployable (CLAUDE.md Task 4).
"""
from __future__ import annotations
import argparse, csv, os, uuid
import numpy as np

from beatstate_af.config import load_config
from beatstate_af.data.synthetic import make_cohort, ATRIAL_IDX, RHYTHM_IDX, N_FEATURES
from beatstate_af.data.patient_split import patient_disjoint_split
from beatstate_af.models.reservoir import build_model
from beatstate_af.training.readout import LogisticReadout
from beatstate_af.memory.ledger import memory_report
from beatstate_af.evaluation.metrics import patient_metrics, discordance_accuracy
from beatstate_af.statistics.stats import superiority, equivalent
from beatstate_af.provenance import RESULT_ROW_COLUMNS, git_commit, environment_hash

OUT = "results/primary"
_COHORT_CACHE = {}


def states_for(model, patient):
    model.reset_state()
    return np.array([model.step(x) for x in patient["feat"]])


def _row(kind, dim, seed, p, mem, m, d):
    return dict(model_id=kind, total_state_dim=dim, atrial_state_dim=mem["atrial_state_dim"],
                rhythm_state_dim=mem["rhythm_state_dim"], seed=seed, patient_id=p,
                recurrent_state_bytes=mem["recurrent_state_bytes"],
                recurrent_param_count=mem["recurrent_param_count"],
                readout_param_count=mem["readout_param_count"], dtype_bytes=mem["dtype_bytes"],
                macro_f1=m["macro_f1"], sensitivity=m["sensitivity"],
                specificity=m["specificity"], ppv=m["ppv"],
                acc_concordant=d["concordant"], acc_discordant=d["discordant"])


def run_cell(kind, dim, seed, cohort, train, test, family="reservoir", hp=None):
    """Train one model arm and score every test patient. Family swaps the model
    factory + readout; row schema and metric flow are identical."""
    rows, per_patient_f1 = [], {}
    if family == "gru":
        from beatstate_af.models.gru import fit_gru, predict_proba_seq
        model, mu, sd = fit_gru(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, dim, seed,
                                cohort, train, hp or {})
        mem = memory_report(model, model.readout_param_count(), dtype_bytes=4)
        for p in test:
            yhat = (predict_proba_seq(model, cohort[p]["feat"], mu, sd) >= 0.5).astype(int)
            m = patient_metrics(cohort[p]["y"], yhat)
            d = discordance_accuracy(cohort[p]["y"], yhat, cohort[p]["disc"])
            per_patient_f1[p] = m["macro_f1"]
            rows.append(_row(kind, dim, seed, p, mem, m, d))
        return rows, per_patient_f1, mem

    # --- reservoir family (walking skeleton) ---
    model = build_model(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, dim, seed)
    Xtr = np.vstack([states_for(model, cohort[p]) for p in train])
    ytr = np.concatenate([cohort[p]["y"] for p in train])
    readout = LogisticReadout(seed=seed).fit(Xtr, ytr)
    mem = memory_report(model, readout.n_params, dtype_bytes=4)
    for p in test:
        H = states_for(model, cohort[p])
        yhat = (readout.predict_proba(H) >= 0.5).astype(int)
        m = patient_metrics(cohort[p]["y"], yhat)
        d = discordance_accuracy(cohort[p]["y"], yhat, cohort[p]["disc"])
        per_patient_f1[p] = m["macro_f1"]
        rows.append(_row(kind, dim, seed, p, mem, m, d))
    return rows, per_patient_f1, mem


def load_cohort(cfg, seed):
    ds = cfg["dataset_id"]
    if ds.startswith("synthetic"):
        return make_cohort(cfg["n_patients"], base_seed=1000 * seed, n_beats=cfg["n_beats"],
                           p_af=cfg["p_af"], p_disc=cfg["p_discordance"])
    if ds.startswith("afdb"):
        if ds not in _COHORT_CACHE:                      # real data: seed-independent
            from beatstate_af.data.wfdb_io import load_afdb_cohort
            dcfg = load_config(cfg["dataset_config"])
            mode = "oracle" if ds.endswith("oracle") else "deployable"
            _COHORT_CACHE[ds] = load_afdb_cohort(dcfg, qrs_mode=mode)
        return _COHORT_CACHE[ds]
    raise ValueError(f"unknown dataset_id: {ds}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/kill_gate_pilot.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    os.makedirs(OUT, exist_ok=True)
    run_id = uuid.uuid4().hex[:12]; commit = git_commit(); envh = environment_hash()

    family = cfg.get("model_family", "reservoir")
    hp = cfg.get("gru", {})
    ARM_M = cfg.get("arm_monolithic", "monolithic")
    ARM_F = cfg.get("arm_factored", "factored")
    ARM_R = cfg.get("arm_random", "random")

    all_rows, f1 = [], {}            # f1[(kind,dim)][ (seed,patient) ] = macro_f1
    for seed in cfg["seeds"]:
        cohort = load_cohort(cfg, seed)
        train, test = patient_disjoint_split(cohort.keys(), cfg["train_frac"], seed=seed)
        for kind in cfg["models"]:
            for dim in cfg["state_dims"]:
                rows, ppf1, mem = run_cell(kind, dim, seed, cohort, train, test, family, hp)
                all_rows.extend(rows)
                f1.setdefault((kind, dim), {})
                for p, v in ppf1.items():
                    f1[(kind, dim)][(seed, p)] = v

    # ---- write per-patient result rows ----
    tag = cfg.get("output_tag", "")
    pref = f"kill_gate_{tag}_" if tag else "kill_gate_"
    pcsv = os.path.join(OUT, f"{pref}patient_seed.csv")
    with open(pcsv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=RESULT_ROW_COLUMNS)
        w.writeheader()
        for r in all_rows:
            r.update(run_id=run_id, experiment_id=cfg["experiment_id"], config_hash=cfg["_config_hash"],
                     commit_sha=commit, environment_hash=envh, dataset_id=cfg["dataset_id"], status="PASS")
            w.writerow({k: r.get(k, "") for k in RESULT_ROW_COLUMNS})

    # ---- aggregate + statistics ----
    def mean_f1(kind, dim):
        return float(np.mean(list(f1[(kind, dim)].values())))
    def paired(kind_a, dim_a, kind_b, dim_b):
        keys = set(f1[(kind_a, dim_a)]) & set(f1[(kind_b, dim_b)])
        return [f1[(kind_a, dim_a)][k] - f1[(kind_b, dim_b)][k] for k in sorted(keys)]

    scsv = os.path.join(OUT, f"{pref}summary.csv")
    with open(scsv, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["model_id", "total_state_dim", "mean_macro_f1",
                                        "recurrent_param_count", "recurrent_state_bytes"])
        for kind in cfg["models"]:
            for dim in cfg["state_dims"]:
                any_row = next(r for r in all_rows if r["model_id"] == kind and r["total_state_dim"] == dim)
                w.writerow([kind, dim, round(mean_f1(kind, dim), 4),
                            any_row["recurrent_param_count"], any_row["recurrent_state_bytes"]])

    dims = cfg["state_dims"]; top = max(dims)
    sup = superiority(paired(ARM_F, top, ARM_M, top), margin=cfg["superiority_margin"])
    spars = superiority(paired(ARM_F, top, ARM_R, top), margin=0.0)
    comp = equivalent(paired(ARM_F, cfg["compression_low_dim"], ARM_M, cfg["compression_high_dim"]),
                      margin=cfg["equivalence_margin"])
    disc_f = float(np.nanmean([r["acc_discordant"] for r in all_rows if r["model_id"] == ARM_F and r["total_state_dim"] == top]))
    disc_m = float(np.nanmean([r["acc_discordant"] for r in all_rows if r["model_id"] == ARM_M and r["total_state_dim"] == top]))

    # ---- auto kill decision (roadmap section 3.5) ----
    verdict = ("CONTINUE" if (sup["superior"] or comp["equivalent"]) else "STOP")
    is_synth = cfg["dataset_id"].startswith("synthetic")
    note = ("> NOTE: dataset is SYNTHETIC. This validates the harness end-to-end. The\n"
            "> decision below is only meaningful once dataset_id = afdb_deployable (see CLAUDE.md).\n"
            if is_synth else
            f"> Dataset is REAL ({cfg['dataset_id']}). This verdict is confirmatory per roadmap 3.5.\n")
    mono_rp = next(r['recurrent_param_count'] for r in all_rows if r['model_id']==ARM_M and r['total_state_dim']==top)
    fac_rp = next(r['recurrent_param_count'] for r in all_rows if r['model_id']==ARM_F and r['total_state_dim']==top)
    with open(os.path.join(OUT, f"{pref}decision.md"), "w") as fh:
        fh.write(f"""# Kill-gate decision (auto-generated)

run_id: {run_id} | config_hash: {cfg['_config_hash']} | commit: {commit} | dataset: {cfg['dataset_id']} | family: {family}

{note}
## Mean macro-F1 (state dim = {top})
- monolithic ({ARM_M}): {mean_f1(ARM_M, top):.4f}   (recurrent params {mono_rp})
- factored   ({ARM_F}): {mean_f1(ARM_F, top):.4f}   (recurrent params {fac_rp})
- random     ({ARM_R}): {mean_f1(ARM_R, top):.4f}

## Success routes
- Superiority (factored - monolithic @ {top}): mean {sup['mean']:+.4f}, 95% CI [{sup['ci'][0]:+.4f}, {sup['ci'][1]:+.4f}] -> superior: {sup['superior']}
- Sparsity control (factored - random @ {top}): mean {spars['mean']:+.4f}, 95% CI [{spars['ci'][0]:+.4f}, {spars['ci'][1]:+.4f}] -> semantics beat sparsity: {spars['superior']}
- Compression (factored@{cfg['compression_low_dim']} vs monolithic@{cfg['compression_high_dim']}): mean {comp['mean']:+.4f}, 95% CI [{comp['ci'][0]:+.4f}, {comp['ci'][1]:+.4f}] -> equivalent at fewer bytes: {comp['equivalent']}
- Discordance accuracy @ {top}: factored {disc_f:.3f} vs monolithic {disc_m:.3f} (delta {disc_f-disc_m:+.3f})

## VERDICT: {verdict}
Per roadmap 3.5: continue only if superiority OR compression is supported. Do not
rescue a failure by adding attention/Transformer/Mamba/branches.
""")
    print(f"[kill-gate] wrote {pcsv}, {scsv}, and {pref}decision.md")
    print(f"[kill-gate] monolithic={mean_f1(ARM_M,top):.3f} factored={mean_f1(ARM_F,top):.3f} random={mean_f1(ARM_R,top):.3f}")
    print(f"[kill-gate] superiority={sup['superior']} sparsity_semantics={spars['superior']} compression={comp['equivalent']} -> {verdict}")


if __name__ == "__main__":
    main()
