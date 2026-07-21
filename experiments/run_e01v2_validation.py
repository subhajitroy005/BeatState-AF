#!/usr/bin/env python3
"""Train/validation-only hyperparameter selection for E01v2."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.config import load_config
from beatstate_af.e01v2 import (
    ATRIAL_IDX,
    N_FEATURES,
    RHYTHM_IDX,
    load_afdb_cohort_v2,
    load_split,
    training_signal_normalizer,
)
from beatstate_af.models.gru import fit_gru_with_validation


def _route_options(model_id: str, routing_seeds: list[int]):
    return routing_seeds if model_id == "gru_random" else [""]


def _base_hp(cfg: dict, search: dict) -> dict:
    hp = dict(cfg.get("gru", {}))
    hp["lr"] = float(search["learning_rates"][0])
    hp["weight_decay"] = float(search["weight_decays"][0])
    hp["epochs"] = int(max(search["epoch_counts"]))
    hp["grad_clip"] = float(search["gradient_clips"][0])
    hp["chunk_len"] = int(search["chunk_len"])
    hp["train_max_beats"] = int(search["train_max_beats"])
    return hp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/e01v2_afdb_deployable.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    protocol = load_config(cfg["protocol_config"])
    search = load_config(protocol["training_selection_policy"]["search_config"])
    split = load_split(cfg["split_manifest"])
    dataset_cfg = load_config(cfg["dataset_config"])
    normalizer = training_signal_normalizer(dataset_cfg, split["train"])
    cohort, _ = load_afdb_cohort_v2(dataset_cfg, split, cfg["qrs_mode"], normalizer)
    candidate_epochs = [int(x) for x in search["epoch_counts"]]
    hp = _base_hp(cfg, search)

    curves = []
    selected = {
        "selection_scope": "train_validation_only",
        "selection_metric": search["selection_metric"],
        "search_config_hash": search["_config_hash"],
        "normalizer": {"median": normalizer.median, "scale": normalizer.scale},
        "arms": {},
    }
    for model_id in cfg["models"]:
        for dim in cfg["state_dims"]:
            for model_seed in cfg["model_seeds"]:
                for routing_seed in _route_options(model_id, cfg["routing_seeds"]):
                    route_arg = None if routing_seed == "" else int(routing_seed)
                    _, _, _, history = fit_gru_with_validation(
                        model_id,
                        N_FEATURES,
                        ATRIAL_IDX,
                        RHYTHM_IDX,
                        int(dim),
                        int(model_seed),
                        cohort,
                        split["train"],
                        split["validation"],
                        hp,
                        routing_seed=route_arg,
                    )
                    candidates = [h for h in history if h["epoch"] in candidate_epochs]
                    best = max(candidates, key=lambda h: h["validation_present_class_macro_f1"])
                    key = f"{model_id}|{dim}|{model_seed}|{routing_seed}"
                    selected["arms"][key] = {
                        "model_id": model_id,
                        "total_state_dim": int(dim),
                        "model_seed": int(model_seed),
                        "routing_seed": routing_seed,
                        "epochs": int(best["epoch"]),
                        "lr": hp["lr"],
                        "weight_decay": hp["weight_decay"],
                        "chunk_len": hp["chunk_len"],
                        "grad_clip": hp["grad_clip"],
                        "train_max_beats": hp["train_max_beats"],
                        "selected_validation_present_class_macro_f1": float(best["validation_present_class_macro_f1"]),
                        "convergence_status": "selected_from_frozen_validation",
                    }
                    for h in history:
                        curves.append({
                            "experiment_id": cfg["experiment_id"],
                            "qrs_mode": cfg["qrs_mode"],
                            "model_id": model_id,
                            "total_state_dim": int(dim),
                            "model_seed": int(model_seed),
                            "routing_seed": routing_seed,
                            "epoch": int(h["epoch"]),
                            "train_loss": h["train_loss"],
                            "validation_present_class_macro_f1": h["validation_present_class_macro_f1"],
                            "selected_epoch": int(best["epoch"]),
                            "selected": bool(h["epoch"] == best["epoch"]),
                            "convergence_status": "selected_from_frozen_validation",
                        })

    out = Path(protocol["outputs"]["validation_curves"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        fieldnames = list(curves[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(curves)

    sel_path = Path(protocol["training_selection_policy"]["selected_hyperparameters"])
    sel_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sel_path, "w") as fh:
        yaml.safe_dump(selected, fh, sort_keys=False)

    fig = Path("figures/audit/e01v2_training_curves.svg")
    fig.parent.mkdir(parents=True, exist_ok=True)
    mean_val = sum(float(r["validation_present_class_macro_f1"]) for r in curves) / len(curves)
    fig.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="640" height="160">'
        f'<rect width="640" height="160" fill="white"/>'
        f'<text x="20" y="40" font-family="monospace" font-size="16">E01v2 validation curves</text>'
        f'<text x="20" y="75" font-family="monospace" font-size="13">rows: {len(curves)}</text>'
        f'<text x="20" y="105" font-family="monospace" font-size="13">mean validation present-class macro-F1: {mean_val:.4f}</text>'
        f'</svg>\n'
    )
    print(f"wrote {out}, {sel_path}, and {fig}")


if __name__ == "__main__":
    main()
