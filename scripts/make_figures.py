#!/usr/bin/env python3
"""Generate figures/tables from released CSVs only (never hand-place numbers).

Kill-gate figure: mean macro-F1 by model x state-dim, with the memory ledger. If
matplotlib is available a PNG is written; otherwise a Markdown table is emitted so
the figure still regenerates from the CSV with zero heavy deps.
"""
from __future__ import annotations
import csv, os, sys

FIGDIR = "figures/paper"


def load_summary(path):
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        r["total_state_dim"] = int(r["total_state_dim"])
        r["mean_macro_f1"] = float(r["mean_macro_f1"])
        r["recurrent_param_count"] = int(r["recurrent_param_count"])
        r["recurrent_state_bytes"] = int(r["recurrent_state_bytes"])
    return rows


def main(summary="results/primary/kill_gate_summary.csv"):
    os.makedirs(FIGDIR, exist_ok=True)
    rows = load_summary(summary)
    dims = sorted({r["total_state_dim"] for r in rows})
    models = sorted({r["model_id"] for r in rows})
    base = os.path.basename(summary)
    tag = base[len("kill_gate_"):-len("summary.csv")].strip("_") or "primary"

    md = ["# Kill-gate figure (generated from %s)\n" % summary,
          "Mean patient-level macro-F1 by model and state dimension; recurrent",
          "params / state bytes from the memory ledger.\n",
          "| model | " + " | ".join(f"dim {d} (F1)" for d in dims) + " | recur.params (max dim) | state bytes (max dim) |",
          "|---|" + "---|" * (len(dims) + 2)]
    for m in models:
        cells = []
        for d in dims:
            v = next((r["mean_macro_f1"] for r in rows if r["model_id"] == m and r["total_state_dim"] == d), None)
            cells.append(f"{v:.4f}" if v is not None else "-")
        top = max(dims)
        rp = next(r["recurrent_param_count"] for r in rows if r["model_id"] == m and r["total_state_dim"] == top)
        sb = next(r["recurrent_state_bytes"] for r in rows if r["model_id"] == m and r["total_state_dim"] == top)
        md.append(f"| {m} | " + " | ".join(cells) + f" | {rp} | {sb} |")
    out_md = os.path.join(FIGDIR, f"kill_gate_{tag}.md")
    open(out_md, "w").write("\n".join(md) + "\n")
    written = [out_md]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        x = np.arange(len(dims)); w = 0.8 / max(1, len(models))
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, m in enumerate(models):
            ys = [next((r["mean_macro_f1"] for r in rows if r["model_id"] == m and r["total_state_dim"] == d), 0) for d in dims]
            ax.bar(x + i * w, ys, w, label=m)
        ax.set_xticks(x + w * (len(models) - 1) / 2); ax.set_xticklabels([f"dim {d}" for d in dims])
        ax.set_ylabel("mean macro-F1 (patient unit)"); ax.set_ylim(0, 1)
        ax.set_title(f"Matched-capacity kill gate ({tag})"); ax.legend()
        png = os.path.join(FIGDIR, f"kill_gate_{tag}.png")
        fig.tight_layout(); fig.savefig(png, dpi=140); written.append(png)
    except Exception as e:
        print(f"[figures] matplotlib unavailable ({e}); wrote Markdown table only")
    print("[figures] wrote " + ", ".join(written))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/primary/kill_gate_summary.csv")
