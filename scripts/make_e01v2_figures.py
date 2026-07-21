#!/usr/bin/env python3
"""Generate lightweight SVG audit figures from E01v2 CSV artifacts."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _svg(path: str, title: str, lines: list[str]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    height = 80 + 24 * len(lines)
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="{height}">',
        '<rect width="900" height="100%" fill="white"/>',
        f'<text x="24" y="36" font-family="monospace" font-size="18">{title}</text>',
    ]
    y = 70
    for line in lines:
        body.append(f'<text x="24" y="{y}" font-family="monospace" font-size="13">{line}</text>')
        y += 24
    body.append("</svg>")
    p.write_text("\n".join(body) + "\n")


def main():
    dep_summary = _read("results/e01v2/deployable/e01v2_model_summary.csv")
    ora_summary = _read("results/e01v2/oracle/e01v2_model_summary.csv")
    dep_pairs = _read("results/e01v2/deployable/e01v2_paired_tests.csv")
    delta = _read("results/e01v2/e01v2_delta_qrs.csv")

    _svg(
        "figures/e01v2/state_memory_comparison.svg",
        "E01v2 State/Memory Comparison",
        [f"{r['model_id']}-{r['total_state_dim']}: peak bytes {float(r['total_peak_inference_bytes']):.0f}" for r in dep_summary],
    )
    _svg(
        "figures/e01v2/patient_paired_differences.svg",
        "E01v2 Patient Paired Differences",
        [f"{r['comparison']}: mean {float(r['mean_diff']):+.4f} CI [{float(r['ci_low']):+.4f}, {float(r['ci_high']):+.4f}]" for r in dep_pairs],
    )
    _svg(
        "figures/e01v2/semantic_vs_random.svg",
        "E01v2 Semantic vs Random",
        [f"{r['comparison']}: success={r['success']}" for r in dep_pairs if "random" in r["comparison"]],
    )
    _svg(
        "figures/e01v2/oracle_vs_deployable.svg",
        "E01v2 Oracle vs Deployable",
        [f"{r['model_id']}-{r['total_state_dim']}: deployable-oracle {float(r['deployable_minus_oracle']):+.4f}" for r in delta[:18]],
    )
    _svg(
        "figures/e01v2/training_convergence.svg",
        "E01v2 Training Convergence",
        [f"deployable summary rows={len(dep_summary)} oracle summary rows={len(ora_summary)}"],
    )
    print("wrote figures/e01v2/*.svg")


if __name__ == "__main__":
    main()
