#!/usr/bin/env python3
"""Post-run E01v2 cross-mode statistics."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.e01v2 import PRIMARY_METRIC, write_rows


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    dep = _read("results/e01v2/deployable/e01v2_patient_aggregated.csv")
    ora = _read("results/e01v2/oracle/e01v2_patient_aggregated.csv")
    olookup = {(r["model_id"], r["total_state_dim"], r["patient_id"]): r for r in ora}
    rows = []
    for d in dep:
        key = (d["model_id"], d["total_state_dim"], d["patient_id"])
        if key not in olookup:
            continue
        o = olookup[key]
        rows.append({
            "model_id": d["model_id"],
            "total_state_dim": d["total_state_dim"],
            "patient_id": d["patient_id"],
            "metric": PRIMARY_METRIC,
            "deployable": d[PRIMARY_METRIC],
            "oracle": o[PRIMARY_METRIC],
            "deployable_minus_oracle": float(d[PRIMARY_METRIC]) - float(o[PRIMARY_METRIC]),
        })
    write_rows("results/e01v2/e01v2_delta_qrs.csv", rows, list(rows[0].keys()))
    print(f"wrote results/e01v2/e01v2_delta_qrs.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
