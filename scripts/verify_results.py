#!/usr/bin/env python3
"""Verify every result row carries the full provenance schema and valid metrics."""
import csv, sys
from beatstate_af.provenance import RESULT_ROW_COLUMNS

def main(path):
    with open(path) as f:
        r = csv.DictReader(f); rows = list(r)
        assert r.fieldnames == RESULT_ROW_COLUMNS, f"schema mismatch:\n{r.fieldnames}"
    assert rows, "no rows"
    for row in rows:
        for k in ("run_id", "config_hash", "model_id", "patient_id"):
            assert row[k], f"empty {k}"
        assert 0.0 <= float(row["macro_f1"]) <= 1.0
    print(f"OK: {len(rows)} rows, schema + metrics valid ({path})")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/primary/kill_gate_patient_seed.csv")
