#!/usr/bin/env python3
"""Verify required E01v2 artifacts and preserved E01v1 hashes."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.e01v2 import E01V2_RESULT_COLUMNS
from beatstate_af.provenance import file_sha256


E01V1_HASHES = {
    "results/primary/kill_gate_patient_seed.csv": "bf4ee1333a90915544a9d6d57d28efbb1f2d7b243e43e2f91c5c10b3f5405158",
    "results/primary/kill_gate_summary.csv": "3285868ccb8d100e51167d0fccbe1b7e6220a1d55885985490d5d11b976d93dc",
    "results/primary/kill_gate_decision.md": "0342a1083404e26119d114953fa26af997b6aa6824a735fb12488df9268c9c74",
    "results/primary/kill_gate_oracle_patient_seed.csv": "4e037216712d5e448a95ce7558f5d4b8edc233afac8141310b595b2869f44bea",
    "results/primary/kill_gate_oracle_summary.csv": "8904452c892b1f40de4bdd63027947de0aba1aa6e6e0eef0c459b6c96741a077",
    "results/primary/kill_gate_oracle_decision.md": "b53c7070514d363868a216785d42a77b617eabe2e61bbe4188c4485d97481268",
}

REQUIRED = [
    "results/e01v2/deployable/e01v2_seed_patient.csv",
    "results/e01v2/deployable/e01v2_patient_aggregated.csv",
    "results/e01v2/deployable/e01v2_model_summary.csv",
    "results/e01v2/deployable/e01v2_paired_tests.csv",
    "results/e01v2/deployable/e01v2_memory_ledger.csv",
    "results/e01v2/deployable/e01v2_qrs_metrics.csv",
    "results/e01v2/deployable/e01v2_discordance.csv",
    "results/e01v2/deployable/e01v2_decision.md",
    "results/e01v2/oracle/e01v2_seed_patient.csv",
    "results/e01v2/oracle/e01v2_patient_aggregated.csv",
    "results/e01v2/oracle/e01v2_model_summary.csv",
    "results/e01v2/oracle/e01v2_paired_tests.csv",
    "results/e01v2/oracle/e01v2_memory_ledger.csv",
    "results/e01v2/oracle/e01v2_discordance.csv",
    "results/e01v2/oracle/e01v2_decision.md",
    "results/e01v2/e01v2_delta_qrs.csv",
    "results/e01v2/e01v2_final_decision.md",
    "results/e01v2/e01v2_environment.json",
    "results/e01v2/e01v2_artifact_manifest.json",
    "figures/e01v2/state_memory_comparison.svg",
    "figures/e01v2/patient_paired_differences.svg",
    "figures/e01v2/semantic_vs_random.svg",
    "figures/e01v2/oracle_vs_deployable.svg",
    "figures/e01v2/training_convergence.svg",
]


def main():
    for path, expected in E01V1_HASHES.items():
        actual = file_sha256(path)
        if actual != expected:
            raise SystemExit(f"E01v1 artifact changed: {path}")
    for path in REQUIRED:
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            raise SystemExit(f"missing or empty artifact: {path}")
    if Path("CLAUDE.md").exists():
        raise SystemExit("context file CLAUDE.md is present in the repo")
    with open("results/e01v2/deployable/e01v2_seed_patient.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != E01V2_RESULT_COLUMNS:
            raise SystemExit("deployable seed-patient schema mismatch")
        rows = list(reader)
    if not rows:
        raise SystemExit("no deployable seed-patient rows")
    if any(r["status"] != "PASS" for r in rows):
        raise SystemExit("deployable rows are not confirmatory PASS rows")
    print("E01v2 artifacts verified")


if __name__ == "__main__":
    main()
