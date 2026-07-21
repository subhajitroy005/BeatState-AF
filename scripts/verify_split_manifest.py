#!/usr/bin/env python3
"""Verify the frozen AFDB split is complete, disjoint, and shared."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split_manifest")
    ap.add_argument("--record-manifest", default="manifests/afdb_records.json")
    args = ap.parse_args()

    with open(args.split_manifest) as fh:
        split = json.load(fh)
    with open(args.record_manifest) as fh:
        records = set(json.load(fh)["records"])

    required = {"dataset_id", "split_seed", "train", "validation", "test"}
    missing = required - set(split)
    if missing:
        raise SystemExit(f"missing split fields: {sorted(missing)}")
    sets = {k: set(split[k]) for k in ("train", "validation", "test")}
    if sets["train"] & sets["validation"] or sets["train"] & sets["test"] or sets["validation"] & sets["test"]:
        raise SystemExit("split leakage: train/validation/test are not disjoint")
    union = sets["train"] | sets["validation"] | sets["test"]
    if union != records:
        raise SystemExit(f"split does not cover manifest records: missing={sorted(records - union)} extra={sorted(union - records)}")
    if split["dataset_id"] != "afdb_v2":
        raise SystemExit(f"unexpected dataset_id: {split['dataset_id']}")
    if int(split["split_seed"]) != 20260721:
        raise SystemExit(f"unexpected split_seed: {split['split_seed']}")
    print(f"verified {args.split_manifest}: {len(sets['train'])}/{len(sets['validation'])}/{len(sets['test'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
