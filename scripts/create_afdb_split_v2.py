#!/usr/bin/env python3
"""Create the frozen E01v2 AFDB train/validation/test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260721)
    ap.add_argument("--manifest", default="manifests/afdb_records.json")
    ap.add_argument("--output", default="manifests/afdb_split_v2.json")
    args = ap.parse_args()

    with open(args.manifest) as fh:
        records = list(json.load(fh)["records"])
    ids = records[:]
    rng = np.random.default_rng(args.seed)
    rng.shuffle(ids)
    train = sorted(ids[:14])
    validation = sorted(ids[14:18])
    test = sorted(ids[18:])
    split = {
        "dataset_id": "afdb_v2",
        "split_id": "afdb_split_v2",
        "split_seed": args.seed,
        "source_manifest": args.manifest,
        "train": train,
        "validation": validation,
        "test": test,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(split, fh, indent=2)
        fh.write("\n")
    print(f"wrote {args.output}: train={len(train)} validation={len(validation)} test={len(test)}")


if __name__ == "__main__":
    main()
