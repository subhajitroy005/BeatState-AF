#!/usr/bin/env python3
"""Manifest-driven AFDB download (CLAUDE.md Task 1).

Downloads ONLY the frozen manifest records (signal + .atr + .qrs) into the local
cache and writes a checksum record. Does not recursively pull the whole database.
"""
from __future__ import annotations
import hashlib, json, os, sys

MANIFEST = "manifests/afdb_records.json"
CACHE = "data/afdb"
ANNOTATORS = ["atr", "qrs"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    import wfdb
    with open(MANIFEST) as f:
        manifest = json.load(f)
    recs = manifest["records"]
    os.makedirs(CACHE, exist_ok=True)
    todo = [r for r in recs if not os.path.exists(os.path.join(CACHE, f"{r}.dat"))]
    print(f"[download] {len(recs)} manifest records, {len(todo)} missing -> fetching")
    if todo:
        wfdb.dl_database("afdb", CACHE, records=todo, annotators=ANNOTATORS)

    sums = {}
    for r in recs:
        for ext in ("dat", "hea", "atr", "qrs"):
            p = os.path.join(CACHE, f"{r}.{ext}")
            if os.path.exists(p):
                sums[f"{r}.{ext}"] = {"sha256": _sha256(p), "bytes": os.path.getsize(p)}
    out = "manifests/afdb_checksums.json"
    with open(out, "w") as f:
        json.dump(sums, f, indent=2, sort_keys=True)
    missing = [r for r in recs if not os.path.exists(os.path.join(CACHE, f"{r}.dat"))]
    print(f"[download] wrote {out} ({len(sums)} files); missing .dat: {missing}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
