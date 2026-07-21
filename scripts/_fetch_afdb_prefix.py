#!/usr/bin/env python3
"""Fetch capped AFDB record prefixes via curl range requests (network-robust).

wfdb's .dat streaming stalls behind PhysioNet rate-limiting; curl range GETs are
reliable. We download the first PREFIX_SAMPLES timesteps of each record's format-212
.dat (3 bytes/timestep, 2 channels), the full small .atr/.qrs annotations, and a
header patched to the prefix length so wfdb reads it locally. Idempotent.
"""
from __future__ import annotations
import concurrent.futures as cf
import json, os, subprocess, sys

BASE = "https://physionet.org/files/afdb/1.0.0"
CACHE = "data/afdb"
PREFIX_SAMPLES = 2_700_000          # 3 h at 250 Hz
PREFIX_BYTES = 3 * PREFIX_SAMPLES   # format 212: 2 channels -> 3 bytes / timestep


def curl(url, out, byte_range=None, tries=4):
    for k in range(tries):
        cmd = ["curl", "-sS", "--fail", "--max-time", "300"]
        if byte_range:
            cmd += ["-r", byte_range]
        cmd += ["-o", out, url]
        if subprocess.run(cmd).returncode == 0 and os.path.getsize(out) > 0:
            return True
    return False


def fetch_record(rec):
    hea = os.path.join(CACHE, f"{rec}.hea")
    dat = os.path.join(CACHE, f"{rec}.dat")
    ok = True
    if not (os.path.exists(dat) and os.path.getsize(dat) >= PREFIX_BYTES - 3):
        ok &= curl(f"{BASE}/{rec}.dat", dat, byte_range=f"0-{PREFIX_BYTES - 1}")
    ok &= curl(f"{BASE}/{rec}.atr", os.path.join(CACHE, f"{rec}.atr"))
    ok &= curl(f"{BASE}/{rec}.qrs", os.path.join(CACHE, f"{rec}.qrs"))
    ok &= curl(f"{BASE}/{rec}.hea", hea)
    # patch header sig_len -> prefix length so wfdb reads the truncated .dat
    if ok and os.path.exists(hea):
        lines = open(hea).read().splitlines()
        p = lines[0].split()
        p[3] = str(PREFIX_SAMPLES)
        if len(p) > 4:
            p = p[:4]                # drop base-time/date (now inconsistent with prefix)
        lines[0] = " ".join(p)
        open(hea, "w").write("\n".join(lines) + "\n")
    status = "OK" if (ok and os.path.getsize(dat) >= PREFIX_BYTES - 3) else "FAIL"
    print(f"  [{status}] {rec}  dat={os.path.getsize(dat) if os.path.exists(dat) else 0}B", flush=True)
    return rec, status == "OK"


def main():
    os.makedirs(CACHE, exist_ok=True)
    recs = json.load(open("manifests/afdb_records.json"))["records"]
    print(f"[fetch] {len(recs)} records, {PREFIX_SAMPLES} samples each ({PREFIX_BYTES/1e6:.1f} MB/rec)", flush=True)
    results = {}
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        for rec, ok in ex.map(fetch_record, recs):
            results[rec] = ok
    bad = [r for r, ok in results.items() if not ok]
    print(f"[fetch] done. failed: {bad}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
