#!/usr/bin/env python3
"""Write the E01v2 environment/provenance report."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.e01v2 import split_manifest_hash, write_json
from beatstate_af.provenance import (
    environment_hash,
    environment_report,
    file_sha256,
    full_git_commit,
    git_tree_dirty,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/e01v2/e01v2_environment.json")
    args = ap.parse_args()
    report = environment_report()
    report.update({
        "environment_hash": environment_hash(),
        "full_commit_sha": full_git_commit(),
        "git_tree_dirty": git_tree_dirty(),
        "protocol_hash": file_sha256("configs/protocol_v2.yaml"),
        "split_manifest_hash": split_manifest_hash(),
        "dependency_lock_hash": file_sha256("requirements-lock.txt"),
    })
    write_json(args.output, report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
