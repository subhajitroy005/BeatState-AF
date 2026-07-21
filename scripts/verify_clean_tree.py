#!/usr/bin/env python3
"""Fail unless the git working tree is clean."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL).decode()
    except Exception as exc:
        raise SystemExit(f"cannot inspect git status: {exc}")
    if out.strip():
        print(out, end="")
        raise SystemExit("working tree is not clean")
    print("working tree clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
