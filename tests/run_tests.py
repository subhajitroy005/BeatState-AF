#!/usr/bin/env python3
"""Zero-dependency test runner (used when pytest is unavailable)."""
import importlib, sys, traceback, os
sys.path.insert(0, os.getcwd())
mods = ["tests.test_matched_capacity", "tests.test_patient_disjoint", "tests.test_smoke_kill_gate"]
fails = 0
for name in mods:
    m = importlib.import_module(name)
    for fn in [f for f in dir(m) if f.startswith("test_")]:
        try:
            getattr(m, fn)(); print(f"PASS {name}.{fn}")
        except Exception:
            fails += 1; print(f"FAIL {name}.{fn}"); traceback.print_exc()
print(f"\n{'ALL PASSED' if not fails else str(fails)+' FAILED'}")
sys.exit(1 if fails else 0)
