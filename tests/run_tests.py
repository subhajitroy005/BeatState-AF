#!/usr/bin/env python3
"""Zero-dependency test runner (used when pytest is unavailable). Skip-aware."""
import importlib, sys, traceback, os
sys.path.insert(0, os.getcwd())
try:
    from _pytest.outcomes import Skipped
except Exception:                       # pragma: no cover
    class Skipped(Exception):
        pass
mods = ["tests.test_matched_capacity", "tests.test_patient_disjoint",
        "tests.test_smoke_kill_gate", "tests.test_qrs", "tests.test_gru_interface",
        "tests.test_wfdb_io"]
fails = skipped = 0
for name in mods:
    m = importlib.import_module(name)
    for fn in [f for f in dir(m) if f.startswith("test_")]:
        try:
            getattr(m, fn)(); print(f"PASS {name}.{fn}")
        except Skipped as e:
            skipped += 1; print(f"SKIP {name}.{fn} ({e})")
        except Exception:
            fails += 1; print(f"FAIL {name}.{fn}"); traceback.print_exc()
print(f"\n{'ALL PASSED' if not fails else str(fails)+' FAILED'} ({skipped} skipped)")
sys.exit(1 if fails else 0)
