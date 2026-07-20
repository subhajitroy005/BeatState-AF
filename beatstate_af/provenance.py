"""Result-row provenance: every row carries enough to reproduce it."""
from __future__ import annotations
import hashlib, platform, subprocess, sys

RESULT_ROW_COLUMNS = [
    "run_id", "experiment_id", "config_hash", "commit_sha", "environment_hash",
    "dataset_id", "model_id", "total_state_dim", "atrial_state_dim", "rhythm_state_dim",
    "dtype_bytes", "seed", "patient_id", "recurrent_state_bytes", "recurrent_param_count",
    "readout_param_count", "macro_f1", "sensitivity", "specificity", "ppv",
    "acc_concordant", "acc_discordant", "status",
]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def environment_hash():
    s = f"{sys.version}|{platform.platform()}"
    try:
        import numpy; s += f"|numpy{numpy.__version__}"
    except Exception:
        pass
    return hashlib.sha256(s.encode()).hexdigest()[:16]
