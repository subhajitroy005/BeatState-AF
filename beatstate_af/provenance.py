"""Result-row provenance: every row carries enough to reproduce it."""
from __future__ import annotations
import hashlib, json, os, platform, subprocess, sys

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


def full_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"


def git_tree_dirty(ignore_paths: tuple[str, ...] = ()) -> bool:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode().splitlines()
    except Exception:
        return True
    if not ignore_paths:
        return bool(out)
    ignored = tuple(p.rstrip("/") + "/" for p in ignore_paths)
    for line in out:
        path = line[3:] if len(line) > 3 else line
        if not path.startswith(ignored):
            return True
    return False


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_paths(paths) -> str:
    h = hashlib.sha256()
    for path in sorted(paths):
        if not os.path.exists(path):
            continue
        h.update(path.encode())
        h.update(file_sha256(path).encode())
    return h.hexdigest()


def environment_hash():
    return hashlib.sha256(json.dumps(environment_report(), sort_keys=True).encode()).hexdigest()[:16]


def environment_report() -> dict:
    report = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "blas_backend": "unknown",
        "cuda_version_if_present": "",
        "deterministic_algorithms_enabled": False,
    }
    try:
        import numpy
        report["numpy_version"] = numpy.__version__
        try:
            report["blas_backend"] = str(numpy.__config__.show(mode="dicts"))
        except Exception:
            report["blas_backend"] = "available"
    except Exception:
        report["numpy_version"] = ""
    try:
        import scipy
        report["scipy_version"] = scipy.__version__
    except Exception:
        report["scipy_version"] = ""
    try:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_version_if_present"] = torch.version.cuda or ""
        report["deterministic_algorithms_enabled"] = bool(torch.are_deterministic_algorithms_enabled())
    except Exception:
        report["torch_version"] = ""
    try:
        import wfdb
        report["wfdb_version"] = wfdb.__version__
    except Exception:
        report["wfdb_version"] = ""
    return report
