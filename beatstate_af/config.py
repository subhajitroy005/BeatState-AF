"""Frozen-YAML config loading + content hash (every experiment value lives here)."""
from __future__ import annotations
import hashlib, json, yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    blob = json.dumps(cfg, sort_keys=True).encode()
    cfg["_config_hash"] = hashlib.sha256(blob).hexdigest()[:16]
    return cfg
