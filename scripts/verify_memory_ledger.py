#!/usr/bin/env python3
"""Reconcile memory ledger byte totals from model/detector shapes."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.data.synthetic import ATRIAL_IDX, RHYTHM_IDX, N_FEATURES
from beatstate_af.memory.ledger import memory_report
from beatstate_af.models.gru import build_gru_model
from beatstate_af.preprocessing.features import feature_memory_report
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector


def main():
    detector = StreamingQRSDetector(250)
    fmem = feature_memory_report(250, dtype_bytes=4)
    model = build_gru_model("gru_factored", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 32, seed=0, routing_seed=101)
    mem = memory_report(
        model,
        dtype_bytes=4,
        normalizer_bytes=72,
        filter_constant_bytes=detector.filter_constant_bytes(),
        qrs_persistent_bytes=detector.persistent_state_bytes(),
        qrs_scratch_bytes=detector.scratch_bytes(),
        feature_persistent_bytes=fmem["feature_persistent_bytes"],
        feature_scratch_bytes=fmem["feature_scratch_bytes"],
    )
    static = mem["model_weight_bytes"] + mem["model_bias_bytes"] + mem["readout_bytes"] + mem["normalizer_bytes"] + mem["filter_constant_bytes"]
    persistent = mem["recurrent_state_bytes"] + mem["qrs_persistent_bytes"] + mem["feature_persistent_bytes"]
    peak = static + persistent + mem["qrs_scratch_bytes"] + mem["feature_scratch_bytes"] + mem["recurrent_step_activation_bytes"] + mem["output_bytes"]
    assert mem["total_static_bytes"] == static
    assert mem["total_persistent_bytes"] == persistent
    assert mem["total_peak_inference_bytes"] == peak
    assert mem["recurrent_state_bytes"] == model.h.size * model.h.itemsize
    print("memory ledger reconciled")


if __name__ == "__main__":
    main()
