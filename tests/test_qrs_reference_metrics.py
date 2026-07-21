from pathlib import Path

import numpy as np
import pytest

from beatstate_af.e01v2 import load_raw_signal, qrs_reference_metrics
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector


def test_qrs_reference_metrics_compute_against_afdb_qrs():
    record = Path("data/afdb/04015.hea")
    if not record.exists():
        pytest.skip("AFDB cache not present")
    sig, sampto = load_raw_signal("data/afdb/04015", fs=250, channel=0, max_beats=1000)
    det = StreamingQRSDetector(250)
    peaks = det.detect(sig)
    metrics = qrs_reference_metrics("data/afdb/04015", peaks, fs=250, sampto=sampto)
    assert 0.0 <= metrics["qrs_sensitivity"] <= 1.0
    assert np.isnan(metrics["qrs_ppv"]) or 0.0 <= metrics["qrs_ppv"] <= 1.0
    assert metrics["qrs_matched_peaks"] + metrics["qrs_missed_peaks"] == metrics["qrs_reference_peaks"]
    assert metrics["qrs_matched_peaks"] + metrics["qrs_false_peaks"] == metrics["qrs_detected_peaks"]
    assert det.persistent_state_bytes() == det.state_bytes(4)
