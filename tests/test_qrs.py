"""Causal QRS detector: finds beats on a synthetic ECG and reports bounded state."""
import numpy as np
from beatstate_af.preprocessing.qrs import CausalQRSDetector


def _synthetic_ecg(fs=250, n_beats=40, hr=1.0):
    """Impulse-like QRS train at ~hr Hz on a noisy baseline."""
    rng = np.random.default_rng(0)
    T = int(n_beats / hr * fs) + fs
    x = rng.normal(0, 0.02, T)
    peaks = []
    t = fs
    while t < T - fs:
        # a sharp biphasic QRS-like deflection
        for k, amp in ((-2, -0.3), (0, 1.0), (2, -0.4)):
            if 0 <= t + k < T:
                x[t + k] += amp
        peaks.append(t)
        t += int(fs / hr) + rng.integers(-5, 6)
    return x, np.array(peaks)


def test_detects_most_beats_causally():
    fs = 250
    x, true_peaks = _synthetic_ecg(fs=fs, n_beats=40)
    det = CausalQRSDetector(fs)
    rp = det.detect(x)
    # match detected to true within 60 ms
    tol = int(0.06 * fs)
    matched = sum(np.any(np.abs(true_peaks - r) <= tol) for r in rp)
    sens = matched / len(true_peaks)
    assert sens > 0.8, f"detector sensitivity too low: {sens:.2f} ({len(rp)} peaks)"


def test_state_bytes_bounded_and_positive():
    det = CausalQRSDetector(250)
    b = det.state_bytes(4)
    assert b > 0
    # O(1) persistent state: independent of signal length
    _ = det.detect(np.zeros(250 * 30))
    assert det.state_bytes(4) == b
