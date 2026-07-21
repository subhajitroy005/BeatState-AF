import numpy as np

from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector


def _synthetic_ecg(fs=250, n=5000):
    rng = np.random.default_rng(1)
    sig = rng.normal(0, 0.02, n).astype(np.float32)
    for r in range(fs, n - fs, fs):
        sig[r - 1:r + 2] += np.array([-0.2, 1.0, -0.2], dtype=np.float32)
    return sig


def test_step_outputs_match_detect_helper():
    sig = _synthetic_ecg()
    det = StreamingQRSDetector(250)
    batch = det.detect(sig)
    det.reset()
    stepped = []
    for sample in sig:
        stepped.extend(det.step(float(sample)))
    assert np.array_equal(batch, np.asarray(sorted(set(stepped)), dtype=int))


def test_streaming_qrs_state_is_bounded():
    det = StreamingQRSDetector(250)
    before = det.persistent_state_bytes()
    _ = det.detect(np.zeros(2500, dtype=np.float32))
    middle = det.persistent_state_bytes()
    _ = det.detect(np.zeros(25000, dtype=np.float32))
    after = det.persistent_state_bytes()
    assert before == middle == after
    assert before > 0
