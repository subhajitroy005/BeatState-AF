import numpy as np

from beatstate_af.preprocessing.features import SignalNormalizer, extract_features


def test_future_samples_do_not_change_past_features():
    fs = 250
    rng = np.random.default_rng(0)
    sig = rng.normal(0, 0.05, 2500).astype(np.float32)
    rpeaks = np.arange(300, 2200, 250)
    for r in rpeaks:
        sig[r] += 1.0
    normalizer = SignalNormalizer(median=0.0, scale=1.0)
    feat_a, samples, _ = extract_features(sig, rpeaks, fs, normalizer=normalizer)
    t = min(3, len(samples) - 1)
    perturbed = sig.copy()
    perturbed[samples[t] + 1:] += rng.normal(10.0, 2.0, perturbed[samples[t] + 1:].shape).astype(np.float32)
    feat_b, _, _ = extract_features(perturbed, rpeaks, fs, normalizer=normalizer)
    assert np.allclose(feat_a[: t + 1], feat_b[: t + 1], atol=1e-6)
