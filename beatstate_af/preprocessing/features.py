"""Per-beat token extraction from ECG + R-peaks (causal), matching synthetic layout.

Emits one 8-dim token per beat, identical column semantics to
beatstate_af.data.synthetic:
  atrial  [0:4] = p_wave_score, fwave_energy, morph_amp, morph_zcr
  rhythm  [4:8] = rr, drr, rr_over_median, local_var

All features at beat i use only samples up to R-peak i and RR intervals of past
beats -> strictly causal. Atrial morphology is read from the pre-QRS (PR/TQ)
segment where organized P-waves (non-AF) or fibrillatory f-waves (AF) live;
rhythm features are RR-interval statistics over a trailing window. A per-beat
signal-quality index (SQI) is returned for the prespecified low-SQI discordant
subset (see docs/discordance_definition.md).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SignalNormalizer:
    """Frozen signal scaling constants.

    E01v2 uses one instance computed from training patients only. The legacy
    no-argument path below remains for old diagnostic code, but confirmatory v2
    feature extraction must pass explicit constants.
    """

    median: float
    scale: float
    dtype_bytes: int = 4

    @property
    def nbytes(self) -> int:
        return 2 * self.dtype_bytes

    def apply(self, sig: np.ndarray) -> np.ndarray:
        x = np.asarray(sig, dtype=np.float32)
        return (x - np.float32(self.median)) / np.float32(self.scale)


def robust_signal_normalizer(sig: np.ndarray, dtype_bytes: int = 4) -> SignalNormalizer:
    x = np.asarray(sig, dtype=np.float32)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    scale = max(1.4826 * mad, 1e-9)
    return SignalNormalizer(median=med, scale=scale, dtype_bytes=dtype_bytes)


def pooled_signal_normalizer(signals, dtype_bytes: int = 4) -> SignalNormalizer:
    """Median/MAD over training-patient samples only."""
    arrays = [np.asarray(s, dtype=np.float32).ravel() for s in signals]
    if not arrays:
        raise ValueError("cannot compute a signal normalizer from no signals")
    x = np.concatenate(arrays)
    return robust_signal_normalizer(x, dtype_bytes=dtype_bytes)


def _robust_normalize(sig: np.ndarray, normalizer: SignalNormalizer | None = None) -> np.ndarray:
    if normalizer is None:
        normalizer = robust_signal_normalizer(sig)
    return normalizer.apply(sig)


def _atrial_morphology(seg: np.ndarray):
    """Atrial-organization descriptors on the short (~200 ms) pre-QRS segment.

    Too short to resolve <5 Hz by FFT, so use time-domain organization instead:
      p_wave_score : fraction of energy retained after low-pass smoothing -- a
                     single organized P-wave (non-AF) survives smoothing; noisy
                     fibrillatory f-waves (AF) do not.  High -> non-AF.
      fwave_energy : high-frequency residual fraction (raw minus smoothed).
                     High -> AF fibrillatory activity.
      morph_amp    : atrial-activity amplitude (std) in normalized ECG units.
      morph_zcr    : zero-crossing rate of the detrended segment (oscillatority).
    """
    n = seg.size
    if n < 5:
        return 0.0, 0.0, 0.0, 0.0
    t = np.arange(n)
    seg = seg - np.polyval(np.polyfit(t, seg, 1), t)   # remove slow baseline
    raw_std = float(np.std(seg)) + 1e-9
    k = 5                                              # ~20 ms moving-average low-pass
    kern = np.ones(k) / k
    smooth = np.convolve(seg, kern, mode="same")
    p_wave_score = float(np.std(smooth)) / raw_std
    fwave_energy = float(np.std(seg - smooth)) / raw_std
    morph_amp = raw_std
    morph_zcr = float(np.count_nonzero(np.diff(np.sign(seg)))) / n
    return p_wave_score, fwave_energy, morph_amp, morph_zcr


def _sqi(win: np.ndarray, fs: float) -> float:
    """Relative-power SQI: physiological-band power / total power in a beat window."""
    win = win - win.mean()
    n = win.size
    if n < 8:
        return 0.0
    sp = np.abs(np.fft.rfft(win)) ** 2
    fr = np.fft.rfftfreq(n, 1.0 / fs)
    total = sp[1:].sum() + 1e-12
    phys = sp[1:][(fr[1:] >= 0.5) & (fr[1:] <= 40.0)].sum()
    return float(phys / total)


def feature_memory_report(fs: float, rr_window: int = 8, dtype_bytes: int = 4) -> dict[str, int]:
    """Persistent and one-beat scratch bytes for the causal feature path."""
    seg_a0 = int(0.25 * fs)
    seg_a1 = int(0.05 * fs)
    win_q = int(0.30 * fs)
    atrial_window = max(0, seg_a0 - seg_a1)
    persistent = (rr_window + 1) * dtype_bytes  # RR ring + previous RR scalar
    scratch = (8 + atrial_window + win_q) * dtype_bytes
    return {
        "feature_persistent_bytes": int(persistent),
        "feature_scratch_bytes": int(scratch),
    }


def extract_features(sig: np.ndarray, rpeaks: np.ndarray, fs: float,
                     rr_window: int = 8, normalizer: SignalNormalizer | None = None):
    """Return (feat (T,8) float32, beat_samples (T,) int, sqi (T,) float).

    T is the number of beats for which a full causal context exists (drops the
    first beat, which has no prior RR).
    """
    x = _robust_normalize(np.asarray(sig, dtype=np.float32), normalizer=normalizer)
    rp = np.asarray(rpeaks, dtype=int)
    rp = rp[(rp > int(0.30 * fs)) & (rp < x.size - 1)]
    if rp.size < 3:
        return (np.zeros((0, 8), np.float32), np.zeros(0, int), np.zeros(0))

    seg_a0 = int(0.25 * fs)   # atrial segment start: 250 ms before R
    seg_a1 = int(0.05 * fs)   # atrial segment end:    50 ms before R
    win_q = int(0.30 * fs)    # SQI window: 300 ms up to R

    feats, samples, sqis = [], [], []
    rr_buf = []
    prev_rr = None
    for k in range(1, rp.size):
        r = rp[k]
        rr = (r - rp[k - 1]) / fs
        rr_buf.append(rr)
        if len(rr_buf) > rr_window:
            rr_buf.pop(0)
        drr = 0.0 if prev_rr is None else (rr - prev_rr)
        med = float(np.median(rr_buf))
        rr_over_median = rr / (med + 1e-9)
        local_var = float(np.std(rr_buf)) if len(rr_buf) > 1 else 0.0
        prev_rr = rr

        seg = x[r - seg_a0: r - seg_a1]
        p_wave_score, fwave_energy, morph_amp, morph_zcr = _atrial_morphology(seg)

        feats.append([p_wave_score, fwave_energy, morph_amp, morph_zcr,
                      rr, drr, rr_over_median, local_var])
        samples.append(int(r))
        sqis.append(_sqi(x[r - win_q: r], fs))

    return (np.asarray(feats, np.float32), np.asarray(samples, int), np.asarray(sqis))
