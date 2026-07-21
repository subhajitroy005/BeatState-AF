"""Sample-by-sample causal QRS detector used by E01v2.

The detector exposes the deployable path as reset()/step(sample). The batch
detect() helper is only a thin loop over step(), so parity is exact by
construction and no full-record preprocessing arrays are part of the primary
path.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import butter


@dataclass(frozen=True)
class QRSMetrics:
    sensitivity: float
    ppv: float
    mean_abs_timing_error_samples: float
    matched_peaks: int
    missed_peaks: int
    false_peaks: int
    reference_peaks: int
    detected_peaks: int


class StreamingQRSDetector:
    def __init__(
        self,
        fs: int,
        bandpass=(5.0, 15.0),
        refractory_s: float = 0.20,
        integ_window_s: float = 0.150,
        search_back_s: float = 0.10,
        learn_s: float = 2.0,
    ):
        self.fs = int(fs)
        self.refractory = max(1, int(refractory_s * fs))
        self.win = max(1, int(integ_window_s * fs))
        self.search_back = max(1, int(search_back_s * fs))
        self.refine_back = max(self.search_back, self.win)
        self.learn_samples = max(1, int(learn_s * fs))
        lo, hi = bandpass
        self.b, self.a = butter(1, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        self.b = self.b.astype(np.float32)
        self.a = self.a.astype(np.float32)
        self._order = max(len(self.a), len(self.b)) - 1
        self.reset()

    def reset(self) -> None:
        self.x_hist = np.zeros(len(self.b) - 1, dtype=np.float32)
        self.y_hist = np.zeros(len(self.a) - 1, dtype=np.float32)
        self.mwi_ring = np.zeros(self.win, dtype=np.float32)
        self.bp_values = np.zeros(self.refine_back + 1, dtype=np.float32)
        self.bp_indices = np.full(self.refine_back + 1, -1, dtype=np.int64)
        self.rr_history = np.zeros(8, dtype=np.float32)
        self.rr_count = 0
        self.rr_pos = 0
        self.ring_pos = 0
        self.hist_pos = 0
        self.ring_sum = np.float32(0.0)
        self.prev_bp = np.float32(0.0)
        self.prev2_mwi = np.float32(0.0)
        self.prev1_mwi = np.float32(0.0)
        self.prev1_index = -1
        self.sample_index = -1
        self.spki = np.float32(0.0)
        self.npki = np.float32(0.0)
        self.threshold = np.float32(0.0)
        self.last_candidate = -self.refractory
        self.last_peak = -self.refractory
        self.learn_values: list[float] = []
        self.initialized = False

    def persistent_state_bytes(self) -> int:
        scalar_count = 12
        return int(
            self.x_hist.nbytes
            + self.y_hist.nbytes
            + self.mwi_ring.nbytes
            + self.bp_values.nbytes
            + self.bp_indices.nbytes
            + self.rr_history.nbytes
            + scalar_count * np.dtype(np.float32).itemsize
        )

    def scratch_bytes(self) -> int:
        return int(8 * np.dtype(np.float32).itemsize)

    def filter_constant_bytes(self) -> int:
        return int(self.b.nbytes + self.a.nbytes)

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.persistent_state_bytes()

    def _record_bp(self, bp: float) -> None:
        self.bp_values[self.hist_pos] = np.float32(abs(bp))
        self.bp_indices[self.hist_pos] = self.sample_index
        self.hist_pos = (self.hist_pos + 1) % self.bp_values.size

    def _refine(self, candidate_index: int) -> int:
        lo = candidate_index - self.refine_back
        mask = (self.bp_indices >= lo) & (self.bp_indices <= candidate_index)
        if not np.any(mask):
            return candidate_index
        vals = self.bp_values[mask]
        idxs = self.bp_indices[mask]
        return int(idxs[int(np.argmax(vals))])

    def _update_rr(self, peak: int) -> None:
        if self.last_peak >= 0:
            self.rr_history[self.rr_pos] = np.float32(peak - self.last_peak)
            self.rr_pos = (self.rr_pos + 1) % self.rr_history.size
            self.rr_count = min(self.rr_count + 1, self.rr_history.size)

    def _mean_rr(self) -> float:
        if self.rr_count == 0:
            return float("nan")
        return float(np.mean(self.rr_history[: self.rr_count]))

    def _accept_candidate(self, candidate_index: int, value: float) -> bool:
        if candidate_index - self.last_candidate < self.refractory:
            return False
        self.threshold = self.npki + np.float32(0.25) * (self.spki - self.npki)
        accept = value > self.threshold
        rr = self._mean_rr()
        if (not accept) and np.isfinite(rr):
            overdue = candidate_index - self.last_candidate > int(1.5 * rr)
            accept = bool(overdue and value > np.float32(0.5) * self.threshold)
        if accept:
            self.spki = np.float32(0.125) * value + np.float32(0.875) * self.spki
            self.last_candidate = candidate_index
        else:
            self.npki = np.float32(0.125) * value + np.float32(0.875) * self.npki
        return bool(accept)

    def step(self, sample: float) -> list[int]:
        self.sample_index += 1
        x = np.float32(sample)
        bp = np.float32(self.b[0] * x)
        for i in range(1, len(self.b)):
            bp += np.float32(self.b[i] * self.x_hist[i - 1])
        for i in range(1, len(self.a)):
            bp -= np.float32(self.a[i] * self.y_hist[i - 1])
        if self.x_hist.size:
            self.x_hist[1:] = self.x_hist[:-1]
            self.x_hist[0] = x
        if self.y_hist.size:
            self.y_hist[1:] = self.y_hist[:-1]
            self.y_hist[0] = bp
        deriv = np.float32(bp - self.prev_bp)
        self.prev_bp = bp
        squared = np.float32(deriv * deriv)

        self.ring_sum += squared - self.mwi_ring[self.ring_pos]
        self.mwi_ring[self.ring_pos] = squared
        self.ring_pos = (self.ring_pos + 1) % self.mwi_ring.size
        mwi = np.float32(self.ring_sum / self.mwi_ring.size)
        self._record_bp(float(bp))

        detections: list[int] = []
        if not self.initialized:
            self.learn_values.append(float(mwi))
            if len(self.learn_values) >= self.learn_samples:
                learn = np.asarray(self.learn_values, dtype=np.float32)
                self.spki = np.float32(0.25) * np.max(learn)
                self.npki = np.float32(0.5) * np.mean(learn)
                self.threshold = self.npki + np.float32(0.25) * (self.spki - self.npki)
                self.initialized = True
            self.prev2_mwi = self.prev1_mwi
            self.prev1_mwi = mwi
            self.prev1_index = self.sample_index
            return detections

        candidate_is_peak = self.prev1_mwi > self.prev2_mwi and self.prev1_mwi >= mwi
        if candidate_is_peak and self._accept_candidate(self.prev1_index, self.prev1_mwi):
            peak = self._refine(self.prev1_index)
            if peak - self.last_peak >= self.refractory:
                detections.append(peak)
                self._update_rr(peak)
                self.last_peak = peak

        self.prev2_mwi = self.prev1_mwi
        self.prev1_mwi = mwi
        self.prev1_index = self.sample_index
        return detections

    def detect(self, sig: np.ndarray) -> np.ndarray:
        self.reset()
        peaks: list[int] = []
        for sample in np.asarray(sig, dtype=np.float32):
            peaks.extend(self.step(float(sample)))
        return np.asarray(sorted(set(peaks)), dtype=int)


def match_qrs_detections(reference, detected, tolerance_samples: int) -> QRSMetrics:
    ref = np.asarray(reference, dtype=int)
    det = np.asarray(detected, dtype=int)
    used = np.zeros(det.size, dtype=bool)
    errors: list[int] = []
    matched = 0
    for r in ref:
        candidates = np.where((~used) & (np.abs(det - r) <= tolerance_samples))[0]
        if candidates.size == 0:
            continue
        best_local = int(candidates[np.argmin(np.abs(det[candidates] - r))])
        used[best_local] = True
        matched += 1
        errors.append(int(det[best_local] - r))
    missed = int(ref.size - matched)
    false = int(det.size - matched)
    sens = matched / ref.size if ref.size else float("nan")
    ppv = matched / det.size if det.size else float("nan")
    mae = float(np.mean(np.abs(errors))) if errors else float("nan")
    return QRSMetrics(
        sensitivity=float(sens),
        ppv=float(ppv),
        mean_abs_timing_error_samples=mae,
        matched_peaks=int(matched),
        missed_peaks=missed,
        false_peaks=false,
        reference_peaks=int(ref.size),
        detected_peaks=int(det.size),
    )
