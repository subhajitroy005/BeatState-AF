"""Causal classical QRS detector (deployable R-peaks) + software-memory accounting.

A Pan-Tompkins-style pipeline that is strictly causal (every output sample uses
only present and past input): band-pass -> derivative -> squaring -> causal
moving-window integration -> adaptive thresholding with a refractory period.

This is the DEPLOYABLE front end. Its persistent state (filter states + the
integration ring buffer + the adaptive-threshold scalars) is counted in the
software-memory ledger, because the paper's efficiency claim must account for the
whole streaming pipeline, not only the recurrent cell. `state_bytes()` returns
that persistent footprint (software bytes, not measured SRAM).

Oracle R-peaks (the machine `.qrs` beats shipped with AFDB) bypass this detector
and are used only as an upper-bound diagnostic (Delta_QRS).
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, lfilter


class CausalQRSDetector:
    def __init__(self, fs: int, bandpass=(5.0, 15.0), refractory_s=0.20,
                 integ_window_s=0.150, search_back_s=0.10):
        self.fs = int(fs)
        self.refractory = int(refractory_s * fs)
        self.win = max(1, int(integ_window_s * fs))
        self.search_back = int(search_back_s * fs)
        lo, hi = bandpass
        # 1st-order Butterworth band-pass (causal IIR; 2 states per section).
        self.b, self.a = butter(1, [lo / (fs / 2), hi / (fs / 2)], btype="band")
        self._order = max(len(self.a), len(self.b)) - 1

    # ---- software-memory ledger --------------------------------------------
    def persistent_state_elems(self) -> int:
        """Number of float state elements held between samples (O(1) in signal length)."""
        filt_states = 2 * self._order            # band-pass + derivative filter memories
        ring_buffer = self.win                   # moving-window integrator buffer
        scalars = 4                              # SPKI, NPKI, threshold, last-peak index
        return filt_states + ring_buffer + scalars

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self.persistent_state_elems() * dtype_bytes

    # ---- detection ----------------------------------------------------------
    def detect(self, sig: np.ndarray) -> np.ndarray:
        """Return R-peak sample indices for a 1-D ECG signal (causal pipeline).

        Pan-Tompkins adaptive thresholding over candidate local maxima of the
        integrated signal: SPKI/NPKI update on signal/noise peaks (not every
        sample), with a refractory period and RR-based search-back for missed
        beats. All decisions use only present and past samples.
        """
        x = np.asarray(sig, dtype=float)
        if x.size < self.fs:
            return np.array([], dtype=int)
        bp = lfilter(self.b, self.a, x)                      # causal band-pass
        deriv = np.diff(bp, prepend=bp[0])                   # causal derivative
        sq = deriv * deriv                                   # squaring
        csum = np.cumsum(sq)                                 # causal moving-window integration
        mwi = csum.copy()
        mwi[self.win:] = csum[self.win:] - csum[:-self.win]
        mwi = mwi / self.win

        # candidate peaks = local maxima of the integrated signal
        i0 = np.arange(1, mwi.size - 1)
        cand = i0[(mwi[i0] > mwi[i0 - 1]) & (mwi[i0] >= mwi[i0 + 1])]
        if cand.size == 0:
            return np.array([], dtype=int)

        learn = mwi[: 2 * self.fs]
        spki = 0.25 * float(np.max(learn)) if learn.size else float(np.max(mwi))
        npki = 0.5 * float(np.mean(learn)) if learn.size else float(np.mean(mwi))

        def refine(c):
            s0 = max(0, c - self.search_back)
            return s0 + int(np.argmax(np.abs(bp[s0: c + 1])))

        peaks = []
        rr = []
        last = -self.refractory
        for c in cand:
            if c - last < self.refractory:
                continue
            P = mwi[c]
            thr = npki + 0.25 * (spki - npki)
            accept = P > thr
            if not accept and peaks and rr:
                # search-back: recover a missed beat if overdue and half-threshold
                if (c - last) > int(1.5 * np.mean(rr[-8:])) and P > 0.5 * thr:
                    accept = True
            if accept:
                r = refine(c)
                if peaks:
                    rr.append(r - peaks[-1])
                peaks.append(r)
                last = c
                spki = 0.125 * P + 0.875 * spki
            else:
                npki = 0.125 * P + 0.875 * npki
        return np.array(sorted(set(peaks)), dtype=int)
