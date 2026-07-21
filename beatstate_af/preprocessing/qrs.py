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
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector


class CausalQRSDetector:
    def __init__(self, fs: int, bandpass=(5.0, 15.0), refractory_s=0.20,
                 integ_window_s=0.150, search_back_s=0.10):
        self._stream = StreamingQRSDetector(
            fs=fs,
            bandpass=bandpass,
            refractory_s=refractory_s,
            integ_window_s=integ_window_s,
            search_back_s=search_back_s,
        )
        self.fs = self._stream.fs
        self.refractory = self._stream.refractory
        self.win = self._stream.win
        self.search_back = self._stream.search_back

    # ---- software-memory ledger --------------------------------------------
    def persistent_state_elems(self) -> int:
        """Number of float state elements held between samples (O(1) in signal length)."""
        return self._stream.persistent_state_bytes() // np.dtype(np.float32).itemsize

    def state_bytes(self, dtype_bytes: int = 4) -> int:
        return self._stream.state_bytes(dtype_bytes)

    def reset(self) -> None:
        self._stream.reset()

    def step(self, sample: float) -> list[int]:
        return self._stream.step(sample)

    # ---- detection ----------------------------------------------------------
    def detect(self, sig: np.ndarray) -> np.ndarray:
        """Return R-peak sample indices for a 1-D ECG signal (causal pipeline).

        Convenience wrapper around reset()/step(sample). Primary E01v2 execution
        uses the step path directly; this method remains for diagnostics/tests.
        """
        return self._stream.detect(sig)
