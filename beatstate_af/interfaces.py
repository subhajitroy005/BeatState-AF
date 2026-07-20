"""Core contracts every model and experiment must honor (frozen).

The walking-skeleton reservoir models and the future PyTorch models (built by
Claude Code) must both satisfy these so the kill-gate harness, the memory
ledger, and the fairness controls are identical across architectures.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, TypedDict
import numpy as np


@dataclass
class AFOutput:
    p_af: float
    state_bytes: int


class StreamingAFModel(Protocol):
    total_state_dim: int
    atrial_state_dim: int
    rhythm_state_dim: int
    def reset_state(self, batch_size: int = 1) -> None: ...
    def step(self, feat: np.ndarray) -> np.ndarray: ...
    def recurrent_param_count(self) -> int: ...
    def state_bytes(self, dtype_bytes: int = 4) -> int: ...


class MemoryReport(TypedDict):
    model_id: str
    total_state_dim: int
    atrial_state_dim: int
    rhythm_state_dim: int
    dtype_bytes: int
    recurrent_state_bytes: int
    recurrent_param_count: int
    readout_param_count: int
