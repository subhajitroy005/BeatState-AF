"""Complete software-memory ledger (the paper's second contribution).

Reports recurrent-state bytes AND recurrent-parameter counts so an efficiency
claim can be attributed to physiological assignment only after the
random-assignment control is beaten (matched dimension != matched capacity).
"""
from __future__ import annotations
from beatstate_af.interfaces import MemoryReport


def memory_report(model, readout_param_count: int, dtype_bytes: int = 4) -> MemoryReport:
    return MemoryReport(
        model_id=model.model_id,
        total_state_dim=model.total_state_dim,
        atrial_state_dim=model.atrial_state_dim,
        rhythm_state_dim=model.rhythm_state_dim,
        dtype_bytes=dtype_bytes,
        recurrent_state_bytes=model.state_bytes(dtype_bytes),
        recurrent_param_count=model.recurrent_param_count(),
        readout_param_count=readout_param_count,
    )
