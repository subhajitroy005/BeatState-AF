"""Complete software-memory ledger for E01v2 software claims."""
from __future__ import annotations
from beatstate_af.interfaces import MemoryReport


def _torch_param_bytes(model, include_head: bool):
    weight = 0
    bias = 0
    count = 0
    for name, p in model.named_parameters():
        is_head = name.startswith("head.")
        if is_head != include_head:
            continue
        b = int(p.numel() * p.element_size())
        count += int(p.numel())
        if name.endswith("bias") or ".bias_" in name or "bias" in name:
            bias += b
        else:
            weight += b
    return weight, bias, count


def _reservoir_param_bytes(model, readout_param_count: int, dtype_bytes: int):
    recurrent = int(model.recurrent_param_count())
    return recurrent * dtype_bytes, 0, int(recurrent + readout_param_count)


def memory_report(model, readout_param_count: int | None = None, dtype_bytes: int = 4,
                  normalizer_bytes: int = 0, filter_constant_bytes: int = 0,
                  qrs_persistent_bytes: int = 0, qrs_scratch_bytes: int = 0,
                  feature_persistent_bytes: int = 0, feature_scratch_bytes: int = 0) -> MemoryReport:
    if hasattr(model, "named_parameters"):
        model_weight_bytes, model_bias_bytes, recurrent_plus_input_params = _torch_param_bytes(
            model, include_head=False
        )
        readout_w, readout_b, readout_params = _torch_param_bytes(model, include_head=True)
        readout_bytes = readout_w + readout_b
        total_model_param_count = recurrent_plus_input_params + readout_params
        readout_param_count = readout_params if readout_param_count is None else readout_param_count
    else:
        readout_param_count = int(readout_param_count or 0)
        model_weight_bytes, model_bias_bytes, total_model_param_count = _reservoir_param_bytes(
            model, readout_param_count, dtype_bytes
        )
        readout_bytes = readout_param_count * dtype_bytes

    recurrent_state_bytes = int(model.state_bytes(dtype_bytes))
    recurrent_step_activation_bytes = int(3 * model.total_state_dim * dtype_bytes)
    output_bytes = int(2 * dtype_bytes)
    total_static_bytes = int(
        model_weight_bytes
        + model_bias_bytes
        + readout_bytes
        + normalizer_bytes
        + filter_constant_bytes
    )
    total_persistent_bytes = int(
        recurrent_state_bytes
        + qrs_persistent_bytes
        + feature_persistent_bytes
    )
    peak_working = int(
        qrs_scratch_bytes
        + feature_scratch_bytes
        + recurrent_step_activation_bytes
        + output_bytes
    )
    total_peak_inference_bytes = int(total_static_bytes + total_persistent_bytes + peak_working)
    estimated_ops_per_beat = int(2 * total_model_param_count + 8 * model.total_state_dim)
    return MemoryReport(
        model_id=model.model_id,
        total_state_dim=model.total_state_dim,
        atrial_state_dim=model.atrial_state_dim,
        rhythm_state_dim=model.rhythm_state_dim,
        dtype_bytes=dtype_bytes,
        model_weight_bytes=int(model_weight_bytes),
        model_bias_bytes=int(model_bias_bytes),
        readout_bytes=int(readout_bytes),
        normalizer_bytes=int(normalizer_bytes),
        filter_constant_bytes=int(filter_constant_bytes),
        qrs_persistent_bytes=int(qrs_persistent_bytes),
        qrs_scratch_bytes=int(qrs_scratch_bytes),
        feature_persistent_bytes=int(feature_persistent_bytes),
        feature_scratch_bytes=int(feature_scratch_bytes),
        recurrent_state_bytes=recurrent_state_bytes,
        recurrent_step_activation_bytes=recurrent_step_activation_bytes,
        output_bytes=output_bytes,
        total_static_bytes=total_static_bytes,
        total_persistent_bytes=total_persistent_bytes,
        total_peak_inference_bytes=total_peak_inference_bytes,
        recurrent_param_count=model.recurrent_param_count(),
        total_model_param_count=int(total_model_param_count),
        estimated_ops_per_beat=estimated_ops_per_beat,
        readout_param_count=int(readout_param_count or 0),
    )
