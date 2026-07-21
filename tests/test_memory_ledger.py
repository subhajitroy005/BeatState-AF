from beatstate_af.data.synthetic import ATRIAL_IDX, N_FEATURES, RHYTHM_IDX
from beatstate_af.memory.ledger import memory_report
from beatstate_af.models.gru import build_gru_model
from beatstate_af.preprocessing.features import feature_memory_report
from beatstate_af.preprocessing.qrs_stream import StreamingQRSDetector


def test_memory_ledger_totals_reconcile():
    det = StreamingQRSDetector(250)
    fmem = feature_memory_report(250)
    model = build_gru_model("gru_monolithic", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 32, seed=0)
    mem = memory_report(
        model,
        dtype_bytes=4,
        normalizer_bytes=72,
        filter_constant_bytes=det.filter_constant_bytes(),
        qrs_persistent_bytes=det.persistent_state_bytes(),
        qrs_scratch_bytes=det.scratch_bytes(),
        feature_persistent_bytes=fmem["feature_persistent_bytes"],
        feature_scratch_bytes=fmem["feature_scratch_bytes"],
    )
    assert mem["recurrent_state_bytes"] == model.h.size * model.h.itemsize
    assert mem["total_static_bytes"] == mem["model_weight_bytes"] + mem["model_bias_bytes"] + mem["readout_bytes"] + mem["normalizer_bytes"] + mem["filter_constant_bytes"]
    assert mem["total_persistent_bytes"] == mem["recurrent_state_bytes"] + mem["qrs_persistent_bytes"] + mem["feature_persistent_bytes"]
    assert mem["total_peak_inference_bytes"] == (
        mem["total_static_bytes"]
        + mem["total_persistent_bytes"]
        + mem["qrs_scratch_bytes"]
        + mem["feature_scratch_bytes"]
        + mem["recurrent_step_activation_bytes"]
        + mem["output_bytes"]
    )
