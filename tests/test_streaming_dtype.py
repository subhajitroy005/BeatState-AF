import numpy as np

from beatstate_af.data.synthetic import ATRIAL_IDX, N_FEATURES, RHYTHM_IDX
from beatstate_af.models.gru import build_gru_model


def test_streaming_state_dtype_and_byte_count_are_float32():
    model = build_gru_model("gru_factored", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=0)
    model.reset_state()
    assert model.h.dtype == np.float32
    assert model.state_bytes(4) == model.h.size * model.h.itemsize
    state = model.step(np.zeros(N_FEATURES, dtype=np.float32))
    assert state.dtype == np.float32
