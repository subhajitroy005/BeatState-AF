"""Trained GRU arms honor the matched-capacity + streaming contracts."""
import numpy as np
import torch
from beatstate_af.models.gru import build_gru_model
from beatstate_af.data.synthetic import ATRIAL_IDX, RHYTHM_IDX, N_FEATURES


def test_gru_total_dim_matched():
    D = 32
    for kind in ("gru_monolithic", "gru_factored", "gru_random"):
        m = build_gru_model(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, seed=0)
        assert m.total_state_dim == D
        assert m.atrial_state_dim + m.rhythm_state_dim == D
        assert m.state_bytes(4) == D * 4


def test_gru_sparsity_is_real():
    D = 32
    mono = build_gru_model("gru_monolithic", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    fac = build_gru_model("gru_factored", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    rnd = build_gru_model("gru_random", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    assert mono.recurrent_param_count() == 3 * D * D
    assert fac.recurrent_param_count() == 3 * (2 * (D // 2) ** 2)
    assert fac.recurrent_param_count() == rnd.recurrent_param_count()
    assert mono.recurrent_param_count() == 2 * fac.recurrent_param_count()


def test_streaming_step_matches_batched_forward():
    # the numpy O(1) step() must equal the trained nn.GRU forward recurrence
    D = 16
    rng = np.random.default_rng(1)
    for kind in ("gru_monolithic", "gru_factored", "gru_random"):
        m = build_gru_model(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, seed=3)
        m.eval()
        x = rng.standard_normal((7, N_FEATURES)).astype(np.float32)
        with torch.no_grad():
            if m.dense:
                out, _ = m.gru(torch.from_numpy(x).unsqueeze(0))
                ref = out.squeeze(0).numpy()
            else:
                oa, _ = m.gru_a(torch.from_numpy(x[:, m.atrial_idx]).unsqueeze(0))
                orr, _ = m.gru_r(torch.from_numpy(x[:, m.rhythm_idx]).unsqueeze(0))
                ref = np.concatenate([oa.squeeze(0).numpy(), orr.squeeze(0).numpy()], axis=1)
        m.reset_state()
        stream = np.array([m.step(x[t]) for t in range(x.shape[0])])
        assert np.allclose(stream, ref, atol=1e-4), f"{kind} stream != forward"
