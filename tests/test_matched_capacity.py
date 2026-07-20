from beatstate_af.models.reservoir import build_model
from beatstate_af.data.synthetic import ATRIAL_IDX, RHYTHM_IDX, N_FEATURES

def test_total_dim_matched():
    D = 32
    for kind in ("monolithic", "factored", "random"):
        m = build_model(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, seed=0)
        assert m.total_state_dim == D
        assert m.atrial_state_dim + m.rhythm_state_dim == D
        assert m.state_bytes(4) == D * 4

def test_sparsity_is_real():
    # monolithic (dense) must carry ~2x the recurrent params of the block-diagonal models
    D = 32
    mono = build_model("monolithic", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    fac = build_model("factored", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    rnd = build_model("random", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, D, 0)
    assert mono.recurrent_param_count() == D * D
    assert fac.recurrent_param_count() == 2 * (D // 2) ** 2
    assert fac.recurrent_param_count() == rnd.recurrent_param_count()
    assert mono.recurrent_param_count() > fac.recurrent_param_count()
