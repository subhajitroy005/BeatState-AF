from beatstate_af.statistics.stats import noninferiority, superiority


def test_noninferiority_accepts_better_compressed_model():
    stat = noninferiority([0.01, 0.02, 0.03, 0.04], delta=0.02, n_boot=500, seed=1)
    assert stat["noninferior"]
    assert stat["ci"][0] > -0.02


def test_superiority_requires_lower_bound_above_margin():
    stat = superiority([0.0, 0.01, -0.01, 0.02], margin=0.0, n_boot=500, seed=1)
    assert not stat["superior"]
