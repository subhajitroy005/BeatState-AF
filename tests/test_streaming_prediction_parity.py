import numpy as np

from beatstate_af.data.synthetic import ATRIAL_IDX, N_FEATURES, RHYTHM_IDX
from beatstate_af.models.gru import (
    build_gru_model,
    predict_proba_seq,
    predict_proba_streaming,
    proba_from_state,
)


def test_streaming_probabilities_match_batched_causal_forward():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(40, N_FEATURES)).astype(np.float32)
    mu = np.zeros(N_FEATURES, dtype=np.float32)
    sd = np.ones(N_FEATURES, dtype=np.float32)
    for kind in ("gru_monolithic", "gru_factored", "gru_random"):
        model = build_gru_model(kind, N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=4, routing_seed=101)
        batched = predict_proba_seq(model, x, mu, sd)
        streamed = predict_proba_streaming(model, x, mu, sd)
        assert np.allclose(streamed, batched, atol=1e-4)


def test_streaming_outputs_invariant_to_chunking():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(37, N_FEATURES)).astype(np.float32)
    model = build_gru_model("gru_factored", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=5)
    model.reset_state()
    full = []
    for token in x:
        full.append(proba_from_state(model, model.step(token)))
    model.reset_state()
    chunked = []
    for chunk in (x[:11], x[11:23], x[23:]):
        for token in chunk:
            chunked.append(proba_from_state(model, model.step(token)))
    assert np.allclose(full, chunked, atol=1e-7)
