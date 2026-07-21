"""Model registry. Skeleton = reservoir kinds; trained = PyTorch GRU kinds."""
from beatstate_af.models.reservoir import build_model

SKELETON_KINDS = ("monolithic", "factored", "random")
GRU_KINDS = ("gru_monolithic", "gru_factored", "gru_random")


def build_reservoir(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed):
    return build_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed)


def build_gru(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed):
    from beatstate_af.models.gru import build_gru_model
    return build_gru_model(kind, n_features, atrial_idx, rhythm_idx, total_state_dim, seed)
