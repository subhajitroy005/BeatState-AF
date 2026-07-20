"""Model registry. Skeleton = reservoir kinds; Claude Code adds PyTorch models here."""
from beatstate_af.models.reservoir import build_model

SKELETON_KINDS = ("monolithic", "factored", "random")

# TODO(claude-code): register trained PyTorch models behind the same interface,
# e.g. "gru_monolithic", "gru_factored", "gru_random", "streaming_ssm".
