import json

from beatstate_af.config import load_config
from beatstate_af.data.synthetic import ATRIAL_IDX, N_FEATURES, RHYTHM_IDX
from beatstate_af.models.gru import build_gru_model


def test_frozen_split_is_complete_and_disjoint():
    split = json.load(open("manifests/afdb_split_v2.json"))
    records = set(json.load(open("manifests/afdb_records.json"))["records"])
    train, validation, test = map(set, (split["train"], split["validation"], split["test"]))
    assert split["split_seed"] == 20260721
    assert train.isdisjoint(validation)
    assert train.isdisjoint(test)
    assert validation.isdisjoint(test)
    assert train | validation | test == records


def test_configs_share_split_and_separate_seed_sources():
    dep = load_config("configs/experiments/e01v2_afdb_deployable.yaml")
    ora = load_config("configs/experiments/e01v2_afdb_oracle.yaml")
    assert dep["split_manifest"] == ora["split_manifest"] == "manifests/afdb_split_v2.json"
    assert dep["model_seeds"] == [0, 1, 2, 3, 4]
    assert dep["routing_seeds"] == [101]
    assert "split_seed" not in dep or dep.get("split_seed") != dep["model_seeds"][0]


def test_random_route_reproducible_independent_of_model_seed():
    a = build_gru_model("gru_random", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=0, routing_seed=101)
    b = build_gru_model("gru_random", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=1, routing_seed=101)
    c = build_gru_model("gru_random", N_FEATURES, ATRIAL_IDX, RHYTHM_IDX, 16, seed=0, routing_seed=202)
    assert a.atrial_idx.tolist() == b.atrial_idx.tolist()
    assert a.rhythm_idx.tolist() == b.rhythm_idx.tolist()
    assert a.atrial_idx.tolist() != c.atrial_idx.tolist() or a.rhythm_idx.tolist() != c.rhythm_idx.tolist()
