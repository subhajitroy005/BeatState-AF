import pytest

from beatstate_af.statistics.stats import (
    assert_patient_level_rows,
    average_metric_by_patient,
    paired_differences,
)


def test_seed_patient_rows_must_be_aggregated_before_inference():
    rows = [
        {"patient_id": "p1", "model_seed": 0},
        {"patient_id": "p1", "model_seed": 1},
    ]
    with pytest.raises(ValueError):
        assert_patient_level_rows(rows)


def test_patient_aggregation_then_paired_difference():
    rows = [
        {"model_id": "a", "total_state_dim": 16, "patient_id": "p1", "model_seed": 0, "present_class_macro_f1": 0.6},
        {"model_id": "a", "total_state_dim": 16, "patient_id": "p1", "model_seed": 1, "present_class_macro_f1": 0.8},
        {"model_id": "b", "total_state_dim": 16, "patient_id": "p1", "model_seed": 0, "present_class_macro_f1": 0.5},
        {"model_id": "b", "total_state_dim": 16, "patient_id": "p1", "model_seed": 1, "present_class_macro_f1": 0.7},
    ]
    agg = average_metric_by_patient(rows, "present_class_macro_f1", ["model_id", "total_state_dim"])
    diffs, patients = paired_differences(agg, "present_class_macro_f1", "a", 16, "b", 16)
    assert patients == ["p1"]
    assert diffs.tolist() == pytest.approx([0.1])
