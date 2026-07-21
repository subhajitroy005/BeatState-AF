import numpy as np

from beatstate_af.evaluation.metrics import patient_metrics


def test_af_only_patient_excludes_absent_nonaf_class():
    m = patient_metrics(np.ones(5), np.ones(5))
    assert m["present_class_macro_f1"] == 1.0
    assert m["af_f1"] == 1.0
    assert m["patient_has_af"] and not m["patient_has_nonaf"]


def test_nonaf_only_patient_excludes_absent_af_class():
    m = patient_metrics(np.zeros(5), np.zeros(5))
    assert m["present_class_macro_f1"] == 1.0
    assert np.isnan(m["af_f1"])
    assert not m["patient_has_af"] and m["patient_has_nonaf"]


def test_mixed_patient_and_degenerate_predictions():
    y = np.array([0, 0, 1, 1])
    all_nonaf = patient_metrics(y, np.zeros_like(y))
    all_af = patient_metrics(y, np.ones_like(y))
    mixed = patient_metrics(y, np.array([0, 1, 1, 1]))
    assert 0.0 <= all_nonaf["present_class_macro_f1"] <= 1.0
    assert 0.0 <= all_af["present_class_macro_f1"] <= 1.0
    assert 0.0 <= mixed["present_class_macro_f1"] <= 1.0
    assert np.isnan(all_nonaf["ppv"])
    assert np.isnan(all_af["specificity"]) or 0.0 <= all_af["specificity"] <= 1.0
