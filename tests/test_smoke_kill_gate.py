from beatstate_af.data.synthetic import make_cohort, ATRIAL_IDX, RHYTHM_IDX, N_FEATURES
from beatstate_af.data.patient_split import patient_disjoint_split
from experiments.run_kill_gate import run_cell

def test_kill_gate_cell_produces_valid_rows():
    cohort = make_cohort(16, base_seed=0, n_beats=200)
    train, test = patient_disjoint_split(cohort.keys(), 0.6, seed=0)
    rows, ppf1, mem = run_cell("factored", 16, 0, cohort, train, test)
    assert rows and mem["total_state_dim"] == 16
    for r in rows:
        assert 0.0 <= r["macro_f1"] <= 1.0
