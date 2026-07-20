from beatstate_af.data.synthetic import make_cohort
from beatstate_af.data.patient_split import patient_disjoint_split

def test_disjoint_and_complete():
    cohort = make_cohort(20, base_seed=0, n_beats=50)
    train, test = patient_disjoint_split(cohort.keys(), 0.6, seed=1)
    assert set(train).isdisjoint(test)
    assert set(train) | set(test) == set(cohort.keys())
    assert train and test
