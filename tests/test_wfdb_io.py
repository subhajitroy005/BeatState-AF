"""AFDB loader honors the synthetic token contract on REAL record ids.

Skipped when the local cache (data/afdb) has not been downloaded, so `make test`
stays green without network. When data is present it enforces the (feat,y,disc)
contract and a patient-disjoint split over real record ids.
"""
import glob, os
import numpy as np
import pytest

from beatstate_af.config import load_config
from beatstate_af.data.patient_split import patient_disjoint_split


def _complete_records():
    recs = []
    for dat in sorted(glob.glob("data/afdb/*.dat")):
        r = dat[:-4]
        if all(os.path.exists(r + e) for e in (".hea", ".atr", ".qrs")):
            recs.append(os.path.basename(r))
    return recs


_RECS = _complete_records()
_HAS_DATA = len(_RECS) >= 4


@pytest.mark.skipif(not _HAS_DATA, reason="AFDB cache not downloaded")
def test_contract_and_disjoint_on_real_ids():
    from beatstate_af.data.wfdb_io import load_afdb_record
    dcfg = load_config("configs/datasets/afdb.yaml")
    recs = _RECS[:4]
    ids = []
    for rec in recs:
        out = load_afdb_record("data/afdb/" + rec, qrs_mode="deployable", fs=dcfg["fs_hz"],
                               channel=dcfg["signal_channel"], max_beats=2000,
                               af_rhythms=tuple(dcfg["af_rhythms"]), disc_cfg=dcfg["discordance"])
        T = out["feat"].shape[0]
        assert out["feat"].shape[1] == 8 and T > 0
        assert out["y"].shape == (T,) and out["disc"].shape == (T,)
        assert set(np.unique(out["y"])) <= {0, 1}
        assert set(np.unique(out["disc"])) <= {0, 1}
        assert np.isfinite(out["feat"]).all()
        ids.append(out["patient_id"])
    assert len(set(ids)) == len(ids)  # patient_id == record id, unique
    train, test = patient_disjoint_split(ids, 0.5, seed=0)
    assert set(train).isdisjoint(test) and train and test


@pytest.mark.skipif(not _HAS_DATA, reason="AFDB cache not downloaded")
def test_oracle_and_deployable_both_load():
    from beatstate_af.data.wfdb_io import load_afdb_record
    dcfg = load_config("configs/datasets/afdb.yaml")
    rec = _RECS[0]
    for mode in ("oracle", "deployable"):
        out = load_afdb_record("data/afdb/" + rec, qrs_mode=mode, fs=dcfg["fs_hz"],
                               channel=dcfg["signal_channel"], max_beats=2000,
                               af_rhythms=tuple(dcfg["af_rhythms"]), disc_cfg=dcfg["discordance"])
        assert out["feat"].shape[0] > 0
