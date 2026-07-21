# E01v1 Audit

The six E01v1 primary artifacts are preserved by SHA-256 in
`scripts/verify_e01v2_artifacts.py`. E01v2 writes only versioned outputs under
`results/e01v2/`.

| Defect | Source Area | Corrective Commit | Regression Test | E01v2 Output Field |
|---|---|---|---|---|
| `(seed, patient)` bootstrapped as independent units | `beatstate_af/statistics/stats.py`, `experiments/run_e01v2_kill_gate.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_patient_unit_statistics.py` | `n_model_seeds`, `n_valid_paired_patients` |
| Patient split changed with model seed | `manifests/afdb_split_v2.json`, `scripts/verify_split_manifest.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_e01v2_split_and_seeds.py` | `split_id` |
| Random partition changed with model seed | `beatstate_af/models/gru.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_e01v2_split_and_seeds.py` | `routing_seed` |
| Compression used symmetric equivalence | `beatstate_af/statistics/stats.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_noninferiority.py` | `test`, `practical_margin` in paired tests |
| Whole-record normalization used future samples | `beatstate_af/preprocessing/features.py`, `beatstate_af/e01v2.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_causal_preprocessing.py` | `normalizer_bytes` |
| QRS detection allocated whole-record arrays | `beatstate_af/preprocessing/qrs_stream.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_qrs_streaming_parity.py` | `qrs_persistent_bytes`, `qrs_scratch_bytes` |
| GRU evaluation used full-sequence forward | `beatstate_af/models/gru.py`, `experiments/run_e01v2_kill_gate.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_streaming_prediction_parity.py` | `status` |
| Streaming GRU state used float64 | `beatstate_af/models/gru.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_streaming_dtype.py` | `recurrent_state_bytes` |
| Memory ledger omitted major components | `beatstate_af/memory/ledger.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_memory_ledger.py` | all `*_bytes` fields |
| Macro-F1 assigned zero to absent classes | `beatstate_af/evaluation/metrics.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_metric_policy_v2.py` | `present_class_macro_f1`, `af_f1` |
| Training lacked validation convergence evidence | `experiments/run_e01v2_validation.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_provenance.py` | `results/validation/e01v2_training_curves.csv` |
| Rows referenced non-producing commits | `beatstate_af/provenance.py`, `experiments/run_e01v2_kill_gate.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_provenance.py` | `full_commit_sha`, `git_tree_dirty` |
| Environment hash omitted core dependencies | `beatstate_af/provenance.py`, `requirements-lock.txt` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_provenance.py` | `environment_hash` |
| Discordance lacked paired interaction test | `experiments/run_e01v2_kill_gate.py` | E01v2 implementation commit recorded as `full_commit_sha` | `tests/test_patient_unit_statistics.py` | `n_discordant_beats`, `n_concordant_beats`, `e01v2_discordance.csv` |
