.PHONY: install test lint run-kill-gate verify-results reproduce audit e01v2-split e01v2-validate e01v2-oracle e01v2-deployable e01v2-statistics e01v2-figures e01v2-decision audit-e01v2 reproduce-e01v2 clean help
PY?=python3
export PYTHONPATH=.

help:
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-18s %s\n",$$1,$$2}'

install:  ## install runtime deps (numpy + pyyaml; add torch/wfdb for real data)
	$(PY) -m pip install --break-system-packages -e . || $(PY) -m pip install --break-system-packages numpy pyyaml

test:  ## run correctness + leakage tests
	$(PY) -m pytest -q tests || $(PY) tests/run_tests.py

lint:  ## byte-compile everything (cheap syntax gate)
	$(PY) -m compileall -q beatstate_af experiments scripts tests

run-kill-gate:  ## THE DIRTY KILL: synthetic walking-skeleton kill gate
	$(PY) experiments/run_kill_gate.py --config configs/experiments/kill_gate_pilot.yaml

download-afdb:  ## fetch the frozen AFDB record prefixes into data/afdb (curl range)
	$(PY) scripts/_fetch_afdb_prefix.py

run-kill-gate-afdb:  ## CONFIRMATORY: trained GRU on real AFDB, deployable R-peaks (Task 4)
	$(PY) experiments/run_kill_gate.py --config configs/experiments/kill_gate_afdb.yaml

run-kill-gate-afdb-oracle:  ## Delta_QRS diagnostic: same, oracle R-peaks
	$(PY) experiments/run_kill_gate.py --config configs/experiments/kill_gate_afdb_oracle.yaml

verify-results:  ## check result rows carry full provenance schema
	$(PY) scripts/verify_results.py results/primary/kill_gate_patient_seed.csv

reproduce:  ## clean rerun of the kill gate + verification
	rm -rf results/primary/*.csv results/primary/*.md; $(MAKE) run-kill-gate verify-results

audit:  ## lint + test + verify (run before any release)
	$(MAKE) lint test verify-results

e01v2-split:
	$(PY) scripts/create_afdb_split_v2.py --seed 20260721 --output manifests/afdb_split_v2.json
	$(PY) scripts/verify_split_manifest.py manifests/afdb_split_v2.json

e01v2-validate:
	$(PY) experiments/run_e01v2_validation.py --config configs/experiments/e01v2_afdb_deployable.yaml

e01v2-oracle:
	$(PY) experiments/run_e01v2_kill_gate.py --config configs/experiments/e01v2_afdb_oracle.yaml

e01v2-deployable:
	$(PY) experiments/run_e01v2_kill_gate.py --config configs/experiments/e01v2_afdb_deployable.yaml

e01v2-statistics:
	$(PY) scripts/run_e01v2_statistics.py

e01v2-figures:
	$(PY) scripts/make_e01v2_figures.py

e01v2-decision:
	$(PY) scripts/write_e01v2_decision.py

audit-e01v2:
	$(PY) scripts/verify_clean_tree.py
	$(PY) scripts/verify_split_manifest.py manifests/afdb_split_v2.json
	$(PY) scripts/verify_e01v2_artifacts.py
	$(MAKE) lint test

reproduce-e01v2:
	$(MAKE) e01v2-validate
	$(MAKE) e01v2-oracle
	$(MAKE) e01v2-deployable
	$(MAKE) e01v2-statistics
	$(MAKE) e01v2-figures
	$(MAKE) e01v2-decision
	$(MAKE) audit-e01v2

clean:
	rm -rf **/__pycache__ .pytest_cache
