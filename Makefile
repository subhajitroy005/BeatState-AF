.PHONY: install test lint run-kill-gate verify-results reproduce audit clean help
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

verify-results:  ## check result rows carry full provenance schema
	$(PY) scripts/verify_results.py results/primary/kill_gate_patient_seed.csv

reproduce:  ## clean rerun of the kill gate + verification
	rm -rf results/primary/*.csv results/primary/*.md; $(MAKE) run-kill-gate verify-results

audit:  ## lint + test + verify (run before any release)
	$(MAKE) lint test verify-results

clean:
	rm -rf **/__pycache__ .pytest_cache
