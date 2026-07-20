# Claude Code playbook (condensed; full contract in CLAUDE.md)

Run these in order. Do not skip. Do not stub a runner.

0. `make install && make test && make run-kill-gate` -> confirm result rows + decision.
1. Implement `data/wfdb_io.py` (AFDB) to the (feat,y,disc) contract; add
   `configs/datasets/afdb.yaml`, a frozen record manifest, patient-disjoint split;
   `make test` green on real ids. Start with `afdb_oracle`.
2. Add causal QRS front end (`preprocessing/qrs.py`), count its bytes in the ledger;
   write `docs/discordance_definition.md` from real annotations; add `afdb_deployable`.
3. Add PyTorch `models/gru.py` (`gru_monolithic|factored|random`) implementing
   `StreamingAFModel`; keep matched dims + ledger; do not change run_kill_gate logic.
4. `run_kill_gate.py --config configs/experiments/kill_gate_afdb.yaml`; READ the
   decision. If STOP -> stop and log the null. If CONTINUE -> hardening sprints.
5+. Saturation/crossover (headline figure) -> memory frontier + raw-window baselines
   -> QRS/noise robustness -> external (SHDB-AF/CPSC2021) -> episodes/burden/
   calibration -> secondary personalization -> release. One gate at a time.

After every gate: `make audit`, and append negatives to `docs/failure_registry.md`.
