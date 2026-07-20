# CLAUDE.md — operating instructions for Claude Code on BeatState-AF

You are executing a controlled research study. Read this file fully before acting.
This repo already runs end-to-end on synthetic data (the walking skeleton). Your
job is to carry it to a real, honest result on public ECG — **in the gated order
below**, never skipping ahead, never stubbing a runner.

## The single law this paper tests
> Under a fixed total recurrent-state and software-memory envelope, explicitly
> factorizing temporal state into **atrial-morphology** evidence and
> **ventricular-rhythm** evidence gives greater continuous-AF utility than a
> **capacity-matched monolithic** recurrent state — with the largest gain when
> morphology and rhythm disagree (the *discordance* mechanism).

Two ways to succeed (compression is the **preferred headline**):
1. **Compression (preferred):** factored reaches equivalent utility using ≤ 1/2 the
   monolithic recurrent-state bytes (TOST-style equivalence, patient unit).
2. **Superiority:** factored beats monolithic at the same total state/memory budget.

**Kill criterion (roadmap §3.5):** if neither route holds AND the random-assignment
control is not beaten AND there is no discordance signal, **STOP the novel-model
direction.** Do NOT rescue a failure by adding attention, a Transformer, Mamba, or
extra branches. Write the null to `docs/failure_registry.md` and stop.

## Non-negotiable rules (do not violate)
- **Software-only.** Never claim compiled flash, measured SRAM, latency, or energy.
  Report serialized bytes, persistent state bytes, param/operation counts only.
- **No stub runners.** Every `experiments/run_*.py` must train and evaluate for
  real and write a result row. A validation-only runner is forbidden.
- **Patient is the statistical unit.** Seeds are optimization noise; never treat a
  seed as an extra patient. Use `beatstate_af/statistics`.
- **Matched comparison.** Identical tokens, identical numerical precision, and
  monolithic total dim == atrial_dim + rhythm_dim. The fusion/readout is tiny and
  counted. Enforced by `tests/test_matched_capacity.py`.
- **The sparsity control is mandatory.** Matched *dimension* ≠ matched *capacity*:
  a block-diagonal factored state has ~half the recurrent parameters of a dense
  monolithic state of the same dimension. So you MUST beat the **random-assignment**
  model (same block structure, non-physiological feature routing) before crediting
  any gain to *physiology*. It is already a first-class arm (`kind="random"`).
- **Complete memory ledger.** Report state bytes AND recurrent-param counts for
  every cell (`beatstate_af/memory/ledger.py`).
- **Provenance on every row.** Each result row carries config hash, commit, env
  hash, dataset id, model id, dims, seed, patient id (`RESULT_ROW_COLUMNS`).
- **Everything generated.** Tables/figures/manuscript numbers regenerate from the
  released CSVs; never hand-type a number. Keep every experiment value in frozen
  YAML under `configs/`.

## Execution order — DO THESE IN SEQUENCE

**Task 0 (already works — verify): the dirty kill.**
`make install && make run-kill-gate` → confirm `results/primary/kill_gate_*.csv`
and `kill_gate_decision.md` appear. This is the harness you will keep using.

**Task 1: real data behind the same contract.**
Implement `beatstate_af/data/wfdb_io.py` (`load_afdb_record`) with `wfdb`, emitting
the identical `dict(feat=(T,8), y, disc, patient_id)` the synthetic generator uses.
Add `configs/datasets/afdb.yaml`, a frozen record manifest under `manifests/`, and a
patient-disjoint split. `tests/test_patient_disjoint.py` must pass on real ids.
Point the pilot at oracle R-peaks first (`dataset_id: afdb_oracle`).

**Task 2: deployable front end + discordance from real annotations.**
Add a causal classical QRS detector (`beatstate_af/preprocessing/qrs.py`), count its
state/buffer bytes in the ledger, and define the prespecified discordant subset in
`docs/discordance_definition.md` from AFDB rhythm/beat annotations (PAC/PVC/flutter,
low-SQI). Add `dataset_id: afdb_deployable`. The primary result uses deployable R;
oracle R is only an upper-bound diagnostic (report `Delta_QRS`).

**Task 3: trained recurrent models (PyTorch) behind the interface.**
Add `beatstate_af/models/gru.py` with `gru_monolithic`, `gru_factored`,
`gru_random` implementing `StreamingAFModel` (causal, O(1) state, `state_bytes`,
`recurrent_param_count`). Keep monolithic dim == sum of the two factored dims.
Register in `models/registry.py`. Do NOT change `run_kill_gate.py`'s logic — only
swap the model factory and the readout to a trained head.

**Task 4: rerun the kill gate on real, deployable data.**
`python experiments/run_kill_gate.py --config configs/experiments/kill_gate_afdb.yaml`
Then READ `kill_gate_decision.md`. If VERDICT is STOP, **stop** and record the null.

**Only if CONTINUE — the hardening sprints (roadmap S12–S19), each gated:**
state-memory saturation/crossover (the headline figure), complete-memory frontier
vs raw-window baselines, QRS/noise robustness, external validation
(SHDB-AF/CPSC2021), continuous episodes + AF burden + calibration, then the
secondary evidence-channel personalization, then release. Build one gate at a time;
run `make test` after each; append every negative to `docs/failure_registry.md`.

## How to work
- One experiment at a time. Write/extend a test with each module. `make test` green
  before moving on. `make audit` before any release.
- Never invent results; if something fails, log it and stop the affected branch.
- The full plan and rationale live in the roadmap docx (see repo owner) and mirror
  the governance of https://github.com/subhajitroy005/BudgetCL-ECG .

## Repo map
`beatstate_af/` reusable code · `configs/` frozen YAML · `manifests/` records+splits
`experiments/` runners · `results/` released CSVs · `figures/`, `manuscript/`
`scripts/` data+figures+verification · `tests/` correctness+leakage · `docs/` protocol.
