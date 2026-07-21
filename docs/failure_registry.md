# Failure registry

Every negative result and abandoned branch is recorded here and kept in releases.
An empty registry is evidence of weak checks, not of correctness.

| date | experiment | what failed | decision |
|------|-----------|-------------|----------|
| 2026-07-21 | kill_gate_afdb_deployable (GRU, real AFDB, deployable R) | Neither success route holds; random-assignment control not beaten; discordance signal weak/unstable | **STOP** the novel-model direction (roadmap 3.5) |
| 2026-07-21 | kill_gate_afdb_oracle (GRU, real AFDB, oracle R) | Same null with clean R-peaks (factored nominally worse); confirms it is not a QRS artefact | STOP confirmed; robust to QRS provenance |

## 2026-07-21 — Physiology-factored state does not beat capacity-matched monolithic on AFDB (pilot)

**Setup.** Trained streaming GRUs, matched total state dim (16, 32), identical 8-dim
per-beat tokens (atrial morphology | ventricular rhythm), patient unit = AFDB record
(23 signal records, first 3 h each; 30% AF cohort-wide), paired bootstrap over
(seed, patient), seeds [0,1,2]. Arms: `gru_monolithic` (dense), `gru_factored`
(physiology block-diagonal), `gru_random` (block-diagonal, non-physiological routing).
Training BPTT capped at 6000 beats/patient (compute-bounded pilot); evaluation on full
sequences. Provenance in `results/primary/kill_gate*_patient_seed.csv`.

**Result (mean patient macro-F1 @ dim 32).**
| arm | deployable R | oracle R | recurrent params |
|---|---|---|---|
| monolithic | 0.676 | 0.764 | 3072 |
| factored | 0.695 | 0.756 | 1536 |
| random | 0.638 | 0.738 | 1536 |

- **Superiority** (factored − monolithic @32): deployable +0.020 CI [−0.007, +0.053];
  oracle −0.008 CI [−0.028, +0.009]. Not superior in either.
- **Sparsity control** (factored − random @32): deployable +0.057 CI [−0.018, +0.139];
  oracle +0.019 CI [−0.004, +0.058]. **The mandatory random control is not beaten**, so
  no gain can be credited to physiological routing. At dim 16 the block-diagonal arms
  (factored 0.721, random 0.724) both beat dense monolithic (0.695) — a
  structural-sparsity/regularisation effect, *not* physiology (random ties/wins).
- **Compression** (factored@16, 64 B, vs monolithic@32, 128 B): no TOST equivalence in
  either mode. Any low-dim edge is the sparsity effect above, not physiology.
- **Discordance** Δ @32: deployable +0.022, oracle **−0.026** — the sign flips with
  QRS provenance, i.e. no stable discordance mechanism.
- **Δ_QRS** (oracle − deployable, macro-F1 @32): monolithic +0.089, factored +0.061,
  random +0.099. The causal detector costs ~6–10 F1 points but does not change the verdict.

**Why.** On single-lead AFDB the discriminative signal is dominated by ventricular
rhythm (RR irregularity; per-feature AUCs: rr≈0.07, local RR var≈0.66) while the atrial-
morphology tokens are weak (P/f-wave organisation AUC≈0.42–0.50). Splitting a fixed
budget to protect a weak atrial channel does not help, and the dense monolithic state
(2× recurrent params) matches or beats it. The mechanism the paper predicts (factored
wins under discordance) does not appear.

**Decision.** STOP the novel-model direction per roadmap §3.5. Do **not** add attention /
Transformer / Mamba / extra branches to rescue it. Possible honest next studies (new
preregistration, not a rescue): stronger atrial-evidence tokens (multi-lead, longer
context), or a dataset where atrial morphology is more informative — each would be a new
law, separately gated.

## E01v2 Audit-Corrected AFDB Kill Gate

Outcome: `STOP_CONFIRMED`.

The corrected deployable-primary E01v2 result did not establish a physiology-assigned state advantage over the non-semantic random structural control. The novel BeatState-AF model direction is closed for this AFDB single-lead GRU setting.
