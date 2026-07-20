# BeatState-AF

**Capacity-matched, physiology-factored state allocation for continuous atrial
fibrillation detection.** Software-only research; public datasets; no hardware claims.

This repository is built to be executed by **Claude Code**. It already runs
end-to-end on synthetic data (the *walking skeleton*), producing a real result row
and an auto kill-decision. Claude Code then carries it to real ECG in a gated order
(see `CLAUDE.md`). Governance mirrors
[BudgetCL-ECG](https://github.com/subhajitroy005/BudgetCL-ECG).

## The one law
Under a fixed recurrent-state / software-memory budget, does splitting state into an
**atrial-morphology** channel and a **ventricular-rhythm** channel beat a
**capacity-matched monolithic** state — most of all when the two channels disagree?
Success = **compression** (same utility at ≤½ the state bytes; preferred) or
**superiority** (better at matched budget). Otherwise the model direction is killed.

## Quickstart (the dirty kill, ~seconds, pure numpy)
```bash
make install          # numpy + pyyaml (torch/wfdb added by Claude Code for real data)
make run-kill-gate    # runs the synthetic walking-skeleton kill gate
cat results/primary/kill_gate_decision.md
```
You will see three matched-capacity models compared — `monolithic` (dense state),
`factored` (physiology block-diagonal), and `random` (the sparsity control) — with
a per-patient result CSV, a summary, a memory ledger, and an auto CONTINUE/STOP
verdict. On synthetic data this only proves the pipeline; the verdict is meaningful
once `CLAUDE.md` Task 4 runs it on real AFDB with deployable R-peaks.

## What is real vs to-be-built
- **Real now:** interfaces, three matched-capacity state models, synthetic streams
  with a discordance mechanism, patient-disjoint splitting, trained readout, memory
  ledger, patient-level metrics, paired-bootstrap + equivalence stats, the kill-gate
  runner, tests.
- **Claude Code builds (gated):** AFDB `wfdb` loader, causal QRS front end, trained
  PyTorch GRU/factored/random models, saturation/crossover, external validation,
  episodes/burden/calibration, secondary personalization, release. See `CLAUDE.md`.

## Layout
`beatstate_af/` code · `configs/` frozen YAML · `experiments/` runners · `results/`
released CSVs · `scripts/` · `tests/` · `docs/` protocol & playbook · `manuscript/`.

## License
MIT for the software (`LICENSE`). PhysioNet datasets keep their own terms and are
never redistributed here.
