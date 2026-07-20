# Protocol

The machine-readable, frozen protocol is `configs/protocol_v1.yaml`. The full
rationale is in the roadmap docx; this repo executes it. Key points:

- **One law**, two success routes; **compression preferred** over superiority.
- **Patient is the statistical unit.** Paired bootstrap CI + TOST equivalence
  (`beatstate_af/statistics/stats.py`).
- **Matched capacity + the sparsity control**: because a block-diagonal factored
  state has ~half the recurrent parameters of a dense monolithic state of the same
  dimension, any efficiency win must beat the **random-assignment** model before it
  is credited to physiology.
- **Deployable R-peaks** for the primary result; oracle R is an upper bound
  (report `Delta_QRS = U_oracle - U_deployable`).
- **Software-only**: report bytes/params/ops, never compiled flash/SRAM/latency/energy.
