# Kill-gate figure (generated from results/primary/kill_gate_summary.csv)

Mean patient-level macro-F1 by model and state dimension; recurrent
params / state bytes from the memory ledger.

| model | dim 16 (F1) | dim 32 (F1) | recur.params (max dim) | state bytes (max dim) |
|---|---|---|---|---|
| gru_factored | 0.7210 | 0.6951 | 1536 | 128 |
| gru_monolithic | 0.6954 | 0.6756 | 3072 | 128 |
| gru_random | 0.7236 | 0.6384 | 1536 | 128 |
