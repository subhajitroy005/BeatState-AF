# Kill-gate figure (generated from results/primary/kill_gate_oracle_summary.csv)

Mean patient-level macro-F1 by model and state dimension; recurrent
params / state bytes from the memory ledger.

| model | dim 16 (F1) | dim 32 (F1) | recur.params (max dim) | state bytes (max dim) |
|---|---|---|---|---|
| gru_factored | 0.7552 | 0.7565 | 1536 | 128 |
| gru_monolithic | 0.7482 | 0.7642 | 3072 | 128 |
| gru_random | 0.7336 | 0.7377 | 1536 | 128 |
