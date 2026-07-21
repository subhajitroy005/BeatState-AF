# Kill-gate decision (auto-generated)

run_id: 6b6eaf9e0d61 | config_hash: 7b7925ce9eaea2df | commit: 5fc093a | dataset: afdb_oracle | family: gru

> Dataset is REAL (afdb_oracle). This verdict is confirmatory per roadmap 3.5.

## Mean macro-F1 (state dim = 32)
- monolithic (gru_monolithic): 0.7642   (recurrent params 3072)
- factored   (gru_factored): 0.7565   (recurrent params 1536)
- random     (gru_random): 0.7377

## Success routes
- Superiority (factored - monolithic @ 32): mean -0.0077, 95% CI [-0.0281, +0.0092] -> superior: False
- Sparsity control (factored - random @ 32): mean +0.0188, 95% CI [-0.0043, +0.0581] -> semantics beat sparsity: False
- Compression (factored@16 vs monolithic@32): mean -0.0090, 95% CI [-0.0295, +0.0076] -> equivalent at fewer bytes: False
- Discordance accuracy @ 32: factored 0.900 vs monolithic 0.925 (delta -0.026)

## VERDICT: STOP
Per roadmap 3.5: continue only if superiority OR compression is supported. Do not
rescue a failure by adding attention/Transformer/Mamba/branches.
