# Kill-gate decision (auto-generated)

run_id: dbd487988692 | config_hash: fd8b7cf420abbc7d | commit: 5fc093a | dataset: afdb_deployable | family: gru

> Dataset is REAL (afdb_deployable). This verdict is confirmatory per roadmap 3.5.

## Mean macro-F1 (state dim = 32)
- monolithic (gru_monolithic): 0.6756   (recurrent params 3072)
- factored   (gru_factored): 0.6951   (recurrent params 1536)
- random     (gru_random): 0.6384

## Success routes
- Superiority (factored - monolithic @ 32): mean +0.0195, 95% CI [-0.0065, +0.0532] -> superior: False
- Sparsity control (factored - random @ 32): mean +0.0566, 95% CI [-0.0180, +0.1391] -> semantics beat sparsity: False
- Compression (factored@16 vs monolithic@32): mean +0.0454, 95% CI [+0.0141, +0.0834] -> equivalent at fewer bytes: False
- Discordance accuracy @ 32: factored 0.852 vs monolithic 0.830 (delta +0.022)

## VERDICT: STOP
Per roadmap 3.5: continue only if superiority OR compression is supported. Do not
rescue a failure by adding attention/Transformer/Mamba/branches.
