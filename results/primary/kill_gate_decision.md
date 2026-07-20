# Kill-gate decision (auto-generated)

run_id: 29dc44e8f78c | config_hash: d29d14d3b194469e | commit: 27b85cd | dataset: synthetic_v1

> NOTE: dataset is SYNTHETIC. This validates the harness end-to-end. The decision
> below is only meaningful once dataset_id = afdb_deployable (see CLAUDE.md).

## Mean macro-F1 (state dim = 32)
- monolithic: 0.9917   (recurrent params 1024)
- factored:   0.9932   (recurrent params 512)
- random:     0.9945

## Success routes
- Superiority (factored - monolithic @ 32): mean +0.0015, 95% CI [+0.0004, +0.0027] -> superior: True
- Sparsity control (factored - random @ 32): mean -0.0013, 95% CI [-0.0024, -0.0002] -> semantics beat sparsity: False
- Compression (factored@16 vs monolithic@32): mean -0.0030, 95% CI [-0.0044, -0.0017] -> equivalent at fewer bytes: True
- Discordance accuracy @ 32: factored 0.982 vs monolithic 0.985 (delta -0.003)

## VERDICT: CONTINUE
Per roadmap 3.5: continue only if superiority OR compression is supported. Do not
rescue a failure by adding attention/Transformer/Mamba/branches.
