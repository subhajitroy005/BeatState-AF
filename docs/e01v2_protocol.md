# E01v2 Protocol

E01v2 is the audit-corrected AFDB confirmatory kill gate. It preserves the three
GRU identities (`gru_monolithic`, `gru_factored`, `gru_random`) and state
dimensions (`16`, `32`) while repairing the evaluation and reproducibility
defects found after E01v1.

Frozen protocol values live in `configs/protocol_v2.yaml`. The split manifest is
`manifests/afdb_split_v2.json` with split seed `20260721`; the same train,
validation, and test patients are used for every model, seed, routing seed, and
QRS mode.

Primary metric: `present_class_macro_f1`.

Primary QRS mode: deployable streaming QRS. Oracle QRS is diagnostic only.

Compression test: `factored-16 - monolithic-32` is non-inferior when the paired
patient bootstrap lower 95% CI is greater than `-0.02`.

Same-budget superiority: `factored-D - monolithic-D` and `factored-D - random-D`
must both have lower 95% CI greater than `0.0`.

Random control: `gru_random` uses the same block recurrence and state allocation
as `gru_factored`, with feature routing controlled by frozen `routing_seed`
instead of model initialization seed.

Preprocessing policy: signal median/MAD constants are computed from training
patients only; token standardization mean/std is also computed from training
patients only. Test-patient future samples are never used for normalization.

Inference policy: primary test predictions are generated beat by beat through
`model.reset_state()` and `model.step(token)` with float32 recurrent state.

Memory policy: reported bytes are software accounting for serialized/static
arrays, persistent algorithmic state, and one-step scratch tensors. They are not
claimed as measured MCU SRAM, latency, energy, or battery life.

Decision: `CONTINUE` requires a deployable primary success route that survives
the random structural control. Otherwise the final result is `STOP_CONFIRMED`.
