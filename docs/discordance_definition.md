# Discordance definition (FROZEN before the confirmatory kill gate)

The discordant subset is where **atrial-morphology** evidence and
**ventricular-rhythm** evidence disagree; the mechanistic hypothesis is
`Delta_U_discordant > Delta_U_concordant` (the factored model helps most here).

## Synthetic (walking skeleton)
Injected directly (`beatstate_af/data/synthetic.py`): PAC-like (irregular rhythm,
present P-wave, label non-AF) and regularised-AF (regular rhythm, absent
P-wave/f-waves, label AF).

## AFDB (real data) — prespecified rule
Implemented in `beatstate_af/data/wfdb_io._labels_and_disc`, parameterised in
`configs/datasets/afdb.yaml`. A beat is **discordant** (`disc = 1`) if ANY holds:

1. **Peri-transition** — the beat falls within `transition_window_s = 5 s` of an
   AFIB onset/offset boundary (from `.atr` rhythm change points). Around a rhythm
   change, morphology and rhythm evidence lag each other.
2. **Non-AFIB arrhythmia** — the beat lies in an `AFL` (atrial flutter) or `J`
   (junctional) rhythm segment. These are rhythm/morphology-atypical non-AF beats
   where a single-channel readout is most tempted to mislabel.
3. **Low SQI** — the beat's atrial-segment relative-power SQI
   (`beatstate_af.preprocessing.features._sqi`) is at or below the per-record
   `low_sqi_percentile = 10`th percentile (noisy atrial evidence).

All other beats are **concordant** (`disc = 0`).

## Labels
`AFIB -> y = 1`; `N`, `AFL`, `J -> y = 0` (standard AFDB AF-detection labeling,
`af_rhythms: [AFIB]`). AFL/J are therefore non-AF **and** flagged discordant.

## Power
The per-patient discordant-beat count is recorded (each result row carries
`acc_discordant`, computed only over `disc = 1` beats). Records with no discordant
beats contribute `NaN` to the discordant accuracy and are dropped from that mean.

**Freeze note:** this rule and its parameters are fixed prior to the confirmatory
run. Any change forks a v2 dataset id.
