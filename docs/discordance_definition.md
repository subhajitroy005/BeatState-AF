# Discordance definition (CONTRACT — freeze before final testing)

The discordant subset is where atrial-morphology evidence and ventricular-rhythm
evidence disagree; the mechanistic hypothesis is Delta_U_discordant > Delta_U_concordant.
On synthetic data these are injected (PAC-like: irregular rhythm, present P-wave,
label non-AF; regularised-AF: regular rhythm, absent P-wave/f-waves, label AF).

TODO(claude-code): define from AFDB annotations before any confirmatory run, e.g.
- irregular RR with beat annotations indicating frequent PAC/PVC (rhythm looks AF, not AF),
- AF rhythm segments with regularised ventricular response,
- low-SQI windows.
Freeze the exact rule and record the discordant-segment count per patient (power).
