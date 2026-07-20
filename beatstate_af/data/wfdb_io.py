"""Real-data loader CONTRACT (Claude Code implements this first).

Load a PhysioNet AFDB record into the same per-beat token layout the synthetic
generator produces, so nothing downstream changes.

Required output per record: dict(feat=(T,8) float32, y=(T,) in {0,1}, disc=(T,)
prespecified from the discordance definition, patient_id=str). Rhythm features
must be causal and derived from DEPLOYABLE R-peaks (see docs/protocol.md);
oracle R-peaks are an upper-bound diagnostic only.
"""
def load_afdb_record(record_path, qrs_mode="deployable"):
    raise NotImplementedError(
        "Claude Code: implement AFDB loading with wfdb; keep the (feat,y,disc) "
        "contract identical to beatstate_af.data.synthetic. See CLAUDE.md step 2.")
