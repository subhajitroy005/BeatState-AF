#!/usr/bin/env python3
"""Write the final E01v2 deployable-primary verdict."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beatstate_af.provenance import file_sha256, full_git_commit


def _read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _success(rows, name):
    for r in rows:
        if r["comparison"] == name:
            return r["success"] == "True", r
    raise KeyError(name)


def main():
    dep = _read("results/e01v2/deployable/e01v2_paired_tests.csv")
    ora = _read("results/e01v2/oracle/e01v2_paired_tests.csv")
    comp, comp_row = _success(dep, "compression_factored16_minus_monolithic32")
    sem16, sem16_row = _success(dep, "same_budget_factored16_minus_random16")
    mono32, mono32_row = _success(dep, "same_budget_factored32_minus_monolithic32")
    rand32, rand32_row = _success(dep, "same_budget_factored32_minus_random32")
    same32 = mono32 and rand32
    continue_route = (comp and sem16) or same32
    verdict = "CONTINUE" if continue_route else "STOP_CONFIRMED"
    text = (
        "# E01v2 Final Decision\n\n"
        "Primary result: deployable R-peaks. Oracle R-peaks are diagnostic only.\n\n"
        f"- Compression route: noninferiority={comp}, random-control superiority at 16={sem16}\n"
        f"- Same-budget 32 route: factored>monolithic={mono32}, factored>random={rand32}\n"
        f"- Oracle diagnostic paired tests available: {len(ora)} rows\n\n"
        f"## VERDICT: {verdict}\n\n"
    )
    if verdict == "STOP_CONFIRMED":
        text += (
            "In a corrected patient-level AFDB evaluation with frozen splits, causal preprocessing, "
            "complete algorithmic-memory accounting, validation-selected training, and a non-semantic "
            "block-diagonal control, physiology-assigned recurrent state did not demonstrate a valid "
            "deployable primary advantage that survived the random control in this run.\n"
        )
    else:
        text += (
            "Physiology-assigned recurrent state achieved a deployable patient-level route that survived "
            "the random structural control under the frozen AFDB protocol.\n"
        )
    out = Path("results/e01v2/e01v2_final_decision.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)

    if verdict == "STOP_CONFIRMED":
        registry = Path("docs/failure_registry.md")
        block = (
            "\n## E01v2 Audit-Corrected AFDB Kill Gate\n\n"
            "Outcome: `STOP_CONFIRMED`.\n\n"
            "The corrected deployable-primary E01v2 result did not establish a physiology-assigned "
            "state advantage over the non-semantic random structural control. The novel BeatState-AF "
            "model direction is closed for this AFDB single-lead GRU setting.\n"
        )
        existing = registry.read_text() if registry.exists() else "# Failure Registry\n"
        marker = "## E01v2 Audit-Corrected AFDB Kill Gate"
        if marker not in existing:
            registry.write_text(existing.rstrip() + "\n" + block)
    artifact_paths = sorted(
        str(p)
        for root in ("results/e01v2", "results/validation", "figures/e01v2", "figures/audit")
        for p in Path(root).glob("**/*")
        if p.is_file()
    )
    manifest = Path("results/e01v2/e01v2_artifact_manifest.json")
    rows = ["{\n", f'  "full_commit_sha": "{full_git_commit()}",\n', '  "artifacts": {\n']
    for i, path in enumerate(artifact_paths):
        comma = "," if i + 1 < len(artifact_paths) else ""
        rows.append(f'    "{path}": "{file_sha256(path)}"{comma}\n')
    rows.append("  }\n}\n")
    manifest.write_text("".join(rows))
    print(f"wrote {out} -> {verdict}")


if __name__ == "__main__":
    main()
