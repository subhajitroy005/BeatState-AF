from pathlib import Path

from beatstate_af.provenance import (
    environment_hash,
    environment_report,
    file_sha256,
    full_git_commit,
    git_tree_dirty,
)


def test_environment_report_contains_required_versions():
    report = environment_report()
    for key in ("python_version", "platform", "numpy_version", "scipy_version", "torch_version", "wfdb_version"):
        assert key in report
    assert len(environment_hash()) == 16


def test_hashes_and_git_helpers_are_available():
    assert len(file_sha256("configs/protocol_v2.yaml")) == 64
    assert full_git_commit() == "nogit" or len(full_git_commit()) == 40
    assert git_tree_dirty(ignore_paths=("results/e01v2", "figures/e01v2", "figures/audit")) in (True, False)
    assert Path("requirements-lock.txt").exists()
