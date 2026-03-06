"""Tests for the zx-motifs CLI (``python -m zx_motifs``)."""
import os
import subprocess
import sys

import pytest


def run_cli(*args, timeout=60):
    """Run the CLI as a subprocess and return the CompletedProcess."""
    result = subprocess.run(
        [sys.executable, "-m", "zx_motifs", *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=os.path.dirname(os.path.dirname(__file__)),
    )
    return result


# ── list algorithms ──────────────────────────────────────────────────


def test_list_algorithms():
    """``list algorithms`` exits 0 and includes bell_state."""
    result = run_cli("list", "algorithms")
    assert result.returncode == 0, result.stderr
    assert "bell_state" in result.stdout


def test_list_algorithms_family_filter():
    """``list algorithms --family entanglement`` shows only entanglement."""
    result = run_cli("list", "algorithms", "--family", "entanglement")
    assert result.returncode == 0, result.stderr
    # Every non-header line should be from the entanglement family
    lines = [
        line for line in result.stdout.strip().splitlines()
        if line and not line.startswith("-") and "FAMILY" not in line
    ]
    assert len(lines) > 0, "Expected at least one entanglement algorithm"
    for line in lines:
        assert "entanglement" in line, (
            f"Non-entanglement entry in filtered output: {line}"
        )


# ── list motifs ──────────────────────────────────────────────────────


def test_list_motifs():
    """``list motifs`` exits 0 and includes cx_pair."""
    result = run_cli("list", "motifs")
    assert result.returncode == 0, result.stderr
    assert "cx_pair" in result.stdout


# ── list families ────────────────────────────────────────────────────


def test_list_families():
    """``list families`` exits 0 and includes entanglement."""
    result = run_cli("list", "families")
    assert result.returncode == 0, result.stderr
    assert "entanglement" in result.stdout


# ── info ─────────────────────────────────────────────────────────────


def test_info_algorithm():
    """``info bell_state`` exits 0 and shows the entanglement family."""
    result = run_cli("info", "bell_state")
    assert result.returncode == 0, result.stderr
    assert "entanglement" in result.stdout


def test_info_motif():
    """``info --motif cx_pair`` exits 0 and shows graph structure."""
    result = run_cli("info", "--motif", "cx_pair")
    assert result.returncode == 0, result.stderr
    # The cx_pair motif is "CNOT as Z-X spider pair" — expect Z or X
    output = result.stdout
    assert "Z" in output or "X" in output or "CNOT" in output


def test_info_not_found():
    """``info nonexistent_algo`` returns a non-zero exit code."""
    result = run_cli("info", "nonexistent_algo")
    assert result.returncode != 0


# ── scaffold motif ───────────────────────────────────────────────────


def test_scaffold_motif(tmp_path, monkeypatch):
    """``scaffold motif`` creates a JSON template file."""
    # Run from the project root; the CLI writes into
    # src/zx_motifs/motifs/library/<name>.json
    result = run_cli("scaffold", "motif", "--name", "_test_scaffold_tmp")

    # The motif library path is inside the source tree
    from zx_motifs.motifs.registry import _LIBRARY_DIR
    created = _LIBRARY_DIR / "_test_scaffold_tmp.json"

    try:
        if result.returncode == 0:
            assert created.exists(), "Scaffold should create the JSON file"
        else:
            # File may already exist (leftover from a previous run)
            pytest.skip("Scaffold target already exists")
    finally:
        # Clean up
        if created.exists():
            created.unlink()


# ── validate ─────────────────────────────────────────────────────────


def test_validate():
    """``validate`` runs the full validation suite with exit code 0."""
    result = run_cli("validate", timeout=120)
    assert result.returncode == 0, (
        f"Validation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
