# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ZX-Web Discovery Pipeline: a Python framework for discovering novel quantum algorithms by mining, recombining, and benchmarking sub-diagram motifs from the ZX-calculus. It implements a 7-step pipeline that starts from 22 known quantum algorithms, converts them to ZX-diagrams, mines recurring sub-diagram patterns ("ZX-Webs"), glues them into new candidate circuits, and benchmarks survivors against the originals.

## Running the Pipeline

```bash
# Full pipeline with defaults
uv run python run_pipeline.py

# With custom parameters
uv run python run_pipeline.py --max-subgraph-size 5 --min-frequency 2 --max-candidates 500 --seed 42 --max-qubits 8

# Quiet mode with JSON output
uv run python run_pipeline.py --quiet --output-json results.json
```

Output is saved to `pipeline_report.json` by default.

## Dependencies

Dependencies are declared in `pyproject.toml` and managed by `uv`:

```bash
uv sync
```

Tested with: Qiskit 2.3.0, PyZX 0.9.0, NetworkX 3.6, Python 3.11+

## Architecture

The pipeline has 7 sequential steps, each in its own module under `zx_discovery/`. Data flows linearly through them, orchestrated by `pipeline.py`:

1. **`corpus.py`** — Builds 22 quantum algorithms as Qiskit circuits, exports to OpenQASM 2.0. All circuits are unitary-only (no measurements). New algorithms go in `ALGORITHM_BUILDERS` dict.
2. **`zx_convert.py`** — QASM → PyZX Graph → two simplification levels (Clifford-simplified and fully-reduced). Produces `ZXEntry` dataclass with both snapshots.
3. **`zx_webs.py`** — Core mining step. Converts PyZX graphs to NetworkX, enumerates connected induced subgraphs via BFS, canonicalizes via WL-style hash, and filters by cross-algorithm frequency. Produces `ZXWeb` dataclass.
4. **`composer.py`** — Glues 2-4 ZXWebs together via spider fusion on compatible ports (same-color vertices). Produces `CandidateAlgorithm` with a PyZX Graph.
5. **`extractor.py`** — Attempts `pyzx.extract_circuit()` with fallback strategies. Filters out identity circuits, too-small, and too-wide circuits. ~10-20% survival rate.
6. **`benchmark.py`** — Evaluates on gates/qubit, entanglement entropy, expressibility, depth, and unitary novelty. Expressibility is estimated by evolving random product states and comparing the fidelity distribution to Haar-random. Uses Qiskit statevector simulation (limited to ≤8-10 qubits).
7. **`reporter.py`** — Identifies Pareto-dominant novel candidates (same qubit width, strictly better on ≥1 metric, not worse on any; invalid metrics excluded). Outputs text summary and structured JSON with QASM for outperformers.

`run_pipeline.py` is the CLI entry point with argparse; `pipeline.py` is the programmatic orchestrator.

## Key Design Patterns

- **Phase classification** (`zx_webs.py`): Phases are coarsened into `zero`, `clifford`, `t_like`, `arb` categories for motif matching.
- **Canonical hashing** (`zx_webs.py:_canonical_hash`): Subgraph identity uses a WL-1 hash over sorted adjacency with node/edge labels — fast but not guaranteed collision-free for all graphs.
- **Spider fusion** (`composer.py`): Port compatibility requires matching spider color (Z-Z or X-X); boundary vertices are universally compatible.
- **Gate palette** (`corpus.py`): Only gates supported by both Qiskit QASM export and PyZX QASM import: `h, x, z, s, t, sdg, tdg, cx, cz, ccx, rz, rx, ry, swap`.

## Project Structure

All pipeline modules live in the `zx_discovery/` package with relative imports. `run_pipeline.py` at the repo root is the CLI entry point. Dependencies are declared in `pyproject.toml` and managed by `uv`.
