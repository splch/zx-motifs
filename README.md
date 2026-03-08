# ZX-Web Discovery Pipeline

A Python framework for discovering novel quantum algorithms by mining, recombining, and benchmarking sub-diagram motifs from the ZX-calculus.

## Overview

This pipeline implements a 7-step process that starts from known quantum algorithms and attempts to synthesize new ones by finding and recombining structural patterns at the ZX-diagram level.

```
┌──────────────────┐    ┌─────────────────┐    ┌────────────────┐
│  Step 1: Corpus  │───▶│  Step 2: ZX     │───▶│  Step 3: Mine  │
│  22 Qiskit algos │    │  Convert+Reduce │    │  ZX-Web motifs │
└──────────────────┘    └─────────────────┘    └───────┬────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌────────▼────────┐
│  Step 6: Bench  │◀───│  Step 5: Filter │◀───│  Step 4: Compose│
│  mark suite     │    │  extractable    │    │  new candidates │
└────────┬────────┘    └─────────────────┘    └─────────────────┘
         │
┌────────▼────────┐
│  Step 7: Report │
│  outperformers  │
└─────────────────┘
```

## Requirements

Dependencies are declared in `pyproject.toml`. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Tested with: Qiskit 2.3.0, PyZX 0.9.0, NetworkX 3.6, Python 3.11+

## Quick Start

```bash
uv run python run_pipeline.py
```

With custom parameters:
```bash
uv run python run_pipeline.py \
  --max-subgraph-size 5 \
  --min-frequency 2 \
  --max-candidates 500 \
  --web-combo-size 3 \
  --seed 42 \
  --max-qubits 8 \
  --output-json results.json
```

Or programmatically:
```python
from zx_discovery.pipeline import run_pipeline
report = run_pipeline(max_candidates=300, seed=42)
print(f"Found {report.n_outperformers} outperformers")
```

## Pipeline Steps in Detail

### Step 1 — Algorithm Corpus (`corpus.py`)

Implements 22 quantum algorithms in Qiskit using only gates compatible with both Qiskit's QASM exporter and PyZX's importer:

| Category | Algorithms |
|---|---|
| Entanglement | Bell states (2,3-pair), GHZ (4,5-qubit), W-state (3,4-qubit) |
| Oracles | Deutsch-Jozsa (3,4), Bernstein-Vazirani |
| Transforms | QFT (3,4), Inverse QFT (4) |
| Search | Grover oracle (3q), Grover diffusion (3,4) |
| Estimation | Phase estimation core (3 counting qubits) |
| Comparison | SWAP test |
| Variational | Hardware-efficient ansatz (4q, depth-2) |
| Arithmetic | Quantum adder (2-bit ripple carry) |
| Primitives | Toffoli decomposition, Quantum walk step, Teleportation (unitary part) |

Each circuit is exported to OpenQASM 2.0 via `qiskit.qasm2.dumps()`.

### Step 2 — ZX Conversion (`zx_convert.py`)

Converts each QASM circuit to a PyZX `Graph` (ZX-diagram) and applies two levels of simplification:

- **Clifford simplification** — fuses adjacent Clifford spiders, preserving circuit-like structure
- **Full reduction** — applies `full_reduce()` for maximum compression using local complementation and pivoting rules

The fully-reduced diagrams (typically 3-5× smaller) are the primary input for motif mining.

### Step 3 — ZX-Web Mining (`zx_webs.py`)

**Core concept:** A *ZX-Web* is a small, recurring labelled sub-diagram found across multiple algorithms' ZX-diagrams.

The mining process:
1. Convert each PyZX graph to a NetworkX graph with typed nodes (Z/X/Boundary + phase class) and typed edges (Simple/Hadamard)
2. Enumerate all connected induced subgraphs of size 2-k via bounded BFS
3. Canonicalise each subgraph using a colour-aware Weisfeiler-Leman hash
4. Count how many distinct source algorithms contain each motif
5. Keep motifs appearing in ≥ `min_frequency` algorithms

Phase classes: `zero`, `clifford` (multiples of π/2), `t_like` (multiples of π/4), `arb` (other)

### Step 4 — Candidate Composition (`composer.py`)

Constructs new ZX-diagrams by gluing 2-4 ZX-Webs together:

- **Port identification** — vertices on the boundary of a subgraph with external neighbours
- **Spider fusion** — compatible ports (same spider colour, or boundary) are merged per the ZX-calculus spider fusion rule, adding their phases
- **I/O fixup** — dangling ports receive fresh boundary vertices
- **Diversity controls** — max web reuse, distinct source algorithms required

### Step 5 — Circuit Extraction Filter (`extractor.py`)

Not every composed ZX-diagram yields a valid quantum circuit. This step:

1. Attempts `pyzx.extract_circuit()` with graph-like preprocessing
2. Falls back to lighter simplification strategies
3. Filters out trivial circuits (identity, too few gates, too many qubits)
4. Exports survivors to QASM

Typical survival rate: ~10-20% of candidates.

### Step 6 — Benchmarking (`benchmark.py`)

Evaluates every surviving candidate and every corpus algorithm on:

| Metric | Direction | Method |
|---|---|---|
| Gates per qubit | Lower ↓ | Gate count / qubit width |
| Entanglement entropy | Higher ↑ | Average bipartite von Neumann entropy of output state |
| Expressibility | Higher ↑ | Fidelity distribution of random product-state inputs vs Haar random |
| Circuit depth | Lower ↓ | Total and 2-qubit critical path |
| Novelty | Boolean | Unitary not equivalent (up to global phase) to any corpus circuit |

### Step 7 — Reporting (`reporter.py`)

Flags any candidate that is:
1. **Novel** — not unitarily equivalent to a known algorithm
2. **Pareto-dominant** — strictly better on ≥1 metric and not worse on any, compared to at least one corpus circuit of the same width

Outputs human-readable text and structured JSON with full QASM for every outperformer.

## Architecture

```
zx_discovery/
├── __init__.py          # Package metadata
├── corpus.py            # Step 1: algorithm implementations
├── zx_convert.py        # Step 2: QASM → ZX-diagram → simplified
├── zx_webs.py           # Step 3: sub-diagram mining
├── composer.py          # Step 4: ZX-Web gluing
├── extractor.py         # Step 5: circuit extraction filter
├── benchmark.py         # Step 6: application-suite benchmarks
├── reporter.py          # Step 7: outperformer detection + reporting
└── pipeline.py          # Orchestrator (runs Steps 1-7)
run_pipeline.py          # CLI entry point
```

## Extending the Pipeline

**Add algorithms:** Add builder functions to `corpus.py` and register them in `ALGORITHM_BUILDERS`.

**Adjust mining:** Increase `max_subgraph_size` (up to ~7) for richer motifs at the cost of exponential enumeration time. Lower `min_frequency` to capture rarer patterns.

**Tune composition:** Increase `max_candidates` and vary `seed` to explore more of the combinatorial space. Set `web_combo_size=4` for larger composed circuits.

**Custom benchmarks:** Add task-specific benchmarks in `benchmark.py` (e.g., VQE energy estimation, QAOA MaxCut approximation ratio).

## Interpreting Results

With sufficient candidates and web combo size (e.g. `--max-candidates 1000 --web-combo-size 4 --max-subgraph-size 6`), the pipeline can discover novel circuits that Pareto-dominate known algorithms. The framework provides:

- **Infrastructure** for systematic exploration
- **Quantitative comparison** between candidates and known algorithms
- **Reproducible** runs via seed control
- **Scalable** to larger corpora and more candidates

For broader exploration, try `max_candidates=5000+`, multiple seeds, and `max_subgraph_size=6`.

## Key Dependencies

| Package | Role |
|---|---|
| **Qiskit** (≥2.3) | Circuit construction, QASM export, statevector simulation |
| **PyZX** (≥0.9) | ZX-calculus: diagram conversion, simplification, circuit extraction |
| **NetworkX** (≥3.6) | Subgraph enumeration, canonical hashing, graph composition |
| **NumPy** | Linear algebra for entanglement entropy, unitary comparison |

## References

- Kissinger & van de Wetering, "PyZX: Large Scale Automated Diagrammatic Reasoning", QPL 2019
- van de Wetering, "ZX-calculus for the working quantum computer scientist", 2020
- Fischbach et al., "Exhaustive Search for Quantum Circuit Optimization using ZX Calculus", 2025
- Coecke & Kissinger, *Picturing Quantum Processes*, Cambridge University Press, 2017
