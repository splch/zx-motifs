# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

zx-motifs converts quantum circuits (Qiskit) into ZX-calculus diagrams (PyZX), discovers recurring structural patterns (motifs) via subgraph isomorphism (NetworkX VF2), and produces fingerprint matrices for algorithm classification. The core pipeline:

```
Qiskit QuantumCircuit → QASM2 → PyZX Circuit/Graph → NetworkX Graph (labeled) → Motif matching → Fingerprint matrix
```

## Commands

```bash
# Install
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file or test
pytest tests/test_registry.py
pytest tests/test_matcher.py -k test_name

# Validate all algorithms and motifs (also runs in CI)
python -m zx_motifs validate

# CLI
zx-motifs list algorithms
zx-motifs list families
zx-motifs list motifs
zx-motifs info <algorithm_name>
zx-motifs info --motif <motif_id>
zx-motifs scaffold algorithm --name <name> --family <family>
zx-motifs scaffold motif --name <name>
```

CI runs `pytest` and `python -m zx_motifs validate` on Python 3.10 and 3.12.

## Architecture

### Algorithm Registry (`src/zx_motifs/algorithms/`)

Algorithms are registered via the `@register_algorithm` decorator in `_registry_core.py`. Each decorated function in `families/` receives `(n_qubits, **kwargs)` and returns a `QuantumCircuit`. Importing `algorithms/__init__.py` triggers all decorator registrations, then rebuilds `ALGORITHM_FAMILY_MAP`. There are 19 family modules with 78 algorithms total.

### Motif Registry (`src/zx_motifs/motifs/`)

Motifs are declarative JSON files in `library/` validated against `schema/motif.schema.json`. The registry auto-discovers and loads all JSON files at import time, converting them to `MotifPattern` objects (which wrap a NetworkX graph). Motif graphs must be connected.

### Pipeline (`src/zx_motifs/pipeline/`)

- **converter.py**: Qiskit → PyZX via QASM2 export. Applies 6 simplification levels (RAW through TELEPORT_REDUCE), producing `ZXSnapshot` objects.
- **featurizer.py**: PyZX graph → labeled NetworkX graph. Phase coarsening (`classify_phase`) maps exact phases to classes: zero, pauli, clifford, t_like, arbitrary. Node attributes: `vertex_type` (Z/X/H_BOX/BOUNDARY), `phase_class`, `degree`. Edge attributes: `edge_type` (SIMPLE/HADAMARD).
- **matcher.py**: VF2 subgraph isomorphism with semantic node/edge matching. Supports phase wildcards (`any`, `any_nonzero`, `any_nonclifford`) for parametric motifs. Has pre-filtering (`can_possibly_match`) and approximate matching.
- **motif_generators.py**: Three discovery strategies — hand-crafted, bottom-up enumeration, and neighborhood extraction. Uses WL hashing for deduplication with VF2 confirmation.
- **fingerprint.py**: `build_corpus()` converts all registry algorithms at all simplification levels. `build_fingerprint_matrix()` counts motif occurrences per algorithm.
- **decomposer.py**: Greedy set-cover decomposition of a ZX graph into non-overlapping motif placements.

### Key Data Flow Conventions

- Corpus dict keys are `(instance_name, simplification_level)` tuples mapping to NetworkX graphs.
- The default analysis level is `"spider_fused"`.
- BOUNDARY nodes are excluded from motif matching by default.

## Adding New Algorithms

Use `zx-motifs scaffold algorithm --name <name> --family <family>`, then implement the circuit generator. The function must be decorated with `@register_algorithm(...)` and return a `QuantumCircuit`. Validate with `zx-motifs validate` and test with `pytest tests/test_registry.py`.

## Adding New Motifs

Use `zx-motifs scaffold motif --name <name>`, then edit the JSON. Key node attributes: `vertex_type` (Z/X/H_BOX), `phase_class` (zero/pi/pi_2/pi_4/arbitrary/any/any_nonzero/any_nonclifford). Edge types: SIMPLE or HADAMARD. The graph must be connected. Validate with `zx-motifs validate`.

## PR Checklist

- `zx-motifs validate` passes
- `pytest` passes
- New algorithms have docstrings
- New motifs have descriptions
