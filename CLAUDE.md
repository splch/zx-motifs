# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev]"

# Tests
.venv/bin/python -m pytest tests/ -v          # all tests
.venv/bin/python -m pytest tests/test_composer.py -v  # one file
.venv/bin/python -m pytest tests/test_featurizer.py::TestClassifyPhase -v  # one class
.venv/bin/python -m pytest tests/test_converter.py::TestQiskitToZx::test_roundtrip_semantics -v  # one test

# Notebooks
uv pip install -e ".[notebooks]"
jupyter notebook notebooks/
```

No linter or type checker is configured.

## Architecture

Pipeline converts quantum circuits into ZX diagrams, finds recurring subgraph motifs, and provides composable building blocks:

```
registry.py ──► converter.py ──► featurizer.py ──► matcher.py ──► catalog.py
(Qiskit QC)     (PyZX Graph)     (NetworkX Graph)   (VF2 matches)  (JSON)
                                                  ◄── motif_generators.py
                                                       (candidate patterns)
                     converter.py ──► composer.py
                     (PyZX Graph)     (ZXBox composition + validation)
```

**Data types that flow through the pipeline:**
- `QuantumCircuit` (Qiskit) → `zx.Circuit` / `Graph` (PyZX) → `nx.Graph` (NetworkX)
- `ZXSnapshot`: a PyZX graph at a specific simplification level with metadata
- `MotifPattern`: a small NetworkX graph used as a search template; accumulates `MotifMatch` occurrences
- `ZXBox`: a PyZX graph with explicit `left_boundary` / `right_boundary` vertex ID lists and `BoundarySpec`

**Simplification levels** (enum `SimplificationLevel` in converter.py): RAW → SPIDER_FUSED → INTERIOR_CLIFFORD → CLIFFORD_SIMP → FULL_REDUCE → TELEPORT_REDUCE. Default for motif detection is SPIDER_FUSED. Phase classification in featurizer.py groups phases into: zero, pauli (pi), clifford (pi/2), t_like (pi/4), arbitrary.

## PyZX API Pitfalls

These are the non-obvious behaviors of PyZX 0.9.x that have caused bugs in this codebase:

- **`g.neighbors(v)` returns `dict_keys`**, not a list. Always wrap: `list(g.neighbors(v))`.
- **`g.add_vertex()` auto-assigns IDs.** There is no `index=` parameter in 0.9.x. Capture the return value: `new_id = g.add_vertex(ty=..., phase=...)`.
- **`teleport_reduce()` returns a graph** unlike all other simplifiers which only mutate in-place. Must reassign: `g = zx.simplify.teleport_reduce(g)`.
- **`spider_simp` / `interior_clifford_simp` preserve boundary vertices.** `clifford_simp` and `full_reduce` may destroy them. The composer validates this and raises `BoundaryDestroyedError`.
- **`Graph.compose(other)` mutates self** and requires each boundary vertex to have exactly one neighbor. The composer falls back to manual vertex remapping (`_compose_manual`) when this fails.
- **Qiskit QASM export:** use `qiskit.qasm2.dumps(qc)`, not the removed `qc.qasm()`.

## Key Conventions

- Motif matching uses coarsened phase classes (not exact phases) for structural similarity. Semantic equivalence is confirmed separately via `zx.compare_tensors()`.
- Bottom-up motif discovery uses MD5 hash for fast deduplication, then VF2 isomorphism to confirm (guards against collisions).
- The matcher excludes BOUNDARY-type nodes from host graphs before searching, so motifs describe interior structure only.
- `ZXBox` boundary lists store PyZX vertex IDs. These IDs must survive any simplification applied to the box; `simplify_box()` enforces this.
