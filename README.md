# ZX-Motifs

A pipeline for discovering recurring structural patterns in quantum algorithms using ZX-calculus diagrams.

Converts quantum circuits from Qiskit into ZX diagrams via PyZX, detects recurring subgraph motifs across algorithm families, and provides a composable box library for reconstructing and exploring novel algorithm compositions.

## Motivation

Gate-sequence analysis misses the structural relationships that ZX-calculus makes explicit. In ZX diagrams, commutation relations are absorbed into the topology, so subgraph isomorphism on ZX graphs gives you commutation-aware pattern detection for free. This project uses that property to find recurring motifs across quantum algorithm families and formalize them into composable building blocks.

## Pipeline Overview

```
Qiskit Circuits ──► ZX Diagrams ──► Labeled Graphs ──► Motif Detection ──► Box Library
                    (PyZX)          (NetworkX)          (VF2 matching)      (composition)
```

**Stage 1 -- Convert.** Quantum circuits are exported as QASM and imported into PyZX. Each circuit is snapshotted at six simplification levels (raw, spider-fused, interior Clifford, full Clifford, full reduce, teleport reduce) to study how ZX rewrites reshape algorithmic structure.

**Stage 2 -- Featurize.** PyZX graphs are converted to NetworkX graphs with labeled nodes (vertex type, phase class) and edges (simple vs. Hadamard). Phase classification groups angles into five classes: zero, Pauli, Clifford, T-like, and arbitrary.

**Stage 3 -- Detect.** Three motif-finding strategies run against the corpus:
- *Top-down*: Hand-crafted patterns from known primitives (phase gadgets, CX pairs, ZZ interactions)
- *Bottom-up*: Recursive enumeration of small connected subgraphs, deduplicated by canonical hash with VF2 confirmation
- *Hybrid*: Neighborhood extraction around high-interest vertices (non-Clifford phases, high degree, color boundaries)

**Stage 4 -- Compose.** Confirmed motifs are wrapped as ZX boxes with explicit boundary contracts (input/output wire counts and types). Boxes compose sequentially (`;`) and in parallel (`tensor`), and simplification validates that boundaries survive.

## Included Algorithms

| Family | Algorithms |
|--------|-----------|
| Entanglement | Bell state, GHZ state, W state, Cluster state |
| Protocol | Teleportation, Superdense coding, Entanglement swapping |
| Transform | QFT, Phase Estimation |
| Oracle | Grover, Bernstein-Vazirani, Deutsch-Jozsa, Simon, Quantum counting |
| Variational | QAOA MaxCut, VQE UCCSD fragment, Hardware-efficient ansatz |
| Error Correction | Bit-flip code, Phase-flip code, Steane code, Shor code |
| Simulation | Trotter Ising, Trotter Heisenberg, Quantum walk |
| Arithmetic | Ripple-carry adder, QFT adder |
| Machine Learning | Quantum kernel, Data re-uploading |

## Setup

Requires Python >= 3.10.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Verify the installation:

```python
from zx_motifs.algorithms.registry import REGISTRY
from zx_motifs.pipeline.converter import convert_at_all_levels

qc = REGISTRY[0].generator(n_qubits=2)
snapshots = convert_at_all_levels(qc, "bell_q2")
print(f"{len(snapshots)} snapshots across simplification levels")
```

## Tests

```bash
pytest tests/ -v
```

Tests covering phase classification, conversion roundtrips, composition semantics, boundary preservation, simplification correctness, and parametrized validation of all algorithm generators.

## Usage

### Convert an algorithm to ZX at multiple simplification levels

```python
from zx_motifs.algorithms.registry import make_grover
from zx_motifs.pipeline.converter import convert_at_all_levels

qc = make_grover(n_qubits=3, marked_state=0, n_iterations=1)
snapshots = convert_at_all_levels(qc, "grover_q3")

for snap in snapshots:
    print(f"  {snap.level.value:20s}  V={snap.num_vertices:3d}  E={snap.num_edges:3d}")
```

### Search for motifs across a corpus

```python
from zx_motifs.pipeline.featurizer import pyzx_to_networkx
from zx_motifs.pipeline.matcher import find_motif_across_corpus
from zx_motifs.pipeline.motif_generators import HANDCRAFTED_MOTIFS

# Build corpus (dict of (name, level) -> NetworkX graph)
nx_graphs = {}
for snap in snapshots:
    nxg = pyzx_to_networkx(snap.graph, coarsen_phases=True)
    nx_graphs[(snap.algorithm_name, snap.level.value)] = nxg

# Search
for motif in HANDCRAFTED_MOTIFS:
    find_motif_across_corpus(motif, nx_graphs, target_level="spider_fused")
    print(f"  {motif.motif_id}: {len(motif.occurrences)} matches")
```

### Compose boxes and validate

```python
import pyzx as zx
from qiskit import QuantumCircuit
from zx_motifs.pipeline.composer import make_box_from_circuit, compose_sequential

# Build sub-circuit boxes
qc_h = QuantumCircuit(2)
qc_h.h(0)
qc_cx = QuantumCircuit(2)
qc_cx.cx(0, 1)

box_h = make_box_from_circuit("hadamard", qc_h)
box_cx = make_box_from_circuit("cnot", qc_cx)
composed = compose_sequential(box_h, box_cx)

# Validate against a monolithic Bell circuit
qc_bell = QuantumCircuit(2)
qc_bell.h(0)
qc_bell.cx(0, 1)
box_bell = make_box_from_circuit("bell", qc_bell)

assert zx.compare_tensors(composed.graph, box_bell.graph)
```

## Notebooks

Four Jupyter notebooks walk through the pipeline interactively:

| Notebook | Purpose |
|----------|---------|
| `01_conversion_explorer` | Batch conversion, simplification cascade visualization, feature clustering |
| `02_motif_detection` | Hand-crafted and bottom-up motif scans, cross-algorithm heatmap |
| `03_box_library` | Grover and QFT decomposition into boxes, reconstruction validation |
| `04_novel_compositions` | Box substitution experiments, systematic exploration sweep |

```bash
uv pip install -e ".[notebooks]"
jupyter notebook notebooks/
```

## Project Structure

```
src/zx_motifs/
  algorithms/
    registry.py          # Algorithm generators, family map
  pipeline/
    converter.py         # Qiskit -> ZX at 6 simplification levels
    featurizer.py        # ZX -> labeled NetworkX graph
    matcher.py           # VF2 subgraph isomorphism with semantic matching
    motif_generators.py  # Motif-finding strategies
    catalog.py           # JSON-serialized motif catalog
    composer.py          # ZX box composition (sequential, parallel)
notebooks/               # Exploration notebooks
tests/                   # Tests
```

## Key Design Decisions

**Simplification level for motif detection.** Spider-fused is the default. Raw preserves compiler artifacts, not algorithmic structure. Full-reduce destroys too much. Spider fusion absorbs trivial identities while keeping compositional structure intact.

**Phase coarsening.** Motif matching uses five phase classes (zero / Pauli / Clifford / T-like / arbitrary) rather than exact phase values. This finds structural motifs that share topology but differ in specific angles. Semantic equivalence is always confirmed separately via `compare_tensors`.

**Hash-then-verify for bottom-up discovery.** Subgraph enumeration uses a fast MD5-based canonical hash for deduplication, followed by VF2 isomorphism confirmation to guard against hash collisions.

**Boundary-safe simplification.** `simplify_box()` validates that all boundary vertices survive after simplification. `interior_clifford_simp` and `spider_simp` are safe; `clifford_simp` and `full_reduce` raise `BoundaryDestroyedError` if boundaries are lost.

## Limitations

- **Scalability.** VF2 subgraph isomorphism is exponential worst-case. Practical for ZX graphs under ~500 vertices; beyond that, restrict pattern size or use frequent subgraph mining (gSpan).
- **Rewrite equivalence.** Two structurally different subgraphs can be ZX-equivalent. The matcher finds literal structural matches, not semantic equivalence classes.
- **Clifford+T bias.** PyZX is strongest on Clifford+T circuits. Variational algorithms with arbitrary rotations produce "arbitrary" phase classes that match too broadly.
- **Circuit extraction.** Aggressive simplification of composed boxes may lose generalized flow, making circuit extraction back from ZX hard. Use `interior_clifford` when you need to extract circuits.

## Dependencies

- [PyZX](https://github.com/zxcalc/pyzx) >= 0.8.0 -- ZX-calculus graph rewriting and simplification
- [Qiskit](https://github.com/Qiskit/qiskit) >= 1.0 -- Quantum circuit construction and QASM export
- [NetworkX](https://networkx.org/) >= 3.0 -- Graph data structures and VF2 isomorphism
- matplotlib, pandas, tqdm -- Visualization and data handling

## References

- Coecke & Kissinger, *Picturing Quantum Processes* (Cambridge, 2017) -- Categorical foundations
- Kissinger & van de Wetering, *Reducing T-count with the ZX-calculus* (2020) -- PyZX optimization
- van de Wetering, *ZX-calculus for the working quantum computer scientist* (2020) -- Tutorial introduction
- Peham et al., *ZX-DB: Graph database approach to ZX rewriting* (2024) -- Pattern matching on ZX diagrams
