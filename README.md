# zx-motifs

ZX-calculus motif discovery pipeline for quantum algorithm analysis.

## Overview

zx-motifs converts quantum circuits into ZX-calculus diagrams, discovers recurring structural patterns (motifs), and uses them for algorithm classification and new circuit design. The pipeline starts from Qiskit circuits, converts them to PyZX diagrams, extracts graph features via NetworkX, matches against a library of known motifs, and produces fingerprint matrices for downstream classification and discovery tasks.

## Quick Start

```bash
pip install -e .
# Or with notebook support
pip install -e ".[notebooks]"
```

## CLI

```bash
zx-motifs list algorithms
zx-motifs list families
zx-motifs list motifs
zx-motifs info bell_state
zx-motifs info --motif cx_pair
zx-motifs validate
```

Use `zx-motifs scaffold` to generate templates for new contributions:

```bash
zx-motifs scaffold algorithm --name my_algo --family my_family
zx-motifs scaffold motif --name my_motif
```

## Project Structure

| Directory | Description |
|-----------|-------------|
| `src/zx_motifs/algorithms/families/` | 78 quantum algorithm circuit generators across 19 families |
| `src/zx_motifs/motifs/library/` | Declarative JSON motif definitions (15 hand-crafted) |
| `src/zx_motifs/pipeline/` | Core analysis pipeline (conversion, featurization, matching, fingerprinting, decomposition) |
| `notebooks/` | Interactive analysis notebooks |

## Architecture

```
Qiskit circuits -> PyZX diagrams -> NetworkX graphs -> Motif matching -> Fingerprint matrix -> Classification / Discovery
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, how to add algorithms and motifs, and the scaffold commands that generate boilerplate for you.
