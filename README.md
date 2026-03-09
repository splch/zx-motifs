# ZX-Webs

A seven-stage pipeline for programmatic quantum algorithm discovery via
ZX-calculus sub-diagram mining and composition.

## Pipeline Overview

```
Stage 1          Stage 2          Stage 3           Stage 4
Qiskit    ──▶   PyZX      ──▶   Sub-Diagram  ──▶  Compose
Corpus          ZX-Diagrams      Mining             Candidates
                                 (ZX-Webs)

Stage 5          Stage 6          Stage 7
Extract   ──▶   Benchmark  ──▶   Report
Filter          vs Baselines      Novel Algorithms
```

## Quick Start

```bash
# Install in editable mode
pip install -e ".[dev]"

# Run the full pipeline (once implemented)
python -m src.pipeline --config config.yaml

# Run a single stage
python -m src.pipeline --stage 1 --config config.yaml

# Run tests
pytest tests/
```

## Project Structure

```
zx-webs/
├── config.yaml          # Pipeline configuration
├── pyproject.toml       # Project metadata & dependencies
├── src/
│   ├── __init__.py
│   ├── pipeline.py      # End-to-end orchestrator & CLI
│   ├── corpus.py        # Stage 1: algorithm builders, registry, QASM export
│   ├── zx.py            # Stage 2: ZX conversion, simplification, storage
│   ├── mining.py        # Stage 3: fingerprinting, gSpan mining, web library
│   ├── compose.py       # Stage 4: template-based composition
│   ├── extract.py       # Stage 5: gFlow detection, circuit extraction, filtering
│   ├── benchmark.py     # Stage 6: metrics, statevector simulation, comparison
│   └── report.py        # Stage 7: novelty assessment, provenance, export
├── tests/
│   └── test_pipeline.py # One test class per module + integration
└── data/                # Pipeline artifacts (gitignored)
    ├── corpus/          #   QASM files
    ├── diagrams/        #   Serialized ZX-diagrams
    ├── webs/            #   Mined ZX-Web library
    ├── candidates/      #   Composed candidate diagrams
    └── results/         #   Benchmarking outputs
```

## Dependencies

- **qiskit** – Circuit construction and transpilation
- **pyzx** – ZX-calculus manipulation
- **networkx** – Graph operations and isomorphism
- **numpy** – Numerical computation
- **pyyaml** – Configuration parsing
- **gspan-mining** – Frequent subgraph mining

Optional:
- **qiskit-aer** – Statevector simulation
