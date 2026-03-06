# Contributing to zx-motifs

## Development Setup

```bash
git clone https://github.com/splch/zx-motifs.git
cd zx-motifs
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Adding an Algorithm

1. **Scaffold** the boilerplate:
   ```bash
   zx-motifs scaffold algorithm --name my_algo --family my_family
   ```

2. **Edit** the generated file in `src/zx_motifs/algorithms/families/`. The function must be decorated with `@register_algorithm(...)` and return a `QuantumCircuit`:

   ```python
   @register_algorithm(
       name="my_algo",
       family="my_family",
       qubit_range=(2, 8),
       tags=["tag1"],
       description="Short description of the algorithm",
   )
   def make_my_algo(n_qubits=2, **kwargs) -> QuantumCircuit:
       """Docstring explaining the circuit."""
       qc = QuantumCircuit(n_qubits)
       # ... build circuit ...
       return qc
   ```

3. **Validate**:
   ```bash
   zx-motifs validate
   ```

4. **Test**:
   ```bash
   pytest tests/test_registry.py
   ```

5. **Submit a PR**.

## Adding a Motif

1. **Scaffold** the boilerplate:
   ```bash
   zx-motifs scaffold motif --name my_motif
   ```

2. **Edit** the generated JSON file in `src/zx_motifs/motifs/library/`. Key node attributes:
   - `vertex_type`: `"Z"`, `"X"`, or `"H_BOX"`
   - `phase_class`: `"zero"`, `"pi"`, `"pi/2"`, `"pi/4"`, `"arbitrary"`, etc.
   - `edge_type`: `"SIMPLE"` or `"HADAMARD"`

   The graph must be connected.

3. **Validate**:
   ```bash
   zx-motifs validate
   ```

4. **Submit a PR**.

## Running Tests

```bash
pytest
# Or with verbose output, stopping at first failure
pytest -x -v
```

## Code Style

Standard Python conventions. Type hints are encouraged. All algorithm generator functions must include a docstring.
