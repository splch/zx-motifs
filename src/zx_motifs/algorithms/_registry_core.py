"""Core registry infrastructure: AlgorithmEntry dataclass and @register_algorithm decorator."""
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AlgorithmEntry:
    name: str
    family: str
    generator: Callable  # (n_qubits, **kwargs) -> QuantumCircuit
    qubit_range: tuple  # (min_qubits, max_qubits)
    tags: list = field(default_factory=list)
    description: str = ""


_REGISTRY: list[AlgorithmEntry] = []


def register_algorithm(name, family, qubit_range, tags=None, description=""):
    """Decorator that registers a generator function in the algorithm registry."""
    def decorator(fn):
        entry = AlgorithmEntry(
            name=name,
            family=family,
            generator=fn,
            qubit_range=qubit_range,
            tags=tags if tags is not None else [],
            description=description,
        )
        _REGISTRY.append(entry)
        return fn
    return decorator


REGISTRY = _REGISTRY
ALGORITHM_FAMILY_MAP = {}


def _rebuild_family_map():
    """Rebuild the family map from current registry contents."""
    ALGORITHM_FAMILY_MAP.clear()
    ALGORITHM_FAMILY_MAP.update({entry.name: entry.family for entry in REGISTRY})
