"""Quantum algorithm circuit generators."""

from ._registry_core import AlgorithmEntry, REGISTRY, ALGORITHM_FAMILY_MAP
from . import families  # triggers decorator registration
from ._registry_core import _rebuild_family_map as _rebuild

# Build the family map after all decorators have fired
_rebuild()
