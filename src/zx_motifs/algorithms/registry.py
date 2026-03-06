"""Backward-compatibility shim -- re-exports from modularised registry.

All public names (REGISTRY, ALGORITHM_FAMILY_MAP, AlgorithmEntry, and every
make_* generator) are available here so that existing imports continue to work.
"""
# Core registry objects
from zx_motifs.algorithms import REGISTRY, ALGORITHM_FAMILY_MAP, AlgorithmEntry  # noqa: F401

# Re-export every make_* generator from family modules
from zx_motifs.algorithms.families.entanglement import *  # noqa: F401, F403
from zx_motifs.algorithms.families.protocol import *  # noqa: F401, F403
from zx_motifs.algorithms.families.oracle import *  # noqa: F401, F403
from zx_motifs.algorithms.families.transform import *  # noqa: F401, F403
from zx_motifs.algorithms.families.variational import *  # noqa: F401, F403
from zx_motifs.algorithms.families.error_correction import *  # noqa: F401, F403
from zx_motifs.algorithms.families.simulation import *  # noqa: F401, F403
from zx_motifs.algorithms.families.arithmetic import *  # noqa: F401, F403
from zx_motifs.algorithms.families.distillation import *  # noqa: F401, F403
from zx_motifs.algorithms.families.machine_learning import *  # noqa: F401, F403
from zx_motifs.algorithms.families.linear_algebra import *  # noqa: F401, F403
from zx_motifs.algorithms.families.cryptography import *  # noqa: F401, F403
from zx_motifs.algorithms.families.sampling import *  # noqa: F401, F403
from zx_motifs.algorithms.families.error_mitigation import *  # noqa: F401, F403
from zx_motifs.algorithms.families.topological import *  # noqa: F401, F403
from zx_motifs.algorithms.families.metrology import *  # noqa: F401, F403
from zx_motifs.algorithms.families.differential_equations import *  # noqa: F401, F403
from zx_motifs.algorithms.families.tda import *  # noqa: F401, F403
from zx_motifs.algorithms.families.communication import *  # noqa: F401, F403
