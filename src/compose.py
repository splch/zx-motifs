"""
compose.py — Template-based candidate algorithm composition.

Merges stage4_compose/{templates,composer}.py. Evolutionary/random-walk
strategies removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.mining import ZXWeb, WebLibrary


# ── Templates ───────────────────────────────────────────────────────


@dataclass
class Slot:
    """One position in an algorithm template."""

    role: str
    min_qubits: int | None = None
    max_qubits: int | None = None


@dataclass
class AlgorithmTemplate:
    """Ordered sequence of slots defining an algorithm's high-level shape."""

    name: str
    slots: list[Slot]
    description: str = ""


def load_templates_from_config(
    template_specs: list[list[str]],
) -> list[AlgorithmTemplate]:
    """Convert config.yaml template specs to AlgorithmTemplate objects."""
    raise NotImplementedError


SEARCH_TEMPLATE = AlgorithmTemplate(
    name="search",
    slots=[
        Slot(role="state_prep"),
        Slot(role="oracle"),
        Slot(role="amplify"),
        Slot(role="readout"),
    ],
    description="Grover-like search: prepare, oracle + amplification, measure.",
)

SIMULATION_TEMPLATE = AlgorithmTemplate(
    name="simulation",
    slots=[
        Slot(role="state_prep"),
        Slot(role="evolve"),
        Slot(role="evolve"),
        Slot(role="readout"),
    ],
    description="Hamiltonian simulation: prepare, evolve steps, measure.",
)

QEC_TEMPLATE = AlgorithmTemplate(
    name="error_correction",
    slots=[
        Slot(role="encode"),
        Slot(role="entangle"),
        Slot(role="decode"),
    ],
    description="Error correction: encode, syndrome extraction, decode.",
)


# ── Composition ─────────────────────────────────────────────────────


@dataclass
class CompositionRecipe:
    """Records exactly how a candidate was assembled."""

    candidate_id: str
    template_name: str | None
    web_sequence: list[str]
    connections: list[tuple[int, int]] = field(default_factory=list)


def compose_from_template(
    template: AlgorithmTemplate,
    library: WebLibrary,
    max_qubits: int,
    enforce_flow: bool = True,
) -> list[tuple[Any, CompositionRecipe]]:
    """Generate candidate diagrams by filling a template's slots."""
    raise NotImplementedError


def connect_webs(
    web_a: ZXWeb,
    web_b: ZXWeb,
    boundary_map: list[tuple[int, int]],
) -> Any:  # pyzx.Graph
    """Connect two ZXWebs at specified boundary pairs."""
    raise NotImplementedError


def validate_candidate(graph: Any, expected_qubits: int) -> bool:
    """Quick sanity check on a composed candidate."""
    raise NotImplementedError
