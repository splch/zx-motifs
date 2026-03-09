"""
mining.py — Sub-diagram discovery: webs, fingerprinting, mining, and library.

Merges stage3_mining/{web,mining,fingerprint,library}.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── ZXWeb ───────────────────────────────────────────────────────────


@dataclass
class Boundary:
    """Describes one open leg of a ZXWeb."""

    index: int
    spider_type: str
    phase: float | None
    edge_type: str


@dataclass
class ZXWeb:
    """A reusable sub-diagram fragment from one or more quantum algorithms."""

    web_id: str
    graph: Any  # pyzx.Graph
    boundaries: list[Boundary]
    spider_count: int
    sources: list[str] = field(default_factory=list)
    support: int = 0
    role: str | None = None
    phase_class: str = "mixed"

    def n_inputs(self) -> int:
        """Number of input-side boundaries."""
        raise NotImplementedError

    def n_outputs(self) -> int:
        """Number of output-side boundaries."""
        raise NotImplementedError

    def is_compatible(self, other: "ZXWeb") -> bool:
        """Check whether this web's outputs can connect to other's inputs."""
        raise NotImplementedError

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dictionary."""
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "ZXWeb":
        """Reconstruct a ZXWeb from a serialized dictionary."""
        raise NotImplementedError


# ── Fingerprinting ──────────────────────────────────────────────────


@dataclass
class DiagramFingerprint:
    """Compact feature vector summarizing a ZX-diagram's structure."""

    n_z_spiders: int
    n_x_spiders: int
    n_hadamard_edges: int
    n_simple_edges: int
    degree_histogram: dict[int, int]
    phase_histogram: dict[str, int]


def compute_fingerprint(graph: Any) -> DiagramFingerprint:
    """Compute the fingerprint of a ZX-diagram."""
    raise NotImplementedError


def fingerprints_compatible(
    parent: DiagramFingerprint,
    sub: DiagramFingerprint,
) -> bool:
    """Quick check whether sub could be a sub-diagram of parent."""
    raise NotImplementedError


# ── Mining ──────────────────────────────────────────────────────────


def mine_webs(
    diagrams: list[tuple[str, Any]],  # (algorithm_key, pyzx.Graph)
    min_support: int,
    min_spiders: int,
    max_spiders: int,
    phase_abstraction: str,
) -> list[ZXWeb]:
    """Discover frequent sub-diagrams using gSpan.

    Converts ZX-diagrams to gSpan's input format, runs gSpan, and
    converts results back to ZXWebs with boundaries.
    """
    raise NotImplementedError


# ── Library ─────────────────────────────────────────────────────────


class WebLibrary:
    """Manages a collection of ZXWebs on disk."""

    def __init__(self, directory: str | Path) -> None:
        self._dir = Path(directory)
        self._index: dict[str, dict] = {}

    def add(self, web: ZXWeb) -> None:
        """Add a ZXWeb to the library."""
        raise NotImplementedError

    def get(self, web_id: str) -> ZXWeb:
        """Load a single ZXWeb by its ID."""
        raise NotImplementedError

    def search(
        self,
        min_boundaries: int | None = None,
        max_boundaries: int | None = None,
        role: str | None = None,
        phase_class: str | None = None,
        min_support: int | None = None,
    ) -> list[ZXWeb]:
        """Search the library with optional filters."""
        raise NotImplementedError

    def all_webs(self) -> list[ZXWeb]:
        """Load and return every web in the library."""
        raise NotImplementedError

    def save_index(self) -> None:
        """Write the index to library.json."""
        raise NotImplementedError

    def load_index(self) -> None:
        """Read the index from library.json."""
        raise NotImplementedError
