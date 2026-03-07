"""
Compose-simplify-rediscover loop for emergent motif generation.

Composes pairs of existing motifs, simplifies the result via ZX rewrite
rules, and runs motif discovery on the simplified diagram. Novel substructures
(those with WL hashes not in the current library) are compositional emergents.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .composer import (
    ZXBox,
    compose_parallel,
    compose_sequential,
    make_box_from_circuit,
    simplify_box,
    BoundaryDestroyedError,
)
from .featurizer import pyzx_to_networkx
from .matcher import MotifPattern
from .motif_generators import wl_hash, is_isomorphic, enumerate_connected_subgraphs

logger = logging.getLogger(__name__)


@dataclass
class EmergentMotif:
    """A novel motif discovered through composition."""

    motif: MotifPattern
    provenance: str  # e.g. "compose(m1, m2) at level spider_fusion"
    wl_hash: str
    parent_motifs: list[str] = field(default_factory=list)


def _is_novel(
    candidate_hash: str,
    candidate_graph,
    known_hashes: set[str],
    known_motifs: list[MotifPattern],
) -> bool:
    """Check if a candidate graph is structurally novel."""
    if candidate_hash not in known_hashes:
        return True
    # Hash collision: verify with VF2
    for km in known_motifs:
        km_hash = wl_hash(km.graph)
        if km_hash == candidate_hash and is_isomorphic(km.graph, candidate_graph):
            return False
    return True


def compose_and_discover(
    motif_a: MotifPattern,
    motif_b: MotifPattern,
    box_a: ZXBox,
    box_b: ZXBox,
    known_hashes: set[str],
    known_motifs: list[MotifPattern],
    simplification_levels: list[str] | None = None,
    min_size: int = 3,
    max_size: int = 7,
    max_subgraphs: int = 200,
) -> list[EmergentMotif]:
    """Compose two motif boxes and discover novel substructures.

    Tries both sequential and parallel composition, simplifies at each level,
    and extracts novel motifs from the simplified result.
    """
    if simplification_levels is None:
        simplification_levels = ["spider_fusion", "interior_clifford"]

    emergents: list[EmergentMotif] = []
    modes = []

    # Try sequential composition if boundaries match
    if box_a.n_outputs == box_b.n_inputs:
        try:
            composed = compose_sequential(box_a, box_b)
            modes.append(("sequential", composed))
        except (ValueError, Exception):
            pass

    # Try parallel composition
    try:
        composed = compose_parallel(box_a, box_b)
        modes.append(("parallel", composed))
    except Exception:
        pass

    for mode_name, composed_box in modes:
        for level in simplification_levels:
            try:
                simplified = simplify_box(composed_box, level=level)
            except BoundaryDestroyedError:
                continue
            except Exception:
                continue

            # Convert to NetworkX and enumerate subgraphs
            nxg = pyzx_to_networkx(simplified.graph, coarsen_phases=True)

            # Remove boundary nodes for motif discovery
            boundary_nodes = [
                n for n, d in nxg.nodes(data=True)
                if d.get("is_boundary", False)
            ]
            interior = nxg.copy()
            interior.remove_nodes_from(boundary_nodes)

            if interior.number_of_nodes() < min_size:
                continue

            subgraphs = enumerate_connected_subgraphs(
                interior,
                min_size=min_size,
                max_size=max_size,
                max_subgraphs=max_subgraphs,
                exclude_boundary=False,  # already removed
            )

            for sg in subgraphs:
                h = wl_hash(sg)
                if _is_novel(h, sg, known_hashes, known_motifs):
                    motif_id = f"emergent_{motif_a.motif_id}_{motif_b.motif_id}_{h[:8]}"
                    em = EmergentMotif(
                        motif=MotifPattern(
                            motif_id=motif_id,
                            graph=sg,
                            source="emergent",
                            description=(
                                f"Emergent from {mode_name}("
                                f"{motif_a.motif_id}, {motif_b.motif_id}) "
                                f"at {level}"
                            ),
                        ),
                        provenance=f"{mode_name}({motif_a.motif_id}, {motif_b.motif_id}) at {level}",
                        wl_hash=h,
                        parent_motifs=[motif_a.motif_id, motif_b.motif_id],
                    )
                    emergents.append(em)
                    known_hashes.add(h)

    return emergents


def run_discovery_loop(
    motifs: list[MotifPattern],
    boxes: dict[str, ZXBox],
    max_library_size: int = 200,
    max_rounds: int = 3,
    min_size: int = 3,
    max_size: int = 7,
) -> list[EmergentMotif]:
    """Run the compose-simplify-rediscover loop until convergence.

    Args:
        motifs: Current motif library.
        boxes: Dict mapping motif_id to ZXBox (pre-built from circuits).
        max_library_size: Cap on total library size.
        max_rounds: Maximum discovery rounds.
        min_size: Minimum subgraph size for discovered motifs.
        max_size: Maximum subgraph size.

    Returns:
        List of all discovered emergent motifs.
    """
    known_hashes = {wl_hash(m.graph) for m in motifs}
    all_emergents: list[EmergentMotif] = []

    for round_num in range(max_rounds):
        round_emergents: list[EmergentMotif] = []

        # Current library = original + discovered so far
        current_motifs = list(motifs) + [e.motif for e in all_emergents]
        if len(current_motifs) >= max_library_size:
            logger.info("Library cap reached at %d motifs", len(current_motifs))
            break

        # Compose all pairs with available boxes
        motif_ids_with_boxes = [m.motif_id for m in current_motifs if m.motif_id in boxes]

        for i, mid_a in enumerate(motif_ids_with_boxes):
            for mid_b in motif_ids_with_boxes[i:]:
                ma = next(m for m in current_motifs if m.motif_id == mid_a)
                mb = next(m for m in current_motifs if m.motif_id == mid_b)
                emergents = compose_and_discover(
                    ma, mb, boxes[mid_a], boxes[mid_b],
                    known_hashes, current_motifs,
                    min_size=min_size, max_size=max_size,
                )
                round_emergents.extend(emergents)

                if len(current_motifs) + len(round_emergents) >= max_library_size:
                    break
            if len(current_motifs) + len(round_emergents) >= max_library_size:
                break

        if not round_emergents:
            logger.info("No new emergents in round %d, converged", round_num)
            break

        logger.info("Round %d: discovered %d emergent motifs", round_num, len(round_emergents))
        all_emergents.extend(round_emergents)

    return all_emergents
