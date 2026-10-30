"""The atom lattice and the check that a piece really fits it."""
import json, math
from importlib import resources

import networkx as nx

BLOCKADE_UM = 6.6   # Pasqal's run script; only nearest neighbours (5 um) interact, next-nearest is 8.66 um
CENTER_TRAP = 30    # trap at (0, 0) on fresnel_61


def load_coords(name: str = "fresnel_61") -> dict[int, tuple[float, float]]:
    raw = json.loads(resources.files("learned_selection.data").joinpath(f"{name}.json").read_text())
    return {int(k): tuple(v) for k, v in raw.items()}


def lattice_graph(coords: dict, blockade: float = BLOCKADE_UM) -> nx.Graph:
    L = nx.Graph()
    for t, p in coords.items():
        L.add_node(t, pos=p)
    ids = list(coords)
    L.add_edges_from((u, v) for i, u in enumerate(ids) for v in ids[i + 1:]
                     if math.dist(coords[u], coords[v]) < blockade)
    return L


def is_valid_piece(graph: nx.Graph, mapping: dict, lattice: nx.Graph) -> bool:
    """True iff mapping is an induced-subgraph embedding: adjacent in graph exactly when on neighbouring traps."""
    if len(set(mapping.values())) != len(mapping):
        return False
    if any(v not in graph for v in mapping) or any(t not in lattice for t in mapping.values()):
        return False
    nodes = list(mapping)
    return all(graph.has_edge(u, v) == lattice.has_edge(mapping[u], mapping[v])
               for i, u in enumerate(nodes) for v in nodes[i + 1:])
