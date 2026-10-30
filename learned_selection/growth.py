"""GLS piece growth without per-vertex graph copies.

Mirrors the vendored GreedyMapping.generate_greedy_ud_subgraph_with / extend_mapping_with_nodes /
greedy_node_scoring (drop vertices that do not fit, rank candidates), step for step, including the
order of RNG draws and the insertion order of the mapping. Vertices the vendored code deletes from its
working copy go into `Growth.removed` instead, so the state graph is never copied.
"""
import math, random
from statistics import mean
from typing import Callable

import networkx as nx

from .lattice import BLOCKADE_UM, lattice_graph


class Growth:
    """The working state of one growing piece."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.mapping: dict = {}      # vertex -> trap, so far
        self.removed: set = set()    # vertices dropped after failing to fit

    @property
    def n(self) -> int:
        return len(self.graph) - len(self.removed)

    def deg(self, v) -> int:
        nbrs = self.graph._adj[v]
        return len(nbrs) - len(nbrs.keys() & self.removed)


# state graph -> scorer for every growth step on it: (growth, candidate vertices) -> one score per vertex
VertexScore = Callable[[nx.Graph], Callable[[Growth, list], list[float]]]


def lattice_adjacency(coords: dict, blockade: float = BLOCKADE_UM) -> tuple[list[list[int]], int]:
    """Neighbour lists in the vendored Lattice's adjacency order, and its integer average degree."""
    L = lattice_graph(coords, blockade)
    return [list(L.neighbors(t)) for t in L], int(mean(d for _, d in L.degree()))


def code_start_trap(n_traps: int) -> int:
    """The trap the vendored code puts the starting vertex on."""
    return int(n_traps / 2 + int(math.sqrt(n_traps)) / 4)


def gls_score(avg_degree: int) -> VertexScore:
    """The vendored score: prefer vertices whose degree is close to the lattice's average degree."""
    return lambda g: lambda grow, nodes: [1 - abs(avg_degree - grow.deg(v)) / grow.n for v in nodes]


def grow_piece(g: nx.Graph, start, adj: list[list[int]], start_trap: int, rng: random.Random,
               scorer: Callable[[Growth, list], list[float]]) -> dict:
    """Grow one piece from `start`: breadth-first over the graph, placing each vertex's best-scored
    neighbours on the free traps around it, and dropping the neighbours that find no trap."""
    grow = Growth(g)
    mapping, removed, nbrs = grow.mapping, grow.removed, g._adj
    unmapping = {start_trap: start}
    mapping[start] = start_trap
    unexpanded = {start}
    v, trap = start, start_trap
    while unexpanded:
        unexpanded.remove(v)
        free = [t for t in adj[trap] if t not in unmapping]
        cands = [u for u in nbrs[v] if u not in removed and u not in mapping]
        if cands:
            scores = scorer(grow, cands)
            draws = [rng.random() for _ in cands]
            order = sorted(range(len(cands)), key=lambda i: (scores[i], draws[i]), reverse=True)
            cands = [cands[i] for i in order]
            for t in free:
                # u fits on t iff its placed neighbours are exactly the vertices on the traps next to t
                near = {unmapping[x] for x in adj[t] if x in unmapping}
                for i, u in enumerate(cands):
                    if nbrs[u].keys() & mapping.keys() == near:
                        mapping[u] = t
                        unmapping[t] = u
                        unexpanded.add(u)
                        del cands[i]
                        break
            removed.update(cands)
        if unexpanded:
            v = next(iter(unexpanded))
            trap = mapping[v]
    return mapping
