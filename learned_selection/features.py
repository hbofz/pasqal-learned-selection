"""Structural features of the vertices of a state graph and of candidate pieces. Graph structure only: nothing
here computes an independent set, so a model built on these features never solves anything itself.

Adjacency sets and C-level set operations keep piece_features in the low milliseconds per state, because a
learned candidate-choice model calls it on every generate call."""
import math

import networkx as nx
import numpy as np

VERTEX_FEATURES = ["deg_frac", "deg_rel", "nbr_deg_mean", "nbr_deg_min", "core_rel", "gls_term", "lower_nbrs",
                   "log_n", "density"]
PIECE_FEATURES = ["size", "size_frac", "piece_density", "deg_mean", "deg_min", "deg_max", "closed_nbhd",
                  "boundary", "interior", "local_min", "size_rank", "log_n", "density"]
LATTICE_DEGREE = 5   # the average lattice degree GLS aims for when it scores a vertex


def vertex_features(g: nx.Graph) -> tuple[list, np.ndarray]:
    """The nodes in g order and an n x 9 matrix, one row per node. Degrees are taken in g."""
    nodes, n = list(g), len(g)
    adj = g._adj
    deg = {v: len(adj[v]) for v in nodes}
    max_deg = max(deg.values(), default=0)
    core = nx.core_number(g)
    max_core = max(core.values(), default=0)
    log_n, density = math.log(max(1, n)), nx.density(g)
    X = np.zeros((n, len(VERTEX_FEATURES)))
    for i, v in enumerate(nodes):
        d = deg[v]
        nbr_deg = [deg[u] for u in adj[v]]
        X[i] = (d / n, d / max(1, max_deg),
                sum(nbr_deg) / d / n if d else 0, min(nbr_deg) / n if d else 0,
                core[v] / max(1, max_core), 1 - abs(LATTICE_DEGREE - d) / n,
                sum(x < d for x in nbr_deg) / max(1, d), log_n, density)
    return nodes, X


def piece_features(g: nx.Graph, maps: list[dict]) -> np.ndarray:
    """A len(maps) x 13 matrix, one row per candidate piece (the vertex set of each mapping). Degrees are
    taken in g, and column 0 is the piece size."""
    n = len(g)
    adj = g._adj
    deg = {v: len(adj[v]) for v in g}
    log_n, density = math.log(max(1, n)), nx.density(g)
    X = np.zeros((len(maps), len(PIECE_FEATURES)))
    for i, m in enumerate(maps):
        P = set(m)
        size = len(P)
        nbrs = [adj[v].keys() for v in P]
        degs = [deg[v] for v in P]
        inside = sum(len(a & P) for a in nbrs) / 2
        total_deg = sum(degs)
        X[i] = (size, size / n, inside / max(1, size * (size - 1) / 2),
                total_deg / max(1, size) / n, min(degs, default=0) / n, max(degs, default=0) / n,
                len(P.union(*nbrs)) / n, (total_deg - 2 * inside) / max(1, total_deg),
                sum(a <= P for a in nbrs) / max(1, size),
                sum(deg[v] <= min(map(deg.__getitem__, a), default=deg[v])
                    for v, a in zip(P, nbrs)) / max(1, size),
                0, log_n, density)
    sizes = X[:, 0]
    X[:, 10] = (len(maps) - np.searchsorted(np.sort(sizes), sizes, side="right")) / max(1, len(maps))
    return X


GRAPH_FEATURES = ["n", "edges", "density", "caro_wei", "isolated", "deg_max", "deg_mean", "deg_std", "core_mean",
                  "core_max"]


def graph_features(g: nx.Graph) -> np.ndarray:
    """A vector of len(GRAPH_FEATURES) raw counts and means over a graph, zeros on the empty graph. caro_wei is
    the sum over vertices of 1 / (degree + 1), a closed form in the degrees."""
    deg = np.array([d for _, d in g.degree()], dtype=np.float64)
    core = np.array(list(nx.core_number(g).values()), dtype=np.float64)
    f = lambda a, op: float(op(a)) if len(a) else 0.0
    return np.array([len(g), g.number_of_edges(), nx.density(g), f(1 / (deg + 1), np.sum), float((deg == 0).sum()),
                     f(deg, np.max), f(deg, np.mean), f(deg, np.std), f(core, np.mean), f(core, np.max)])
