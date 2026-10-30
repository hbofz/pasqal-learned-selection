"""Tree search over pieces under one global budget of counted solves.

PUCT from the root, with the rank hook as the prior over a node's pieces and the value hook as the leaf value of the
graph a solve leaves behind. Counts runs as run_beam does; it does not call run_beam, so the checks are repeated here.
"""
import math, time

import networkx as nx
import numpy as np

from .engine import RunResult, Rank, Value, _remove_closed, exact_piece_solver, gls_generator
from .growth import VertexScore
from .lattice import is_valid_piece, lattice_graph, load_coords
from .solvers import canonical, first_maximum_set, is_independent


class _Node:
    __slots__ = ("g", "sol", "depth", "v", "N", "W", "kids", "cands", "expanded", "done")

    def __init__(self, g, sol, depth, v):
        self.g, self.sol, self.depth, self.v = g, sol, depth, v   # v: partial size plus predicted value-to-go
        self.N, self.W = 0, 0.0
        self.kids: list[tuple[_Node, float]] = []    # solved children, each with its share of the prior
        self.cands: list[tuple[dict, float]] = []    # unsolved pieces, each with its prior
        self.expanded = self.done = False


def run_mcts(graph: nx.Graph, k: int, s: int, budget: int, score: VertexScore, rank: Rank, value: Value,
             c: float = 1.0, solve=exact_piece_solver, seed: int = 0, small: int = 5, trace: list | None = None,
             noise: tuple | None = None) -> RunResult:
    """Search until `budget` solves are counted or the tree is exhausted, then dive along the most visited children
    to a terminal. The answer is the largest complete set seen. `trace` and `noise` are for self-play and do
    nothing here."""
    start = time.perf_counter()
    L = lattice_graph(load_coords())
    if all(type(v) is int for v in graph):
        g0, back = graph, None
    else:
        g0 = nx.convert_node_labels_to_integers(canonical(graph), label_attribute="orig")
        back = nx.get_node_attributes(g0, "orig")

    scores: dict[int, float] = {}   # id of a grown piece -> its rank score, filled by the generator's rank call

    def capture(g, maps):
        out = list(rank(g, maps))
        scores.update((id(m), x) for m, x in zip(maps, out))
        return out

    generate = gls_generator(score=score, rank=capture, seed=seed)
    best: list = []
    quantum_runs = pasqal_runs = atoms = deepest = 0
    run_sizes: list[int] = []
    solver_s = 0.0

    def expand(node):
        """The first k pieces of distinct vertex sets, with the softmax of their rank scores as priors."""
        scores.clear()
        picked, sets = [], set()
        for m in generate(node.g, len(node.g)):
            if (key := frozenset(m)) not in sets:
                sets.add(key)
                picked.append(m)
            if len(picked) == k:
                break
        x = np.array([scores[id(m)] for m in picked], dtype=float)
        p = np.exp(x - x.max())
        node.cands = list(zip(picked, (p / p.sum()).tolist()))
        node.expanded = True

    def finish(node) -> float:
        """A terminal: counted once if non-empty, its complete set is a candidate answer."""
        nonlocal quantum_runs, best, deepest
        if len(node.g):
            quantum_runs += 1
            run_sizes.append(len(node.g))
        total = node.sol + first_maximum_set(node.g)
        if len(total) > len(best):
            best = total
        node.done = True
        deepest = max(deepest, node.depth)
        return float(len(total))

    def solve_piece(node, i):
        """Solve candidate i of node (one counted solve) and hang its children under it."""
        nonlocal quantum_runs, pasqal_runs, atoms, solver_s
        m, prior = node.cands.pop(i)
        if not m:
            raise ValueError("generator returned an empty piece")
        if not is_valid_piece(node.g, m, L):
            raise AssertionError(f"generator produced a piece that does not fit the lattice: {m}")
        piece = node.g.subgraph(m)
        quantum_runs += 1
        run_sizes.append(len(m))
        if len(m) <= small:
            sets = [first_maximum_set(piece)]
        else:
            pasqal_runs += 1
            atoms += len(m)
            t = time.perf_counter()
            sets = solve(piece, m, s)
            solver_s += time.perf_counter() - t
            if not sets:
                raise ValueError("piece solver returned no sets for a non-empty piece")
            for S in sets:
                if not S or not set(S) <= m.keys() or not is_independent(piece, S):
                    raise ValueError(f"piece solver returned an invalid set: {sorted(S, key=repr)}")
        rest = [_remove_closed(node.g, S) for S in sets]
        kids = [_Node(h, node.sol + list(S), node.depth + 1, len(node.sol) + len(S) + float(v))
                for h, S, v in zip(rest, sets, value(rest), strict=True)]
        node.kids += [(q, prior / len(kids)) for q in kids]
        return kids

    def select(node):
        """PUCT over the open children (returns a node) and the unsolved candidates (returns an index). Ties go first."""
        sq = math.sqrt(node.N + 1)
        qs = [q.W / q.N if q.N else q.v for q, _ in node.kids]
        lo = min(qs, default=0.0)
        span = (max(qs, default=0.0) - lo) or 1.0
        top, pick = -math.inf, None
        for (q, share), x in zip(node.kids, qs):
            if q.done:
                continue
            u = (x - lo) / span + c * share * sq / (1 + q.N)
            if u > top:
                top, pick = u, q
        fpu = 0.5 if node.kids else 0.0
        for i, (_, prior) in enumerate(node.cands):
            u = fpu + c * prior * sq
            if u > top:
                top, pick = u, i
        return pick

    def simulate(node):
        path = [node]
        while True:
            if len(node.g) <= small:
                v = finish(node)
                break
            if not node.expanded:
                expand(node)
                v = node.v
                break
            pick = select(node)
            if isinstance(pick, _Node):
                node = pick
                path.append(node)
                continue
            node = max(solve_piece(node, pick), key=lambda q: q.v)
            path.append(node)
            v = finish(node) if len(node.g) <= small else node.v
            break
        for x in path:
            x.N += 1
            x.W += v
        for x in reversed(path):
            if x.expanded and not x.cands and all(q.done for q, _ in x.kids):
                x.done = True

    root = _Node(g0, [], 1, float(value([g0])[0]))   # depth counts levels as run_beam's rounds do
    while quantum_runs < budget and not root.done:
        simulate(root)
    node = root
    while len(node.g) > small:
        while not node.kids:
            simulate(node)
        node = max((q for q, _ in node.kids), key=lambda q: (q.N, q.W / q.N if q.N else q.v))
    if not node.done:
        finish(node)

    solution = frozenset(best) if back is None else frozenset(back[v] for v in best)
    if not is_independent(graph, solution):
        raise AssertionError("final answer is not an independent set of the original graph")
    total_s = time.perf_counter() - start
    return RunResult(solution, len(solution), quantum_runs, pasqal_runs, atoms, deepest, total_s - solver_s,
                     solver_s, run_sizes)
