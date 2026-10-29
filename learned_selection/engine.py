"""The beam engine: one loop, three injected callables (generate pieces, solve a piece, prune branches).

Code-faithful mode mirrors vendored greedy_subgraph_solver_vv.solve_breadthfirst call for call,
including the order of calls into the global `random` state that the vendored code reseeds.
"""
import copy, time
from dataclasses import dataclass
from typing import Callable

import networkx as nx

from .lattice import BLOCKADE_UM, load_coords, lattice_graph, is_valid_piece
from .solvers import first_maximum_set, is_independent, top_maximal_sets
from .vendored import GreedyMapping, Lattice, greedy_subgraph_solver_vv, weighted_generate_different_mis

Mapping = dict                                                   # graph node -> trap id
Generator = Callable[[nx.Graph, int], list[Mapping]]             # (current graph, k) -> up to k pieces
PieceSolver = Callable[[nx.Graph, Mapping, int], list[frozenset]]  # (piece, mapping, s) -> up to s sets
Branch = tuple[nx.Graph, list]                                   # (remaining graph, partial solution)
Pruner = Callable[[list[Branch], int], list[Branch]]             # (next level, solver children) -> kept


@dataclass
class RunResult:
    solution: frozenset
    size: int
    quantum_runs: int   # every subproblem solve: pieces of any size plus non-empty leftovers (R3)
    pasqal_runs: int    # piece-solver calls only, Pasqal's convention
    atoms: int          # sum of piece sizes over piece-solver calls
    rounds: int
    classical_s: float
    solver_s: float


def gls_generator(coords=None, blockade=BLOCKADE_UM, seed=0) -> Generator:
    """Vendored obtain_embeddable_subgraphs: grow a piece from every vertex, keep the k largest."""
    coords = load_coords() if coords is None else coords
    lattice = Lattice(coords, blockade, seed=seed)

    def generate(g: nx.Graph, k: int) -> list[Mapping]:
        maps = [GreedyMapping(g, copy.deepcopy(lattice), [], seed=seed).generate_greedy_ud_subgraph_with(v)
                for v in g.nodes()]
        return sorted(maps, key=len, reverse=True)[:k]
    return generate


def pasqal_greedy_piece_solver(coords=None, blockade=BLOCKADE_UM) -> PieceSolver:
    """Pasqal's classical piece solver, adapted to our interface. Exists to test the exact match."""
    coords = load_coords() if coords is None else coords
    lattice = Lattice(coords, blockade)   # built once: building it per call would reseed the global RNG

    def solve(piece: nx.Graph, mapping: Mapping, s: int) -> list[frozenset]:
        on_lattice = greedy_subgraph_solver_vv.generate_graph_to_solve(None, piece, lattice.lattice, mapping)
        inverse = {t: v for v, t in mapping.items()}
        return [frozenset(inverse[t] for t in mis) for mis in weighted_generate_different_mis(on_lattice, coords, s)]
    return solve


def exact_piece_solver(piece: nx.Graph, mapping: Mapping, s: int) -> list[frozenset]:
    return top_maximal_sets(piece, s)


def code_prune(k: int, l: int) -> Pruner:
    """Pasqal's code: prune to k only once the running count of solver children passes l."""
    counter = 0

    def prune(level: list[Branch], children: int) -> list[Branch]:
        nonlocal counter
        counter += children
        if counter > l:
            counter = 0
            return sorted(level, key=lambda b: len(b[1]), reverse=True)[:k]
        return level
    return prune


def _remove_closed(g: nx.Graph, S) -> nx.Graph:
    h = copy.deepcopy(g)   # deepcopy, as vendored: it keeps adjacency order, which GLS tie-breaks depend on
    h.remove_nodes_from({*S, *(u for v in S for u in g[v])})
    return h


def run_beam(graph: nx.Graph, generate: Generator, solve: PieceSolver, k: int, s: int, prune: Pruner,
             small: int = 5, max_branches: int = 10_000, lattice: nx.Graph | None = None) -> RunResult:
    start = time.perf_counter()
    L = lattice_graph(load_coords()) if lattice is None else lattice
    if list(graph) == list(range(len(graph))):
        g0, back = graph, None   # keep the original adjacency order: the exact-match test depends on it
    else:
        # GLS iterates Python sets; string hashing is randomised per process, so run on integer labels
        g0 = nx.convert_node_labels_to_integers(graph, label_attribute="orig")
        back = nx.get_node_attributes(g0, "orig")

    queue: list[Branch] = [(g0, [])]
    best: list = []
    quantum_runs = pasqal_runs = atoms = rounds = 0
    solver_s = 0.0
    while queue:
        next_level: list[Branch] = []
        children = 0
        for g, sol in queue:
            if len(g) <= small:
                if len(g):
                    quantum_runs += 1
                total = sol + first_maximum_set(g)
                if len(total) > len(best):
                    best = total
                continue
            for m in generate(g, k):   # all mappings first, then solve: same RNG order as vendored
                if not is_valid_piece(g, m, L):
                    raise AssertionError(f"generator produced a piece that does not fit the lattice: {m}")  # R1
                piece = g.subgraph(m)
                quantum_runs += 1
                if piece.number_of_nodes() <= small:
                    S = first_maximum_set(piece)
                    next_level.append((_remove_closed(g, S), sol + S))
                    continue
                pasqal_runs += 1
                atoms += len(m)
                t = time.perf_counter()
                sets = solve(piece, m, s)
                solver_s += time.perf_counter() - t
                for S in sets:
                    next_level.append((_remove_closed(g, S), sol + list(S)))
                    children += 1
        queue = prune(next_level, children)
        rounds += 1
        if len(queue) > max_branches:
            raise RuntimeError(f"beam exploded: {len(queue)} branches after round {rounds} (max {max_branches})")

    solution = frozenset(best) if back is None else frozenset(back[v] for v in best)
    if not is_independent(graph, solution):
        raise AssertionError("final answer is not an independent set of the original graph")   # R2
    total_s = time.perf_counter() - start
    return RunResult(solution, len(solution), quantum_runs, pasqal_runs, atoms, rounds,
                     total_s - solver_s, solver_s)


def run_code_faithful(graph: nx.Graph, k: int, s: int, l: int, solve: PieceSolver = exact_piece_solver,
                      seed: int = 0, small: int = 5) -> RunResult:
    return run_beam(graph, gls_generator(seed=seed), solve, k, s, code_prune(k, l), small)
