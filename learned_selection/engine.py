"""The beam engine: one loop, three injected callables (generate pieces, solve a piece, prune branches).

Code-faithful mode mirrors vendored greedy_subgraph_solver_vv.solve_breadthfirst call for call.
gls_generator is a fast port of the vendored growth that leaves the global `random` state alone;
vendored_gls_generator is the original, which reseeds it, and Pasqal's own piece solver needs it for an exact match.
"""
import copy, random, time
from dataclasses import dataclass
from typing import Callable, Sequence

import networkx as nx

from .growth import VertexScore, code_start_trap, gls_score, grow_piece, lattice_adjacency
from .lattice import BLOCKADE_UM, CENTER_TRAP, load_coords, lattice_graph, is_valid_piece
from .solvers import canonical, first_maximum_set, is_independent, top_maximal_sets
from .vendored import GreedyMapping, Lattice, greedy_subgraph_solver_vv, weighted_generate_different_mis

Mapping = dict                                                   # graph node -> trap id
Generator = Callable[[nx.Graph, int], list[Mapping]]             # (current graph, k) -> up to k pieces
PieceSolver = Callable[[nx.Graph, Mapping, int], list[frozenset]]  # (piece, mapping, s) -> up to s sets
Branch = tuple[nx.Graph, list]                                   # (remaining graph, partial solution)
Pruner = Callable[[list[Branch], int], list[Branch]]             # (next level, solver children) -> kept
Rank = Callable[[nx.Graph, list[Mapping]], Sequence[float]]      # (current graph, candidate pieces) -> one rank each


@dataclass
class RunResult:
    solution: frozenset
    size: int
    quantum_runs: int   # every subproblem solve: pieces of any size plus non-empty leftovers
    pasqal_runs: int    # piece-solver calls only, Pasqal's convention
    atoms: int          # sum of piece sizes over piece-solver calls
    rounds: int
    classical_s: float
    solver_s: float
    run_sizes: list[int]   # node count of every counted solve, in the order they happen


def _check_trap_order(coords: dict) -> None:
    # GreedyMapping relabels the lattice to positional indices; those equal trap ids only for keys 0..n-1 in order
    if list(coords) != list(range(len(coords))):
        raise ValueError("lattice trap ids must be 0..n-1 in insertion order")


def vendored_gls_generator(coords=None, blockade=BLOCKADE_UM, seed=0) -> Generator:
    """Vendored obtain_embeddable_subgraphs: grow a piece from every vertex, keep the k largest.
    Slow (it deep-copies the graph for every start vertex); kept as the reference the fast port is tested against."""
    coords = load_coords() if coords is None else coords
    _check_trap_order(coords)
    lattice = Lattice(coords, blockade, seed=seed)

    def generate(g: nx.Graph, k: int) -> list[Mapping]:
        maps = [GreedyMapping(g, copy.deepcopy(lattice), [], seed=seed).generate_greedy_ud_subgraph_with(v)
                for v in g.nodes()]
        return sorted(maps, key=len, reverse=True)[:k]
    return generate


def gls_generator(coords=None, blockade=BLOCKADE_UM, seed=0, score: VertexScore | None = None,
                  rank: Rank | None = None) -> Generator:
    """GLS as the vendored code runs it: grow a piece from every vertex, keep the k of highest rank
    (by default the k largest). `score` replaces the vertex priority used during growth."""
    coords = load_coords() if coords is None else coords
    _check_trap_order(coords)
    adj, avg_degree = lattice_adjacency(coords, blockade)
    start_trap = code_start_trap(len(adj))
    make_scorer = score or gls_score(avg_degree)

    def generate(g: nx.Graph, k: int) -> list[Mapping]:
        scorer = make_scorer(g)
        maps = [grow_piece(g, v, adj, start_trap, random.Random(seed), scorer) for v in g.nodes()]
        ranks = [len(m) for m in maps] if rank is None else rank(g, maps)
        return [maps[i] for i in sorted(range(len(maps)), key=ranks.__getitem__, reverse=True)[:k]]
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


def code_prune(k: int, l: int, count_small: bool = False) -> Pruner:
    """Pasqal's code: prune to k only once the running count of solver children passes l.
    With count_small every child counts, so levels made of small pieces still get pruned."""
    counter = 0

    def prune(level: list[Branch], children: int) -> list[Branch]:
        nonlocal counter
        counter += len(level) if count_small else children
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
             small: int = 5, max_branches: int = 1_000, lattice: nx.Graph | None = None) -> RunResult:
    """Generators and pruners carry state, so build fresh ones for each run, and parallelise with processes."""
    start = time.perf_counter()
    L = lattice_graph(load_coords()) if lattice is None else lattice
    if all(type(v) is int for v in graph):
        g0, back = graph, None   # as-is, like the vendored code: int hashing is deterministic, adjacency order kept
    else:
        # GLS iterates Python sets and str hashing is randomised per process: run on a canonical integer copy
        g0 = nx.convert_node_labels_to_integers(canonical(graph), label_attribute="orig")
        back = nx.get_node_attributes(g0, "orig")

    queue: list[Branch] = [(g0, [])]
    best: list = []
    quantum_runs = pasqal_runs = atoms = rounds = 0
    run_sizes: list[int] = []
    solver_s = 0.0
    while queue:
        next_level: list[Branch] = []
        children = 0
        for g, sol in queue:
            if len(g) <= small:
                if len(g):
                    quantum_runs += 1
                    run_sizes.append(len(g))
                total = sol + first_maximum_set(g)
                if len(total) > len(best):
                    best = total
                continue
            for m in generate(g, k):   # all mappings first, then solve: same RNG order as vendored
                if not m:
                    raise ValueError("generator returned an empty piece")   # would never shrink the branch
                if not is_valid_piece(g, m, L):
                    raise AssertionError(f"generator produced a piece that does not fit the lattice: {m}")
                piece = g.subgraph(m)
                quantum_runs += 1
                run_sizes.append(len(m))
                if piece.number_of_nodes() <= small:
                    S = first_maximum_set(piece)
                    next_level.append((_remove_closed(g, S), sol + S))
                    continue
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
                    next_level.append((_remove_closed(g, S), sol + list(S)))
                    children += 1
        queue = prune(next_level, children)
        rounds += 1
        if len(queue) > max_branches:
            raise RuntimeError(f"beam exploded: {len(queue)} branches after round {rounds} (max {max_branches})")

    solution = frozenset(best) if back is None else frozenset(back[v] for v in best)
    if not is_independent(graph, solution):
        raise AssertionError("final answer is not an independent set of the original graph")
    total_s = time.perf_counter() - start
    return RunResult(solution, len(solution), quantum_runs, pasqal_runs, atoms, rounds,
                     total_s - solver_s, solver_s, run_sizes)


def run_code_faithful(graph: nx.Graph, k: int, s: int, l: int, solve: PieceSolver = exact_piece_solver,
                      seed: int = 0, small: int = 5, max_branches: int = 1_000) -> RunResult:
    return run_beam(graph, gls_generator(seed=seed), solve, k, s, code_prune(k, l), small, max_branches)


def run_code_bounded(graph: nx.Graph, k: int, s: int, l: int, score: VertexScore | None = None,
                     rank: Rank | None = None, solve: PieceSolver = exact_piece_solver, seed: int = 0,
                     small: int = 5, max_branches: int = 1_000) -> RunResult:
    """The code loop with a pruner that also counts small-piece children, so graphs that split into many
    small pieces (isolated vertices, say) stay bounded. `score` and `rank` are the generator's hooks."""
    return run_beam(graph, gls_generator(seed=seed, score=score, rank=rank), solve, k, s,
                    code_prune(k, l, count_small=True), small, max_branches)


class CenterSeededMapping(GreedyMapping):
    """GLS growth as the paper describes it: the starting item goes on the central trap."""

    def initialize_mapping(self, starting_node, mapping, unmapping, unexpanded_nodes) -> int:
        mapping[starting_node] = CENTER_TRAP
        unmapping[CENTER_TRAP] = starting_node
        unexpanded_nodes.add(starting_node)
        return CENTER_TRAP


def paper_generator(coords=None, blockade=BLOCKADE_UM, seed=0) -> Generator:
    """The paper's loop: k pieces grown from k random starting items, each placed at the lattice centre.
    The ranking score stays the vendored one, since the paper does not pin its sort direction."""
    coords = load_coords() if coords is None else coords
    _check_trap_order(coords)
    lattice = Lattice(coords, blockade)
    rng = random.Random(seed)

    def generate(g: nx.Graph, k: int) -> list[Mapping]:
        starts = rng.sample(sorted(g, key=repr), min(k, len(g)))
        return [CenterSeededMapping(g, copy.deepcopy(lattice), [], seed=rng.randrange(2**31))
                .generate_greedy_ud_subgraph_with(v) for v in starts]
    return generate


def paper_prune(l: int) -> Pruner:
    """The paper's beam: always keep the l best partial solutions."""
    return lambda level, children: sorted(level, key=lambda b: len(b[1]), reverse=True)[:l]


def run_paper_faithful(graph: nx.Graph, k: int, s: int, l: int, solve: PieceSolver = exact_piece_solver,
                       seed: int = 0, small: int = 5, max_branches: int = 1_000) -> RunResult:
    return run_beam(graph, paper_generator(seed=seed), solve, k, s, paper_prune(l), small, max_branches)


def best_of_restarts(run: Callable[[int], RunResult], restarts: int) -> RunResult:
    """Best answer over runs with seeds 0..restarts-1 (ties go to the earlier seed); costs are summed."""
    rs = [run(seed) for seed in range(restarts)]
    best = max(rs, key=lambda r: r.size)
    total = lambda f: sum(getattr(r, f) for r in rs)
    return RunResult(best.solution, best.size, total("quantum_runs"), total("pasqal_runs"), total("atoms"),
                     total("rounds"), total("classical_s"), total("solver_s"),
                     [x for r in rs for x in r.run_sizes])
