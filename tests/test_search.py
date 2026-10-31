import math, time

import networkx as nx, numpy as np, pytest
from test_bench import KEYS
from learned_selection.bench import run_instance
from learned_selection.engine import exact_piece_solver, gls_generator
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.instances import erdos_renyi
from learned_selection.models import fit_linear, fit_value_linear, save
from learned_selection import search
from learned_selection.search import run_mcts
from learned_selection.solvers import is_independent

score = lambda g: lambda grow, cands: [-g.degree(v) for v in cands]
rank = lambda g, maps: [len(m) for m in maps]
value = lambda gs: [len(g) / 3 for g in gs]


def test_mcts_gives_a_valid_answer_within_budget_plus_a_dive():
    G = erdos_renyi(60, 0.15, 1)
    r = run_mcts(G, 4, 2, 20, score, rank, value)
    assert is_independent(G, r.solution) and r.size == len(r.solution) > 0
    assert 20 <= r.quantum_runs <= 20 + 60 and r.pasqal_runs <= r.quantum_runs and len(r.run_sizes) == r.quantum_runs


def test_mcts_ends_when_the_tree_is_exhausted():
    for G in (erdos_renyi(12, 0.9, 0), nx.complete_graph(8), nx.path_graph(7)):
        nx.set_node_attributes(G, 1, "weight")
        r = run_mcts(G, 4, 2, 10_000, score, rank, value)
        assert is_independent(G, r.solution) and r.quantum_runs < 10_000


def test_mcts_more_budget_never_hurts_on_average():
    gaps = {b: [] for b in (6, 200)}
    for seed in range(6):
        G = erdos_renyi(60, 0.15, seed)
        for b in gaps:
            gaps[b].append(run_mcts(G, 4, 2, b, score, rank, value).size)
    assert sum(gaps[200]) > sum(gaps[6])


def test_mcts_counts_small_pieces_and_terminals_like_the_beam():
    G = nx.empty_graph(4); nx.set_node_attributes(G, 1, "weight")
    r = run_mcts(G, 4, 2, 10, score, rank, value)         # a terminal straight away: one solve, no piece-solver call
    assert (r.size, r.quantum_runs, r.pasqal_runs, r.atoms) == (4, 1, 0, 0)
    G = nx.empty_graph(0)
    assert run_mcts(G, 4, 2, 10, score, rank, value).quantum_runs == 0
    G = nx.empty_graph(40); nx.set_node_attributes(G, 1, "weight")
    r = run_mcts(G, 4, 2, 30, score, rank, value)
    assert r.size == 40 and all(x >= 1 for x in r.run_sizes) and r.quantum_runs == len(r.run_sizes)


def test_mcts_string_labels_match_their_integer_twin():
    G = erdos_renyi(30, 0.2, 2)
    H = nx.relabel_nodes(G, {v: f"v{v:02d}" for v in G})
    a, b = run_mcts(G, 4, 2, 20, score, rank, value), run_mcts(H, 4, 2, 20, score, rank, value)
    assert a.size == b.size and a.quantum_runs == b.quantum_runs and all(type(v) is str for v in b.solution)


def test_mcts_is_deterministic_for_a_seed():
    G = erdos_renyi(50, 0.2, 3)
    a, b = run_mcts(G, 4, 2, 30, score, rank, value), run_mcts(G, 4, 2, 30, score, rank, value)
    assert (a.solution, a.quantum_runs) == (b.solution, b.quantum_runs)


def test_mcts_prior_steers_the_first_solve():
    # all the prior on the smallest piece: the first counted solve is that piece, where a size rank picks the largest
    G = erdos_renyi(60, 0.15, 1)
    sizes = sorted(len(m) for m in gls_generator(score=score)(G, len(G)))
    assert 5 < sizes[0] < sizes[-1]
    for steer, want in ((lambda g, maps: [-1000.0 * len(m) for m in maps], sizes[0]), (rank, sizes[-1])):
        seen = []

        def solve(piece, m, s):
            seen.append(len(m))
            return exact_piece_solver(piece, m, s)
        run_mcts(G, 4, 2, 3, score, steer, value, solve=solve)
        assert seen[0] == want


def test_mcts_rejects_an_invalid_piece_set():
    G = erdos_renyi(40, 0.3, 0)
    with pytest.raises(ValueError, match="invalid set"):
        run_mcts(G, 4, 2, 10, score, rank, value, solve=lambda piece, m, s: [frozenset(piece.nodes)])
    with pytest.raises(ValueError, match="no sets"):
        run_mcts(G, 4, 2, 10, score, rank, value, solve=lambda piece, m, s: [])


def test_bench_routes_search_names_with_budget_as_l(tmp_path):
    rng = np.random.default_rng(0)
    for hook, d in (("h2", len(VERTEX_FEATURES)), ("h4", len(PIECE_FEATURES))):
        X, g, y = rng.standard_normal((60, d)), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
        save(fit_linear(X, y, g), tmp_path / f"lin-{hook}.json")
    save(fit_value_linear(rng.standard_normal((40, len(GRAPH_FEATURES))), rng.integers(0, 9, 40)), tmp_path / "lin-h5.json")
    recs = run_instance("er", {"p": 0.25}, 60, 0, ["lin-h2h4+lin-h5+mcts"], widths=[(4, 2, 15)], restarts=[1],
                        time_limit_s=10, models_dir=tmp_path)
    r = recs[0]
    assert len(recs) == 1 and KEYS <= r.keys() and r["l"] == 15 and r["quantum_runs"] >= 15
    assert r["track"] == "structure" and r["restarts"] == 1 and r["size"] > 0


def test_mcts_counts_each_solve_once_whatever_its_kind(monkeypatch):
    # every piece solve and every non-empty terminal is one quantum run; only pieces over `small` reach the solver
    first_max, solved, finished = search.first_maximum_set, [], []

    def spy_first_max(g):
        finished.append(len(g))
        return first_max(g)

    def spy_solve(piece, m, s):
        solved.append(len(m))
        time.sleep(0.002)
        return exact_piece_solver(piece, m, s)
    monkeypatch.setattr(search, "first_maximum_set", spy_first_max)
    for G in (erdos_renyi(60, 0.15, 1), erdos_renyi(40, 0.75, 0), erdos_renyi(80, 0.05, 2)):
        solved.clear(); finished.clear()
        r = run_mcts(G, 4, 2, 40, score, rank, value, solve=spy_solve)
        assert any(x <= 5 for x in r.run_sizes) and solved
        assert r.quantum_runs == len(solved) + sum(x > 0 for x in finished) == len(r.run_sizes)
        assert r.pasqal_runs == len(solved) == sum(x > 5 for x in r.run_sizes)
        assert r.atoms == sum(solved) == sum(x for x in r.run_sizes if x > 5)
        assert r.solver_s >= 0.002 * len(solved) and r.classical_s > 0


def test_mcts_refuses_a_piece_that_does_not_fit_the_lattice(monkeypatch):
    monkeypatch.setattr(search, "is_valid_piece", lambda *a: False)
    with pytest.raises(AssertionError, match="does not fit"):
        run_mcts(erdos_renyi(40, 0.3, 0), 4, 2, 10, score, rank, value)


def test_mcts_refuses_an_empty_piece(monkeypatch):
    def empty(score, rank, seed):
        calls = []

        def generate(g, n):
            calls.append(1)
            assert len(calls) < 20, "an empty piece was accepted"
            maps = [{}]
            rank(g, maps)
            return maps
        return generate
    monkeypatch.setattr(search, "gls_generator", empty)
    with pytest.raises(ValueError, match="empty piece"):
        run_mcts(erdos_renyi(40, 0.3, 0), 4, 2, 10, score, rank, value)


def test_mcts_checks_the_final_answer_against_the_input_graph(monkeypatch):
    G = erdos_renyi(40, 0.3, 0)
    real = search.is_independent
    monkeypatch.setattr(search, "is_independent", lambda g, S: False if g is G else real(g, S))
    with pytest.raises(AssertionError, match="not an independent set"):
        run_mcts(G, 4, 2, 10, score, rank, value)


def test_mcts_fails_loudly_on_a_value_hook_that_returns_too_few_values():
    G = erdos_renyi(60, 0.15, 1)
    with pytest.raises(ValueError):
        run_mcts(G, 4, 2, 10, score, rank, lambda gs: [0.0] * min(len(gs), 1))


# A scripted world: the root grows the pieces listed with the weights given (the rank is their log, so the prior is
# the weight over the total), every other graph grows one piece, itself. Solves are scripted by the piece's vertex set
# and fall back to the exact solver. The graph is four disjoint edges, so a set S of m vertices removes 2m of them.
MATCHING = nx.Graph([(0, 1), (2, 3), (4, 5), (6, 7)])
A, B, C = frozenset(range(6)), frozenset(range(2, 8)), frozenset({0, 1, 2, 6, 7})
SOLS = {A: [{0, 2, 4}, {0, 2}, {0}], B: [{2, 4, 6}, {2, 4}], C: [{0, 2, 6}]}
A1, A2, A3 = frozenset({6, 7}), frozenset({4, 5, 6, 7}), frozenset(range(2, 8))   # what each set of A leaves, 2 nodes
                                                                                   # being terminal at small = 2


def scripted(monkeypatch, G, root, sols, vals, budget, k=3, small=2, c=1.0):
    """(result, the vertex set of every piece sent to the solver, in order)."""
    weights = {frozenset(vs): w for vs, w in root}

    def grower(score, rank, seed):
        def generate(g, n):
            maps = [{v: v for v in sorted(vs)} for vs, _ in root] if set(g) == set(G) else [{v: v for v in g}]
            rank(g, maps)
            return maps
        return generate
    monkeypatch.setattr(search, "gls_generator", grower)
    monkeypatch.setattr(search, "is_valid_piece", lambda *a: True)
    solved = []

    def solve(piece, m, s):
        solved.append(frozenset(m))
        return [frozenset(S) for S in sols[frozenset(m)]] if frozenset(m) in sols else exact_piece_solver(piece, m, s)
    r = run_mcts(G, k, 3, budget, lambda g: None, lambda g, maps: [math.log(weights.get(frozenset(m), 1.0)) for m in maps],
                 lambda gs: [vals.get(frozenset(g), 0.0) for g in gs], solve=solve, small=small, c=c)
    return r, solved


def test_mcts_expands_the_child_with_the_best_static_value(monkeypatch):
    # A leaves three children valued (|S| + value) 8 (terminal), 10 and 9.5. The best, A2, is entered first; the
    # first or the one with the largest value alone (8, 9.5) would not be. At the root's 3rd visit (sqrt(4) = 2) the
    # open child A2 scores (10 - 8) / 2 + (0.5 / 3) * 2 / 3 = 1.111, above A3 1.083 and the piece B 0.5 + 0.3 * 2 = 1.1,
    # so A2 is expanded and its own piece solved next, with 2 runs spent.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 5, A2: 8, A3: 8.5}, 2)
    assert solved == [A, A2] and r.quantum_runs == 2 and r.run_sizes == [6, 4] and r.rounds == 3


def test_mcts_uses_the_static_value_of_an_unvisited_child(monkeypatch):
    # A's children are valued 12 (terminal, true size 4 once visited), 10 and 8. At the root's 3rd visit (sqrt(3)) with
    # lo = 4, span 6, the unvisited child valued 10 scores 1 + (0.5 / 3) * 1.732 = 1.289, above B 0.5 + 0.3 * 1.732 = 1.02.
    # A child worth 0 until visited would leave B ahead.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 9, A2: 8, A3: 7}, 3)
    assert solved[:2] == [A, A2]


def test_mcts_gives_an_unsolved_piece_a_first_play_value_and_splits_the_prior_over_children(monkeypatch):
    # Priors 0.45, 0.45, 0.1. After A (3 children, 0.15 each) and its terminal, the best open child scores
    # 1 + 0.15 * 1.732 = 1.260 and the piece B 0.5 + 0.45 * 1.732 = 1.279, so B is solved second. With no first-play
    # value B would score 0.779, with the prior unsplit the child 1.779, with Q unnormalised the child 10.3, with a
    # uniform prior B 1.077 against the child 1.192. A cheaper third child (valued 5) keeps those wrong choices from
    # catching up through free expansions: they would solve the child's piece before B.
    for third in (7, 4):
        r, solved = scripted(monkeypatch, MATCHING, [(A, 0.45), (B, 0.45), (C, 0.1)], SOLS, {A1: 9, A2: 8, A3: third}, 3)
        assert solved[:2] == [A, B]


def test_mcts_skips_an_exhausted_child_but_counts_it_in_the_normalisation(monkeypatch):
    # The terminal child (worth 4) has the top Q, 1 + 0.144 = 1.144 against B 1.02 and the open children 0.79 and 0.29,
    # so it would be picked again if it were not exhausted, and spend a run. Skipped, B is solved second.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 9, A2: 1, A3: 1}, 3)
    assert solved[:2] == [A, B]


def test_mcts_scales_exploration_by_c_and_visits_by_sqrt_of_n_plus_one(monkeypatch):
    # The priors and values of the first-play case with c = 0.5: the open child scores 1 + 0.5 * 0.15 * 1.732 = 1.13
    # against B 0.5 + 0.5 * 0.45 * 1.732 = 0.89, then at the next visit 1.075 against 0.95, so the child's own piece
    # is solved second, where c = 1 solves B.
    pieces, vals = [(A, 0.45), (B, 0.45), (C, 0.1)], {A1: 9, A2: 8, A3: 4}
    assert scripted(monkeypatch, MATCHING, pieces, SOLS, vals, 3, c=0.5)[1][:2] == [A, A2]
    # And c = 2 on the first case's priors: B 1.539 against the open child 1.577 at the 3rd visit, so that child is
    # expanded first (1.7 over 1.333 then solves B). It keeps its visit, and the dive, tied on one visit with the
    # terminal and B's child, takes it for its mean of 10 and solves its piece: A, B, A2 and 5 runs. A bonus on
    # the children that ignored c (1.289) would send the 3rd visit to B and the dive to the terminal instead.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 9, A2: 8, A3: 7}, 3, c=2.0)
    assert solved == [A, B, A2] and r.quantum_runs == 5
    # The root's bonus uses sqrt(N + 1). The 5th simulation enters the unvisited child valued 8 (1.373) and the 6th
    # solves B (1.235 over that child's 1.204), after A and A2, for 4 runs; sqrt(N) stops one run short.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 0, A2: 8, A3: 7}, 3)
    assert solved == [A, A2, B] and r.quantum_runs == 4


def test_mcts_solves_equal_vertex_sets_once(monkeypatch):
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.4), (A, 0.4), (B, 0.2)], SOLS, {}, 10)
    assert solved.count(A) == 1 and B in solved


def test_mcts_dive_follows_the_most_visited_child_and_counts_a_terminal_once(monkeypatch):
    # Budget 2 is spent by A and its terminal child; the dive enters that child (1 visit, worth 4), not the unvisited
    # one valued 10, and does not count the terminal a second time.
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {A1: 9, A2: 8, A3: 7}, 2)
    assert solved == [A] and r.quantum_runs == 2 and r.run_sizes == [6, 2] and r.size == 4 and r.rounds == 2


def test_mcts_dive_finishes_a_terminal_it_reaches_first():
    G = nx.empty_graph(4)
    r = run_mcts(G, 4, 2, 0, score, rank, value)
    assert r.size == 4 and r.quantum_runs == 1 and r.rounds == 1


def test_mcts_keeps_the_largest_answer_seen_and_ends_with_the_tree(monkeypatch):
    # A star: the first piece (4 leaves) ends in a terminal of 4 + 3 = 7, the second (the centre) in an empty
    # graph and an answer of 1. The empty terminal costs nothing. Both pieces solved, the root is exhausted.
    G = nx.star_graph(7)
    P1, P2 = frozenset({1, 2, 3, 4}), frozenset({0, 1, 2, 3})
    r, solved = scripted(monkeypatch, G, [(P1, 0.6), (P2, 0.4)], {P1: [{1, 2, 3, 4}], P2: [{0}]}, {}, 100, k=2, small=3)
    assert solved == [P1, P2] and r.size == 7 and r.run_sizes == [4, 3, 4] and r.quantum_runs == 3 and r.rounds == 2
    assert (r.pasqal_runs, r.atoms) == (2, 8)


def test_mcts_keeps_only_the_first_k_pieces(monkeypatch):
    r, solved = scripted(monkeypatch, MATCHING, [(A, 0.5), (B, 0.3), (C, 0.2)], SOLS, {}, 40, k=2)
    assert C not in solved and B in solved


def test_mcts_takes_a_build_hook():
    G = erdos_renyi(30, 0.2, 2)
    build = lambda g: [{v: 32} for v in g]
    r = run_mcts(G, 4, 2, 10, score, rank, value, build=build)
    assert is_independent(G, r.solution) and all(x == 1 or x <= 5 for x in r.run_sizes)


def test_mcts_needs_rank_value_and_a_way_to_grow():
    G = erdos_renyi(20, 0.2, 1)
    for kw in (dict(score=score, value=value), dict(score=score, rank=rank), dict(rank=rank, value=value)):
        with pytest.raises(ValueError):
            run_mcts(G, 4, 2, 5, **kw)
