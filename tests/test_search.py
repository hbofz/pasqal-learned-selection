import networkx as nx, numpy as np, pytest
from test_bench import KEYS
from learned_selection.bench import run_instance
from learned_selection.engine import exact_piece_solver, gls_generator
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.instances import erdos_renyi
from learned_selection.models import fit_linear, fit_value_linear, save
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
