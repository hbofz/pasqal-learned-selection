import contextlib, io, itertools, networkx as nx, pytest
from learned_selection.engine import (run_code_faithful, run_beam, code_prune, gls_generator,
                                      pasqal_greedy_piece_solver, exact_piece_solver)
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import load_coords
from learned_selection.solvers import exact_mis, is_independent, min_degree_greedy
from learned_selection.vendored import greedy_subgraph_solver_vv, weighted_generate_different_mis

CASES = list(itertools.product([20, 30, 40], [0.25, 0.5], [0, 1, 2], [(4, 2, 10), (2, 2, 5), (3, 1, 10)]))
# On the small cases above, big pieces only appear in round 1, so pruning never acts. These larger ones
# were chosen because pruning to l, always pruning to k, or never resetting the counter each changes the result.
PRUNING_CASES = [(120, 0.2, 2, (4, 2, 10)), (150, 0.1, 1, (2, 2, 5)), (150, 0.1, 2, (3, 1, 10)),
                 (150, 0.15, 0, (4, 2, 10)), (150, 0.1, 3, (2, 2, 5)), (150, 0.15, 3, (4, 2, 10))]
CASES += PRUNING_CASES

@pytest.mark.parametrize("n,p,seed,ksl", CASES)
def test_matches_vendored_breadthfirst_exactly(n, p, seed, ksl):
    k, s, l = ksl
    G = erdos_renyi(n, p, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        best, _, nodes = greedy_subgraph_solver_vv(G, load_coords(), 6.6, weighted_generate_different_mis).solve(
            "breadthfirst", exact_solving_threshold=5, subgraph_quantity=k,
            mis_sample_quantity=s, exploration_threshold=l)
    r = run_code_faithful(G, k, s, l, solve=pasqal_greedy_piece_solver())
    assert r.solution == frozenset(best) and r.pasqal_runs == len(nodes)

def test_counts_small_pieces():
    r = run_code_faithful(erdos_renyi(40, 0.75, 0), 4, 2, 10)
    assert r.quantum_runs > r.pasqal_runs and r.rounds >= 1

def test_own_rng_survives_vendored_reseeding():
    G = erdos_renyi(50, 0.2, 2)
    before = min_degree_greedy(G, 7)
    gls_generator()(G, 4)
    assert min_degree_greedy(G, 7) == before

def test_engine_on_string_labels():
    G = erdos_renyi(30, 0.25, 0)
    S = nx.relabel_nodes(G, lambda i: f"S{i}")
    r_int, r_str = run_code_faithful(G, 4, 2, 10), run_code_faithful(S, 4, 2, 10)
    assert is_independent(S, r_str.solution)
    assert r_str.solution == frozenset(f"S{i}" for i in r_int.solution)

@pytest.mark.parametrize("G", [nx.empty_graph(0), nx.empty_graph(7), nx.path_graph(4),
                               nx.disjoint_union_all([nx.path_graph(3), nx.path_graph(4), nx.empty_graph(3)])])
def test_engine_degenerate_graphs(G):
    nx.set_node_attributes(G, 1, "weight")
    r = run_code_faithful(G, 4, 2, 10)
    assert r.size == len(exact_mis(G)[0]) and is_independent(G, r.solution)

def test_engine_refuses_beam_explosion():
    G = nx.empty_graph(40)
    nx.set_node_attributes(G, 1, "weight")
    with pytest.raises(RuntimeError, match="beam exploded"):
        run_beam(G, gls_generator(), exact_piece_solver, 4, 2, code_prune(4, 10), max_branches=50)

def test_engine_accepts_short_solver_output():
    one = lambda piece, m, s: exact_piece_solver(piece, m, 1)
    r = run_code_faithful(erdos_renyi(40, 0.25, 1), 4, 3, 10, solve=one)
    assert is_independent(erdos_renyi(40, 0.25, 1), r.solution)
