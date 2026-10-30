import contextlib, io, itertools, networkx as nx, pytest
from learned_selection.engine import (run_code_faithful, run_beam, code_prune, gls_generator, vendored_gls_generator,
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
    # Pasqal's piece solver draws from the global RNG, which only the vendored generator reseeds
    r = run_beam(G, vendored_gls_generator(), pasqal_greedy_piece_solver(), k, s, code_prune(k, l))
    assert r.solution == frozenset(best) and r.pasqal_runs == len(nodes)

@pytest.mark.parametrize("n,p,seed,ksl", CASES[:54:9] + PRUNING_CASES)
def test_port_reproduces_vendored_runs(n, p, seed, ksl):
    k, s, l = ksl
    G = erdos_renyi(n, p, seed)
    a = run_code_faithful(G, k, s, l)
    b = run_beam(G, vendored_gls_generator(), exact_piece_solver, k, s, code_prune(k, l))
    assert (a.solution, a.quantum_runs, a.pasqal_runs, a.atoms) == (b.solution, b.quantum_runs, b.pasqal_runs, b.atoms)

def test_counts_small_pieces():
    r = run_code_faithful(erdos_renyi(40, 0.75, 0), 4, 2, 10)
    assert r.quantum_runs > r.pasqal_runs and r.rounds >= 1

def test_own_rng_survives_vendored_reseeding():
    import random
    from learned_selection.engine import paper_generator
    G = erdos_renyi(50, 0.2, 2)
    random.seed(123); before, maps = min_degree_greedy(G, 7), paper_generator(seed=4)(G, 4)
    gls_generator()(G, 4); random.seed(999)
    assert min_degree_greedy(G, 7) == before and paper_generator(seed=4)(G, 4) == maps

def test_code_prune_running_counter():
    p = code_prune(2, 10)
    lv = [(None, [0] * i) for i in range(6)]
    assert p(lv, 6) == lv                                   # counter 6: no prune
    assert [len(b[1]) for b in p(lv, 6)] == [5, 4]         # counter 12 > 10: keep k=2 best, reset
    assert p(lv, 6) == lv                                   # counter back to 6

def test_only_solver_children_count_toward_pruning():
    seen, returned = [], []
    inner = code_prune(4, 10)
    spy_prune = lambda level, children: (seen.append(children), inner(level, children))[1]
    def spy_solve(piece, m, s):
        out = exact_piece_solver(piece, m, s)
        returned.append(len(out))
        return out
    r = run_beam(erdos_renyi(40, 0.75, 0), gls_generator(), spy_solve, 4, 2, spy_prune)
    assert r.quantum_runs > r.pasqal_runs          # small pieces did occur
    assert sum(seen) == sum(returned)              # yet only solver children were counted

def test_generators_require_trap_ids_in_order():
    c = load_coords()
    shuffled = {t: c[t] for t in reversed(list(c))}
    with pytest.raises(ValueError):
        gls_generator(coords=shuffled)

def test_engine_on_string_labels():
    G = erdos_renyi(30, 0.25, 0)
    S = nx.relabel_nodes(G, lambda i: f"S{i:02d}")   # zero-padded: repr order equals int order
    r_int, r_str = run_code_faithful(G, 4, 2, 10), run_code_faithful(S, 4, 2, 10)
    assert is_independent(S, r_str.solution)
    assert r_str.solution == frozenset(f"S{i:02d}" for i in r_int.solution)

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

GAP_CASES = list(itertools.product([30, 40], range(6), ["drop0", "drop3711", "shift100"]))

def _gap_variant(G, variant):
    if variant == "drop0":
        G.remove_node(0)
    elif variant == "drop3711":
        G.remove_nodes_from([3, 7, 11])
    else:
        G = nx.relabel_nodes(G, lambda v: v + 100)
    return G

@pytest.mark.parametrize("n,seed,variant", GAP_CASES)
def test_matches_vendored_on_gapped_and_shifted_int_labels(n, seed, variant):
    G = _gap_variant(erdos_renyi(n, 0.25, seed), variant)
    with contextlib.redirect_stdout(io.StringIO()):
        best, _, nodes = greedy_subgraph_solver_vv(G, load_coords(), 6.6, weighted_generate_different_mis).solve(
            "breadthfirst", exact_solving_threshold=5, subgraph_quantity=4, mis_sample_quantity=2, exploration_threshold=10)
    r = run_beam(G, vendored_gls_generator(), pasqal_greedy_piece_solver(), 4, 2, code_prune(4, 10))
    assert r.solution == frozenset(best) and r.pasqal_runs == len(nodes)

@pytest.mark.parametrize("n,seed,variant", GAP_CASES[::3])
def test_port_reproduces_vendored_runs_on_gapped_labels(n, seed, variant):
    G = _gap_variant(erdos_renyi(n, 0.25, seed), variant)
    a = run_code_faithful(G, 4, 2, 10)
    b = run_beam(G, vendored_gls_generator(), exact_piece_solver, 4, 2, code_prune(4, 10))
    assert (a.solution, a.quantum_runs, a.atoms) == (b.solution, b.quantum_runs, b.atoms)

HASH_SCRIPT = """
import networkx as nx
from learned_selection.engine import run_code_faithful
from learned_selection.instances import erdos_renyi
from learned_selection.solvers import top_maximal_sets
S = nx.relabel_nodes(erdos_renyi(120, 0.25, 0), lambda i: f"T{i:03d}")
view = S.subgraph([f"T{i:03d}" for i in range(0, 120, 2)])     # < half the nodes: iterates a set of str
r = run_code_faithful(view, 4, 2, 10)
print(sorted(r.solution), r.pasqal_runs, r.quantum_runs)
print([sorted(x) for x in top_maximal_sets(S.subgraph([f"T{i:03d}" for i in range(0, 50, 2)]), 3)])
"""

def test_string_label_views_reproduce_across_processes():
    import os, subprocess, sys
    outs = {subprocess.run([sys.executable, "-W", "ignore", "-c", HASH_SCRIPT], capture_output=True, text=True,
                           env={**os.environ, "PYTHONHASHSEED": h}, check=True).stdout for h in ("1", "2", "3")}
    assert len(outs) == 1, outs

def _bad_solver(kind):
    def solve(piece, m, s):
        good = exact_piece_solver(piece, m, s)
        return {"none": [], "empty": [frozenset()], "outside": [good[0] | {"stranger"}],
                "dependent": [frozenset(list(piece.edges())[0])]}[kind]
    return solve

@pytest.mark.parametrize("kind", ["none", "empty", "outside", "dependent"])
def test_engine_rejects_bad_solver_output(kind):
    with pytest.raises(ValueError):
        run_code_faithful(erdos_renyi(40, 0.25, 0), 4, 2, 10, solve=_bad_solver(kind))

def test_engine_rejects_empty_generator_output():
    with pytest.raises(ValueError):
        run_beam(erdos_renyi(40, 0.25, 0), lambda g, k: [{}], exact_piece_solver, 4, 2, code_prune(4, 10))
