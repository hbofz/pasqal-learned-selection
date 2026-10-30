from learned_selection.engine import (paper_generator, paper_prune, run_paper_faithful, run_code_faithful,
                                      best_of_restarts, gls_generator)
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import CENTER_TRAP, load_coords, lattice_graph, is_valid_piece
from learned_selection.solvers import is_independent

def test_paper_generator_seeds_at_center_and_is_valid():
    G, L = erdos_renyi(50, 0.25, 0), lattice_graph(load_coords())
    maps = paper_generator(seed=3)(G, 4)
    assert len(maps) == 4 and all(CENTER_TRAP in m.values() and is_valid_piece(G, m, L) for m in maps)

def test_paper_prune_keeps_l():
    branches = [(None, list(range(i))) for i in range(25)]
    kept = paper_prune(10)(branches, 25)
    assert len(kept) == 10 and [len(b[1]) for b in kept] == list(range(24, 14, -1))

def test_paper_loop_valid_and_deterministic():
    G = erdos_renyi(60, 0.25, 1)
    a, b = run_paper_faithful(G, 4, 2, 10, seed=5), run_paper_faithful(G, 4, 2, 10, seed=5)
    assert a.solution == b.solution and is_independent(G, a.solution)

def test_restart_seed_changes_gls_tiebreaks():
    graphs = [erdos_renyi(40, 0.25, s) for s in range(5)]
    assert any(gls_generator(seed=0)(G, 4) != gls_generator(seed=1)(G, 4) for G in graphs)

def test_best_of_restarts_sums_costs():
    G = erdos_renyi(50, 0.25, 2)
    runs = [run_code_faithful(G, 4, 2, 10, seed=i) for i in range(5)]
    r = best_of_restarts(lambda i: run_code_faithful(G, 4, 2, 10, seed=i), 5)
    assert r.size == max(x.size for x in runs) and r.quantum_runs == sum(x.quantum_runs for x in runs)
    assert best_of_restarts(lambda i: run_code_faithful(G, 4, 2, 10, seed=i), 1).solution == runs[0].solution
