import json

import networkx as nx

from learned_selection.engine import gls_generator, run_code_bounded
from learned_selection.evolve import evolve, objective
from learned_selection.growth import Growth
from learned_selection.instances import erdos_renyi
from learned_selection.models import GLS_THETA, decode, load_theta, theta_score
from learned_selection.solvers import exact_mis

def test_gls_theta_reproduces_gls():
    for seed in range(3):
        G = erdos_renyi(80, 0.25, seed)
        a, b = run_code_bounded(G, 4, 2, 10), run_code_bounded(G, 4, 2, 10, score=theta_score(GLS_THETA))
        assert (a.solution, a.quantum_runs, a.atoms) == (b.solution, b.quantum_runs, b.atoms)

def test_decode_origin_is_gls():
    assert decode([0, 0, 0]) == GLS_THETA
    assert decode([1, -2, 3]) == (10.0, -2.0, 3.0)

def test_theta_terms_by_hand():
    grow = Growth(nx.path_graph(4))      # vertex 0 is placed; candidates 1 (degree 2, one placed neighbour) and 3
    grow.mapping[0] = 0
    scores = lambda theta: theta_score(theta)(grow.graph)(grow, [1, 3])
    assert scores((5.0, 0.0, 0.0)) == [0.25, 0.0]
    assert scores((5.0, 1.0, 0.0)) == [0.5, 0.0]
    assert scores((5.0, 0.0, 1.0)) == [0.75, 0.25]
    assert scores((3.0, 0.0, 0.0)) == [0.75, 0.5]

def test_theta_changes_pieces():
    G = erdos_renyi(80, 0.25, 0)
    assert gls_generator(score=theta_score((20.0, 0.0, 0.0)))(G, 4) != gls_generator()(G, 4)

def test_evolve_never_returns_worse_than_gls():
    insts = [erdos_renyi(40, 0.25, s) for s in range(3)]
    optima = [len(exact_mis(G)[0]) for G in insts]
    theta, best, history = evolve(insts, optima, generations=3, popsize=4, workers=1)
    assert len(theta) == 3 and best <= objective(GLS_THETA, insts, optima) and len(history) == 3

def test_evolve_returns_best_seen_and_pool_matches_serial():
    insts = [erdos_renyi(30, 0.25, s) for s in range(2)]
    optima = [len(exact_mis(G)[0]) for G in insts]
    serial = evolve(insts, optima, generations=2, popsize=4, workers=1)
    assert serial[1] == min(objective(GLS_THETA, insts, optima), *serial[2])
    assert serial[1] == objective(serial[0], insts, optima)
    assert evolve(insts, optima, generations=2, popsize=4, workers=2) == serial

def test_load_theta_reads_saved_result(tmp_path):
    path = tmp_path / "cma.json"
    path.write_text(json.dumps({"theta": [6.5, 0.25, -1.0], "objective": 3.0}))
    assert load_theta(path) == (6.5, 0.25, -1.0)
