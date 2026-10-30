import networkx as nx, pytest
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import load_coords
from learned_selection.solvers import (is_independent, exact_mis, top_maximal_sets,
                                       first_maximum_set, min_degree_greedy, repair)
from learned_selection.vendored import greedy_subgraph_solver_vv

def is_maximal(G, S):
    return is_independent(G, S) and all(v in S or any(u in S for u in G[v]) for v in G)

def test_erdos_renyi_is_deterministic_and_weighted():
    a, b = erdos_renyi(30, 0.25, 3), erdos_renyi(30, 0.25, 3)
    assert list(a.edges()) == list(b.edges()) and list(a) == list(range(30))
    assert all(a.nodes[v]["weight"] == 1 for v in a)

@pytest.mark.parametrize("G,alpha", [(nx.cycle_graph(5), 2), (nx.petersen_graph(), 4),
                                     (nx.complete_graph(6), 1), (nx.empty_graph(0), 0)])
def test_exact_mis_known_values(G, alpha):
    S, proven = exact_mis(G)
    assert len(S) == alpha and proven and is_independent(G, S)

def test_top_maximal_sets():
    P4 = nx.path_graph(4)
    sets = top_maximal_sets(P4, 5)
    assert sets[0] in ({0, 2}, {0, 3}, {1, 3}) and len(sets) == 3 and len(set(sets)) == 3
    assert all(len(S) == 2 and is_maximal(P4, S) for S in sets)
    assert top_maximal_sets(nx.star_graph(3), 2) == [frozenset({1, 2, 3}), frozenset({0})]

def test_first_maximum_set_matches_vendored():
    for seed in range(20):
        G = erdos_renyi(7, 0.4, seed)
        vendored = greedy_subgraph_solver_vv(G, load_coords(), 6.6, None)
        assert first_maximum_set(G) == vendored.find_maximum_independent_set(G)

def test_min_degree_greedy_and_repair():
    G = erdos_renyi(60, 0.2, 1)
    assert is_maximal(G, min_degree_greedy(G, 0))
    assert min_degree_greedy(G, 4) == min_degree_greedy(G, 4)
    T = nx.complete_graph(3)
    assert len(repair(T, {0, 1, 2})) == 1 and is_maximal(T, repair(T, {0, 1, 2}))
    assert repair(T, {0, 1, 2}) == repair(T, {0, 1, 2})
    assert is_maximal(nx.path_graph(4), repair(nx.path_graph(4), {0, 1}))

def test_solvers_accept_string_labels():
    G = nx.relabel_nodes(nx.petersen_graph(), {i: f"T{i}" for i in range(10)})
    G.remove_node("T3")
    assert len(exact_mis(G)[0]) == 4
    assert all(is_maximal(G, S) for S in top_maximal_sets(G, 3))
    assert is_maximal(G, min_degree_greedy(G, 0))

def test_top_maximal_sets_is_deterministic_on_gls_pieces():
    from learned_selection.engine import gls_generator
    for seed in range(4):
        G = erdos_renyi(60, 0.25, seed)
        for m in gls_generator()(G, 4):
            P = G.subgraph(m)
            assert len({tuple(top_maximal_sets(P, 2)) for _ in range(20)}) == 1
