import math, networkx as nx
from collections import Counter
from learned_selection.lattice import load_coords, lattice_graph, is_valid_piece, CENTER_TRAP
from learned_selection.vendored import Lattice, GreedyMapping

def test_fresnel_lattice_shape():
    c = load_coords()
    L = lattice_graph(c)
    assert sorted(c) == list(range(61))
    assert c[30] == (0.0, 0.0) and c[32] == (0.0, 17.320508) and CENTER_TRAP == 30
    assert Counter(d for _, d in L.degree()) == {3: 6, 4: 18, 6: 37}
    assert all(abs(math.dist(c[u], c[v]) - 5.0) < 1e-6 for u, v in L.edges())

def test_lattice_matches_vendored():
    c = load_coords()
    V = Lattice(c, 6.6)
    assert set(map(frozenset, V.lattice.edges())) == set(map(frozenset, lattice_graph(c).edges()))
    assert V.avg_degree == 5

def test_fit_checker_rejects_phantom_and_missing_edges():
    L = lattice_graph(load_coords())
    a = CENTER_TRAP
    b, c2 = next((u, v) for u in L[a] for v in L[a] if u < v and L.has_edge(u, v))   # a triangle
    path = nx.path_graph(3)                                                           # 0-1-2
    assert not is_valid_piece(path, {0: b, 1: a, 2: c2}, L)        # phantom edge 0-2
    far = next(t for t in L if t != a and not L.has_edge(a, t))
    assert not is_valid_piece(nx.path_graph(2), {0: a, 1: far}, L)  # missing edge
    assert not is_valid_piece(nx.path_graph(2), {0: a, 1: a}, L)    # two vertices on one trap
    assert is_valid_piece(nx.path_graph(2), {0: a, 1: b}, L)

def test_vendored_greedy_mappings_are_valid():
    c, L = load_coords(), lattice_graph(load_coords())
    for seed in range(5):
        G = nx.gnp_random_graph(40, 0.25, seed=seed)
        nx.set_node_attributes(G, 1, "weight")
        for v in G:
            m = GreedyMapping(G, Lattice(c, 6.6), []).generate_greedy_ud_subgraph_with(v)
            assert v in m and is_valid_piece(G, m, L)
