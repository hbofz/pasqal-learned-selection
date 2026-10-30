import inspect, networkx as nx, numpy as np, pytest
from learned_selection import features
from learned_selection.dataset import collect_states, label_instance, mis_membership, q_values
from learned_selection.engine import _remove_closed, gls_generator
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES, piece_features, vertex_features
from learned_selection.instances import TEST_SEEDS, erdos_renyi
from learned_selection.solvers import exact_mis, first_maximum_set, is_independent, top_maximal_sets

def test_features_shapes_and_finite_on_degenerate_states():
    for G in (nx.empty_graph(8), nx.star_graph(7), erdos_renyi(40, 0.25, 0)):
        nx.set_node_attributes(G, 1, "weight")
        nodes, X = vertex_features(G)
        maps = gls_generator()(G, len(G))
        P = piece_features(G, maps)
        assert nodes == list(G) and X.shape == (len(G), len(VERTEX_FEATURES)) and np.isfinite(X).all()
        assert P.shape == (len(maps), len(PIECE_FEATURES)) and np.isfinite(P).all()
        assert list(P[:, 0]) == [len(m) for m in maps]

def test_feature_values_on_a_star():
    G = nx.star_graph(3)   # centre 0, leaves 1 to 3
    nodes, X = vertex_features(G)
    f = dict(zip(VERTEX_FEATURES, X.T))
    assert nodes == [0, 1, 2, 3]
    assert list(f["deg_frac"]) == [.75, .25, .25, .25] and list(f["nbr_deg_mean"]) == [.25, .75, .75, .75]
    assert list(f["gls_term"]) == [.5, 0, 0, 0] and list(f["lower_nbrs"]) == [1, 0, 0, 0]
    assert np.allclose(f["deg_rel"][:2], [1, 1 / 3]) and list(f["core_rel"]) == [1] * 4 and f["density"][0] == .5
    P = dict(zip(PIECE_FEATURES, piece_features(G, [{0: 0, 1: 1}, {1: 0}]).T))
    assert list(P["size"]) == [2, 1] and list(P["size_frac"]) == [.5, .25] and list(P["piece_density"]) == [1, 0]
    assert list(P["deg_mean"]) == [.5, .25] and list(P["deg_min"]) == [.25, .25] and list(P["deg_max"]) == [.75, .25]
    assert list(P["closed_nbhd"]) == [1, .5] and list(P["boundary"]) == [.5, 1]
    assert list(P["interior"]) == [.5, 0] and list(P["local_min"]) == [.5, 1] and list(P["size_rank"]) == [0, .5]

def test_features_never_solve():
    src = inspect.getsource(features)
    assert "solvers" not in src and "vendored" not in src

def test_q_values_on_whole_components():
    G = nx.disjoint_union(nx.path_graph(4), nx.complete_graph(3))
    nx.set_node_attributes(G, 1, "weight")
    maps = gls_generator()(G, len(G))
    assert q_values(G, maps) == [3] * len(maps)

def test_q_values_match_definition():
    G = erdos_renyi(30, 0.25, 0)
    maps = gls_generator()(G, len(G))
    q = q_values(G, maps)
    sets = top_maximal_sets(G.subgraph(maps[0]), 2)
    assert q[0] == max(len(S) + len(exact_mis(_remove_closed(G, S))[0]) for S in sets)
    assert max(q) <= len(exact_mis(G)[0])

def test_labels_are_none_when_unproven():
    G = erdos_renyi(200, 0.25, 0)
    assert mis_membership(G, time_limit_s=0.01) is None
    S, proven = exact_mis(G, time_limit_s=0.001, workers=1)
    assert not proven and is_independent(G, S)

def test_q_values_use_the_committed_set_for_small_pieces():
    # piece 0-1-2 with pendants 3 (on 0) and 4 (on 2). The engine commits first_maximum_set = [0, 2], which removes
    # everything (total 2). Taking {1} instead would leave 3 and 4 (total 3), but the engine never branches on it.
    G = nx.Graph([(0, 1), (1, 2), (0, 3), (2, 4)])
    nx.set_node_attributes(G, 1, "weight")
    assert first_maximum_set(G.subgraph([0, 1, 2])) == [0, 2]
    assert q_values(G, [{0: 0, 1: 1, 2: 2}]) == [2]

def test_label_instance_defers_big_states():
    r = label_instance("er", {"p": 0.25}, 40, 0, big=30)
    assert r["deferred"] and all(len(g) > 30 for g, _ in r["deferred"])
    assert all(x[2] <= 30 for x in r["h2"] + r["h4"])

def test_collect_states_deterministic_and_distinct():
    G = erdos_renyi(80, 0.25, 1)
    a, b = collect_states(G, max_states=12, seed=0), collect_states(G, max_states=12, seed=0)
    assert [set(g) for g in a] == [set(g) for g in b]
    assert len({frozenset(g) for g in a}) == len(a) <= 12 and all(len(g) > 5 for g in a)

def test_label_instance_refuses_test_seeds():
    with pytest.raises(ValueError, match="test seed"):
        label_instance("er", {"p": 0.25}, 40, TEST_SEEDS[0])
