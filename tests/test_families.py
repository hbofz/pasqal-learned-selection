import math, networkx as nx
from learned_selection.instances import (make, erdos_renyi, factor_market, sbm, unit_disk, TRAIN_SETTINGS,
                                         HELD_OUT_SETTINGS, TRAIN_SEEDS, VAL_SEEDS, TEST_SEEDS)

def test_families_are_deterministic_weighted_and_int_labelled():
    for fam, params in TRAIN_SETTINGS + HELD_OUT_SETTINGS:
        a, b = make(fam, 60, 3, **params), make(fam, 60, 3, **params)
        assert list(a) == list(range(60)) and sorted(a.edges()) == sorted(b.edges())
        assert all(a.nodes[v]["weight"] == 1 for v in a)

def test_make_matches_erdos_renyi():
    assert sorted(make("er", 30, 3, p=0.25).edges()) == sorted(erdos_renyi(30, 0.25, 3).edges())

def test_crisis_is_denser_than_calm():
    assert nx.density(factor_market(150, 0.3, 0, market=1.0)) > 2 * nx.density(factor_market(150, 0.3, 0))

def test_sbm_blocks_are_denser_inside():
    G = sbm(120, 4, 0.3, 0.03, 0)
    block = nx.get_node_attributes(G, "block")
    inside = sum(block[u] == block[v] for u, v in G.edges())
    assert inside > 2 * (G.number_of_edges() - inside)

def test_unit_disk_edges_follow_distance():
    G = unit_disk(80, 0.15, 0)
    pos = nx.get_node_attributes(G, "pos")
    assert all((math.dist(pos[u], pos[v]) <= 0.15) == G.has_edge(u, v) for u in G for v in G if u < v)

def test_splits_do_not_overlap():
    assert not set(TRAIN_SEEDS) & set(VAL_SEEDS) and not (set(TRAIN_SEEDS) | set(VAL_SEEDS)) & set(TEST_SEEDS)
    assert not {f for f, _ in TRAIN_SETTINGS} & {"unit_disk"}
    assert all(p.get("market", 0.5) == 0.5 for f, p in TRAIN_SETTINGS if f == "factor")
