import networkx as nx, numpy as np, pytest
from learned_selection.dataset import children_values, mis_membership, q_values
from learned_selection.engine import _remove_closed, gls_generator
from learned_selection.graphdata import graph_of, label_graphs, state_record
from learned_selection.instances import TEST_SEEDS, erdos_renyi
from learned_selection.solvers import exact_mis, first_maximum_set, top_maximal_sets

def _sets(g, m):
    piece = g.subgraph(m)
    return [first_maximum_set(piece)] if len(m) <= 5 else top_maximal_sets(piece, 2)

def test_children_values_match_q_values_and_their_definition():
    g = erdos_renyi(30, 0.25, 0)
    maps = gls_generator()(g, len(g))
    q, children = children_values(g, maps)
    assert q == q_values(g, maps)
    first = [(list(_remove_closed(g, S)), len(exact_mis(_remove_closed(g, S))[0]), len(S)) for S in _sets(g, maps[0])]
    assert children[:len(first)] == first
    distinct = list({frozenset(m): m for m in maps}.values())
    assert len(children) == sum(len(_sets(g, m)) for m in distinct)

def test_children_values_stop_at_the_first_unproven_child(monkeypatch):
    import learned_selection.dataset as dataset
    calls, real = [], dataset.exact_mis
    def flaky(graph, time_limit_s, workers):
        calls.append(1)
        return (frozenset(), False) if len(calls) == 3 else real(graph, time_limit_s, workers)
    monkeypatch.setattr(dataset, "exact_mis", flaky)
    g = erdos_renyi(30, 0.25, 0)
    q, children = children_values(g, gls_generator()(g, len(g)))
    assert q is None and len(children) == 2 and len(calls) == 3

def test_commit_never_raises_outside_a_repository(monkeypatch, tmp_path):
    import subprocess
    import learned_selection.dataset as dataset
    def no_git(*a, **kw):
        raise subprocess.CalledProcessError(128, "git")
    monkeypatch.setattr(dataset.subprocess, "run", no_git)
    assert isinstance(dataset._commit(), str) and dataset._commit()

def test_state_record_roundtrips_a_mid_run_state():
    g = _remove_closed(erdos_renyi(50, 0.2, 1), [0, 7])         # labels with gaps
    nodes = list(g)
    rec = state_record(g, False, 20.0, 1)
    h = graph_of(rec)
    assert rec["n"] == len(g) and list(h) == list(range(len(g)))
    assert sorted(map(sorted, h.edges())) == sorted(sorted((nodes.index(u), nodes.index(v))) for u, v in g.edges())
    maps = gls_generator()(g, len(g))
    assert [p.tolist() for p in rec["pieces"]] == [[nodes.index(v) for v in m] for m in maps]
    assert rec["q"].tolist() == q_values(g, maps)
    member = mis_membership(g)
    assert rec["member"].tolist() == [member[v] for v in nodes] and rec["alpha"] == sum(member.values())
    _, children = children_values(g, maps)
    assert [(c.tolist(), a, s) for c, a, s in rec["children"]] == [([nodes.index(v) for v in r], a, s)
                                                                  for r, a, s in children]

def test_unproven_state_gets_no_labels():
    rec = state_record(erdos_renyi(200, 0.25, 0), True, 0.01, 1)
    assert rec["member"] is None and rec["alpha"] is None and rec["q"] is None

def test_label_graphs_defers_big_states_and_refuses_test_seeds():
    r = label_graphs("er", {"p": 0.1}, 40, 0, big=30)        # states of 40, 10, 8, 11, 9, 11, 11, 11 nodes
    assert r["deferred"] and all(len(g) > 30 for g, _ in r["deferred"])
    assert r["states"] and all(rec["n"] <= 30 for rec in r["states"])
    with pytest.raises(ValueError, match="test seed"):
        label_graphs("er", {"p": 0.25}, 40, TEST_SEEDS[0])
