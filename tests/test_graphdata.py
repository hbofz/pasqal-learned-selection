import networkx as nx, numpy as np, pytest
from learned_selection.dataset import children_values, collect_states, mis_membership, q_values
from learned_selection.engine import _remove_closed, code_prune, exact_piece_solver, gls_generator, run_beam
from learned_selection.features import VERTEX_FEATURES
from learned_selection.graphdata import graph_of, label_graphs, state_record
from learned_selection.instances import TEST_SEEDS, erdos_renyi, make
from learned_selection.models import LinearModel, h2_score, save
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

def test_hook_counts_read_rows_from_the_records(capsys):
    from learned_selection.graphdata import hook_counts, summarize
    a = dict(n=10, member=np.zeros(10, np.int8), q=np.zeros(3, np.int64), pieces=[None] * 3, first=True, children=[],
             setting="s", instance_n=10)
    b = dict(n=7, member=np.zeros(7, np.int8), q=None, pieces=[None] * 2, first=False, children=[],
             setting="s", instance_n=10)
    c = dict(n=5, member=None, q=None, pieces=[None], first=False, children=[], setting="s", instance_n=10)
    assert hook_counts([a, b, c]) == {"h2": dict(states=2, rows=17), "h4": dict(states=1, rows=3)}
    meta = dict(instances=1, dropped=[], settings=[dict(setting="s", n=10, instances=1)], label_cpu_s=0.0)
    summarize([meta], [a, b, c])
    out = capsys.readouterr().out
    assert "h2: 2 states labelled (17 rows)" in out and "h4: 1 states labelled (3 rows)" in out

def test_no_policy_keeps_the_gls_states_and_candidates():
    G = make("er", 40, 3, p=0.1)
    a = collect_states(G, seed=3, max_states=1000)
    seen, generate = {}, gls_generator()     # the plain GLS beam, written out here so this test does not read its own code
    def record(g, k):
        seen.setdefault(frozenset(g), g)
        return generate(g, k)
    run_beam(G, record, exact_piece_solver, 4, 2, code_prune(4, 10, count_small=True))
    assert [set(x) for x in a] == [set(x) for x in seen.values()]
    rec = state_record(a[0], True, 20.0, 1)
    assert [set(m) for m in gls_generator()(a[0], len(a[0]))] == [set(p.tolist()) for p in rec["pieces"]]

def test_policy_changes_the_states_and_the_candidates():
    G = make("er", 60, 3, p=0.1)
    score = lambda g: (lambda grow, cands: [-g.degree(v) for v in cands])    # prefer low degree
    states = collect_states(G, seed=3, score=score)
    assert {frozenset(x) for x in states} != {frozenset(x) for x in collect_states(G, seed=3)}
    rec = state_record(G, True, 20.0, 1, score=score)
    gls = state_record(G, True, 20.0, 1)
    assert {frozenset(p.tolist()) for p in rec["pieces"]} != {frozenset(p.tolist()) for p in gls["pieces"]}
    assert (rec["member"] == gls["member"]).all() and rec["alpha"] == gls["alpha"]

def test_label_graphs_with_a_policy_uses_its_hooks(tmp_path):
    w = np.random.default_rng(0).normal(size=len(VERTEX_FEATURES))
    model = LinearModel(np.zeros(len(VERTEX_FEATURES)), np.ones(len(VERTEX_FEATURES)), w)
    save(model, tmp_path / "lin-h2.json")
    r = label_graphs("er", {"p": 0.1}, 40, 3, policy="lin-h2", models_dir=tmp_path)
    G = make("er", 40, 3, p=0.1)
    score = h2_score(model)
    states = collect_states(G, seed=3, score=score)
    assert not r["deferred"] and [rec["n"] for rec in r["states"]] == [len(g) for g in states]
    for rec, g in zip(r["states"], states):
        want = state_record(g, set(g) == set(G), 60.0, 1, score=score)
        assert [p.tolist() for p in rec["pieces"]] == [p.tolist() for p in want["pieces"]]
        assert rec["first"] == want["first"] and rec["alpha"] == want["alpha"]
    plain = label_graphs("er", {"p": 0.1}, 40, 3)
    assert [p.tolist() for p in r["states"][0]["pieces"]] != [p.tolist() for p in plain["states"][0]["pieces"]]

def test_label_graphs_refuses_selection_seeds():
    with pytest.raises(ValueError):
        label_graphs("er", {"p": 0.1}, 40, 105)
