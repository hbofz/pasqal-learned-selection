import inspect, networkx as nx, numpy as np, pytest, torch
from learned_selection import gnn
from learned_selection.engine import gls_generator, run_code_bounded
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.gnn import (CONVS, GNN, gnn_rank, gnn_score, gnn_value, graph_values, load_gnn, piece_scores,
                                   save_gnn, vertex_scores)
from learned_selection.instances import erdos_renyi

def _net(conv, head, seed=0):
    torch.manual_seed(seed)
    d, p = len(VERTEX_FEATURES), len(PIECE_FEATURES)
    return GNN(conv, head, np.zeros(d), np.ones(d), np.zeros(p), np.ones(p)).eval()

def _weighted(G):
    nx.set_node_attributes(G, 1, "weight")
    return G

DEGENERATE = [nx.empty_graph(0), nx.empty_graph(1), nx.empty_graph(6), nx.star_graph(7), nx.complete_graph(5)]

@pytest.mark.parametrize("conv", CONVS)
def test_heads_are_finite_on_degenerate_graphs(conv):
    for G in map(_weighted, DEGENERATE):
        s = vertex_scores(_net(conv, "vertex"), G)
        assert s.shape == (len(G),) and np.isfinite(s).all()
        maps = gls_generator()(G, len(G)) if len(G) else []
        p = piece_scores(_net(conv, "piece"), G, maps)
        assert p.shape == (len(maps),) and np.isfinite(p).all()
    v = graph_values(_net(conv, "value"), [_weighted(G) for G in DEGENERATE])
    assert v.shape == (len(DEGENERATE),) and np.isfinite(v).all() and v[0] == 0.0

@pytest.mark.parametrize("conv", CONVS)
def test_outputs_do_not_depend_on_node_labels_or_order(conv):
    G = erdos_renyi(30, 0.2, 0)
    perm = [int(x) for x in np.random.default_rng(0).permutation(30)]
    H = nx.Graph()
    H.add_nodes_from(perm[v] for v in reversed(range(30)))
    H.add_edges_from((perm[u], perm[v]) for u, v in G.edges())
    _weighted(H)
    a = dict(zip(G, vertex_scores(_net(conv, "vertex"), G)))
    b = dict(zip(H, vertex_scores(_net(conv, "vertex"), H)))
    assert all(a[v] == pytest.approx(b[perm[v]], abs=1e-5) for v in G)
    maps = gls_generator()(G, 10)
    pa = piece_scores(_net(conv, "piece"), G, maps)
    pb = piece_scores(_net(conv, "piece"), H, [{perm[v]: t for v, t in m.items()} for m in maps])
    assert np.allclose(pa, pb, atol=1e-5)
    va, vb = graph_values(_net(conv, "value"), [G, H])
    assert va == pytest.approx(vb, abs=1e-4)

@pytest.mark.parametrize("conv", CONVS)
def test_batched_values_equal_one_at_a_time(conv):
    graphs = [erdos_renyi(n, 0.3, n) for n in (0, 5, 17, 40)]
    net = _net(conv, "value")
    assert np.allclose(graph_values(net, graphs), [graph_values(net, [g])[0] for g in graphs], atol=1e-5)

def test_save_load_roundtrip(tmp_path):
    G = erdos_renyi(25, 0.2, 3)
    maps = gls_generator()(G, 6)
    out = {"vertex": lambda m: vertex_scores(m, G), "piece": lambda m: piece_scores(m, G, maps),
           "value": lambda m: graph_values(m, [G])}
    for conv in CONVS:
        for head, f in out.items():
            net = _net(conv, head, seed=1)
            save_gnn(net, tmp_path / f"{conv}-{head}.pt")
            assert np.array_equal(f(load_gnn(tmp_path / f"{conv}-{head}.pt")), f(net))

def test_gnn_never_solves():
    src = inspect.getsource(gnn)
    assert "solvers" not in src and "vendored" not in src

def test_gnn_hooks_run_in_the_engine_and_survive_string_labels():
    G = erdos_renyi(50, 0.2, 0)
    S = nx.relabel_nodes(G, lambda i: f"T{i:03d}")                  # zero-padded: repr order equals int order
    hooks = dict(score=gnn_score(_net("gin", "vertex")), rank=gnn_rank(_net("sage", "piece")),
                 value=gnn_value(_net("gat", "value")))
    a, b = run_code_bounded(G, 4, 2, 10, **hooks), run_code_bounded(S, 4, 2, 10, **hooks)
    assert b.solution == frozenset(f"T{i:03d}" for i in a.solution) and a.quantum_runs == b.quantum_runs
