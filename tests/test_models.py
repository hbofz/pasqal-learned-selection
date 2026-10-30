import networkx as nx, numpy as np, pytest
from learned_selection.engine import run_code_bounded
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES, VERTEX_FEATURES, graph_features
from learned_selection.instances import erdos_renyi
from learned_selection.models import (LinearModel, fit_lgbm, fit_linear, fit_value_linear, h2_score, h4_rank,
                                      linear_value, load, relevance, save, topk_regret)
from learned_selection.solvers import is_independent

def _planted(rng, groups=50, size=30, d=5):
    X = rng.standard_normal((groups * size, d))
    X[:, 3] = 1.0                                                  # a constant column
    g = np.repeat(np.arange(groups), size)
    y = np.round(2 * X[:, 0] - X[:, 1] + np.repeat(rng.integers(0, 5, groups), size)).astype(int)
    return X, y, g

def test_fit_linear_recovers_planted_order():
    X, y, g = _planted(np.random.default_rng(0))
    m = fit_linear(X, y, g)
    assert np.isfinite(m.w).all() and topk_regret(m.predict(X), y, g, 1) < 0.5

def test_group_weights_steer_the_fit():
    # two groups disagree about feature 0; the heavier group wins
    X = np.array([[1.0], [0.0], [1.0], [0.0]])
    y, g = np.array([1, 0, 0, 1]), np.array([0, 0, 1, 1])
    assert fit_linear(X, y, g, weights=np.array([9.0, 9.0, 1.0, 1.0])).w[0] > 0
    assert fit_linear(X, y, g, weights=np.array([1.0, 1.0, 9.0, 9.0])).w[0] < 0

def test_lgbm_beats_random():
    rng = np.random.default_rng(1)
    (X, y, g), (Xv, yv, gv) = _planted(rng), _planted(rng, groups=20)
    b = fit_lgbm(X, relevance(y, g), g, Xv, relevance(yv, gv), gv, eval_at=1)
    assert topk_regret(b.predict(Xv), yv, gv, 1) < topk_regret(rng.random(len(yv)), yv, gv, 1)

def test_topk_regret_by_hand():
    pred, y, g = np.array([0, 1, 2, 1, 0]), np.array([3, 1, 2, 5, 5]), np.array([0, 0, 0, 1, 1])
    assert topk_regret(pred, y, g, 1) == 0.5 and topk_regret(pred, y, g, 3) == 0.0

def test_size_only_h4_reproduces_gls():
    w = np.zeros(len(PIECE_FEATURES))
    w[0] = 1.0
    size_rule = LinearModel(np.zeros_like(w), np.ones_like(w), w)
    for seed in range(3):
        G = erdos_renyi(80, 0.25, seed)
        a, b = run_code_bounded(G, 4, 2, 10), run_code_bounded(G, 4, 2, 10, rank=h4_rank(size_rule))
        assert (a.solution, a.quantum_runs) == (b.solution, b.quantum_runs)

def test_learned_hooks_run_and_survive_string_labels():
    rng = np.random.default_rng(2)
    rand = lambda d: LinearModel(np.zeros(d), np.ones(d), rng.standard_normal(d))
    m2, m4 = rand(len(VERTEX_FEATURES)), rand(len(PIECE_FEATURES))
    G = erdos_renyi(60, 0.2, 0)
    S = nx.relabel_nodes(G, lambda i: f"T{i:03d}")                  # zero-padded: repr order equals int order
    a = run_code_bounded(G, 4, 2, 10, score=h2_score(m2), rank=h4_rank(m4))
    b = run_code_bounded(S, 4, 2, 10, score=h2_score(m2), rank=h4_rank(m4))
    assert is_independent(G, a.solution) and b.solution == frozenset(f"T{i:03d}" for i in a.solution)

def test_save_load_roundtrip(tmp_path):
    X, y, g = _planted(np.random.default_rng(3))
    models = {"m.json": fit_linear(X, y, g), "m.txt": fit_lgbm(X, relevance(y, g), g, X, relevance(y, g), g, eval_at=1)}
    for name, m in models.items():
        save(m, tmp_path / name)
        assert np.allclose(load(tmp_path / name).predict(X), m.predict(X), rtol=0, atol=1e-12)


def test_value_model_fits_an_intercept_and_old_files_still_load(tmp_path):
    X = np.random.default_rng(0).standard_normal((50, 10))
    y = X @ np.arange(10) + 3.0
    m = fit_value_linear(X, y, ridge=1e-9)
    assert np.allclose(m.predict(X), y, atol=1e-5)
    save(m, tmp_path / "v.json")
    assert np.allclose(load(tmp_path / "v.json").predict(X), m.predict(X))
    assert load("models/lin-h2.json").b == 0.0

def test_graph_features_on_a_star_and_on_nothing():
    f = dict(zip(GRAPH_FEATURES, graph_features(nx.star_graph(4))))
    assert f["n"] == 5 and f["edges"] == 4 and f["caro_wei"] == pytest.approx(4 * 0.5 + 0.2) and f["isolated"] == 0
    assert np.isfinite(graph_features(nx.empty_graph(0))).all()

def test_linear_value_hook_scores_empty_graphs_zero():
    m = LinearModel(np.zeros(10), np.ones(10), np.ones(10), 2.5)
    v = linear_value(m)([nx.empty_graph(0), nx.empty_graph(3)])
    assert v[0] == 0.0 and v[1] == pytest.approx(11.5) and linear_value(m)([]) == []
