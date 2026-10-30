from pathlib import Path

import networkx as nx, numpy as np, pytest, torch
from learned_selection.az import az_name, fine_tune_policy, self_play_job, soft_listwise_loss
from learned_selection.engine import exact_piece_solver, gls_generator
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.gnn import GNN, collate
from learned_selection.instances import erdos_renyi
from learned_selection.models import fit_linear, fit_value_linear, save
from learned_selection.search import run_mcts
from learned_selection.train import _encode_ranker, listwise_loss

score = lambda g: lambda grow, cands: [-g.degree(v) for v in cands]
rank = lambda g, maps: [len(m) for m in maps]
value = lambda gs: [len(g) / 3 for g in gs]


def test_trace_records_visits_over_expanded_candidates_of_all_pieces():
    # (seed 1's tree is exhausted at 14 root visits, under the 16 a trace needs)
    G = erdos_renyi(50, 0.15, 0); trace = []
    run_mcts(G, 4, 2, 40, score, rank, value, trace=trace)
    root = trace[0]
    assert root["n"] == 50 and root["visits"].sum() > 0 and len(root["visits"]) == len(root["expanded"]) <= 4
    assert len(root["pieces"]) == 50 and all(p.max() < e["n"] for e in trace for p in e["pieces"])


def test_trace_is_off_and_noise_is_seeded_by_default():
    G = erdos_renyi(50, 0.15, 1)
    assert run_mcts(G, 4, 2, 40, score, rank, value).solution == run_mcts(G, 4, 2, 40, score, rank, value, trace=[]).solution
    n = lambda: (0.25, 0.3, np.random.default_rng(5))
    assert run_mcts(G, 4, 2, 40, score, rank, value, noise=n()).solution == run_mcts(G, 4, 2, 40, score, rank, value, noise=n()).solution


def test_trace_keeps_a_node_only_from_sixteen_visits_and_indexes_the_generators_list():
    G = erdos_renyi(60, 0.15, 0); trace = []
    run_mcts(G, 4, 2, 60, score, rank, value, trace=trace)
    assert len(trace) > 1
    for e in trace:
        assert e["visits"].sum() >= 15   # a node of 16 visits sent its 15 later ones into its candidates
        # indices refer to the full ordered list, without duplicates among the expanded ones
        assert e["expanded"].dtype.kind == "i" and e["expanded"].max() < len(e["pieces"])
        assert len({tuple(sorted(e["pieces"][i].tolist())) for i in e["expanded"]}) == len(e["expanded"])
        assert (e["visits"] >= 0).all() and e["edges"].shape[1] == 2 and e["edges"].max() < e["n"]


def test_trace_positions_survive_string_labels():
    G = erdos_renyi(60, 0.15, 1)
    H = nx.relabel_nodes(G, {v: f"v{v:02d}" for v in G})
    a, b = [], []
    run_mcts(G, 4, 2, 40, score, rank, value, trace=a)
    run_mcts(H, 4, 2, 40, score, rank, value, trace=b)
    assert len(a) > 1 and len(a) == len(b) and all(
        x["n"] == y["n"] and (x["visits"] == y["visits"]).all() and (x["expanded"] == y["expanded"]).all()
        for x, y in zip(a, b))


def test_noise_perturbs_only_the_root_with_the_number_of_its_priors():
    G = erdos_renyi(50, 0.15, 1)
    firsts = []
    for hot in (0, 3):
        seen, solved = [], []

        class Rng:
            def dirichlet(self, alpha):
                seen.append(len(alpha))
                return np.eye(len(alpha))[hot]

        def solve(piece, m, s):
            solved.append(frozenset(m))
            return exact_piece_solver(piece, m, s)
        run_mcts(G, 4, 2, 6, score, rank, value, solve=solve, noise=(1.0, 0.3, Rng()))
        assert seen == [4]
        firsts.append(solved[0])
    assert firsts[0] != firsts[1]


def _linear_models(tmp_path):
    rng = np.random.default_rng(0)
    for hook, d in (("h2", len(VERTEX_FEATURES)), ("h4", len(PIECE_FEATURES))):
        X, g, y = rng.standard_normal((60, d)), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
        save(fit_linear(X, y, g), tmp_path / f"lin-{hook}.json")
    save(fit_value_linear(rng.standard_normal((40, len(GRAPH_FEATURES))), rng.integers(0, 9, 40)), tmp_path / "lin-h5.json")


def test_self_play_refuses_seeds_outside_training():
    for seed in (105, 1000, 200):
        with pytest.raises(ValueError):
            self_play_job(("er", {"p": 0.1}, 40, seed, "lin-h2h4+lin-h5+mcts", "models", 20, 4))


def test_self_play_returns_tagged_numpy_examples_with_visits(tmp_path):
    _linear_models(tmp_path)
    out = self_play_job(("er", {"p": 0.25}, 60, 3, "lin-h2h4+lin-h5+mcts", tmp_path, 60, 4))
    assert out and all(e["setting"] == "er(p=0.25)" and e["instance_n"] == 60 and e["seed"] == 3 for e in out)
    for e in out:
        assert e["visits"].sum() > 0 and len(e["visits"]) == len(e["expanded"]) <= 4
        assert all(isinstance(v, np.ndarray) for v in (e["edges"], e["expanded"], e["visits"], *e["pieces"]))


def test_soft_listwise_loss_agrees_with_listwise_on_uniform_best_targets():
    scores = torch.tensor([3.0, 0.5, 3.0, 1.0, 0.0, 2.0, -1.0], requires_grad=True)
    y = torch.tensor([2.0, 1.0, 2.0, 5.0, 0.0, 0.0, 0.0])
    group = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    w = torch.tensor([1.0, 2.0, 0.5])
    target = torch.tensor([0.5, 0.0, 0.5, 1.0, 0.0, 0.5, 0.5])
    a, b = listwise_loss(scores, y, group, w), soft_listwise_loss(scores, target, group, w)
    assert abs(float(a) - float(b)) < 1e-6
    (g1,), (g2,) = torch.autograd.grad(a, scores), torch.autograd.grad(b, scores)
    assert torch.allclose(g1, g2, atol=1e-6)


def test_soft_listwise_loss_is_the_weighted_cross_entropy_of_soft_targets():
    scores = torch.tensor([1.0, 2.0, 0.0, 0.0])
    target, group = torch.tensor([0.25, 0.75, 1.0, 0.0]), torch.tensor([0, 0, 1, 1])
    want = [-(0.25 * np.log(1 / (1 + np.e)) + 0.75 * np.log(np.e / (1 + np.e))), -np.log(0.5)]
    assert abs(float(soft_listwise_loss(scores, target, group, torch.tensor([1.0, 3.0]))) - (want[0] + 3 * want[1]) / 4) < 1e-6


def _example(seed=0, n=24, expanded=(0, 1, 2, 3), visits=(0, 0, 9, 0)):
    g = erdos_renyi(n, 0.2, seed)
    at = {v: i for i, v in enumerate(g)}
    maps = gls_generator(score=score)(g, len(g))
    return dict(n=n, edges=np.array([(at[u], at[v]) for u, v in g.edges()], dtype=np.int32).reshape(-1, 2),
                pieces=[np.array([at[v] for v in m], dtype=np.int16) for m in maps],
                expanded=np.array(expanded), visits=np.array(visits, dtype=float))


def _encode(ex):
    return collate([_encode_ranker((ex["n"], ex["edges"], ex["pieces"]))])


def test_fine_tune_policy_moves_mass_to_the_visited_piece():
    torch.manual_seed(0)
    ex = _example()
    net = GNN("gin", "piece", np.zeros(len(VERTEX_FEATURES)), np.ones(len(VERTEX_FEATURES)))
    b = _encode(ex)

    def scores():
        net.eval()
        with torch.inference_mode():
            return net(b.to("cpu")).numpy()[ex["expanded"]]
    margin = lambda x: x[2] - np.delete(x, 2).max()
    before = margin(scores())
    info = fine_tune_policy(net, [], [ex], epochs=10, seed=0)
    after = scores()
    assert after.argmax() == 2 and margin(after) > before
    assert info["states"] == 0 and info["examples"] == 1 and len(info["loss"]) == 10 and info["loss"][-1] < info["loss"][0]


def test_fine_tune_policy_trains_on_both_kinds_and_weights_the_best_q_uniformly():
    torch.manual_seed(0)
    ex = _example(1)
    g = erdos_renyi(24, 0.2, 2)
    at = {v: i for i, v in enumerate(g)}
    maps = gls_generator(score=score)(g, len(g))
    q = np.zeros(len(maps), dtype=np.int64); q[[1, 4]] = 7
    qs = dict(n=24, edges=np.array([(at[u], at[v]) for u, v in g.edges()], dtype=np.int32).reshape(-1, 2),
              pieces=[np.array([at[v] for v in m], dtype=np.int16) for m in maps], q=q)
    net = GNN("gin", "piece", np.zeros(len(VERTEX_FEATURES)), np.ones(len(VERTEX_FEATURES)))
    info = fine_tune_policy(net, [qs, dict(qs, q=None)], [ex], epochs=2, batch=2)
    assert info["states"] == 1 and info["examples"] == 1 and all(np.isfinite(info["loss"])) and not net.training


def test_az_name_swaps_the_h4_part():
    assert az_name("lin-h2+gin-h4+sage-h5+mcts-c0.5") == "lin-h2+az-h4+sage-h5+mcts-c0.5"
    assert az_name("lin-h2h4+sage-h5+mcts") == "lin-h2+az-h4+sage-h5+mcts"
    assert az_name("gin-h2h4h5+mcts") == "gin-h2+az-h4+gin-h5+mcts"


def test_loop_tunes_on_every_iteration_of_self_play_so_far(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from learned_selection import az
    models = tmp_path / "models"
    models.mkdir()
    for name in ("gin-h4.pt", "lin-h2.json", "sage-h5.pt"):
        (models / name).touch()
    calls = []

    def fake_run(*args):
        args = [str(x) for x in args]
        calls.append(args)
        target = Path(args[args.index("--out") + 1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.touch()
    monkeypatch.setattr(az, "_run", fake_run)
    az.loop(SimpleNamespace(method="lin-h2+gin-h4+sage-h5+mcts", models=models, iters=3, budget=15, k=4,
                            base_train=["base.pkl"], out=tmp_path / "out", cache=tmp_path / "cache", workers=1,
                            threads=1, shard="0/160", max_states=None))
    tunes = [c for c in calls if c[0] == "tune"]
    examples = lambda c: [Path(x).name for x in c[c.index("--examples") + 1:c.index("--out")]]
    assert [examples(c) for c in tunes] == [
        [f"selfplay_it{i}_shard0of160.pkl" for i in range(1, t + 1)] for t in (1, 2, 3)]
    plays = [c for c in calls if c[0] == "selfplay"]
    assert [c[c.index("--method") + 1] for c in plays] == ["lin-h2+gin-h4+sage-h5+mcts"] + ["lin-h2+az-h4+sage-h5+mcts"] * 2
