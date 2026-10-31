import json
import networkx as nx, numpy as np, pytest, torch
from test_builder import _lattice, _models, _net, _rec, _write_states
from learned_selection import builder, gfn, solvers
from learned_selection.bench import known_method, parse_method, run_instance
from learned_selection.builder import build_piece, builder_hook, load_builder, pair_weights, save_builder
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES
from learned_selection.gfn import gfn_hook, main, parent_count, path_logp, state_seed, tb_loss, tb_rollout
from learned_selection.gnn import encode
from learned_selection.instances import erdos_renyi
from learned_selection.lattice import is_valid_piece, lattice_graph, load_coords
from learned_selection.models import fit_linear, fit_value_linear, save


def _sampled(G, net, start, seed):
    H = net.embed(encode(G).to("cpu")).detach().numpy()
    return build_piece(G, H, pair_weights(net), start, *_lattice(), np.random.default_rng(seed))


def test_parent_count_on_a_path_and_a_triangle():
    adj, start_trap, _ = _lattice()
    assert start_trap == 32 and 23 in adj[32] and 14 in adj[23] and 14 not in adj[32]
    assert parent_count({0: 32, 1: 23, 2: 14}, adj, 32) == 1          # only the far end can go, the middle one holds the rest
    assert 27 in adj[32] and 27 in adj[23]
    assert parent_count({0: 32, 1: 23, 2: 27}, adj, 32) == 2          # either of the two others can go
    assert parent_count({0: 32}, adj, 32) == 1                        # at least 1


def _by_networkx(place, L, start_trap):
    """Placed traps, not the start, whose removal leaves the rest connected on the lattice graph."""
    occ = set(place.values())
    return max(sum(nx.is_connected(L.subgraph(occ - {t})) for t in occ - {start_trap}), 1)


def test_parent_count_matches_removal_and_connectivity_on_built_pieces():
    adj, start_trap, _ = _lattice()
    L, net = lattice_graph(load_coords()), _net()
    for seed, G in enumerate((erdos_renyi(30, 0.2, 3), erdos_renyi(40, 0.1, 4), erdos_renyi(25, 0.3, 5))):
        order = list(_sampled(G, net, seed, seed)[0].items())
        for i in range(1, len(order) + 1):
            assert parent_count(dict(order[:i]), adj, start_trap) == _by_networkx(dict(order[:i]), L, start_trap)
    far = next(t for t in range(len(adj)) if t not in (32, 23, 14) and not any(t in adj[o] for o in (32, 23, 14)))
    apart = {0: 32, 1: 23, 2: 14, 3: far}      # the last one is not on the lattice next to the rest
    assert not nx.is_connected(L.subgraph(apart.values()))
    assert parent_count(apart, adj, 32) == _by_networkx(apart, L, 32) == 1      # only the stray one can go


def _hand_logp(net, G, decisions):
    """The path's log probability from the pair network's logits and an independent log softmax."""
    H, (w1, b1, w2, b2) = net.embed(encode(G).to("cpu")).detach().numpy(), pair_weights(net)
    total = 0.0
    for idx, phi, chosen in decisions:
        logit = (np.maximum(np.c_[H[idx], phi] @ w1.T + b1, 0) @ w2.T + b2).ravel().astype(np.float64)
        total += logit[chosen] - (logit.max() + np.log(np.exp(logit - logit.max()).sum()))
    return total


def test_path_logp_is_the_sum_of_log_softmaxes():
    G, net, rec = erdos_renyi(30, 0.2, 3), _net(), _rec(30, 0.2, 3)
    paths = [_sampled(G, net, v, 2 * v)[1] for v in (0, 1, 2)]
    rows = [(0, 0.0, 0.0, d, 0.0, 9) for d in paths]
    got = path_logp(net, rec["n"], rec["edges"], rows)
    for lp, d in zip(got, paths):
        assert float(lp) == pytest.approx(_hand_logp(net, G, d), abs=1e-4) and float(lp) < 0


def test_rollout_log_pb_follows_the_insertion_order(tmp_path):
    import math, random
    from learned_selection.graphdata import graph_of
    _models(tmp_path)
    rec, net = _rec(30, 0.15, 2), _net()
    n, edges, rows = tb_rollout((builder.blob_of(net), rec["n"], rec["edges"], 2, 3, 7, 2.0, "gin-h5", str(tmp_path)))
    g, (adj, start_trap, trap_x) = graph_of(dict(n=n, edges=edges)), _lattice()
    H, weights, rng, want = net.embed(encode(g).to("cpu")).detach().numpy(), pair_weights(net), np.random.default_rng(7), []
    for v in random.Random(7).sample(list(g), 2):
        for _ in range(3):
            piece, decisions = build_piece(g, H, weights, v, adj, start_trap, trap_x, rng)
            order = list(piece.items())
            want.append(-sum(math.log(parent_count(dict(order[:i]), adj, start_trap)) for i in range(2, len(order) + 1)))
    assert len(rows) == 6 and [r[2] for r in rows] == pytest.approx(want) and min(want) < 0


def test_loss_penalises_the_backward_term_with_the_right_sign():
    G, net, rec = erdos_renyi(30, 0.2, 3), _net(), _rec(30, 0.2, 3)
    a, b = _sampled(G, net, 0, 1)[1], _sampled(G, net, 0, 2)[1]
    la, lb = (float(x) for x in path_logp(net, rec["n"], rec["edges"], [(0, 0, 0, a, 0, 9), (0, 0, 0, b, 0, 9)]))
    rows = [(0, 0.5, -1.0, a, 0.0, 9), (0, 0.5, -4.0, b, 0.0, 9)]       # same flow but for the backward term
    loss, _ = tb_loss(net, rec["n"], rec["edges"], rows)
    assert float(loss) == pytest.approx(((la - lb) - (-1.0 - -4.0)) ** 2 / 2, abs=1e-4)


def test_state_seed_follows_the_label_set_and_not_the_order():
    G = erdos_renyi(30, 0.2, 1)
    H = nx.Graph(); H.add_nodes_from(reversed(list(G))); H.add_edges_from(G.edges)
    assert state_seed(G) == state_seed(H) and state_seed(G)[:2] == [30, G.number_of_edges()]
    assert state_seed(G) != state_seed(erdos_renyi(30, 0.2, 2))


def test_flow_loss_is_zero_when_residuals_agree():
    G, net, rec = erdos_renyi(30, 0.2, 3), _net(), _rec(30, 0.2, 3)
    path = _sampled(G, net, 0, 1)[1]
    other = _sampled(G, net, 5, 2)[1]
    same = [(0, 0.7, -1.5, path, 1.0, 9), (0, 0.7, -1.5, path, 1.0, 9)]       # equal paths, equal flows: equal residuals
    loss, count = tb_loss(net, rec["n"], rec["edges"], same)
    assert float(loss) == pytest.approx(0, abs=1e-5) and count == 2
    d = 1.3
    shifted = [same[0], (0, 0.7 + d, -1.5, path, 1.0, 9)]
    loss, count = tb_loss(net, rec["n"], rec["edges"], shifted)
    assert float(loss) == pytest.approx(d ** 2 / 2, abs=1e-5) and count == 2
    alone = shifted + [(5, 3.0, -9.0, other, 1.0, 9)]                          # a start with one sample adds nothing
    loss, count = tb_loss(net, rec["n"], rec["edges"], alone)
    assert float(loss) == pytest.approx(d ** 2 / 2, abs=1e-5) and count == 2
    assert tb_loss(net, rec["n"], rec["edges"], [])[1] == 0


def test_gfn_hook_is_deterministic_for_a_state():
    G = erdos_renyi(40, 0.2, 5)
    hook = gfn_hook(_net())
    a = hook(G)
    hook(erdos_renyi(30, 0.2, 1))
    b = hook(G)
    assert a == b


def test_gfn_hook_differs_from_the_greedy_builder_and_fits():
    G, net = erdos_renyi(60, 0.1, 0), _net()
    L = lattice_graph(load_coords())
    maps, greedy = gfn_hook(net)(G), builder_hook(net)(G)
    assert [next(iter(m)) for m in maps] == list(G)
    assert all(is_valid_piece(G, m, L) for m in maps)
    assert any(m != b for m, b in zip(maps, greedy))


def test_gfn_hook_handles_isolated_vertices():
    G = nx.empty_graph(12); nx.set_node_attributes(G, 1, "weight")
    assert all(len(m) == 1 for m in gfn_hook(_net())(G))
    assert gfn_hook(_net())(nx.empty_graph(0)) == []


def test_gfn_hook_never_calls_a_solver(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the deployed hook called a solver")
    for name in ("exact_mis", "top_maximal_sets", "first_maximum_set", "min_degree_greedy"):
        monkeypatch.setattr(solvers, name, refuse)
        monkeypatch.setattr(builder, name, refuse, raising=False)
        monkeypatch.setattr(gfn, name, refuse, raising=False)
    assert len(gfn_hook(_net())(erdos_renyi(30, 0.2, 1))) == 30


def test_tb_step_raises_the_probability_of_a_high_reward_piece():
    G, net = erdos_renyi(40, 0.15, 3), _net()
    rec = _rec(40, 0.15, 3)
    first = _sampled(G, net, 0, 0)
    other = next(r for r in (_sampled(G, net, 0, s) for s in range(1, 20)) if list(r[0].items()) != list(first[0].items()))
    a, b = other[1], first[1]
    rows = [(0, 1.0, -2.0, a, 0.0, 9), (0, -1.0, -2.0, b, 0.0, 9)]            # log_r differs by 2, same backward cost
    gap = lambda: (lambda lp: float(lp[0] - lp[1]))(path_logp(net, rec["n"], rec["edges"], rows))
    opt, before = torch.optim.Adam(net.parameters(), lr=1e-2), gap()
    for _ in range(5):
        opt.zero_grad()
        tb_loss(net, rec["n"], rec["edges"], rows)[0].backward()
        opt.step()
    assert gap() > before


def test_tb_rollout_returns_flows_from_the_value_and_the_sampled_paths(tmp_path):
    _models(tmp_path)
    rec, net = _rec(30, 0.15, 2), _net()
    job = (builder.blob_of(net), rec["n"], rec["edges"], 3, 2, 7, 2.0, "gin-h5", str(tmp_path))
    n, edges, rows = tb_rollout(job)
    assert n == 30 and len(rows) == 6 and len({r[0] for r in rows}) == 3
    for start, log_r, log_pb, decisions, reward, size in rows:
        assert decisions and size == len(decisions) + 1 and log_pb <= 0 and isinstance(log_r, float)
    assert len({round(r[1] - 2.0 * r[4], 6) for r in rows}) == 1             # log_r is beta * (reward - value of the state)
    assert [r[:3] for r in tb_rollout(job)[2]] == [r[:3] for r in rows]


def _args(tmp_path, out, iters=2, init=None):
    return ["--init", str(init or tmp_path / "init.pt"), "--value", "gin-h5", "--rank", "gin-h4", "--models", str(tmp_path),
            "--train", str(tmp_path / "s.pkl"), "--beta", "2", "--iters", str(iters), "--every", "1", "--states", "2",
            "--starts", "2", "--samples", "2", "--min-n", "20", "--seed", "3", "--out", str(out), "--threads", "1",
            "--workers", "1"]


def test_training_writes_checkpoints_with_companions_and_provenance(tmp_path):
    _models(tmp_path)
    _write_states(tmp_path / "s.pkl", [_rec(24, 0.15, s) for s in range(4)])
    save_builder(_net(), tmp_path / "init.pt")
    main(_args(tmp_path, tmp_path / "o"))
    out = tmp_path / "o"
    for t in (1, 2):
        assert sorted(p.name for p in (out / f"it{t}").iterdir()) == [
            "gfn-h3.pt", "gin-h4.pt", "gin-h5.pt", "train_meta_gfn-h3.json"]
    meta = json.loads((out / "it2" / "train_meta_gfn-h3.json").read_text())
    assert meta["iter"] == 2 and meta["beta"] == 2 and meta["seed"] == 3 and meta["commit"] and meta["cpu_s"] > 0
    lines = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert [l["iter"] for l in lines] == [1, 2] and all(np.isfinite(l["loss"]) for l in lines)
    moved = load_builder(out / "it2" / "gfn-h3.pt").state_dict()
    start = load_builder(tmp_path / "init.pt").state_dict()
    assert any(not torch.allclose(moved[k], start[k], atol=1e-7) for k in moved)


def test_training_refuses_every_seed_outside_training(tmp_path):
    _models(tmp_path)
    save_builder(_net(), tmp_path / "init.pt")
    for seed in (105, 1000, 100):
        _write_states(tmp_path / "s.pkl", [_rec(24, 0.15, 0), _rec(24, 0.15, 1) | dict(seed=seed)])
        with pytest.raises(ValueError):
            main(_args(tmp_path, tmp_path / "o"))


def test_bench_routes_a_gfn_name(tmp_path):
    from test_bench import KEYS
    assert parse_method("gfn-h3+dgin-h4+dgin-h5") == [("gfn", "h3"), ("dgin", "h4"), ("dgin", "h5")]
    assert not known_method("gfn-h2") and not known_method("gfn-h3+gin-h2+gin-h4")
    rng = np.random.default_rng(0)
    save_builder(_net(), tmp_path / "gfn-h3.pt")
    X, g, y = rng.standard_normal((60, len(PIECE_FEATURES))), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
    save(fit_linear(X, y, g), tmp_path / "lin-h4.json")
    save(fit_value_linear(rng.standard_normal((40, len(GRAPH_FEATURES))), rng.integers(0, 9, 40)), tmp_path / "lin-h5.json")
    (r,) = run_instance("er", {"p": 0.1}, 40, 0, ["gfn-h3+lin-h4+lin-h5"], widths=[(2, 2, 5)], restarts=[1],
                        time_limit_s=10, models_dir=tmp_path)
    assert KEYS <= r.keys() and r["track"] == "structure" and r["size"] > 0 and not r["exploded"]
