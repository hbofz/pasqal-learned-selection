import networkx as nx, numpy as np, pytest, torch
from learned_selection import builder, solvers
from learned_selection.bench import run_instance
from learned_selection.builder import (PAIR_FEATURES, PieceBuilder, build_piece, builder_hook, clone_examples, clone_fit,
                                       decision_loss, load_builder, main, one_step_reward, pair_options, pair_weights,
                                       rollout_job, save_builder, start_advantages, trap_features)
from learned_selection.cloud import EMULATION_DEVICE, piece_sequence
from learned_selection.engine import run_code_bounded
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES
from learned_selection.gnn import GNN, encode, save_gnn
from learned_selection.growth import code_start_trap, gls_score, lattice_adjacency
from learned_selection.instances import erdos_renyi, power_law, sbm
from learned_selection.lattice import is_valid_piece, lattice_graph, load_coords
from learned_selection.models import fit_linear, fit_value_linear, save
from learned_selection.solvers import is_independent


def _net(seed=0):
    torch.manual_seed(seed)
    body = GNN("gin", "vertex", np.zeros(9), np.ones(9))
    return PieceBuilder(body).eval()


def _lattice():
    coords = load_coords()
    adj, _ = lattice_adjacency(coords)
    return adj, code_start_trap(len(adj)), trap_features(coords, adj)


def test_every_built_piece_fits_the_lattice():
    L, build = lattice_graph(load_coords()), builder_hook(_net())
    for G in (erdos_renyi(60, 0.1, 0), erdos_renyi(40, 0.75, 1), power_law(60, 2, 2), sbm(60, 4, 0.3, 0.03, 3)):
        maps = build(G)
        assert [next(iter(m)) for m in maps] == list(G)          # one piece per start vertex, the start placed first
        assert all(is_valid_piece(G, m, L) for m in maps)
        assert all(m[next(iter(m))] == code_start_trap(61) for m in maps)


def test_trap_features_are_centred_and_scaled():
    adj, _, trap_x = _lattice()
    assert len(PAIR_FEATURES) == 10 and trap_x.shape == (len(adj), 4)
    assert np.allclose(trap_x[:, :2].mean(0), 0, atol=1e-9) and np.abs(trap_x[:, :2]).max() == pytest.approx(1.0)
    assert np.allclose(trap_x[:, 2], np.hypot(*trap_x[:, :2].T))
    assert np.allclose(trap_x[:, 3], [len(a) / 6 for a in adj])


def test_pair_options_are_exactly_the_placements_that_keep_the_piece_valid():
    adj, start_trap, trap_x = _lattice()
    L = lattice_graph(load_coords())
    for seed in (0, 1):
        G = erdos_renyi(25, 0.3, seed)
        start = max(G, key=G.degree)
        place, occ = {start: start_trap}, {start_trap: start}
        for _ in range(5):
            near_filled = {t for s in occ for t in adj[s] if t not in occ}
            brute = [(u, t) for t in sorted(near_filled) for u in sorted(G) if u not in place
                     and is_valid_piece(G, {**place, u: t}, L)]
            options = pair_options(G, place, occ, adj, trap_x)
            assert [o[:2] for o in options] == brute
            assert all(len(o[2]) == len(PAIR_FEATURES) for o in options)
            if not brute:
                break
            u, t = brute[0]
            place[u], occ[t] = t, u
        assert len(place) >= 3


def test_pair_option_features_follow_their_formulae():
    adj, start_trap, trap_x = _lattice()
    G = erdos_renyi(25, 0.3, 0)
    start = max(G, key=G.degree)
    place, occ = {start: start_trap}, {start_trap: start}
    u, t, phi = pair_options(G, place, occ, adj, trap_x)[0]
    d, n = G.degree(u), len(G)
    near = len({x for x in adj[t] if x in occ})
    assert phi[:4] == pytest.approx(tuple(trap_x[t]))
    assert phi[4:] == pytest.approx((near / 6, (len(adj[t]) - near) / 6, 1 / 61, d / n, (d - near) / n, d / (n - 1)))


def test_builder_is_deterministic_and_sampling_is_seeded():
    G, net = erdos_renyi(40, 0.2, 5), _net()
    assert builder_hook(net)(G) == builder_hook(net)(G)
    H = net.embed(encode(G).to("cpu")).detach().numpy()
    args = (G, H, pair_weights(net), 0, *_lattice())
    a, _ = build_piece(*args, rng=np.random.default_rng(1))
    b, _ = build_piece(*args, rng=np.random.default_rng(1))
    assert a == b


def test_greedy_build_takes_the_first_of_tied_logits():
    G, net = erdos_renyi(30, 0.2, 5), _net()
    for p in net.pair.parameters():
        torch.nn.init.zeros_(p)                          # every pair scores 0: the first option is always taken
    H = net.embed(encode(G).to("cpu")).detach().numpy()
    adj, start_trap, trap_x = _lattice()
    piece, decisions = build_piece(G, H, pair_weights(net), 0, adj, start_trap, trap_x)
    assert decisions == []
    place, occ = {0: start_trap}, {start_trap: 0}
    while options := pair_options(G, place, occ, adj, trap_x):
        u, t, _ = options[0]
        place[u], occ[t] = t, u
    assert piece == place and list(piece) == list(place)


def test_sampled_decisions_reproduce_the_piece_and_match_the_torch_logits():
    G, net = erdos_renyi(40, 0.15, 3), _net()
    H = net.embed(encode(G).to("cpu")).detach().numpy()
    adj, start_trap, trap_x = _lattice()
    w1, b1, w2, b2 = pair_weights(net)
    piece, decisions = build_piece(G, H, pair_weights(net), 0, adj, start_trap, trap_x, rng=np.random.default_rng(4))
    assert len(decisions) == len(piece) - 1 > 3
    at = {v: i for i, v in enumerate(G)}
    place, occ = {0: start_trap}, {start_trap: 0}
    for idx, phi, chosen in decisions:
        options = pair_options(G, place, occ, adj, trap_x)
        assert [at[o[0]] for o in options] == list(idx)
        with torch.no_grad():
            torch_logits = net.pair_logits(torch.from_numpy(H), idx, torch.as_tensor(phi)).numpy()
        numpy_logits = (np.maximum(np.c_[H[idx], phi] @ w1.T + b1, 0) @ w2.T + b2).ravel()
        assert np.allclose(torch_logits, numpy_logits, atol=1e-5)
        u, t, _ = options[chosen]
        place[u], occ[t] = t, u
    assert place == piece and list(place) == list(piece)


def test_pair_logits_carry_gradient_to_the_scorer_and_the_body():
    G, net = erdos_renyi(30, 0.2, 1), _net()
    adj, start_trap, trap_x = _lattice()
    H = net.embed(encode(G).to("cpu")).detach().numpy()
    _, decisions = build_piece(G, H, pair_weights(net), 0, adj, start_trap, trap_x, rng=np.random.default_rng(0))
    idx, phi, chosen = decisions[0]
    h = net.embed(encode(G).to("cpu"))
    logits = net.pair_logits(h, idx, torch.as_tensor(phi))
    (logits[chosen] - torch.logsumexp(logits, 0)).backward()
    assert net.pair[0].weight.grad.abs().sum() > 0 and net.body.embed.weight.grad.abs().sum() > 0


def test_builder_handles_isolated_vertices_and_tiny_graphs():
    G = nx.empty_graph(12); nx.set_node_attributes(G, 1, "weight")
    assert all(len(m) == 1 for m in builder_hook(_net())(G))
    r = run_code_bounded(G, 2, 2, 5, build=builder_hook(_net()))
    assert r.size == 12
    H = nx.path_graph(6); nx.set_node_attributes(H, 1, "weight")
    assert is_independent(H, run_code_bounded(H, 2, 2, 5, build=builder_hook(_net())).solution)
    one = nx.empty_graph(1); nx.set_node_attributes(one, 1, "weight")
    assert builder_hook(_net())(one) == [{0: code_start_trap(61)}]
    assert builder_hook(_net())(nx.empty_graph(0)) == []


def test_builder_string_labels_match_their_integer_twin():
    G = erdos_renyi(30, 0.2, 2)
    H = nx.relabel_nodes(G, {v: f"v{v:02d}" for v in G})
    build = builder_hook(_net())
    a, b = run_code_bounded(G, 4, 2, 10, build=build), run_code_bounded(H, 4, 2, 10, build=build)
    assert a.size == b.size and a.quantum_runs == b.quantum_runs


def test_deployed_hook_never_calls_a_solver(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("the deployed hook called a solver")
    for name in ("exact_mis", "top_maximal_sets", "first_maximum_set", "min_degree_greedy"):
        monkeypatch.setattr(solvers, name, refuse)
        monkeypatch.setattr(builder, name, refuse, raising=False)
    maps = builder_hook(_net())(erdos_renyi(30, 0.2, 1))
    assert len(maps) == 30


def test_built_pieces_become_real_registers():
    G = erdos_renyi(40, 0.1, 0)
    big = [m for m in builder_hook(_net())(G) if len(m) >= 5][:3]
    assert len(big) == 3
    for m in big:
        piece = G.subgraph(m)
        seq = piece_sequence(piece, m, EMULATION_DEVICE)      # building it runs Pulser's device validation
        assert list(seq.register.qubits) == [str(v) for v in piece] and len(seq.register.qubits) == len(m)


def test_save_and_load_round_trip(tmp_path):
    net = _net(); save_builder(net, tmp_path / "bld-h3.pt")
    G = erdos_renyi(30, 0.2, 2)
    loaded = load_builder(tmp_path / "bld-h3.pt")
    assert not loaded.training and builder_hook(loaded)(G) == builder_hook(net)(G)
    assert [p.name for p in tmp_path.iterdir()] == ["bld-h3.pt"]


def test_bench_routes_a_builder_name(tmp_path):
    from test_bench import KEYS
    rng = np.random.default_rng(0)
    save_builder(_net(), tmp_path / "bld-h3.pt")
    X, g, y = rng.standard_normal((60, len(PIECE_FEATURES))), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
    save(fit_linear(X, y, g), tmp_path / "lin-h4.json")
    save(fit_value_linear(rng.standard_normal((40, len(GRAPH_FEATURES))), rng.integers(0, 9, 40)), tmp_path / "lin-h5.json")
    for name, width in (("bld-h3+lin-h4+lin-h5", (2, 2, 5)), ("bld-h3+lin-h4+lin-h5+mcts", (4, 2, 15))):
        (r,) = run_instance("er", {"p": 0.1}, 40, 0, [name], widths=[width], restarts=[1], time_limit_s=10,
                            models_dir=tmp_path)
        assert KEYS <= r.keys() and r["track"] == "structure" and r["size"] > 0 and not r["exploded"]


def _rec(n, p, seed, **kw):
    G = erdos_renyi(n, p, seed)
    return dict(n=n, edges=np.array(G.edges(), dtype=np.int32).reshape(-1, 2), setting="er(p=0.1)", seed=seed, **kw)


def _gls_factory():
    return gls_score(lattice_adjacency(load_coords())[1])


def test_clone_examples_replay_the_expert_placement():
    n, edges, ds = clone_examples(_rec(30, 0.2, 1), _gls_factory(), *_lattice())
    assert n == 30 and len(edges) and ds and all(0 <= c < len(idx) for idx, _, c in ds)    # the expert's pair is among the options
    assert all(phi.shape == (len(idx), len(PAIR_FEATURES)) for idx, phi, _ in ds)


def test_decision_loss_matches_a_written_out_softmax():
    rec, net = _rec(20, 0.2, 3), _net()
    idx, phi = np.array([2, 5, 7]), np.random.default_rng(0).random((3, len(PAIR_FEATURES))).astype(np.float32)
    ds = [(idx, phi, 1), (idx[:2], phi[:2], 0)]
    loss, logits = decision_loss(net, rec["n"], rec["edges"], ds, np.array([2.0, -0.5]))
    z = logits.numpy()
    logp = [z[1] - np.log(np.exp(z[:3]).sum()), z[3] - np.log(np.exp(z[3:]).sum())]
    assert float(loss) == pytest.approx(-(2.0 * logp[0] - 0.5 * logp[1]) / 2, abs=1e-5)
    assert not logits.requires_grad and loss.requires_grad


def test_cloning_raises_agreement_on_the_expert():
    examples = [clone_examples(_rec(25, 0.15, s), _gls_factory(), *_lattice()) for s in range(3)]
    history = clone_fit(_net(1).train(), examples, epochs=30, seed=0)
    assert len(history) == 30 and history[-1][1] > history[0][1] and history[-1][0] < history[0][0]


def test_one_step_reward_uses_the_engine_sets_and_the_value():
    G = nx.path_graph(8); nx.set_node_attributes(G, 1, "weight")
    assert one_step_reward(G, {0: 32, 1: 33}, lambda gs: [len(g) for g in gs]) == 1 + 6     # one vertex taken, six left
    H = nx.empty_graph(9); H.add_edges_from([(0, 1), (2, 3), (4, 5)]); nx.set_node_attributes(H, 1, "weight")
    big = {v: 30 + v for v in range(6)}                                                      # over small: the top 2 sets
    assert one_step_reward(H, big, lambda gs: [len(g) for g in gs]) == 3 + 3                 # any maximum set leaves 6, 7, 8
    assert one_step_reward(G, {v: v for v in range(8)}, lambda gs: [len(g) for g in gs]) == 4   # nothing left


def test_start_advantages_centre_per_start_and_scale_by_the_batch():
    adv = start_advantages([1.0, 3.0, 10.0, 10.0], [("s", 0), ("s", 0), ("s", 1), ("s", 1)])
    assert adv[0] < 0 < adv[1] and adv[2] == adv[3] == 0
    assert adv[1] == pytest.approx(1 / np.std([-1.0, 1.0, 0.0, 0.0]))
    assert (start_advantages([5.0, 5.0], [(0, 1), (0, 1)]) == 0).all()      # equal rewards: zeros, not NaN


def test_rl_step_moves_mass_towards_the_better_pair():
    rec, net = _rec(20, 0.2, 3), _net()
    idx, phi = np.array([2, 5]), np.random.default_rng(0).random((2, len(PAIR_FEATURES))).astype(np.float32)
    ds = [(idx, phi, 0), (idx, phi, 1)]
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    before = decision_loss(net, rec["n"], rec["edges"], ds, np.array([1.0, -1.0]))[1]
    for _ in range(3):
        opt.zero_grad()
        decision_loss(net, rec["n"], rec["edges"], ds, np.array([1.0, -1.0]))[0].backward()
        opt.step()
    after = decision_loss(net, rec["n"], rec["edges"], ds, np.array([1.0, -1.0]))[1]
    assert after[0] - after[1] > before[0] - before[1]


def _models(path):
    body = lambda head: GNN("gin", head, np.zeros(9), np.ones(9), np.zeros(len(PIECE_FEATURES)), np.ones(len(PIECE_FEATURES)))
    torch.manual_seed(5)
    save_gnn(body("piece"), path / "gin-h4.pt")
    save_gnn(body("value"), path / "gin-h5.pt")
    save_gnn(body("vertex"), path / "gin-h2.pt")
    return path


def _write_states(path, states):
    import pickle
    path.write_bytes(pickle.dumps(dict(meta={}, states=states)))
    return str(path)


def test_rollout_job_returns_numpy_rows_with_rewards_and_decisions(tmp_path):
    _models(tmp_path)
    rec, net = _rec(30, 0.15, 2), _net()
    blob = builder.blob_of(net)
    n, edges, rows = rollout_job((blob, rec["n"], rec["edges"], 3, 2, 7, "gin-h5", "gin-h4", str(tmp_path)))
    assert n == 30 and len(rows) == 6 and len({r[0] for r in rows}) == 3
    assert all(isinstance(r[1], float) and r[2] == len(r[3]) + 1 for r in rows)
    again = rollout_job((blob, rec["n"], rec["edges"], 3, 2, 7, "gin-h5", "gin-h4", str(tmp_path)))[2]
    assert [r[:3] for r in rows] == [r[:3] for r in again]                   # the seed fixes the rollout


def _rl_args(tmp_path, out, iters, init):
    return ["rl", "--init", str(init), "--value", "gin-h5", "--rank", "gin-h4", "--models", str(tmp_path), "--train",
            str(tmp_path / "s.pkl"), "--iters", str(iters), "--states", "2", "--starts", "2", "--samples", "2", "--min-n", "20",
            "--every", "2", "--seed", "3", "--out", str(out), "--threads", "1", "--workers", "1"]


def test_rl_resume_matches_an_unbroken_run(tmp_path):
    _models(tmp_path)
    _write_states(tmp_path / "s.pkl", [_rec(24, 0.15, s) for s in range(4)])
    save_builder(_net(), tmp_path / "init.pt")
    main(_rl_args(tmp_path, tmp_path / "straight", 4, tmp_path / "init.pt"))
    main(_rl_args(tmp_path, tmp_path / "broken", 2, tmp_path / "init.pt"))
    assert not (tmp_path / "broken" / "it4").exists()
    main(_rl_args(tmp_path, tmp_path / "broken", 4, tmp_path / "init.pt"))
    a, b = load_builder(tmp_path / "straight" / "it4" / "bld-h3.pt"), load_builder(tmp_path / "broken" / "it4" / "bld-h3.pt")
    for x, y in zip(a.state_dict().values(), b.state_dict().values()):
        assert torch.allclose(x, y, atol=1e-6)
    start = load_builder(tmp_path / "init.pt")
    assert any(not torch.allclose(x, y, atol=1e-6) for x, y in zip(a.state_dict().values(), start.state_dict().values()))
    import json
    lines = [json.loads(l) for l in (tmp_path / "broken" / "log.jsonl").read_text().splitlines()]
    assert [l["iter"] for l in lines] == [1, 2, 3, 4]
    state = torch.load(tmp_path / "broken" / "state.pt", weights_only=True)
    assert state["t"] == 4 and state["cpu_s"] > 0
    for where in (tmp_path / "broken", tmp_path / "broken" / "it4"):          # provenance beside the model and in the run dir
        meta = json.loads((where / "train_meta_bld-h3.json").read_text())
        assert meta["iter"] == 4 and meta["seed"] == 3 and meta["lr"] == 1e-4 and meta["commit"] and meta["cpu_s"] > 0
        assert meta["init"] == str(tmp_path / "init.pt") and meta["train"] == [str(tmp_path / "s.pkl")]
        assert (meta["states"], meta["starts"], meta["samples"], meta["min_n"]) == (2, 2, 2, 20)
    assert sorted(p.name for p in (tmp_path / "broken").iterdir() if p.is_file()) == [
        "log.jsonl", "state.pt", "train_meta_bld-h3.json"]                   # no stray temp files
    for f in ("bld-h3.pt", "gin-h4.pt", "gin-h5.pt"):                     # so --models resolves bld-h3+gin-h4+gin-h5
        assert (tmp_path / "broken" / "it2" / f).exists()


def test_clone_writes_the_model_with_its_companions(tmp_path):
    _models(tmp_path)
    _write_states(tmp_path / "s.pkl", [_rec(24, 0.15, s) for s in range(3)])
    main(["clone", "--expert", "gin-h2", "--rank", "gin-h4", "--value", "gin-h5", "--models", str(tmp_path), "--train",
          str(tmp_path / "s.pkl"), "--epochs", "1", "--seed", "1", "--out", str(tmp_path / "c"), "--threads", "1",
          "--workers", "1"])
    out = tmp_path / "c"
    assert sorted(p.name for p in out.iterdir()) == ["bld-h3.pt", "gin-h4.pt", "gin-h5.pt", "train_meta_bld-h3.json"]
    import json
    meta = json.loads((out / "train_meta_bld-h3.json").read_text())
    assert meta["seed"] == 1 and meta["states"] == 3 and meta["cpu_s"] > 0 and "commit" in meta
    assert load_builder(out / "bld-h3.pt").pair[0].weight.shape[1] == 64 + len(PAIR_FEATURES)


def test_training_refuses_every_seed_outside_training(tmp_path):
    _models(tmp_path)
    for seed in (105, 1000, 100):
        path = _write_states(tmp_path / f"s{seed}.pkl", [_rec(24, 0.15, 0), _rec(24, 0.15, 1) | dict(seed=seed)])
        with pytest.raises(ValueError):
            main(["clone", "--expert", "gin-h2", "--models", str(tmp_path), "--train", path, "--out", str(tmp_path / "o")])
    save_builder(_net(), tmp_path / "init.pt")
    with pytest.raises(ValueError):
        main(["rl", "--init", str(tmp_path / "init.pt"), "--models", str(tmp_path), "--train", path, "--out", str(tmp_path / "o")])
