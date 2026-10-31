import networkx as nx, numpy as np, pytest, torch
from learned_selection import builder, solvers
from learned_selection.bench import run_instance
from learned_selection.builder import (PAIR_FEATURES, PieceBuilder, build_piece, builder_hook, load_builder, pair_options,
                                       pair_weights, save_builder, trap_features)
from learned_selection.cloud import EMULATION_DEVICE, piece_sequence
from learned_selection.engine import run_code_bounded
from learned_selection.features import GRAPH_FEATURES, PIECE_FEATURES
from learned_selection.gnn import GNN, encode
from learned_selection.growth import code_start_trap, lattice_adjacency
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
