import json, shutil
import networkx as nx, numpy as np, pytest
from learned_selection import bandit
from learned_selection.bandit import (COSTS, FIXED_ARM_L, MENU, ROUND_ARMS, context, fit_ridge, instance_runner,
                                      load_model, pick_arm, reward_table, round_context, round_episode, round_runner,
                                      run_round_beam, save_model, update_round)
from learned_selection.bench import run_instance
from learned_selection.engine import exact_piece_solver, gls_generator, run_beam
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.instances import TRAIN_SETTINGS, erdos_renyi, make
from learned_selection.models import fit_linear, save
from learned_selection.solvers import is_independent

TOY_MENU = [("lin-h2h4", 1, 1, 1), ("lin-h2h4", 2, 2, 5)]


def _lin_models(tmp_path):
    rng = np.random.default_rng(0)
    for hook, d in (("h2", len(VERTEX_FEATURES)), ("h4", len(PIECE_FEATURES))):
        X, g, y = rng.standard_normal((60, d)), np.repeat(np.arange(6), 10), rng.integers(0, 3, 60)
        save(fit_linear(X, y, g), tmp_path / f"lin-{hook}.json")


def _forcing(arm, arms=2):
    """A model whose constant term makes `arm` the pick for every graph."""
    theta = np.zeros((arms, 16))
    theta[arm, -1] = 1.0
    return dict(mu=np.zeros(15), sd=np.ones(15), theta=theta)


def _records(family, params, n, seed, menu=MENU, **change):
    return [dict(family=family, params=params, n=n, seed=seed, method=m, k=k, s=s, l=l, restarts=1, exploded=False,
                 size=20 + i, quantum_runs=10 * (i + 1), **change) for i, (m, k, s, l) in enumerate(menu)]


def _write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_context_is_structure_only_and_named():
    G = erdos_renyi(30, 0.2, 1)
    assert context(G).shape == (15,) and np.isfinite(context(G)).all()


def test_reward_table_refuses_other_seeds_and_held_out_settings(tmp_path):
    (f, p), n = TRAIN_SETTINGS[1], 60
    instances, X, sizes, runs = reward_table([_write(tmp_path / "ok.jsonl", _records(f, p, n, 3))])
    assert instances == [(f, p, n, 3)] and X.shape == (1, 15) and sizes.shape == runs.shape == (1, len(MENU))
    assert sizes[0].tolist() == [20 + i for i in range(len(MENU))] and runs[0, 0] == 10 and runs[0, -1] == 80
    for name, bad in (("test_seed", _records(f, p, n, 1000)), ("sel_seed", _records(f, p, n, 105)),
                      ("val_seed", _records(f, p, n, 100)),
                      ("held_out", _records("unit_disk", {"radius": 0.1}, n, 3)),
                      ("missing_arm", _records(f, p, n, 3)[:-1]),
                      ("exploded", _records(f, p, n, 3)[:-1] + [dict(_records(f, p, n, 3)[-1], exploded=True,
                                                                     size=None, quantum_runs=None)])):
        with pytest.raises(ValueError):
            reward_table([_write(tmp_path / f"{name}.jsonl", bad)])


def test_reward_table_sorts_instances_and_pools_files(tmp_path):
    (fa, pa), (fb, pb) = TRAIN_SETTINGS[:2]
    a = _write(tmp_path / "a.jsonl", _records(fb, pb, 60, 2) + _records(fa, pa, 60, 5))
    b = _write(tmp_path / "b.jsonl", _records(fa, pa, 60, 1))
    instances, X, *_ = reward_table([a, b])
    assert instances == [(fa, pa, 60, 1), (fa, pa, 60, 5), (fb, pb, 60, 2)] and len(X) == 3


def test_ridge_learns_a_planted_rule():
    rng = np.random.default_rng(0)
    def sample(n):
        X = rng.standard_normal((n, 4))
        return X, np.stack([3 * X[:, 0], np.zeros(n)], 1) + rng.normal(0, 0.5, (n, 2))
    X, sizes = sample(400)
    model = fit_ridge(X, sizes, np.zeros((400, 2)), 0.01)
    Xt, st = sample(200)
    assert model["theta"].shape == (2, 5)
    assert np.mean([pick_arm(model, x) == st[i].argmax() for i, x in enumerate(Xt)]) >= 0.9


def test_higher_run_cost_moves_picks_to_cheaper_arms():
    X = np.random.default_rng(1).standard_normal((50, 3))
    sizes, runs = np.tile([20.0, 15.0], (50, 1)), np.tile([100.0, 5.0], (50, 1))   # big and dear, small and cheap
    assert pick_arm(fit_ridge(X, sizes, runs, 0.001), X[0]) == 0
    assert pick_arm(fit_ridge(X, sizes, runs, 0.1), X[0]) == 1


def test_pick_arm_breaks_ties_to_the_first():
    assert pick_arm(dict(mu=np.zeros(2), sd=np.ones(2), theta=np.zeros((3, 3))), np.ones(2)) == 0


def test_model_round_trips_and_a_changed_menu_is_refused(tmp_path, monkeypatch):
    model = fit_ridge(np.random.default_rng(2).standard_normal((30, 15)), np.ones((30, len(MENU))),
                      np.ones((30, len(MENU))), 0.01)
    save_model({3: model}, tmp_path / "m.json")
    back = load_model(tmp_path / "m.json")
    assert set(back) == {3} and np.allclose(back[3]["theta"], model["theta"])
    assert json.loads((tmp_path / "m.json").read_text())["commit"]
    monkeypatch.setattr(bandit, "MENU", MENU[:2])
    with pytest.raises(ValueError, match="menu"):
        load_model(tmp_path / "m.json")


def test_bandit_i_record_equals_the_chosen_arm_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "MENU", TOY_MENU)
    _lin_models(tmp_path)
    save_model({3: _forcing(1)}, tmp_path / "bandit-i.json")
    (rec,) = run_instance("er", {"p": 0.25}, 40, 0, ["bandit-i"], widths=[(4, 2, 3)], restarts=[1], time_limit_s=10,
                          models_dir=tmp_path)
    (direct,) = run_instance("er", {"p": 0.25}, 40, 0, ["lin-h2h4"], widths=[(2, 2, 5)], restarts=[1], time_limit_s=10,
                             models_dir=tmp_path)
    assert (rec["method"], rec["track"], rec["k"], rec["s"], rec["l"]) == ("bandit-i", "structure", 4, 2, 3)
    for key in ("size", "quantum_runs", "pasqal_runs", "atoms", "rounds", "max_atoms", "over_30", "atoms_all"):
        assert rec[key] == direct[key], key
    (first,) = run_instance("er", {"p": 0.25}, 40, 0, ["lin-h2h4"], widths=[(1, 1, 1)], restarts=[1], time_limit_s=10,
                            models_dir=tmp_path)
    assert first["quantum_runs"] != rec["quantum_runs"]    # the arm was not the first


def test_bandit_load_fails_on_a_missing_arm_model(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "MENU", TOY_MENU)
    save_model({3: _forcing(0)}, tmp_path / "bandit-i.json")
    with pytest.raises(FileNotFoundError):
        instance_runner(tmp_path)


def test_an_unfitted_cost_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "MENU", TOY_MENU)
    _lin_models(tmp_path)
    save_model({3: _forcing(0)}, tmp_path / "bandit-i.json")
    with pytest.raises(ValueError, match="l=7"):
        instance_runner(tmp_path)(erdos_renyi(20, 0.2, 0), 4, 2, 7, 0)


def test_bandit_i_handles_tiny_and_string_labelled_graphs(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "MENU", TOY_MENU)
    _lin_models(tmp_path)
    save_model({3: _forcing(1)}, tmp_path / "bandit-i.json")
    run = instance_runner(tmp_path)
    tiny = nx.empty_graph(4)
    nx.set_node_attributes(tiny, 1, "weight")
    assert run(tiny, 4, 2, 3, 0).size == 4
    G = make("er", 30, 0, p=0.25)
    twin = nx.relabel_nodes(G, {v: f"v{v:02d}" for v in G})
    assert list(twin) == [f"v{v:02d}" for v in G]
    a, b = run(G, 4, 2, 3, 0), run(twin, 4, 2, 3, 0)
    assert (a.size, a.quantum_runs, a.atoms) == (b.size, b.quantum_runs, b.atoms)


def test_fit_cli_writes_a_model_per_cost(tmp_path, capsys):
    rng = np.random.default_rng(3)
    rows = []
    for f, p in TRAIN_SETTINGS[:2]:
        for seed in range(3):
            rows += [dict(r, size=int(rng.integers(10, 30)), quantum_runs=int(rng.integers(5, 90)))
                     for r in _records(f, p, 60, seed)]
    path = _write(tmp_path / "train.jsonl", rows)
    bandit.main(["fit", "--records", str(path), "--out", str(tmp_path / "bandit-i.json")])
    assert set(load_model(tmp_path / "bandit-i.json")) == {round(1000 * c) for c in COSTS}
    assert "best fixed arm" in capsys.readouterr().out


# the per-round bandit

ROUND_TOY = "lin-h2h4+lin-h5"


def _round_models(tmp_path, builder=False):
    """Toy files for a score-and-rank stack (lin-h2, lin-h4, lin-h5), or with the real builder file added."""
    from learned_selection.features import GRAPH_FEATURES
    from learned_selection.models import fit_value_linear
    _lin_models(tmp_path)
    rng = np.random.default_rng(1)
    save(fit_value_linear(rng.standard_normal((40, len(GRAPH_FEATURES))), rng.integers(0, 9, 40)), tmp_path / "lin-h5.json")
    if builder:
        shutil.copy("models/bld-h3.pt", tmp_path)
    return tmp_path


def _no_value(graphs):
    return [0.0] * len(graphs)


def _spy_asked(monkeypatch):
    """The k each generate call is asked for."""
    asked, real = [], bandit.gls_generator

    def spy(**kw):
        gen = real(**kw)

        def generate(g, k):
            asked.append(k)
            return gen(g, k)
        return generate
    monkeypatch.setattr(bandit, "gls_generator", spy)
    return asked


def _runs_seen(contexts, n0):
    """The running count of solves each context carries."""
    return [round(x[5] * n0 * 4) for x in contexts]


def test_round_context_has_the_spike_shape():
    G = erdos_renyi(30, 0.2, 1)
    x = round_context([(G, [0, 1])], 40, 12, 3)
    assert x.shape == (8,) and x[0] == 1 and x[4] == pytest.approx(3 / 60)
    assert (round_context([], 40, 12, 3) == 0).all()
    assert np.isfinite(round_context([(nx.empty_graph(0), [])], 0, 0, 0)).all()


def test_run_round_beam_uses_the_chosen_arm_each_round(monkeypatch):
    asked, contexts = _spy_asked(monkeypatch), []

    def choose(x):
        contexts.append(x)
        return 0 if len(contexts) == 1 else 3
    r, seen = run_round_beam(erdos_renyi(100, 0.1, 1), choose, dict(value=_no_value))
    assert asked[0] == 1 and set(asked[1:]) == {8}          # arm 0 on the root, arm 3 after
    assert [arm for _, arm in seen] == [0] + [3] * (len(seen) - 1) and len(seen) == r.rounds
    assert all(np.array_equal(x, c) for (x, _), c in zip(seen, contexts))
    assert round(np.expm1(4 * contexts[1][3])) == 1         # arm 0 keeps one branch
    assert all(round(np.expm1(4 * x[3])) <= 8 for x in contexts[1:])


def test_bandit_r_counts_like_a_beam():
    G = erdos_renyi(30, 0.2, 1)
    r, _ = run_round_beam(G, lambda x: 2, dict(value=_no_value))
    assert r.quantum_runs == len(r.run_sizes) and is_independent(G, r.solution)
    assert r.pasqal_runs == sum(m > 5 for m in r.run_sizes) and r.atoms == sum(m for m in r.run_sizes if m > 5)
    direct = run_beam(G, gls_generator(), lambda piece, m, _: exact_piece_solver(piece, m, 2), 4, 2,
                      lambda level, _: sorted(level, key=lambda b: len(b[1]), reverse=True)[:4])
    for key in ("size", "quantum_runs", "pasqal_runs", "atoms", "rounds", "run_sizes"):
        assert getattr(r, key) == getattr(direct, key), key


def test_bandit_r_handles_tiny_and_string_labelled_graphs():
    hooks = dict(value=_no_value)
    assert run_round_beam(nx.empty_graph(0), lambda x: 1, hooks)[0].size == 0
    r, seen = run_round_beam(nx.empty_graph(4), lambda x: 1, hooks)
    assert (r.size, r.quantum_runs, len(seen)) == (4, 1, 1)
    G = make("er", 30, 0, p=0.2)
    twin = nx.relabel_nodes(G, {v: f"v{v:02d}" for v in G})
    a, b = (run_round_beam(g, lambda x: 2, hooks)[0] for g in (G, twin))
    assert (a.size, a.quantum_runs, a.atoms) == (b.size, b.quantum_runs, b.atoms)


def test_episode_refuses_other_seeds():
    for seed in (100, 105, 200, 1000, 1020):
        with pytest.raises(ValueError):
            round_episode((np.zeros((4, 8)), 0.1, "er", {"p": 0.1}, 40, seed, 0, 0.01, "models"))


def test_round_episode_reports_its_run(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    job = (np.zeros((4, 8)), 1.0, "er", {"p": 0.25}, 40, 3, 7, 0.01, _round_models(tmp_path))
    a, b = round_episode(job), round_episode(job)
    assert set(a) == {"reward", "size", "runs", "seen", "v0"} and isinstance(a["v0"], float)
    assert a["reward"] == pytest.approx(a["size"] - 0.01 * a["runs"]) and a["seen"]
    assert [arm for _, arm in a["seen"]] == [arm for _, arm in b["seen"]] and a["size"] == b["size"]


def test_update_round_uses_the_root_value():
    rng = np.random.default_rng(0)
    x1, x2 = rng.standard_normal(8), rng.standard_normal(8)
    A, b = np.array([np.eye(8)] * 4), np.zeros((4, 8))
    update_round(A, b, [dict(seen=[(x1, 0)], reward=1.0, v0=1.0), dict(seen=[(x2, 0)], reward=3.0, v0=2.0)])
    assert np.allclose(b[0], 0 * x1 + 1 * x2) and not b[1:].any()
    assert np.allclose(A[0], np.eye(8) + np.outer(x1, x1) + np.outer(x2, x2)) and np.allclose(A[1:], np.eye(8))


def test_running_count_equals_quantum_runs():
    for G in (erdos_renyi(40, 0.2, 0), erdos_renyi(100, 0.1, 0), erdos_renyi(12, 0.9, 0), nx.path_graph(4)):
        contexts = []
        r, _ = run_round_beam(G, lambda x: contexts.append(x) or 2, dict(value=_no_value))
        runs = _runs_seen(contexts, len(G))
        assert runs[0] == (1 if 0 < len(G) <= 5 else 0) and runs == sorted(runs)
        assert runs[-1] == r.quantum_runs      # the last level holds only small branches, all counted


def test_a_fixed_arm_ignores_theta_and_the_context(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    asked = _spy_asked(monkeypatch)
    run = round_runner(_round_models(tmp_path))      # no bandit-r directory at all
    run(erdos_renyi(40, 0.2, 0), 4, 2, FIXED_ARM_L + 3, 0)
    assert asked and set(asked) == {ROUND_ARMS[2][0]}


def test_a_missing_theta_fails_when_it_is_needed(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    run = round_runner(_round_models(tmp_path))
    with pytest.raises(FileNotFoundError):
        run(erdos_renyi(20, 0.2, 0), 4, 2, 10, 0)


@pytest.mark.parametrize("stack, builder", [("bld-h3+lin-h4+lin-h5", True), (ROUND_TOY, False)])
def test_bandit_r_record_through_bench(tmp_path, monkeypatch, stack, builder):
    from test_bench import KEYS
    monkeypatch.setattr(bandit, "ROUND_STACK", stack)
    d = _round_models(tmp_path, builder)
    (d / "bandit-r").mkdir()
    (d / "bandit-r" / "theta_10.json").write_text(json.dumps(dict(theta=np.zeros((4, 8)).tolist())))
    (rec,) = run_instance("er", {"p": 0.25}, 40, 0, ["bandit-r"], widths=[(4, 2, 10)], restarts=[1], time_limit_s=10,
                          models_dir=d)
    assert KEYS <= rec.keys() and (rec["method"], rec["track"], rec["k"], rec["s"], rec["l"]) == (
        "bandit-r", "structure", 4, 2, 10)
    assert rec["quantum_runs"] > 0 and not rec["exploded"]


# what the round bandit learns from

def test_round_context_by_hand():
    level = [(nx.path_graph(4), [0, 1]), (nx.complete_graph(6), [0, 1, 2, 3])]      # sizes 4, 6; densities 1/2, 1
    x = round_context(level, 20, 7, 3)
    assert np.allclose(x, [1, 5 / 20, 0.75, np.log1p(2) / 4, 3 / 60, 7 / 20 / 4, 3 / 20, 5 / 20 * 0.75])
    capped = round_context(level, 20, 5000, 100)
    assert capped[4] == 1 and capped[5] == pytest.approx(2000 / 20 / 4)


def test_update_round_credits_every_round_of_an_episode():
    x1, x2, x3 = np.arange(1.0, 9), np.arange(8.0, 0, -1), np.linspace(-1, 1, 8)
    A, b = np.array([np.eye(8)] * 4), np.zeros((4, 8))
    update_round(A, b, [dict(seen=[(x1, 0), (x2, 3), (x3, 0)], reward=5.0, v0=2.0)])
    assert np.allclose(b[0], 3 * (x1 + x3)) and np.allclose(b[3], 3 * x2) and not b[1:3].any()
    assert np.allclose(A[0], np.eye(8) + np.outer(x1, x1) + np.outer(x3, x3))
    assert np.allclose(A[3], np.eye(8) + np.outer(x2, x2)) and np.allclose(A[1:3], np.eye(8))


def test_run_round_beam_matches_a_hand_written_beam_with_a_value_hook():
    G = erdos_renyi(100, 0.1, 1)
    value = lambda graphs: [4.0 * (g.number_of_edges() % 3) - len(g) / 2 for g in graphs]     # not a function of size
    (k, s, beam), gen = ROUND_ARMS[3], gls_generator(seed=0)

    def prune(level, _):
        if len(level) <= beam:
            return level
        keys = [len(sol) + v for (_, sol), v in zip(level, value([g for g, _ in level]))]
        return [level[i] for i in sorted(range(len(level)), key=keys.__getitem__, reverse=True)[:beam]]
    direct = run_beam(G, lambda g, _: gen(g, k), lambda piece, m, _: exact_piece_solver(piece, m, s), 4, 2, prune)
    r, _ = run_round_beam(G, lambda x: 3, dict(value=value))
    for key in ("size", "quantum_runs", "pasqal_runs", "atoms", "rounds", "run_sizes"):
        assert getattr(r, key) == getattr(direct, key), key
    assert r.rounds > 1 and max(r.run_sizes) > 5


def test_round_training_names_its_log_and_model_by_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    out = tmp_path / "out"
    for cost in ("0.003", "0.01"):
        bandit.main(["round", "--cost", cost, "--iters", "1", "--instances", "2", "--seed", "4",
                     "--models", str(_round_models(tmp_path)), "--out", str(out), "--workers", "1"])
    assert sorted(p.name for p in out.iterdir()) == ["log_10.jsonl", "log_3.jsonl", "theta_10.json", "theta_3.json"]
    meta = json.loads((out / "theta_10.json").read_text())
    assert (meta["iters"], meta["seed"], meta["instances"], meta["cost"]) == (1, 4, 2, 0.01)
    assert json.loads((out / "log_3.jsonl").read_text())["t"] == 1


def test_greedy_takes_the_arm_of_largest_theta_x(tmp_path, monkeypatch):
    monkeypatch.setattr(bandit, "ROUND_STACK", ROUND_TOY)
    d, asked = _round_models(tmp_path), _spy_asked(monkeypatch)
    theta = np.zeros((4, 8))
    theta[:, 0] = [0.1, 0.2, 0.9, 0.3]          # x[0] is 1, so arm 2 is largest and arm 0 smallest
    (d / "bandit-r").mkdir()
    (d / "bandit-r" / "theta_10.json").write_text(json.dumps(dict(theta=theta.tolist())))
    round_runner(d)(erdos_renyi(40, 0.2, 0), 4, 2, 10, 0)
    assert asked and set(asked) == {ROUND_ARMS[2][0]}
    out = round_episode((theta, 0.0, "er", {"p": 0.25}, 40, 3, 7, 0.01, d))
    assert out["seen"] and {arm for _, arm in out["seen"]} == {2}
