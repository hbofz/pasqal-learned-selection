import json
import networkx as nx, numpy as np, pytest
from learned_selection import bandit
from learned_selection.bandit import (COSTS, MENU, context, fit_ridge, instance_runner, load_model, pick_arm,
                                      reward_table, save_model)
from learned_selection.bench import run_instance
from learned_selection.features import PIECE_FEATURES, VERTEX_FEATURES
from learned_selection.instances import TRAIN_SETTINGS, erdos_renyi, make
from learned_selection.models import fit_linear, save

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
