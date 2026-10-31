import networkx as nx, numpy as np, pytest, torch
from learned_selection.graphdata import state_record
from learned_selection.instances import erdos_renyi, power_law
from learned_selection.train import listwise_loss, load_states, train_ranker, train_value, value_rows

def _states():
    gs = [erdos_renyi(24, 0.2, s) for s in range(6)] + [power_law(30, 2, s) for s in range(4)]
    return [dict(state_record(g, True, 20.0, 1), setting="er(p=0.25)", instance_n=len(g), seed=s)
            for s, g in enumerate(gs)]

def test_listwise_loss_rewards_mass_on_the_best_items():
    scores = torch.tensor([3.0, 0.0, 0.0, 0.0, 2.0])
    y, group = torch.tensor([1.0, 0, 0, 0, 1]), torch.tensor([0, 0, 0, 1, 1])
    w = torch.ones(2)
    assert listwise_loss(scores, y, group, w) < listwise_loss(-scores, y, group, w)

def test_train_ranker_fits_a_planted_rule():
    states = []
    for s in range(12):
        g = erdos_renyi(20, 0.25, s)
        deg = np.array([g.degree(v) for v in g])
        rec = dict(state_record(g, True, 20.0, 1), setting="er(p=0.25)", instance_n=20, seed=s)
        rec["member"] = (deg == deg.max()).astype(np.int8)          # label: the highest-degree vertices
        states.append(rec)
    _, info = train_ranker("sage", "h2", states, states, "cpu", epochs=30, patience=30, warm_epochs=0, batch=2)
    assert info["val_regret"] < 0.1 and np.isfinite(info["history"]).all()

def test_train_ranker_reports_the_warm_start_and_the_rule():
    states = _states()
    net, info = train_ranker("gin", "h4", states[:7], states[7:], "cpu", epochs=3, patience=3, warm_epochs=2, batch=2)
    assert 0 <= info["imitation_agreement"] <= 1 and info["warm_epochs_run"] <= 2 and info["rule_regret"] >= 0

def test_train_value_beats_predicting_zero():
    states = _states()
    _, info = train_value("gin", states, states, "cpu", epochs=30, patience=30, batch=2)
    assert info["val_mae"] < info["mae_of_zero"] and 0 <= info["prune_regret"] and 0 <= info["rule_prune_regret"]

def test_value_rows_cap_children_and_keep_states():
    rows = value_rows(_states(), cap=3)
    assert all(sum(1 for r in rows if r[2] == i and r[4]) <= 3 for i in range(10))
    assert sum(1 for r in rows if not r[4]) == 10

def test_load_states_refuses_test_seeds_and_held_out_settings(tmp_path):
    import pickle
    ok = dict(setting="er(p=0.25)", seed=0)
    for bad in (dict(ok, seed=1000), dict(ok, setting="unit_disk(radius=0.1)")):
        path = tmp_path / "x.pkl"
        path.write_bytes(pickle.dumps({"meta": {}, "states": [ok, bad]}))
        with pytest.raises(ValueError):
            load_states([path])

def test_value_selection_prefers_lower_regret_then_lower_child_error():
    from learned_selection.train import _better
    a, b = dict(prune_regret=0.0, child_mae=0.3), dict(prune_regret=0.0, child_mae=0.5)
    assert _better(a, b) and not _better(b, a)
    assert _better(dict(prune_regret=0.0, child_mae=9.0), dict(prune_regret=0.1, child_mae=0.1))

def test_load_states_refuses_selection_seeds(tmp_path):
    import pickle
    path = tmp_path / "x.pkl"
    path.write_bytes(pickle.dumps({"meta": {}, "states": [dict(setting="er(p=0.25)", seed=105)]}))
    with pytest.raises(ValueError):
        load_states([path])

def _er_states(count, seed0=0):
    return [dict(state_record(erdos_renyi(30, 0.1, s), True, 20.0, 1), setting="er(p=0.1)", instance_n=30, seed=s)
            for s in range(seed0, seed0 + count)]

def test_jobs_accept_dagger_and_linear_rankers():
    from learned_selection import train
    assert train._jobs("dlin-h2,dsage-h5,lin-h4,gin-h2") == [("dlin-h2", "lin", "h2"), ("dsage-h5", "sage", "h5"),
                                                             ("lin-h4", "lin", "h4"), ("gin-h2", "gin", "h2")]
    for bad in ("lgb-h2", "dlgb-h2", "gin-h3", "dfoo-h2"):
        with pytest.raises(SystemExit):
            train._jobs(bad)

def test_linear_ranker_job_learns_the_planted_order_of_a_state(tmp_path):
    from learned_selection.features import PIECE_FEATURES, piece_features, vertex_features
    from learned_selection.graphdata import graph_of
    from learned_selection.models import load, save, topk_regret
    from learned_selection.train import fit_linear_ranker_job
    # h4: the planted value falls with the size of the closed neighbourhood, so the size rule is wrong
    states = _er_states(16)
    for rec in states:
        pf = piece_features(graph_of(rec), [set(p.tolist()) for p in rec["pieces"]])
        rec["q"] = np.round(-pf[:, PIECE_FEATURES.index("closed_nbhd")]).astype(np.int64)
    model, info = fit_linear_ranker_job("h4", states[:10], states[10:], workers=2)
    assert info["rule_regret"] > 0 and info["val_regret"] < info["rule_regret"] and info["val_regret"] < 0.5
    assert (info["train_states"], info["val_states"]) == (10, 6) and info["seconds"] >= 0
    save(model, tmp_path / "lin-h4.json")
    pf = piece_features(graph_of(states[0]), [set(p.tolist()) for p in states[0]["pieces"]])
    assert np.allclose(load(tmp_path / "lin-h4.json").predict(pf), model.predict(pf))
    # h2: the planted label marks the highest-degree vertices
    for rec in states:
        deg = np.bincount(rec["edges"].ravel(), minlength=rec["n"])
        rec["member"] = (deg == deg.max()).astype(np.int8)
    model, info = fit_linear_ranker_job("h2", states[:10], states[10:])
    assert info["val_regret"] < info["rule_regret"] and info["val_regret"] < 0.1
    assert model.predict(vertex_features(graph_of(states[0]))[1]).shape == (30,)

def test_linear_ranker_job_skips_unproven_states():
    from learned_selection.train import fit_linear_ranker_job
    states = _er_states(8)
    unproven = [dict(s, member=None, q=None, alpha=None) for s in states[:3]]
    no_pieces = [dict(s, pieces=[], q=np.zeros(0, dtype=np.int64)) for s in states[3:5]]
    for hook in ("h2", "h4"):
        _, info = fit_linear_ranker_job(hook, unproven + no_pieces + states[5:], unproven + states[5:])
        assert info["train_states"] == (5 if hook == "h2" else 3)   # candidate-less states only count for h2
        assert info["val_states"] == 3
    with pytest.raises(ValueError):
        fit_linear_ranker_job("h2", unproven, states)

def test_main_names_files_by_job_for_linear_rankers(tmp_path, capsys):
    import json, pickle
    from learned_selection.train import main
    path = tmp_path / "s.pkl"
    path.write_bytes(pickle.dumps({"meta": {}, "states": _er_states(6)}))
    main(["--train", str(path), "--val", str(path), "--jobs", "dlin-h2,lin-h4", "--out", str(tmp_path / "o"),
          "--workers", "1"])
    assert sorted(p.name for p in (tmp_path / "o").iterdir()) == [
        "dlin-h2.json", "lin-h4.json", "train_meta_dlin-h2.json", "train_meta_lin-h4.json"]
    assert json.loads((tmp_path / "o" / "train_meta_dlin-h2.json").read_text())["job"] == "dlin-h2"
    assert "dlin-h2: val regret" in capsys.readouterr().out

def test_jobs_accept_a_builder_stack_prefix():
    from learned_selection import train
    assert train._jobs("bgin-h4,bgin-h5") == [("bgin-h4", "gin", "h4"), ("bgin-h5", "gin", "h5")]
    for bad in ("blin-h2x", "bfoo-h2"):
        with pytest.raises(SystemExit):
            train._jobs(bad)
