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
