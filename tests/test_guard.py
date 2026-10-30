import json, pickle

from learned_selection.bench import run_instance
from learned_selection.graphdata import label_graphs
from learned_selection.guard import labels_guard, records_guard

INST = ("er", {"p": 0.1}, 30, 1000)


def test_records_guard_passes_on_a_fresh_record_and_fails_on_a_changed_one(tmp_path):
    rows = run_instance(*INST, ["gls"], widths=[(2, 2, 5)], restarts=[1], time_limit_s=5)
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert records_guard([good], ["gls"], [INST], workers=1, models_dir=None) == (1, 0, 0)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps(dict(r, size=r["size"] + 1)) + "\n" for r in rows))
    assert records_guard([bad], ["gls"], [INST], workers=1, models_dir=None) == (0, 1, 0)
    assert records_guard([good], ["gls"], [("er", {"p": 0.1}, 30, 1001)], workers=1, models_dir=None) == (0, 0, 1)


def test_labels_guard_passes_on_a_fresh_pickle_and_fails_on_a_changed_one(tmp_path):
    fam, params, n, seed = "er", {"p": 0.25}, 20, 0
    states = label_graphs(fam, params, n, seed)["states"]
    for s in states:
        s.update(setting="er(p=0.25)", instance_n=n, seed=seed)
    good = tmp_path / "good.pkl"
    good.write_bytes(pickle.dumps(dict(meta={}, states=states)))
    assert labels_guard([good], [(fam, params, n, seed)], workers=1) == (len(states), 0)
    nodes, alpha, s = states[-1]["children"][0]
    states[-1]["children"][0] = (nodes, alpha + 1, s)
    bad = tmp_path / "bad.pkl"
    bad.write_bytes(pickle.dumps(dict(meta={}, states=states)))
    assert labels_guard([bad], [(fam, params, n, seed)], workers=1) == (len(states) - 1, 1)
