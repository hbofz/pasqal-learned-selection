import json, pickle

import pytest

from learned_selection.bench import grid, run_instance
from learned_selection.graphdata import label_graphs
from learned_selection.guard import labels_guard, main, records_guard

INST = ("er", {"p": 0.1}, 30, 1000)


def write(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_exit_code_is_one_when_nothing_was_compared_and_zero_when_something_was(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--records", str(tmp_path / "none*.jsonl"), "--methods", "gls", "--count", "1"])
    assert e.value.code == 1 and "0 same" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--labels", str(tmp_path / "none*.pkl")])
    assert e.value.code == 1
    fam, params, n, seed = grid("test")[0]
    rows = run_instance(fam, params, n, seed, ["gls"], widths=[(1, 1, 1)], restarts=[1], time_limit_s=5)
    good = write(tmp_path / "good.jsonl", rows)
    with pytest.raises(SystemExit) as e:
        main(["--records", str(good), "--methods", "gls", "--stride", "100000", "--count", "1"])
    assert e.value.code == 0
    write(tmp_path / "bad.jsonl", [dict(r, size=r["size"] + 1) for r in rows])
    with pytest.raises(SystemExit) as e:
        main(["--records", str(tmp_path / "bad.jsonl"), "--methods", "gls", "--stride", "100000", "--count", "1"])
    assert e.value.code == 1


def test_records_guard_reports_two_records_with_one_key_that_disagree(tmp_path):
    rows = run_instance(*INST, ["gls"], widths=[(2, 2, 5)], restarts=[1], time_limit_s=5)
    a = write(tmp_path / "a.jsonl", rows)
    b = write(tmp_path / "b.jsonl", [dict(r, size=r["size"] + 1) for r in rows])
    assert records_guard([a, a], ["gls"], [INST], workers=1, models_dir=None) == (1, 0, 0)
    assert records_guard([a, b], ["gls"], [INST], workers=1, models_dir=None) == (1, 1, 0)


def test_records_guard_reruns_exactly_the_old_keys(tmp_path):
    rows = (run_instance(*INST, ["gls"], widths=[(2, 2, 5)], restarts=[1], time_limit_s=5)
            + run_instance(*INST, ["gls"], widths=[(4, 2, 10)], restarts=[5], time_limit_s=5))
    f = write(tmp_path / "old.jsonl", rows)
    assert records_guard([f], ["gls"], [INST], workers=1, models_dir=None) == (2, 0, 0)


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


def test_guard_fresh_grid(tmp_path, capsys):
    inst = next(i for i in grid("fresh") if i[3] == 1020)
    rows = run_instance(*inst, ["gls"], widths=[(1, 1, 1)], restarts=[1], time_limit_s=5)
    f = write(tmp_path / "fresh.jsonl", rows)
    with pytest.raises(SystemExit) as e:
        main(["--records", str(f), "--methods", "gls", "--grid", "fresh", "--seeds", "1020-1020", "--stride", "26",
              "--count", "1"])
    assert e.value.code == 0 and "1 same, 0 differ" in capsys.readouterr().out
