import json

import numpy as np
import pandas as pd

from learned_selection import portfolio as pf
from learned_selection.engine import exact_piece_solver
from learned_selection.market import Market


def market():
    idx = pd.bdate_range("2004-01-01", periods=400)
    rng = np.random.default_rng(0)
    names = [f"S{i}" for i in range(40)]
    common = rng.standard_normal((400, 8))
    R = common[:, np.arange(40) % 8] * 0.008 + rng.standard_normal((400, 40)) * 0.01
    return Market(pd.DataFrame(100 * np.exp(R.cumsum(0)), index=idx, columns=names), [("2004-01-01", names)]), str(idx[300].date())


def test_every_method_returns_an_independent_set_of_real_tickers():
    m, date = market()
    G, names = m.graph(date, 0.3)
    for job in [(date, 0.3, "greedy-mindeg", None, None, None), (date, 0.3, "exact", None, None, None), (date, 0.3, "gls", 4, 2, 10)]:
        r = pf.run_job(job, m, exact_piece_solver, "exact-top-s")
        assert r["size"] == len(r["tickers"]) and set(r["tickers"]) <= set(names) and json.dumps(r)
        ids = [names.index(t) for t in r["tickers"]]
        assert not any(G.has_edge(a, b) for a in ids for b in ids if a != b)
    assert pf.run_job((date, 0.3, "exact", None, None, None), m, exact_piece_solver, "x")["proven"] is True


def test_primary_methods_finish_every_date_before_the_secondary_ones_start():
    js = pf.jobs(["2005-01-03", "2005-04-01"], (0.3, 0.5))
    order = list(dict.fromkeys(j[2] for j in js))
    assert order[:5] == ["greedy-mindeg", "exact", "gls", pf.S8, "bandit-r"] and set(order[5:]) == {pf.B8, pf.L}
    assert len(js) == 2 * 2 * len(pf.RUNS)
    positions = {m: [i for i, j in enumerate(js) if j[2] == m] for m in order}
    assert max(positions["bandit-r"]) < min(positions[pf.B8]) < min(positions[pf.L])
