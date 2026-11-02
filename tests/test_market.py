import numpy as np
import pandas as pd
import pytest

from learned_selection import market as mk
from learned_selection.market import Market


def build(n=6, days=600, membership=None, seed=0):
    idx = pd.bdate_range("2004-01-01", periods=days)
    rng = np.random.default_rng(seed)
    common = rng.standard_normal((days, 1)) * 0.01
    noise = rng.standard_normal((days, n)) * 0.01
    # A and B move together, the rest apart
    noise[:, 1] = noise[:, 0]
    names = [chr(65 + i) for i in range(n)]
    close = pd.DataFrame(100 * np.exp(np.cumsum(common + noise, axis=0)), index=idx, columns=names)
    return Market(close, membership or [("2004-01-01", names)])


def test_later_prices_do_not_touch_the_graph_and_earlier_ones_do_not_touch_the_holding():
    m = build()
    date, nxt = m.close.index[300], m.close.index[360]
    G, names = m.graph(date, 0.5)
    before = (set(G.edges), names)
    hold = m.holding(date, nxt).copy()

    late = m.close.copy()
    late.iloc[300:] *= np.random.default_rng(1).uniform(0.5, 1.5, late.iloc[300:].shape)
    G2, names2 = Market(late, [("2004-01-01", list(m.close.columns))]).graph(date, 0.5)
    assert (set(G2.edges), names2) == before

    early = m.close.copy()
    early.iloc[:300] *= np.random.default_rng(2).uniform(0.5, 1.5, early.iloc[:300].shape)
    pd.testing.assert_frame_equal(Market(early, [("2004-01-01", list(m.close.columns))]).holding(date, nxt), hold)


def test_holding_starts_the_day_after_the_rebalance_and_ends_on_the_next_one():
    m = build()
    date, nxt = m.close.index[300], m.close.index[360]
    h = m.holding(date, nxt)
    assert h.index[0] == m.close.index[301] and h.index[-1] == nxt and len(h) == 60
    assert h["A"].iloc[0] == pytest.approx(m.close["A"].iloc[301] / m.close["A"].iloc[300] - 1)
    assert m.formation(date).index[-1] == m.close.index[299] and len(m.formation(date)) == 250


def test_rebalance_dates_are_first_trading_days_of_quarters():
    d = build(days=500).rebalance_dates("2004-01-01", "2005-06-30")
    assert [(x.year, x.month) for x in d] == [(2004, 1), (2004, 4), (2004, 7), (2004, 10), (2005, 1), (2005, 4)]
    assert d[1] == pd.Timestamp("2004-04-01")


def test_membership_as_of_a_date():
    m = build(membership=[("2004-01-01", list("ABC")), ("2005-03-01", list("ABD"))])
    assert m.members_at("2005-02-28") == list("ABC")
    assert m.members_at("2005-03-01") == list("ABD")
    with pytest.raises(ValueError):
        m.members_at("2003-12-31")


def test_universe_drops_a_gap_a_late_start_and_a_non_member():
    m = build(membership=[("2004-01-01", list("ABCD"))])
    date = m.close.index[300]
    close = m.close.copy()
    close.iloc[100:290, close.columns.get_loc("C")] = np.nan     # a gap inside the formation
    close.iloc[:295, close.columns.get_loc("D")] = np.nan        # priced only just before the date
    assert Market(close, [("2004-01-01", list("ABCD"))]).universe(date) == ["A", "B"]


def test_a_missing_day_makes_no_return_across_it():
    m = build()
    close = m.close.copy()
    close.iloc[50, 0] = np.nan
    r = Market(close, [("2004-01-01", list(m.close.columns))])
    assert np.isnan(r.log_returns["A"].iloc[50]) and np.isnan(r.log_returns["A"].iloc[51])
    assert np.isnan(r.simple_returns["A"].iloc[51])


def test_a_takeover_ticker_is_dropped_before_its_date_only():
    names = ["T", "A", "B"]
    idx = pd.bdate_range("2004-01-01", periods=800)
    close = pd.DataFrame(np.exp(np.random.default_rng(0).normal(0, 0.01, (800, 3)).cumsum(0)), index=idx, columns=names)
    m = Market(close, [("2004-01-01", names)])
    assert "T" not in m.universe("2005-12-30")
    assert "T" in m.universe("2006-01-02")


def test_graph_joins_the_correlated_pair_and_only_above_theta():
    G, names = build().graph(build().close.index[300], 0.9)
    assert names[:2] == ["A", "B"] and (0, 1) in G.edges


def test_volatility_and_portfolio_returns_against_hand_values():
    daily = pd.Series([0.01, -0.01, 0.02, 0.0])
    assert mk.volatility(daily) == pytest.approx(np.std([0.01, -0.01, 0.02, 0.0], ddof=1) * 252 ** 0.5)
    r = pd.DataFrame({"A": [0.02, np.nan], "B": [0.0, 0.04]})
    assert mk.portfolio_returns(r, ["A", "B"]).tolist() == pytest.approx([0.01, 0.04])
    assert mk.max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
    assert mk.total_return(pd.Series([0.1, -0.5])) == pytest.approx(-0.45)


def test_random_sets_have_the_size_and_repeat_from_the_key():
    names = [str(i) for i in range(50)]
    a = mk.random_sets(names, 7, 3, "k1")
    assert all(len(set(s)) == 7 for s in a)
    assert a == mk.random_sets(names, 7, 3, "k1") and a != mk.random_sets(names, 7, 3, "k2")


def test_labels_read_the_window_each_question_looks_at():
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    close = pd.DataFrame({"A": np.linspace(100, 200, len(idx))}, index=idx)
    m = Market(close, [("2019-01-01", ["A"])])
    d = lambda s: idx[idx.searchsorted(pd.Timestamp(s))]
    assert m.episode_holding(d("2020-01-02"), d("2020-04-01")) == 1 and m.episode_formation(d("2020-04-01")) is None
    assert m.episode_formation(d("2020-07-01")) == 1 and m.episode_holding(d("2020-07-01"), d("2020-10-01")) is None


def test_block_and_episode_intervals_and_the_word():
    flat = mk.block_ci([1.0] * 20)
    assert flat == pytest.approx((1.0, 1.0, 1.0))
    noisy = np.random.default_rng(0).normal(0, 1, 40)
    m, lo, hi = mk.block_ci(noisy)
    assert lo < m < hi and m == pytest.approx(noisy.mean())
    # a block bootstrap widens when neighbours move together
    runs = np.repeat(np.random.default_rng(1).normal(0, 1, 10), 4)
    assert np.subtract(*mk.block_ci(runs)[:0:-1]) > np.subtract(*mk.block_ci(runs, block=1)[:0:-1])
    m, lo, hi = mk.episode_ci([1, 2, 3, 10, 11, 12], [0, 0, 0, 2, 2, 2])
    assert m == pytest.approx(6.5) and 5 < lo < m < hi < 8
    assert mk.word((-1, -2, -0.5)) == "gain" and mk.word((1, 0.5, 2)) == "worse" and mk.word((0, -1, 1)) == "not resolved"
    assert mk.word((-1, -2, -0.5), [-1, -2, -3]) == "gain"
    assert mk.word((-1, -2, -0.5), [-1, -2, 0.1]) == "not resolved"
    assert mk.word((-1, -2, -0.5), below="lower") == "lower"
